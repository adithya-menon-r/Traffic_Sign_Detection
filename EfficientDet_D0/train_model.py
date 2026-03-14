import os
import time
import torch
import pandas as pd
from torch.utils.data import DataLoader
from effdet import create_model_from_config, get_efficientdet_config, DetBenchPredict
from effdet.helpers import load_pretrained
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from config_data import cfg, TrafficDataset, get_transforms, collate_fn

def get_model(device):
    config = get_efficientdet_config(cfg.MODEL_NAME)
    config.image_size = (cfg.IMG_SIZE, cfg.IMG_SIZE)
    config.num_classes = cfg.NUM_CLASSES
    
    model = create_model_from_config(
        config, 
        bench_task='train',
        bench_labeler=True,
        num_classes=cfg.NUM_CLASSES, 
        pretrained=False 
    )
    
    print("Loading COCO Pretrained Weights...")
    load_pretrained(model, config.url, strict=False)

    
    return model.to(device)

def get_predictor(model_train, device):
    return DetBenchPredict(model_train.model).to(device)

def main():
    print(f"Setup Complete. PyTorch Version: {torch.__version__}")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Compute Device: {device}")

    train_ds = TrafficDataset(cfg.IMG_TRAIN, cfg.LBL_TRAIN, transforms=get_transforms(True))
    val_ds = TrafficDataset(cfg.IMG_VAL, cfg.LBL_VAL, transforms=get_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

    print(f"Data Loaded: {len(train_ds)} Train | {len(val_ds)} Val")

    model = get_model(device)
    print("Architecture Ready. Backbone loaded, Head resized.")

    print(f"Starting Training: {cfg.EPOCHS} Epochs | Batch Size {cfg.BATCH_SIZE}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

    history = {'train_loss': [], 'val_map50': []}
    best_map = 0.0

    for epoch in range(cfg.EPOCHS):
        model.train()
        epoch_loss, epoch_start = 0, time.time()
        torch.cuda.empty_cache() 

        for images, targets in train_loader:
            images = torch.stack([img.to(device) for img in images])
            
            batch_size = len(targets)
            max_boxes = max([len(t['boxes']) for t in targets])
            max_boxes = max(max_boxes, 1) 
            
            padded_bboxes = torch.zeros((batch_size, max_boxes, 4), dtype=torch.float32).to(device)
            padded_labels = torch.full((batch_size, max_boxes), -1.0, dtype=torch.float32).to(device)
            
            for i, t in enumerate(targets):
                num_boxes = len(t['boxes'])
                if num_boxes > 0:
                    padded_bboxes[i, :num_boxes] = t['boxes'].to(device)[:, [1, 0, 3, 2]]
                    padded_labels[i, :num_boxes] = t['labels'].to(device).float()
            
            target_res = {
                'bbox': padded_bboxes,
                'cls': padded_labels,
                'img_size': torch.tensor([(cfg.IMG_SIZE, cfg.IMG_SIZE)] * batch_size).to(device),
                'img_scale': torch.tensor([1.0] * batch_size).to(device)
            }
            
            optimizer.zero_grad()
            output = model(images, target_res)
            loss = output['loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        
        predictor = get_predictor(model, device)
        predictor.eval()
        metric = MeanAveragePrecision()
        
        with torch.no_grad():
            for val_imgs, val_tgts in val_loader:
                val_imgs_stack = torch.stack([img.to(device) for img in val_imgs])
                outputs = predictor(val_imgs_stack)
                preds, gts = [], []
                for i in range(len(outputs)):
                    keep = outputs[i][:, 4] > 0.1
                    preds.append({
                        'boxes': outputs[i][keep, :4], 
                        'scores': outputs[i][keep, 4], 
                        'labels': outputs[i][keep, 5].long()
                    })
                    gts.append({
                        'boxes': val_tgts[i]['boxes'].to(device), 
                        'labels': val_tgts[i]['labels'].long().to(device)
                    })
                metric.update(preds, gts)
                
        res_metrics = metric.compute()
        epoch_map50 = res_metrics['map_50'].item()
        history['val_map50'].append(epoch_map50)
        scheduler.step()
        
        if epoch_map50 > best_map:
            best_map = epoch_map50
            torch.save(model.state_dict(), cfg.MODEL_SAVE_PATH)
            print(f"New Best Model Saved (mAP: {best_map:.4f})")
        
        print(f"Epoch {epoch+1:02d}/{cfg.EPOCHS} | Loss: {avg_loss:.4f} | mAP@50: {epoch_map50:.4f} | Time: {time.time()-epoch_start:.1f}s")

    pd.DataFrame(history).to_csv('training_history.csv', index=False)
    
if __name__ == "__main__":
    main()

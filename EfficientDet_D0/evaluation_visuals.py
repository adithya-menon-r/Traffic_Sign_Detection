import os
import torch
import torchvision
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import DataLoader
from config_data import cfg, TrafficDataset, get_transforms, collate_fn
from train_model import get_model, get_predictor

def generate_density_analysis():
    print("Generating Bounding Box Density & Size Analysis...")
    box_counts, box_areas = [], []
    for file in os.listdir(cfg.LBL_TRAIN):
        if not file.endswith('.txt'): continue
        with open(os.path.join(cfg.LBL_TRAIN, file), 'r') as f:
            lines = f.readlines()
            box_counts.append(len(lines))
            for line in lines:
                data = line.strip().split()
                if len(data) >= 5:
                    w, h = float(data[3]), float(data[4])
                    box_areas.append((w * h) * 100)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(box_counts, discrete=True, color='royalblue')
    plt.title("Sign Density (Signs per Image)")
    plt.xlabel("Number of Signs"); plt.ylabel("Frequency")
    
    plt.subplot(1, 2, 2)
    sns.histplot(box_areas, bins=30, color='darkorange')
    plt.title("Sign Size Distribution")
    plt.xlabel("Size (% of Image Area)"); plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("visualization_1_density_analysis.png")
    print("Saved -> visualization_1_density_analysis.png")


def generate_learning_curves():
    print("Generating Learning Curves...")
    if not os.path.exists('training_history.csv'):
        print("Training history not found. Skipping.")
        return

    history_df = pd.read_csv('training_history.csv')

    history_df['epoch'] = history_df.index + 1
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.lineplot(data=history_df, x='epoch', y='train_loss', ax=axes[0], color='red', linewidth=2, marker='o')
    axes[0].set_title("Training Loss over Epochs", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].grid(True, linestyle='--')

    sns.lineplot(data=history_df, x='epoch', y='val_map50', ax=axes[1], color='blue', linewidth=2, marker='o')
    axes[1].set_title("Validation mAP@0.5 over Epochs", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("mAP Score")
    axes[1].set_ylim(0, 1.0); axes[1].grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig("visualization_2_3_learning_curves.png")
    print("Saved -> visualization_2_3_learning_curves.png")

def generate_class_specific_metrics():
    print("Generating Class-Specific mAP & F1-Score Bar Chart...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    val_ds = TrafficDataset(cfg.IMG_VAL, cfg.LBL_VAL, transforms=get_transforms(False))
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = get_model(device)
    if os.path.exists(cfg.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: Model weights not found. Using untrained weights.")

    predictor = get_predictor(model, device)
    predictor.eval()
    metric = MeanAveragePrecision(class_metrics=True)

    with torch.no_grad():
        for images, targets in val_loader:
            outputs = predictor(torch.stack([img.to(device) for img in images]))
            preds, gts = [], []
            for i in range(len(outputs)):
                keep = outputs[i][:, 4] > 0.1
                preds.append({
                    'boxes': outputs[i][keep, :4], 
                    'scores': outputs[i][keep, 4], 
                    'labels': outputs[i][keep, 5].long()
                })
                gts.append({
                    'boxes': targets[i]['boxes'].to(device), 
                    'labels': targets[i]['labels'].long().to(device)
                })
            metric.update(preds, gts)

    results = metric.compute()
    class_names = cfg.CLASSES[1:]
    class_maps = [0.0] * (cfg.NUM_CLASSES - 1)
    class_f1s = [0.0] * (cfg.NUM_CLASSES - 1)

    if 'classes' in results:
        for c_id, map_cls, mar_cls in zip(results['classes'], results['map_per_class'], results['mar_100_per_class']):
            idx = int(c_id.item()) - 1
            if 0 <= idx < len(class_names):
                p = map_cls.item() if map_cls.item() >= 0 else 0.0
                r = mar_cls.item() if mar_cls.item() >= 0 else 0.0
                class_maps[idx] = p
                class_f1s[idx] = 2 * (p * r) / (p + r + 1e-6)
    else:
        for i, (map_cls, mar_cls) in enumerate(zip(results['map_per_class'], results['mar_100_per_class'])):
            if i < len(class_names):
                p = map_cls.item() if map_cls.item() >= 0 else 0.0
                r = mar_cls.item() if mar_cls.item() >= 0 else 0.0
                class_maps[i] = p
                class_f1s[i] = 2 * (p * r) / (p + r + 1e-6)

    class_names.append("ALL CLASSES")
    overall_map = results['map_50'].item()
    overall_mar = results['mar_100'].item()
    class_maps.append(overall_map)
    class_f1s.append(2 * (overall_map * overall_mar) / (overall_map + overall_mar + 1e-6))

    x = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, class_maps, width, label='mAP@0.5', color='darkorange')
    rects2 = ax.bar(x + width/2, class_f1s, width, label='F1-Score', color='royalblue')

    ax.set_ylabel('Scores', fontweight='bold')
    ax.set_title('Final Detection Accuracy (mAP@0.5 & F1-Score) by Sign Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1.1)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("visualization_4_class_metrics.png")
    print("Saved -> visualization_4_class_metrics.png")

def print_yolo_summary():
    print("\nGenerating YOLO-style Evaluation Summary Table...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    val_ds = TrafficDataset(cfg.IMG_VAL, cfg.LBL_VAL, transforms=get_transforms(False))
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = get_model(device)
    if os.path.exists(cfg.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location=device))
    
    predictor = get_predictor(model, device)
    predictor.eval()
    
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    metric_50 = MeanAveragePrecision(iou_type="bbox", class_metrics=True, iou_thresholds=[0.5])
    
    tp_per_class = {c: 0 for c in range(1, cfg.NUM_CLASSES)}
    fp_per_class = {c: 0 for c in range(1, cfg.NUM_CLASSES)}
    fn_per_class = {c: 0 for c in range(1, cfg.NUM_CLASSES)}
    images_per_class = {c: set() for c in range(1, cfg.NUM_CLASSES)}
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            outputs = predictor(torch.stack([img.to(device) for img in images]))
            preds, gts = [], []
            for i in range(len(outputs)):
                img_global_idx = batch_idx * cfg.BATCH_SIZE + i
                
                pred_boxes = outputs[i][:, :4]
                pred_scores = outputs[i][:, 4]
                pred_labels = outputs[i][:, 5].long()
                
                gt_boxes = targets[i]['boxes'].to(device)
                gt_labels = targets[i]['labels'].long().to(device)
                
                keep = pred_scores > 0.001
                preds.append({'boxes': pred_boxes[keep], 'scores': pred_scores[keep], 'labels': pred_labels[keep]})
                gts.append({'boxes': gt_boxes, 'labels': gt_labels})
                
                for lbl in gt_labels.unique():
                    if lbl.item() in images_per_class:
                        images_per_class[lbl.item()].add(img_global_idx)
                
                keep_50 = pred_scores >= 0.5
                p_boxes = pred_boxes[keep_50]
                p_labels = pred_labels[keep_50]
                p_scores = pred_scores[keep_50]
                
                for c in range(1, cfg.NUM_CLASSES):
                    c_p_boxes = p_boxes[p_labels == c]
                    c_p_scores = p_scores[p_labels == c]
                    c_gt_boxes = gt_boxes[gt_labels == c]
                    
                    fn_per_class[c] += len(c_gt_boxes)
                    if len(c_p_boxes) == 0: continue
                    if len(c_gt_boxes) == 0:
                        fp_per_class[c] += len(c_p_boxes)
                        continue
                        
                    ious = torchvision.ops.box_iou(c_p_boxes, c_gt_boxes)
                    matched_gt = set()
                    indices = torch.argsort(c_p_scores, descending=True)
                    c_p_boxes = c_p_boxes[indices]
                    ious = ious[indices]
                    
                    for i_p in range(len(c_p_boxes)):
                        best_iou = 0
                        best_gt_idx = -1
                        for j_g in range(len(c_gt_boxes)):
                            if j_g not in matched_gt and ious[i_p, j_g] > best_iou:
                                best_iou = ious[i_p, j_g]
                                best_gt_idx = j_g
                        if best_iou >= 0.5:
                            tp_per_class[c] += 1
                            matched_gt.add(best_gt_idx)
                        else:
                            fp_per_class[c] += 1
                            
                    fn_per_class[c] -= len(matched_gt)
            metric.update(preds, gts)
            metric_50.update(preds, gts)
            
    results = metric.compute()
    results_50 = metric_50.compute()
    
    map_all = results['map'].item()
    map50_all = results['map_50'].item()
    
    tp_all = sum(tp_per_class.values())
    fp_all = sum(fp_per_class.values())
    fn_all = sum(fn_per_class.values())
    
    p_all = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0.0
    r_all = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else 0.0
    
    instances_all = tp_all + fn_all
    images_all = len(val_ds)
    
    map_classes = results['classes'].tolist() if 'classes' in results else []
    map_per_class_vals = results['map_per_class'].tolist() if 'map_per_class' in results else []
    map50_per_class_vals = results_50['map_per_class'].tolist() if 'map_per_class' in results_50 else []
    
    map_dict = {int(c): v for c, v in zip(map_classes, map_per_class_vals)}
    map50_dict = {int(c): v for c, v in zip(map_classes, map50_per_class_vals)}
    if not map_dict:
        for i, (map_cls, map50_cls) in enumerate(zip(results.get('map_per_class', []), results_50.get('map_per_class', []))):
            map_dict[i+1] = map_cls.item()
            map50_dict[i+1] = map50_cls.item()
            
    table_data = []
    table_data.append(["all", images_all, instances_all, f"{p_all:.3f}", f"{r_all:.3f}", f"{map50_all:.3f}", f"{map_all:.3f}"])
    
    for c in range(1, cfg.NUM_CLASSES):
        tp = tp_per_class[c]
        fp = fp_per_class[c]
        fn = fn_per_class[c]
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        instances = tp + fn
        images_c = len(images_per_class[c])
        class_name = cfg.CLASSES[c]
        
        mAP50 = map50_dict.get(c, 0.0)
        mAP = map_dict.get(c, 0.0)
        
        table_data.append([class_name.lower().replace(' ', ''), images_c, instances, f"{p:.3f}", f"{r:.3f}", f"{mAP50:.3f}", f"{mAP:.3f}"])
        
    headers = ["Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95"]
    
    try:
        from tabulate import tabulate
        print("\n" + tabulate(table_data, headers=headers, tablefmt="plain"))
    except ImportError:
        row_format = "{:<13}  {:>8}  {:>9}  {:>8}  {:>8}  {:>8}  {:>8}"
        print("\n" + row_format.format(*headers))
        for row in table_data:
            print(row_format.format(*row))

if __name__ == "__main__":
    generate_density_analysis()
    generate_learning_curves()
    generate_class_specific_metrics()
    print_yolo_summary()

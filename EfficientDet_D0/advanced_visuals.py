import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import xml.etree.ElementTree as ET
from pathlib import Path
import kagglehub
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torchvision
try:
    from tabulate import tabulate
except ImportError:
    pass

from train_model import get_model, get_predictor
from config_data import cfg, collate_fn

# --- Weather Degradation Evaluation ---
def generate_weather_degradation_chart():
    print("Downloading Clean Kaggle Dataset for Weather Testing...")
    
    # Hide output stream if preferred or keep defaults
    kaggle_path = kagglehub.dataset_download("andrewmvd/road-sign-detection")
    
    label_map = {'crosswalk': 1, 'trafficlight': 2, 'stop': 3, 'speedlimit': 4}

    class KaggleWeatherDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.image_files = list((Path(root_dir) / 'images').glob('*.png'))[:150] 
            self.anno_dir = Path(root_dir) / 'annotations'
            self.transform = transform

        def __getitem__(self, idx):
            img_name = self.image_files[idx].stem
            img = cv2.cvtColor(cv2.imread(str(self.image_files[idx])), cv2.COLOR_BGR2RGB)

            boxes, labels = [], []
            anno_path = self.anno_dir / f"{img_name}.xml"
            if anno_path.exists():
                for obj in ET.parse(anno_path).getroot().findall('object'):
                    name = obj.find('name').text
                    if name in label_map:
                        bndbox = obj.find('bndbox')
                        boxes.append([float(bndbox.find(k).text) for k in ['xmin', 'ymin', 'xmax', 'ymax']])
                        labels.append(label_map[name])

            boxes = np.array(boxes, dtype=np.float32) if boxes else np.empty((0, 4), dtype=np.float32)
            labels = np.array(labels, dtype=np.int64) if labels else np.empty((0,), dtype=np.int64)

            if self.transform:
                aug = self.transform(image=img, bboxes=boxes, class_labels=labels)
                img = aug['image'].float()
                boxes = torch.as_tensor(aug['bboxes'], dtype=torch.float32) if len(aug['bboxes'])>0 else torch.zeros((0,4))
                labels = torch.as_tensor(aug['class_labels'], dtype=torch.float32) if len(aug['class_labels'])>0 else torch.zeros((0,))

            return img, {'boxes': boxes, 'labels': labels}
            
        def __len__(self): 
            return len(self.image_files)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model(device)
    if os.path.exists(cfg.MODEL_SAVE_PATH):
        try:
            model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location=device), strict=False)
        except Exception as e:
            pass

    def eval_weather(transform, name):
        print(f"\nEvaluating {name}...")
        loader = DataLoader(KaggleWeatherDataset(kaggle_path, transform), batch_size=8, collate_fn=collate_fn, num_workers=0)
        pred = get_predictor(model, device)
        pred.eval()
        metric = MeanAveragePrecision()
        
        tp_per_class = {c: 0 for c in range(1, 5)}
        fp_per_class = {c: 0 for c in range(1, 5)}
        fn_per_class = {c: 0 for c in range(1, 5)}
        
        with torch.no_grad():
            for imgs, tgts in loader:
                outputs = pred(torch.stack([i.to(device) for i in imgs]))
                preds, gts = [], []
                for i, out in enumerate(outputs):
                    keep = out[:, 4] > 0.001
                    preds.append({'boxes': out[keep, :4], 'scores': out[keep, 4], 'labels': out[keep, 5].long()})
                    gts.append({'boxes': tgts[i]['boxes'].to(device), 'labels': tgts[i]['labels'].long().to(device)})
                    
                    keep_50 = out[:, 4] >= 0.5
                    p_boxes = out[keep_50, :4]
                    p_scores = out[keep_50, 4]
                    p_labels = out[keep_50, 5].long()
                    
                    gt_boxes = tgts[i]['boxes'].to(device)
                    gt_labels = tgts[i]['labels'].long().to(device)
                    
                    for c in range(1, 5):
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
                
        res = metric.compute()
        p_all_map = res['map_50'].item()
        r_all_map = res['mar_100'].item()
        overall_f1 = 2 * (p_all_map * r_all_map) / (p_all_map + r_all_map + 1e-6) if (p_all_map+r_all_map)>0 else 0
        
        class_f1s = {}
        class_names_local = {1: 'Crosswalk', 2: 'Trafficlight', 3: 'Stop', 4: 'Speedlimit'}
        
        for c in range(1, 5):
            tp, fp, fn = tp_per_class[c], fp_per_class[c], fn_per_class[c]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
            class_f1s[class_names_local[c]] = f1
            
        return p_all_map, overall_f1, class_f1s

    base_aug = [
        A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE), 
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
    conds = {
        "Clear Weather": A.Compose(base_aug, bbox_params=A.BboxParams('pascal_voc', ['class_labels'])),
        "Light Fog": A.Compose([A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0)] + base_aug, bbox_params=A.BboxParams('pascal_voc', ['class_labels'])),
        "Heavy Fog": A.Compose([A.RandomFog(fog_coef_lower=0.7, fog_coef_upper=0.9, p=1.0)] + base_aug, bbox_params=A.BboxParams('pascal_voc', ['class_labels'])),
        "Light Rain": A.Compose([A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, blur_value=3, rain_type='drizzle', p=1.0)] + base_aug, bbox_params=A.BboxParams('pascal_voc', ['class_labels'])),
        "Heavy Rain": A.Compose([A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=2, blur_value=5, rain_type='heavy', p=1.0)] + base_aug, bbox_params=A.BboxParams('pascal_voc', ['class_labels']))
    }

    perf = {c: eval_weather(t, c) for c, t in conds.items()}
    
    print("\n" + "="*60)
    print(f"{'FCOS Weather Robustness Detailed F1-Scores':^60}")
    print("="*60)
    
    for cond_name, (map50, overall_f1, class_f1s) in perf.items():
        print(f"\n{cond_name.upper()} WEATHER:")
        print(f"Overall mAP@50: {map50:.3f} | Overall F1: {overall_f1:.3f}")
        for cls_name, f1_val in class_f1s.items():
            print(f"  - {cls_name:<15}: {f1_val:.3f}")

    print("\nGenerating Weather Degradation Chart...")
    x = np.arange(len(conds))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    r1 = ax.bar(x - w/2, [v[1] for v in perf.values()], w, label='F1-Score', color='royalblue')
    r2 = ax.bar(x + w/2, [v[0] for v in perf.values()], w, label='mAP@0.5', color='darkorange')
    ax.set_ylabel('Score')
    ax.set_title('EfficientDet Performance vs Weather', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(conds.keys())
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    for r in r1+r2: 
        ax.annotate(f'{r.get_height():.2f}', (r.get_x() + r.get_width()/2, r.get_height()), ha='center', va='bottom')
        
    plt.tight_layout()
    plt.savefig("visualization_5_weather_degradation.png")
    print("Saved -> visualization_5_weather_degradation.png")


if __name__ == "__main__":
    generate_weather_degradation_chart()
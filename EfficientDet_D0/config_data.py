import os
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

class Config:
    DATA_DIR = "Traffic_Project_Data/dataset_weather"
    IMG_TRAIN = f"{DATA_DIR}/images/train"
    LBL_TRAIN = f"{DATA_DIR}/labels/train"
    IMG_VAL = f"{DATA_DIR}/images/val"
    LBL_VAL = f"{DATA_DIR}/labels/val"
    
    MODEL_SAVE_PATH = 'efficientdet_d0_best.pth'
    MODEL_NAME = 'tf_efficientdet_d0'
    IMG_SIZE = 512
    BATCH_SIZE = 8
    EPOCHS = 20      
    LEARNING_RATE = 0.0003
    
    CLASSES = ["Background", "Crosswalk", "Speed Limit", "Stop Sign", "Traffic Light"]
    NUM_CLASSES = len(CLASSES)

cfg = Config()

class TrafficDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, transforms=None):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.transforms = transforms
        self.imgs = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img.shape[:2]

        label_path = os.path.join(self.lbl_dir, self.imgs[idx].rsplit('.', 1)[0] + '.txt')
        boxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    data = list(map(float, line.strip().split()))
                    if len(data) < 5: continue
                    c, cx, cy, w, h = data
                    cls_id = int(c) + 1
                    if cls_id >= cfg.NUM_CLASSES: continue
                    
                    x_min, y_min = max(0, (cx - w/2) * w_orig), max(0, (cy - h/2) * h_orig)
                    x_max, y_max = min(w_orig, (cx + w/2) * w_orig), min(h_orig, (cy + h/2) * h_orig)
                    if x_max > x_min and y_max > y_min:
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(cls_id)
        
        if self.transforms:
            aug = self.transforms(image=img, bboxes=boxes, labels=labels)
            img_tensor = aug['image'].float() 
            boxes, labels = aug['bboxes'], aug['labels']
            
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros((0, 4)),
            'labels': torch.as_tensor(labels, dtype=torch.float32) if len(labels) > 0 else torch.zeros((0,)),
            'img_scale': torch.tensor([1.0]),
            'img_size': torch.tensor([cfg.IMG_SIZE, cfg.IMG_SIZE])
        }
        return img_tensor, target

    def __len__(self): return len(self.imgs)

def collate_fn(batch): return tuple(zip(*batch))

def get_transforms(train=True):
    if train:
        return A.Compose([
            A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE), 
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.4),
            A.Blur(blur_limit=3, p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        return A.Compose([
            A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE), 
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


import albumentations as A
import cv2
import os
import yaml
import shutil
from ultralytics import YOLO

MODEL_PATH = os.path.join("runs", "detect", "yolo_fog_model_v3", "weights", "best.pt")

VAL_IMAGES_DIR = os.path.join(os.getcwd(), "dataset_weather", "images", "val")
VAL_LABELS_DIR = os.path.join(os.getcwd(), "dataset_weather", "labels", "val")

DENSITIES = {
    "clear_weather": A.Compose([]),
    "light_fog": A.Compose([
        A.RandomFog(fog_coef_range=(0.1, 0.3), alpha_coef=0.08, p=1.0)
    ]),
    "heavy_fog": A.Compose([
        A.RandomFog(fog_coef_range=(0.7, 0.9), alpha_coef=0.08, p=1.0)
    ]),
    "light_rain": A.Compose([
        A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, blur_value=3, rain_type='drizzle', p=1.0)
    ]),
    "heavy_rain": A.Compose([
        A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=2, blur_value=5, rain_type='heavy', p=1.0)
    ])
}

def create_density_dataset(density_name, transform):
    print(f"\nGenerating Fixed Subset for: {density_name.upper()}")
    
    new_img_dir = os.path.join(os.getcwd(), f"dataset_{density_name}", "images", "val")
    new_lbl_dir = os.path.join(os.getcwd(), f"dataset_{density_name}", "labels", "val")
    os.makedirs(new_img_dir, exist_ok=True)
    os.makedirs(new_lbl_dir, exist_ok=True)

    for img_name in os.listdir(VAL_IMAGES_DIR):
        if not img_name.endswith(('.png', '.jpg', '.jpeg')): continue
        
        img_path = os.path.join(VAL_IMAGES_DIR, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = transform(image=image)
        aug_image = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(os.path.join(new_img_dir, img_name), aug_image)
        
        lbl_name = img_name.rsplit('.', 1)[0] + '.txt'
        src_lbl = os.path.join(VAL_LABELS_DIR, lbl_name)
        if os.path.exists(src_lbl):
            shutil.copy(src_lbl, os.path.join(new_lbl_dir, lbl_name))

    yaml_path = f"{density_name}_config.yaml"
    yaml_content = {
        'path': os.path.join(os.getcwd(), f"dataset_{density_name}"),
        'train': 'images/val', 
        'val': 'images/val',
        'names': {0: 'crosswalk', 1: 'speedlimit', 2: 'stop', 3: 'trafficlight'}
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f)
        
    return yaml_path

def main():
    model = YOLO(MODEL_PATH)
    results_map = {}

    for density_name, transform in DENSITIES.items():
        yaml_config = create_density_dataset(density_name, transform)
        
        print(f"Testing Model against {density_name.upper()}...")
        metrics = model.val(data=yaml_config, split='val', verbose=False)
        
        map50 = metrics.box.map50
        results_map[density_name] = map50
        print(f"{density_name.upper()} Accuracy (mAP50): {map50 * 100:.2f}%")

    print("\n" + "="*40)
    print("FINAL DENSITY ANALYSIS RESULTS")
    print("="*40)
    for condition, score in results_map.items():
        print(f"{condition.replace('_', ' ').title().ljust(15)}: {score * 100:.2f}% mAP")
    print("="*40)

if __name__ == "__main__":
    main()
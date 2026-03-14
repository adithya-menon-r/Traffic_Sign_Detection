from ultralytics import YOLO
import os
import yaml
import time 

DATASET_DIR = os.path.join(os.getcwd(), "dataset_weather")
MODEL_NAME = "yolov8n.pt"

def create_yaml_config():
    yaml_content = {'path': DATASET_DIR, 'train': 'images/train', 'val': 'images/val', 'names': {0: 'traffic_sign'}}
    existing_yaml = os.path.join(DATASET_DIR, "data.yaml")
    if os.path.exists(existing_yaml):
        with open(existing_yaml, 'r') as f: data = yaml.safe_load(f)
        data['path'] = DATASET_DIR
        data['train'] = 'images/train'
        data['val'] = 'images/val'
        config_path = "yolo_config.yaml"
        with open(config_path, 'w') as f: yaml.dump(data, f)
        return config_path
    config_path = "yolo_config.yaml"
    with open(config_path, 'w') as f: yaml.dump(yaml_content, f)
    return config_path

def main():
    print("Setting up YOLOv8 Retraining (Batch Size 8)...")
    yaml_file = create_yaml_config()
    model = YOLO(MODEL_NAME)

    print("Starting FAST Training on RTX 3050 GPU...")
    
    start_time = time.time()
    
    results = model.train(
        data=yaml_file,
        epochs=20,         
        imgsz=640,
        batch=8,
        name="yolo_official_batch8",
        device=0           
    )

    end_time = time.time()
    total_time = (end_time - start_time) / 60
    
    print(f"\nTraining Complete in {total_time:.2f} minutes!")
    print(f"Official Batch 8 results saved in 'runs/detect/yolo_official_batch8'")

if __name__ == "__main__":
    main()
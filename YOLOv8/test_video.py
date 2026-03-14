from ultralytics import YOLO
import os

MODEL_PATH = os.path.join("runs", "detect", "yolo_fog_model_v3", "weights", "best.pt")

VIDEO_PATH = "all_classes_test.mp4"

def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Could not find '{VIDEO_PATH}'. Please ensure it is in your project folder.")
        return

    print("Loading Model for Video Inference...")
    model = YOLO(MODEL_PATH)

    print("Processing Video... (Watch your screen!)")
    
    results = model.predict(
        source=VIDEO_PATH,
        conf=0.5,
        save=True,
        show=True
    )

    print("\nSUCCESS!")
    print("Go to your 'runs/detect' folder and look for the newest 'predict' folder.")
    print("Your final video with bounding boxes is saved there (usually as an .avi file).")

if __name__ == "__main__":
    main()
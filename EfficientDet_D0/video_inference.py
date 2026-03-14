import os
import cv2
import time
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from train_model import get_model, get_predictor
from config_data import cfg

VIDEO_INPUT = "test_vid.mp4" 
VIDEO_OUTPUT = "output_inference.mp4"

def run_video_inference():
    if not os.path.exists(VIDEO_INPUT):
        print(f"Warning: {VIDEO_INPUT} not found. Please add a video to test.")
        return

    print(f"Processing {VIDEO_INPUT}...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    eval_model = get_model(device)
    if os.path.exists(cfg.MODEL_SAVE_PATH):
        eval_model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: Model weights not found. Using untrained weights for dummy inference.")

    video_predictor = get_predictor(eval_model, device)
    video_predictor.eval()

    cap = cv2.VideoCapture(VIDEO_INPUT)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(VIDEO_OUTPUT, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    total_time, frame_count = 0, 0
    transform = A.Compose([
        A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE), 
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            img_tensor = transform(image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))['image'].unsqueeze(0).float().to(device)
            
            start_infer = time.time()
            outputs = video_predictor(img_tensor)
            infer_ms = (time.time() - start_infer) * 1000
            total_time += infer_ms
            frame_count += 1
            
            keep = outputs[0][:, 4] > 0.35
            colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 165, 0), (255, 0, 255)]
            for box, score, lbl in zip(outputs[0][keep, :4], outputs[0][keep, 4], outputs[0][keep, 5]):
                x1, y1, x2, y2 = box.cpu().numpy()
                x1, x2 = int(x1 * (width/cfg.IMG_SIZE)), int(x2 * (width/cfg.IMG_SIZE))
                y1, y2 = int(y1 * (height/cfg.IMG_SIZE)), int(y2 * (height/cfg.IMG_SIZE))
                c_idx = int(lbl) if int(lbl) < len(colors) else 0
                box_color = colors[c_idx]
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                label_text = f"{cfg.CLASSES[int(lbl)]} {score:.2f}"
                cv2.putText(frame, label_text, (x1, max(y1-10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
                
            cv2.putText(frame, f"Infer: {infer_ms:.1f}ms | {1000/infer_ms:.1f} FPS", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            out.write(frame)

    cap.release()
    out.release()
    if frame_count > 0:
        print(f"SPEED: {total_time/frame_count:.2f} ms/frame ({1000/(total_time/frame_count):.1f} FPS)")
    print(f"Video saved as: {VIDEO_OUTPUT}")

if __name__ == "__main__":
    run_video_inference()

import cv2
import numpy as np
from ultralytics import YOLO

class FireDetector:
    def __init__(self, model_path='yolov8n.pt'):
        print(f"[AI] Đang khởi tạo mạng Neural Network từ {model_path}...")
        self.model = YOLO(model_path)
        
    def get_fire_pixel(self, frame):
        results = self.model(frame, verbose=False)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                u = (x1 + x2) / 2.0
                v = y2 
                return (u, v) 
        return None
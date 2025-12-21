from ultralytics import YOLO
import cv2
import numpy as np

class VehicleDetector:
    def __init__(self, model_path="yolov8n.pt"):
        # Load the YOLOv8 model
        # Using 'yolov8n.pt' will automatically download it if not present
        self.model = YOLO(model_path)
        
        # COCO class IDs for vehicles: 
        # 2: car, 3: motorcycle, 5: bus, 7: truck
        self.vehicle_classes = [2, 3, 5, 7]

    def detect(self, frame):
        """
        Detects vehicles in the frame.
        Returns a list of detections: [x1, y1, x2, y2, class_id, confidence]
        """
        # Enable tracking. persist=True keeps tracks across frames.
        # Tracker configuration can be customized (e.g., tracker="bytetrack.yaml")
        # imgsz=1280 improves detection of small/distant objects (inference will be slower)
        results = self.model.track(frame, persist=True, verbose=False, conf=0.1, imgsz=1280) 
        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id in self.vehicle_classes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    
                    # Extract Track ID if available
                    track_id = -1
                    if box.id is not None:
                        track_id = int(box.id[0])
                    
                    detections.append([int(x1), int(y1), int(x2), int(y2), track_id, cls_id, conf])
        
        return detections

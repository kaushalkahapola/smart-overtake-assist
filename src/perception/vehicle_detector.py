from ultralytics import YOLO
import torch

class VehicleDetector:
    def __init__(self, model_path):
        # Load the model
        self.model = YOLO(model_path)
        
        # Frame Skipping State
        self.frame_count = 0
        self.skip_interval = 3  # Run detection every 3rd frame
        self.last_detections = [] # Cache

    def detect(self, frame):
        """
        Detects vehicles. 
        Optimization: Runs inference only every 'skip_interval' frames.
        Returns cached detections for skipped frames.
        """
        self.frame_count += 1
        
        # 1. Check if we should skip this frame
        if (self.frame_count % self.skip_interval != 0) and (self.last_detections is not None):
            return self.last_detections

        # 2. Run Heavy Inference (Only if not skipped)
        # ensure imgsz=640 for speed
        results = self.model.track(frame, persist=True, verbose=False, conf=0.1, imgsz=640) 
        
        detections = []
        
        if results and results[0].boxes:
            for box in results[0].boxes:
                # Extract coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Get Track ID (if available)
                track_id = int(box.id.item()) if box.id is not None else -1
                
                # Filter for vehicles (car, bus, truck, motorcycle)
                cls_id = int(box.cls.item())
                if cls_id in [2, 3, 5, 7]: 
                    detections.append([int(x1), int(y1), int(x2), int(y2), track_id])
        
        # 3. Update Cache
        self.last_detections = detections
        return detections
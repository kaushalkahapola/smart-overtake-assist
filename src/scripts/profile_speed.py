import time
import cv2
import sys
import os
import numpy as np

# Adjust path to import modules if running from src/scripts directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perception.vehicle_detector import VehicleDetector
from perception.lane_detector import LaneDetector
import config

def profile():
    print("--- SPEED PROFILER ---")
    
    # 1. Setup
    # Force Nano model
    detector = VehicleDetector(config.YOLO_MODEL_PATH) 
    lane_detector = LaneDetector()
    
    # Load video
    video_path = config.TEST_VIDEO_PATH
    cap = cv2.VideoCapture(video_path)
    
    # Warmup
    print("Warming up...")
    for _ in range(10): cap.read()
    
    yolo_times = []
    lane_times = []
    
    print("Profiling 100 frames...")
    count = 0
    while count < 100:
        ret, frame = cap.read()
        if not ret: break
        
        # Resize to benchmark resolution
        frame = cv2.resize(frame, (640, 360))
        
        # TIME YOLO
        t0 = time.perf_counter()
        detector.detect(frame)
        t1 = time.perf_counter()
        yolo_times.append((t1-t0)*1000)
        
        # TIME LANE DETECTION
        t0 = time.perf_counter()
        # Note: We run detect() which does Canny + Hough + Python Loops
        lane_detector.detect(frame)
        t1 = time.perf_counter()
        lane_times.append((t1-t0)*1000)
        
        count += 1
        print(f"Frame {count}: YOLO={yolo_times[-1]:.1f}ms | Lanes={lane_times[-1]:.1f}ms", end='\r')

    print("\n" + "="*30)
    print(f"AVG YOLO TIME: {np.mean(yolo_times):.2f} ms")
    print(f"AVG LANE TIME: {np.mean(lane_times):.2f} ms")
    print("="*30)

if __name__ == "__main__":
    profile()
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Adjust path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from perception.vehicle_detector import VehicleDetector
from perception.lane_detector import LaneDetector
from logic.safety_checker import SafetyChecker
import config

def benchmark_system():
    print("Initializing System for Benchmarking...")
    
    # Force load the model first to avoid counting load time
    vehicle_detector = VehicleDetector(config.YOLO_MODEL_PATH)
    lane_detector = LaneDetector()
    safety_checker = SafetyChecker()
    
    # Load Video
    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(current_dir, "..", "assets", "videos", "test4.mp4")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Video not found.")
        return

    print("Starting Benchmark Loop...")
    
    processing_times = []
    frame_indices = []
    frame_count = 0
    
    # Warmup
    for _ in range(10):
        cap.read()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # --- OPTIMIZATION: Process at 640px width ---
        # 1280x720 is too heavy for CPU Lane Detection.
        # 640x360 is standard for real-time CPU apps.
        processing_frame = cv2.resize(frame, (640, 360))
        
        # --- START TIMER ---
        start_time = time.perf_counter()
        
        # 1. Detect Vehicles (YOLOv8n)
        detections = vehicle_detector.detect(processing_frame)
        
        # 2. Detect Lanes (Hough)
        # We ignore the 3rd return value (debug image) to save time
        left, right, _ = lane_detector.detect(processing_frame)
        
        # 3. Safety Logic
        safety_checker.assess(detections, (left, right))
        
        # --- END TIMER ---
        end_time = time.perf_counter()
        
        duration_ms = (end_time - start_time) * 1000
        processing_times.append(duration_ms)
        frame_indices.append(frame_count)
        
        if frame_count % 100 == 0:
            print(f"Frame {frame_count}: {duration_ms:.1f}ms ({(1000/duration_ms):.1f} FPS)")

        # Limit to 1000 frames for graph (enough for proof)
        if frame_count >= 1000:
            break

    cap.release()
    
    # 4. Generate Graph
    print("Generating Graph...")
    avg_ms = np.mean(processing_times)
    avg_fps = 1000 / avg_ms
    
    print(f"\nFINAL RESULT: {avg_fps:.2f} FPS")
    
    if avg_fps < 10:
        print("WARNING: Still under 10 FPS. Ensure you are using 'yolov8n.pt'")
    else:
        print("SUCCESS: Real-time performance achieved.")

    plt.figure(figsize=(10, 5))
    plt.plot(frame_indices, processing_times, color='#2c3e50', linewidth=1, alpha=0.8, label='Frame Latency')
    plt.axhline(y=avg_ms, color='#e74c3c', linestyle='--', linewidth=2, label=f'Average: {avg_ms:.1f}ms ({avg_fps:.1f} FPS)')
    
    plt.title('System Processing Latency (Optimized)', fontsize=14, fontweight='bold')
    plt.xlabel('Frame Sequence', fontsize=12)
    plt.ylabel('Processing Time (ms)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig("fps_analysis_graph.png", dpi=300)
    print("Graph saved to: fps_analysis_graph.png")

if __name__ == "__main__":
    benchmark_system()
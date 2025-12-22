import cv2
import sys
import numpy as np
import time  # Added time module

# Adjust path to import modules if running from src directory
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from perception.vehicle_detector import VehicleDetector
from perception.lane_detector import LaneDetector
from logic.safety_checker import SafetyChecker
from utils.visualization import draw_results
import config

def main():
    print("Initializing Smart Overtaking Assistant...")
    
    # Initialize Modules
    try:
        vehicle_detector = VehicleDetector(config.YOLO_MODEL_PATH)
        lane_detector = LaneDetector()
        safety_checker = SafetyChecker()
        print("Modules loaded successfully.")
    except Exception as e:
        print(f"Error loading modules: {e}")
        return

    # Video Source
    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(current_dir, "..", "assets", "videos", "test4.mp4")
    video_path = os.path.abspath(video_path)

    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        print(f"Processing video: {video_path}")
    else:
        print(f"Test video not found at: {video_path}")
        print("Attempting to open Webcam (0)...")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("System running... Press 'q' to stop and calculate Average FPS.")

    # --- FPS Variables ---
    frame_count = 0
    start_time = time.time()
    # ---------------------

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break # Stop loop at end of video for accurate calculation

        # Resize
        frame = cv2.resize(frame, (1280, 720))
        frame_count += 1

        # 1. Perception
        detections = vehicle_detector.detect(frame)
        left_line, right_line, debug_view = lane_detector.detect(frame)
        
        lane_info = (left_line, right_line)
        
        # 2. Risk Assessment
        status, divider = safety_checker.assess(detections, lane_info)

        # 3. Visualization
        output_frame = draw_results(frame, detections, lane_info, status, divider)

        # Display
        cv2.imshow("Smart Overtaking Assistant", output_frame)
        cv2.imshow("Debug: Lane Detection", debug_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Final FPS Calculation ---
    end_time = time.time()
    total_time = end_time - start_time
    
    if total_time > 0:
        avg_fps = frame_count / total_time
        print("\n" + "="*40)
        print(f"  Total Frames Processed: {frame_count}")
        print(f"  Total Time Elapsed:     {total_time:.2f} seconds")
        print(f"  AVERAGE FPS:            {avg_fps:.2f}")
        print("="*40 + "\n")
    # -----------------------------

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
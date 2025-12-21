import cv2
import sys
import numpy as np

# Adjust path to import modules if running from src directory
import sys
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
    # Resolve path relative to this script file to avoid CWD issues
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to 'application' (from 'src'), then into 'assets/videos'
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

    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Resize for consistent processing speed (optional)
        frame = cv2.resize(frame, (1280, 720))

        # 1. Perception
        # 1. Perception
        detections = vehicle_detector.detect(frame)
        left_line, right_line, debug_view = lane_detector.detect(frame)
        
        # Pack for safety check
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

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

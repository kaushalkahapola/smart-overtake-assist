import cv2
import sys
import numpy as np
import time
import os
import json

# Ensure the root directory and src are in path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import config
from perception.vehicle_detector import VehicleDetector
from perception.lane_detector import LaneDetector
from logic.safety_checker import SafetyChecker
from utils.visualization import draw_results

def apply_tuning_defaults(lane_detector, params):
    # Apply JSON parameters directly to the Lane Tuning trackbars
    hough_params = params.get("hough", {})
    color_params = params.get("color", {})
    roi_params = params.get("roi", {})
    
    cv2.setTrackbarPos('Canny Low', 'Lane Tuning', hough_params.get('canny_low', 50))
    cv2.setTrackbarPos('Canny High', 'Lane Tuning', hough_params.get('canny_high', 150))
    cv2.setTrackbarPos('Threshold', 'Lane Tuning', hough_params.get('threshold', 50))
    cv2.setTrackbarPos('Min Len', 'Lane Tuning', hough_params.get('min_len', 100))
    cv2.setTrackbarPos('Max Gap', 'Lane Tuning', hough_params.get('max_gap', 50))
    
    cv2.setTrackbarPos('White L', 'Lane Tuning', color_params.get('white_l_min', 200))
    cv2.setTrackbarPos('Yel H Min', 'Lane Tuning', color_params.get('yellow_h_min', 15))
    cv2.setTrackbarPos('Yel H Max', 'Lane Tuning', color_params.get('yellow_h_max', 35))
    cv2.setTrackbarPos('Yel S Min', 'Lane Tuning', color_params.get('yellow_s_min', 100))
    
    cv2.setTrackbarPos('ROI Top W', 'Lane Tuning', roi_params.get('top_w', 200))
    cv2.setTrackbarPos('ROI Bot W', 'Lane Tuning', roi_params.get('bot_w', 600))
    cv2.setTrackbarPos('ROI H %', 'Lane Tuning', roi_params.get('h_pct', 40))
    cv2.setTrackbarPos('ROI Bot Off', 'Lane Tuning', roi_params.get('bot_off', 50))

def get_demo_config(video_id):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "data", "demo_config.json")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            all_configs = json.load(f)
            if video_id in all_configs:
                return all_configs[video_id]
                
    print(f"No specific configuration found in demo_config.json for video {video_id}.")
    print("Run `python3 setup_demo.py` first! Falling back to basic 1-minute demo.\n")
    return {
        "video_file": f"test_{video_id}.mp4",
        "segments": [[0, 1800]],  # ~1 min at 30 fps
        "hough": {}, "color": {}, "roi": {}
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 demo.py <video_id: 1 or 2>")
        sys.exit(1)
        
    video_id = sys.argv[1]
    demo_config = get_demo_config(video_id)
    segments = demo_config.get("segments", [])
    
    if not segments:
        print("No segments defined in config! Defaulting to first 1800 frames.")
        segments = [[0, 1800]]
    
    print(f"Initializing Smart Overtaking Assistant DEMO for Video {video_id}...")
    
    try:
        vehicle_detector = VehicleDetector(config.YOLO_MODEL_PATH)
        lane_detector = LaneDetector()
        safety_checker = SafetyChecker()
        print("Modules loaded successfully.")
    except Exception as e:
        print(f"Error loading modules: {e}")
        return

    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(current_dir, "data", "videos", demo_config["video_file"])

    if not os.path.exists(video_path):
        print(f"Test video not found at: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Set trackbars to loaded JSON values
    apply_tuning_defaults(lane_detector, demo_config)

    # Tuning Phase
    cap.set(cv2.CAP_PROP_POS_FRAMES, segments[0][0])
    print("\n" + "="*40)
    print("ENTERING TUNING MODE (DEMO)")
    print("Default demo parameters have been loaded.")
    print("Press 's' to START demo, or 'q' to abort.")
    print("="*40 + "\n")

    ret, tuning_frame = cap.read()
    if ret:
        tuning_frame = cv2.resize(tuning_frame, (1280, 720))
        while True:
            detections = vehicle_detector.detect(tuning_frame)
            left_line, right_line, debug_view = lane_detector.detect(tuning_frame)
            cv2.imshow("Tuning Preview", debug_view)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                cv2.destroyWindow("Tuning Preview")
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

    print("Demo sequence starting... Press 'q' anytime to exit.")

    # Sort segments by start time
    segments = sorted(segments, key=lambda x: x[0])
    total_frames_to_play = sum([end - start for start, end in segments])
    frames_played = 0

    for i, (start_frame, end_frame) in enumerate(segments):
        print(f"\n--- Playing Segment {i+1}/{len(segments)}: Frames {start_frame} to {end_frame} ---")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (1280, 720))
            frames_played += 1

            detections = vehicle_detector.detect(frame)
            left_line, right_line, debug_view = lane_detector.detect(frame)
            lane_info = (left_line, right_line)
            status, divider = safety_checker.assess(detections, lane_info)

            output_frame = draw_results(frame, detections, lane_info, status, divider)

            cv2.imshow("Smart Overtaking Assistant Demo", output_frame)
            cv2.imshow("Debug: Lane Detection", debug_view)

            if frames_played % 100 == 0:
                print(f"Overall Progress: {frames_played}/{total_frames_to_play} frames", end='\r')

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nDemo interrupted.")
                cap.release()
                cv2.destroyAllWindows()
                return

    print(f"\nDemo completed ({frames_played} frames played across {len(segments)} segments).")
    cv2.waitKey(2000)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

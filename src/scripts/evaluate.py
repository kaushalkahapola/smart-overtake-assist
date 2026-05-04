import csv
import os
import cv2
import numpy as np
import sys
import json

# Adjust path to import modules if running from src/scripts directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perception.vehicle_detector import VehicleDetector
from perception.lane_detector import LaneDetector
from logic.safety_checker import SafetyChecker
from utils.visualization import draw_results
import config

def apply_tuning_defaults(params):
    # Apply JSON parameters directly to the Lane Tuning trackbars
    if not params:
        return
        
    hough_params = params.get("hough", {})
    color_params = params.get("color", {})
    roi_params = params.get("roi", {})
    
    # Ensure window exists before setting trackbars
    cv2.namedWindow('Lane Tuning', cv2.WINDOW_NORMAL)
    
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
    config_path = os.path.join(current_dir, "..", "..", "data", "demo_config.json")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            all_configs = json.load(f)
            if video_id in all_configs:
                return all_configs[video_id]
                
    return None

def calculate_metrics(y_true, y_pred, labels=["SAFE", "RISKY"]):
    # Simple manual confusion matrix
    cm = {l: {l2: 0 for l2 in labels} for l in labels}
    
    for t, p in zip(y_true, y_pred):
        if t in labels and p in labels:
            cm[t][p] += 1
            
    # Print Matrix
    print("\nConfusion Matrix:")
    print(f"{'':10} {'SAFE':10} {'RISKY':10}")
    for l in labels:
        print(f"{l:10} {cm[l]['SAFE']:<10} {cm[l]['RISKY']:<10}")
        
    # Accuracy
    correct = sum([cm[l][l] for l in labels])
    total = len(y_true)
    accuracy = correct / total if total > 0 else 0
    print(f"\nAccuracy: {accuracy:.2%}")
    
    return cm

def evaluate(video_path, csv_path, demo_config=None):
    print(f"Loading Ground Truth from: {csv_path}")
    
    ground_truth = {}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            
            if not headers:
                print("Error: CSV file is empty.")
                return

            frame_idx = -1
            label_idx = -1
            
            for i, h in enumerate(headers):
                h_clean = h.lower().strip()
                if 'frame' in h_clean: frame_idx = i
                if 'label' in h_clean: label_idx = i

            if frame_idx == -1 or label_idx == -1:
                print("Error: Headers missing 'Frame' or 'Label'.")
                return

            for row in reader:
                if len(row) > max(frame_idx, label_idx):
                    try:
                        f_val = row[frame_idx].strip()
                        if not f_val: continue
                        frame_id = int(float(f_val)) 
                        label = row[label_idx].strip().upper()
                        if label in ["SAFE", "RISKY"]:
                            ground_truth[frame_id] = label
                    except ValueError:
                        continue 

    except FileNotFoundError:
        print("CSV file not found.")
        return

    print("Initializing Modules...")
    vehicle_detector = VehicleDetector(config.YOLO_MODEL_PATH)
    
    # 1. Disable Frame Skipping for Evaluation
    vehicle_detector.skip_interval = 1 
    print("Optimization: Frame skipping disabled for static evaluation.")
    
    lane_detector = LaneDetector()
    safety_checker = SafetyChecker()
    
    # Setup Results Directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
    results_dir = os.path.join(base_dir, "results", csv_name)
    success_dir = os.path.join(results_dir, "success")
    fail_dir = os.path.join(results_dir, "fail")
    
    for d in [success_dir, fail_dir]:
        os.makedirs(d, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    y_true = []
    y_pred = []
    target_frames = sorted(ground_truth.keys())
    
    if not target_frames:
        print("No valid frames to evaluate.")
        return
    
    print(f"Evaluating on {len(target_frames)} labeled frames...")
    
    if demo_config:
        print("Applying predefined configuration from demo_config.json...")
        apply_tuning_defaults(demo_config)
    
    # --- Tuning Loop ---
    print("\n" + "="*40)
    print("ENTERING TUNING MODE")
    print("Adjust parameters in 'Lane Tuning' window.")
    print("Press 's' to START evaluation, or 'q' to abort.")
    print("="*40 + "\n")

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frames[0])
    ret, tuning_frame = cap.read()
    
    if ret:
        while True:
            detections = vehicle_detector.detect(tuning_frame)
            left_line, right_line, debug_view = lane_detector.detect(tuning_frame)
            
            cv2.imshow("Tuning Preview", debug_view)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                print("Tuning complete. Starting evaluation...")
                cv2.destroyWindow("Tuning Preview")
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

    # --- Evaluation Loop ---
    WARMUP_FRAMES = 10  # Increased from 5 for better history accumulation

    for i, target in enumerate(target_frames):
        # 1. Start EARLY for Warm-Up
        start_frame = max(0, target - WARMUP_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # 2. DON'T reset history - let adaptive algorithm accumulate knowledge
        # This allows expected_lane_width to be learned and maintained
        # lane_detector.left_history = []  # REMOVED
        # lane_detector.right_history = []  # REMOVED
        
        # 3. Process Warm-Up Frames
        frames_to_read = target - start_frame + 1
        current_frame = None
        
        for _ in range(frames_to_read):
            ret, current_frame = cap.read()
            if not ret: break
            
            # This calls update_params_from_ui() internally.
            # We MUST keep the window alive so params don't reset to 0.
            left_line, right_line, _ = lane_detector.detect(current_frame)
            
            # CRITICAL FIX: Keep UI alive so trackbar values don't die
            cv2.waitKey(1) 

        if not ret:
            print(f"Could not read frame {target}")
            break
            
        # 4. Final Detection (Target Frame)
        detections = vehicle_detector.detect(current_frame)
        lane_info = (left_line, right_line)
        
        # 5. Logic Assessment
        status, _ = safety_checker.assess(detections, lane_info)
        
        final_pred = "SAFE"
        if status == "RISKY" or status == "WARNING":
            final_pred = "RISKY"
        
        final_truth = ground_truth[target]
        y_true.append(final_truth)
        y_pred.append(final_pred)
        
        print(f"[{i+1}/{len(target_frames)}] Frame {target}: True={final_truth}, Pred={final_pred}")
        
        # 6. Save/Show Result
        output_frame = draw_results(current_frame, detections, lane_info, status, right_line)
        
        # Optional: Show progress window to ensure OS knows app is alive
        cv2.imshow("Evaluation Progress", cv2.resize(output_frame, (640, 360)))
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nEvaluation interrupted.")
            break
        elif key == ord('p'):
            print("\nPaused. Press 'p' again to resume.")
            while True:
                if cv2.waitKey(1) & 0xFF == ord('p'):
                    print("Resumed.")
                    break

        filename = f"frame_{target}_truth_{final_truth}_pred_{final_pred}.jpg".lower()
        save_path = os.path.join(success_dir if final_truth == final_pred else fail_dir, filename)
        cv2.imwrite(save_path, output_frame)

    cap.release()
    cv2.destroyAllWindows()
    
    calculate_metrics(y_true, y_pred)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    video_path = None
    csv_path = None
    demo_config = None

    if len(sys.argv) == 2 and sys.argv[1] in ["1", "2"]:
        # Simplified style: python evaluate.py 1
        video_id = sys.argv[1]
        video_path = os.path.join(current_dir, "..", "..", "data", "videos", f"test_{video_id}.mp4")
        csv_path = os.path.join(current_dir, "..", "..", "data", "annotations", f"ground_truth_dense_{video_id}.csv")
        demo_config = get_demo_config(video_id)
        print(f"Using simplified argument mode for video ID: {video_id}")
    else:
        # Standard style
        default_video = os.path.join(current_dir, "..", "..", "data", "videos", "test_1.mp4")
        default_csv = os.path.join(current_dir, "..", "..", "data", "annotations", "ground_truth_dense_1.csv") # Default to standard CSV

        video_path = sys.argv[1] if len(sys.argv) > 1 else default_video
        csv_path = sys.argv[2] if len(sys.argv) > 2 else default_csv

    if not os.path.exists(os.path.abspath(video_path)):
        print(f"Error: Video not found at {video_path}")
        sys.exit(1)

    evaluate(os.path.abspath(video_path), os.path.abspath(csv_path), demo_config)
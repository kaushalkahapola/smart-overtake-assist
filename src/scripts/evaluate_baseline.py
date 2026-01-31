import csv
import os
import cv2
import numpy as np
import sys

# Adjust path to import modules if running from src/scripts directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perception.vehicle_detector import VehicleDetector
from perception.lane_detector import LaneDetector
from logic.baseline_checker import BaselineChecker
import config

def calculate_metrics(y_true, y_pred, labels=["SAFE", "RISKY"]):
    # Simple manual confusion matrix
    cm = {l: {l2: 0 for l2 in labels} for l in labels}
    
    for t, p in zip(y_true, y_pred):
        if t in labels and p in labels:
            cm[t][p] += 1
            
    # Print Matrix
    print("\n--- BASELINE RESULTS (NAIVE LOGIC) ---")
    print("Confusion Matrix (Rows=True, Cols=Pred):")
    print(f"{'':10} {'SAFE':10} {'RISKY':10}")
    for l in labels:
        print(f"{l:10} {cm[l]['SAFE']:<10} {cm[l]['RISKY']:<10}")
        
    # Metrics
    tp = cm['RISKY']['RISKY']
    fn = cm['RISKY']['SAFE']
    fp = cm['SAFE']['RISKY']
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Accuracy
    correct = sum([cm[l][l] for l in labels])
    total = len(y_true)
    accuracy = correct / total if total > 0 else 0
    
    print(f"\nRecall:    {recall:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"F1-Score:  {f1:.2%}")
    print(f"Accuracy:  {accuracy:.2%}")
    
    return cm

def evaluate(video_path, csv_path):
    print(f"Loading Ground Truth from: {csv_path}")
    
    ground_truth = {}
    try:
        with open(csv_path, 'r') as f:
            # 1. Read the Header Line first
            reader = csv.reader(f)
            headers = next(reader, None)
            
            if not headers:
                print("Error: CSV file is empty.")
                return

            # 2. Find which column index is 'Frame' and which is 'Label'
            frame_idx = -1
            label_idx = -1
            
            print(f"Found headers: {headers}")
            
            for i, h in enumerate(headers):
                h_clean = h.lower().strip()
                # Look for 'frame' (matches "Frame ID", "frame_id", "Frame ID (Calculated)")
                if 'frame' in h_clean: 
                    frame_idx = i
                # Look for 'label'
                if 'label' in h_clean:
                    label_idx = i

            if frame_idx == -1 or label_idx == -1:
                print("Error: Could not identify columns. Make sure headers contain 'Frame' and 'Label'.")
                return

            print(f"Reading Frames from Col {frame_idx} and Labels from Col {label_idx}...")

            # 3. Read the Data
            count = 0
            for row in reader:
                # Ensure row has enough columns
                if len(row) > max(frame_idx, label_idx):
                    try:
                        # Get Frame ID (Handle Excel floats like "120.0")
                        f_val = row[frame_idx].strip()
                        if not f_val: continue
                        frame_id = int(float(f_val)) 
                        
                        # Get Label
                        label = row[label_idx].strip().upper()
                        
                        if label in ["SAFE", "RISKY"]:
                            ground_truth[frame_id] = label
                            count += 1
                    except ValueError:
                        continue # Skip bad rows
            
            print(f"Successfully loaded {count} labeled frames.")

    except FileNotFoundError:
        print("CSV file not found.")
        return

    print("Initializing Modules...")
    vehicle_detector = VehicleDetector(config.YOLO_MODEL_PATH)
    lane_detector = LaneDetector()
    safety_checker = BaselineChecker()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    y_true = []
    y_pred = []
    
    # Sort frame indices to reading sequentially
    target_frames = sorted(ground_truth.keys())
    
    if not target_frames:
        print("No valid frames to evaluate. Check CSV.")
        return
    
    print(f"Evaluating on {len(target_frames)} labeled frames...")

    for target in target_frames:
        # Fast forward
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if not ret:
            print(f"Could not read frame {target}")
            break
            
        # 1. Perception
        detections = vehicle_detector.detect(frame)
        left_line, right_line, _ = lane_detector.detect(frame)
        lane_info = (left_line, right_line)
        
        # 2. Logic
        status, _ = safety_checker.assess(detections, lane_info)
        
        # Map Internal Status to Binary Label
        final_pred = "RISKY" if status in ["RISKY", "WARNING"] else "SAFE"
            
        final_truth = ground_truth[target]
        
        y_true.append(final_truth)
        y_pred.append(final_pred)
        
        print(f"Frame {target}: True={final_truth}, Pred={final_pred} (Raw: {status})")

    cap.release()
    cv2.destroyAllWindows()
    
    # Metrics
    calculate_metrics(y_true, y_pred)

if __name__ == "__main__":
    # Video Source Logic (similar to main.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_video = os.path.join(current_dir, "..", "..", "data", "videos", "test_1.mp4")
    default_video = os.path.abspath(default_video)
    
    default_csv = os.path.join(current_dir, "..", "..", "data", "annotations", "ground_truth_dense_1.csv")
    default_csv = os.path.abspath(default_csv)

    video_path = sys.argv[1] if len(sys.argv) > 1 else default_video
    csv_path = sys.argv[2] if len(sys.argv) > 2 else default_csv

    # Final path validation
    video_path = os.path.abspath(video_path)
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        sys.exit(1)

    csv_path = os.path.abspath(csv_path)
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)

    evaluate(video_path, csv_path)
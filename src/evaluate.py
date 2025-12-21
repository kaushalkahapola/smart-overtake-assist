
import csv
import os
import cv2
import numpy as np
import sys

# Import modules
from perception.vehicle_detector import VehicleDetector
from perception.lane_detector import LaneDetector
from logic.safety_checker import SafetyChecker
import config

def calculate_metrics(y_true, y_pred, labels=["SAFE", "RISKY"]):
    # Simple manual confusion matrix
    cm = {l: {l2: 0 for l2 in labels} for l in labels}
    
    for t, p in zip(y_true, y_pred):
        if t in labels and p in labels:
            cm[t][p] += 1
            
    # Print Matrix
    print("\nConfusion Matrix (Rows=True, Cols=Pred):")
    print(f"{'':10} {'SAFE':10} {'RISKY':10}")
    for l in labels:
        print(f"{l:10} {cm[l]['SAFE']:<10} {cm[l]['RISKY']:<10}")
        
    # Accuracy
    correct = sum([cm[l][l] for l in labels])
    total = len(y_true)
    accuracy = correct / total if total > 0 else 0
    print(f"\nAccuracy: {accuracy:.2%}")
    
    return cm

def evaluate(video_path, csv_path):
    print(f"Loading Ground Truth from: {csv_path}")
    
    # Read CSV: frame_id, label
    ground_truth = {}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None) # Skip header
            for row in reader:
                if len(row) >= 2:
                    frame_id = int(row[0])
                    label = row[1].strip().upper()
                    ground_truth[frame_id] = label
    except FileNotFoundError:
        print("CSV file not found. Generating a sample template...")
        with open("evaluation_template.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["frame_id", "label"])
            writer.writerow(["100", "SAFE"])
            writer.writerow(["200", "RISKY"])
        print("Created 'evaluation_template.csv'. Please rename and fill it with real data.")
        return

    print("Initializing Modules...")
    vehicle_detector = VehicleDetector(config.YOLO_MODEL_PATH)
    lane_detector = LaneDetector()
    safety_checker = SafetyChecker()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    y_true = []
    y_pred = []
    
    # Sort frame indices to reading sequentially
    target_frames = sorted(ground_truth.keys())
    
    current_frame = 0
    
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
        
        # Map WAIT -> SAFE for binary classification usually, or handle separately?
        # User asked for SAFE vs RISKY.
        # "WAIT" means "Safe to follow but don't overtake". 
        # For simplicity, map WAIT -> SAFE (since it's not DANGEROUS/RISKY).
        final_pred = "SAFE"
        if status == "RISKY":
            final_pred = "RISKY"
        elif status == "WAIT":
            final_pred = "SAFE" 
            
        final_truth = ground_truth[target]
        
        y_true.append(final_truth)
        y_pred.append(final_pred)
        
        print(f"Frame {target}: True={final_truth}, Pred={final_pred} (Raw: {status})")

    cap.release()
    
    # Metrics
    calculate_metrics(y_true, y_pred)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python evaluate.py <video_path> <csv_path>")
        print("Example: python evaluate.py ../assets/videos/test4.mp4 labels.csv")
    else:
        evaluate(sys.argv[1], sys.argv[2])

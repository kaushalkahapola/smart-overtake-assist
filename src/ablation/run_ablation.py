import sys
import os
import cv2
import csv
import time

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from perception.vehicle_detector import VehicleDetector
from ablation.ablation_lane_detector import AblationLaneDetector
from ablation.ablation_safety_checker import AblationSafetyChecker
from ablation.ablation_configs import ABLATION_CONFIGS

def calculate_metrics(y_true, y_pred, labels=["SAFE", "RISKY"]):
    cm = {l: {l2: 0 for l2 in labels} for l in labels}
    for t, p in zip(y_true, y_pred):
        if t in labels and p in labels:
            cm[t][p] += 1
            
    tp = cm['RISKY']['RISKY']
    fn = cm['RISKY']['SAFE']
    fp = cm['SAFE']['RISKY']
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    correct = sum([cm[l][l] for l in labels])
    total = len(y_true)
    accuracy = correct / total if total > 0 else 0
    
    return accuracy, precision, recall, f1

def run_ablation(video_path, csv_path):
    print(f"--- STARTING ABLATION STUDY ---")
    print(f"Video: {video_path}")
    print(f"Ground Truth: {csv_path}")
    
    # Load Ground Truth
    ground_truth = {}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            if not headers: return
            
            frame_idx = -1
            label_idx = -1
            for i, h in enumerate(headers):
                h_clean = h.lower().strip()
                if 'frame' in h_clean: frame_idx = i
                if 'label' in h_clean: label_idx = i
                
            if frame_idx == -1 or label_idx == -1: return

            for row in reader:
                if len(row) > max(frame_idx, label_idx):
                    try:
                        f_val = row[frame_idx].strip()
                        if not f_val: continue
                        frame_id = int(float(f_val))
                        label = row[label_idx].strip().upper()
                        if label in ["SAFE", "RISKY"]:
                            ground_truth[frame_id] = label
                    except ValueError: continue
    except FileNotFoundError:
        print("CSV not found")
        return

    # --- TUNING PHASE ---
    print("\n" + "="*40)
    print("ENTERING TUNING MODE")
    print("Adjust parameters in 'Lane Tuning' window.")
    print("Press 's' to START experiments, or 'q' to abort.")
    print("="*40 + "\n")

    params_tuned = False
    
    # Use standard LaneDetector for tuning
    from perception.lane_detector import LaneDetector
    tuning_detector = LaneDetector()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video")
        return

    # Jump to first target frame (if available) for better context
    target_frames = sorted(ground_truth.keys())
    if target_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frames[0])
    
    ret, tuning_frame = cap.read()
    if ret:
        while True:
            # We don't strictly need vehicle detection for lane tuning, but visual consistency is nice
            # But let's keep it simple: just show lane detector debug view
            _, _, debug_view = tuning_detector.detect(tuning_frame)
            
            cv2.imshow("Tuning Preview", debug_view)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                print("Tuning complete. Updating configuration...")
                params_tuned = True
                cv2.destroyWindow("Tuning Preview")
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
    cap.release()
    
    if not params_tuned:
        print("Aborted.")
        return

    # Update Config Defaults with Tuned Values
    # This ensures all ablation instances inherit these values
    defaults = config.LANE_DETECTION_DEFAULTS
    defaults["hough_canny_low"] = tuning_detector.hough_canny_low
    defaults["hough_canny_high"] = tuning_detector.hough_canny_high
    defaults["hough_threshold"] = tuning_detector.hough_threshold
    defaults["hough_min_line_len"] = tuning_detector.hough_min_line_len
    defaults["hough_max_line_gap"] = tuning_detector.hough_max_line_gap
    defaults["white_l_min"] = tuning_detector.white_l_min
    defaults["yellow_h_min"] = tuning_detector.yellow_h_min
    defaults["yellow_h_max"] = tuning_detector.yellow_h_max
    defaults["yellow_s_min"] = tuning_detector.yellow_s_min
    defaults["roi_top_width"] = tuning_detector.roi_top_width
    defaults["roi_bottom_width"] = tuning_detector.roi_bottom_width
    defaults["roi_height_pct"] = tuning_detector.roi_height_pct
    defaults["roi_bottom_offset"] = tuning_detector.roi_bottom_offset
    
    print("Configuration updated with tuned parameters.")

    results = []
    
    # Iterate through all configs
    for name, conf in ABLATION_CONFIGS.items():
        print(f"\nRunning Configuration: {name}")
        print(f"Details: {conf}")
        
        # Initialize modules with this config
        vehicle_detector = VehicleDetector(config.YOLO_MODEL_PATH)
        lane_detector = AblationLaneDetector(conf)
        safety_checker = AblationSafetyChecker(conf)
        
        skip_frames_enabled = conf.get('skip_frames', False)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video")
            return
            
        y_true_list = []
        y_pred_list = []
        
        # We process continuously
        last_target = target_frames[-1] if target_frames else 0
        
        current_frame_idx = 0
        detections = []
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Stop if we passed all labelled frames
            if current_frame_idx > last_target:
                break
                
            # --- SKIP FRAMES LOGIC ---
            run_detection = True
            if skip_frames_enabled:
                if current_frame_idx % 3 != 0 and len(detections) > 0:
                    run_detection = False
            
            if run_detection:
                detections = vehicle_detector.detect(frame)
            
            left, right, debug_view = lane_detector.detect(frame)
            
            # Assessment
            status, _ = safety_checker.assess(detections, (left, right))
            final_pred = "RISKY" if status in ["RISKY", "WARNING"] else "SAFE"
            
            # If this is a labeled frame, record it
            if current_frame_idx in ground_truth:
                y_true_list.append(ground_truth[current_frame_idx])
                y_pred_list.append(final_pred)
            
            # Show Progress
            # Reusing debug_view for simple feedback
            cv2.putText(debug_view, f"Config: {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(debug_view, f"Frame: {current_frame_idx}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Ablation Runner", debug_view)
            cv2.waitKey(1)

            current_frame_idx += 1
            
        cap.release()
        cv2.destroyWindow("Ablation Runner") # Close window between runs
        
        elapsed = time.time() - start_time
        fps = current_frame_idx / elapsed if elapsed > 0 else 0
        
        acc, prec, rec, f1 = calculate_metrics(y_true_list, y_pred_list)
        
        print(f"Results for {name}:")
        print(f"  Accuracy:  {acc:.2%}")
        print(f"  Precision: {prec:.2%}")
        print(f"  Recall:    {rec:.2%}")
        print(f"  F1-Score:  {f1:.2%}")
        print(f"  FPS:       {fps:.2f}")
        
        results.append({
            "Config": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "FPS": fps
        })

    # Save Summary
    print("\n--- FINAL SUMMARY ---")
    print(f"{'Config':<15} {'Acc':<10} {'Prec':<10} {'Rec':<10} {'F1':<10} {'FPS':<10}")
    for r in results:
        print(f"{r['Config']:<15} {r['Accuracy']:.2%}   {r['Precision']:.2%}   {r['Recall']:.2%}   {r['F1']:.2%}   {r['FPS']:.2f}")

    # Write to file
    out_file = "ablation_results.txt"
    with open(out_file, "w") as f:
        f.write("Config,Accuracy,Precision,Recall,F1,FPS\n")
        for r in results:
            f.write(f"{r['Config']},{r['Accuracy']},{r['Precision']},{r['Recall']},{r['F1']},{r['FPS']}\n")
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_video = os.path.join(current_dir, "..", "..", "data", "videos", "test_1.mp4")
    default_csv = os.path.join(current_dir, "..", "..", "data", "annotations", "ground_truth_dense_1.csv")
    
    vid = sys.argv[1] if len(sys.argv) > 1 else default_video
    csv_f = sys.argv[2] if len(sys.argv) > 2 else default_csv
    
    run_ablation(vid, csv_f)

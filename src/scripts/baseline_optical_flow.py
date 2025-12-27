import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import time
import os

class OpticalFlowBaseline:
    """
    Classic computer vision approach using dense optical flow (Farneback)
    to detect approaching vehicles based on 'Looming' (Expansion).
    """
    def __init__(self, width=1280, height=720):
        self.prev_gray = None
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2
        
        # Pre-compute the "Expected Expansion Vector" field
        # This speeds up calculation by 100x compared to for-loops
        y, x = np.mgrid[0:height, 0:width]
        self.vec_x = x - self.center_x
        self.vec_y = y - self.center_y
        
        # Normalize vectors to unit length for pure direction checking
        norm = np.sqrt(self.vec_x**2 + self.vec_y**2)
        norm[norm == 0] = 1 # Avoid divide by zero
        self.vec_x = self.vec_x / norm
        self.vec_y = self.vec_y / norm

        # Define Static ROI (Right Lane Trapezoid)
        # Matches your YOLO logic approximately
        self.mask = np.zeros((height, width), dtype=np.uint8)
        pts = np.array([
            [width * 0.55, height * 0.45],  # Top Left
            [width * 1.0, height * 1.0],    # Bottom Right
            [width * 0.45, height * 1.0],   # Bottom Left (Near center)
            [width * 0.52, height * 0.45]   # Top Right (Narrow strip)
        ], np.int32)
        # Actually let's just use the right half for the baseline to be "naive"
        # Or a trapezoid similar to your visualizer
        pts = np.array([
            [int(width*0.55), int(height*0.55)], # Horizon Right
            [width, height],                     # Bottom Right
            [int(width*0.5), height],            # Bottom Center
            [int(width*0.52), int(height*0.55)]  # Horizon Center
        ], np.int32)
        cv2.fillPoly(self.mask, [pts], 1)

    def detect(self, frame, threshold=2.0):
        """
        Returns 'RISKY' or 'SAFE' based on flow expansion.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize for speed (Optical Flow is heavy)
        scale = 0.5
        small_gray = cv2.resize(gray, None, fx=scale, fy=scale)
        
        if self.prev_gray is None:
            self.prev_gray = small_gray
            return "SAFE", 0.0

        # 1. Calculate Dense Optical Flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, small_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # 2. Vectorized Expansion Calculation
        # Resize flow back up to match pre-computed vectors (or resize vectors down)
        # It's faster to process at small scale.
        
        # We need a small mask
        h, w = small_gray.shape
        small_mask = cv2.resize(self.mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Generate small vectors on the fly if needed, or just slice
        # Let's just do math on small scale for speed
        y, x = np.mgrid[0:h, 0:w]
        cx, cy = w//2, h//2
        vec_x = x - cx
        vec_y = y - cy
        norm = np.sqrt(vec_x**2 + vec_y**2)
        norm[norm == 0] = 1
        vec_x = vec_x / norm
        vec_y = vec_y / norm
        
        # Dot Product: Flow * Radial_Vector
        # Positive = Moving AWAY from center (Expansion/Approaching)
        # Negative = Moving TOWARD center (Receding)
        dot_product = flow[..., 0] * vec_x + flow[..., 1] * vec_y
        
        # 3. Apply Mask & Filter
        # We only care about the Right Lane
        lane_activity = dot_product * small_mask
        
        # Filter out small noise (e.g. slight camera jitter)
        lane_activity[lane_activity < 0.5] = 0 
        
        # Score = Average expansion magnitude in the active region
        # We use Mean to be independent of object size, or Sum?
        # Sum is better for "Large Approaching Object" vs "Small Leaves"
        active_pixels = np.count_nonzero(lane_activity)
        if active_pixels > 0:
            score = np.mean(lane_activity[lane_activity > 0]) * (active_pixels / 100.0)
        else:
            score = 0.0
            
        self.prev_gray = small_gray
        
        if score > threshold:
            return "RISKY", score
        else:
            return "SAFE", score

def evaluate_baseline(video_path, csv_path, threshold=5.0):
    if not os.path.exists(video_path) or not os.path.exists(csv_path):
        print("Files not found.")
        return

    print(f"--- Running Optical Flow Baseline on {os.path.basename(video_path)} ---")
    
    # Load Labels
    df = pd.read_csv(csv_path)
    # Filter only relevant frames for speed? Or stream all?
    # We must stream all to maintain 'prev_gray' continuity!
    # Optical flow fails if you jump frames.
    
    cap = cv2.VideoCapture(video_path)
    of = OpticalFlowBaseline()
    
    predictions = []
    ground_truth = []
    
    target_frames = set(df['frame_id'].values)
    results_map = {} # frame_id -> pred
    
    frame_count = 0
    t0 = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Run detection on EVERY frame to keep flow valid
        pred, score = of.detect(frame, threshold=threshold)
        
        if frame_count in target_frames:
            results_map[frame_count] = pred
            
            # Optional: Visualize
            # cv2.putText(frame, f"Flow Score: {score:.2f} | {pred}", (10,30), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            # cv2.imshow("Optical Flow Baseline", frame)
            # if cv2.waitKey(1) == ord('q'): break
            
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...", end='\r')

    cap.release()
    cv2.destroyAllWindows()
    
    # Align Data
    y_true = []
    y_pred = []
    
    for idx, row in df.iterrows():
        fid = row['frame_id']
        if fid in results_map:
            y_true.append(row['label'])
            y_pred.append(results_map[fid])
            
    # Metrics
    labels = ["SAFE", "RISKY"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label="RISKY", zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label="RISKY", zero_division=0)
    
    t_end = time.time()
    fps = frame_count / (t_end - t0)

    print("\n" + "="*40)
    print(f"OPTICAL FLOW BASELINE RESULTS (Thresh={threshold})")
    print("="*40)
    print(f"FPS: {fps:.2f}")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print("-" * 20)
    print("Confusion Matrix:")
    print(f"      Pred:SAFE  Pred:RISKY")
    print(f"Act:SAFE    {cm[0][0]:<5}      {cm[0][1]}")
    print(f"Act:RISKY   {cm[1][0]:<5}      {cm[1][1]}")
    print("="*40)

if __name__ == "__main__":
    # Use your Video 1 and its dense CSV
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_video = os.path.join(current_dir, "..", "..", "data", "videos", "test_1.mp4")
    default_csv = os.path.join(current_dir, "..", "..", "data", "annotations", "ground_truth_dense_1.csv")  
    
    # Run
    evaluate_baseline(default_video, default_csv, threshold=3.5)
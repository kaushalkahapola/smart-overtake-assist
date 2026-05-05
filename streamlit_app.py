import streamlit as st
import cv2
import sys
import os
import time
import csv
import pandas as pd

# Ensure the src directory is in path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import config
from perception.vehicle_detector import VehicleDetector
from perception.lane_detector import LaneDetector
from logic.safety_checker import SafetyChecker
from utils.visualization import draw_results
from scripts.evaluate import get_demo_config, calculate_metrics

st.set_page_config(page_title="Smart Overtaking Assistant", layout="wide")
st.title("Smart Overtaking Assistant Dashboard")

# Mode Selection
mode = st.sidebar.radio("Mode", ["Prototype Demo", "Evaluation Metrics"])

# Initialize Session State
if "video_id" not in st.session_state:
    st.session_state.video_id = "1"

video_id = st.sidebar.selectbox("Select Video", ["1", "2"], index=0)
if video_id != st.session_state.video_id:
    st.session_state.video_id = video_id
    st.experimental_rerun()

demo_config = get_demo_config(video_id)
if not demo_config:
    st.error(f"Config for video {video_id} not found.")
    st.stop()

# Video path
current_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(current_dir, "data", "videos", demo_config["video_file"])

if not os.path.exists(video_path):
    st.error(f"Video not found at: {video_path}")
    st.stop()

st.sidebar.header("Lane Tuning Parameters")

# Hough Parameters
st.sidebar.subheader("Hough & Canny")
hough_params = demo_config.get("hough", {})
canny_low = st.sidebar.slider("Canny Low", 0, 255, hough_params.get("canny_low", 50))
canny_high = st.sidebar.slider("Canny High", 0, 255, hough_params.get("canny_high", 150))
threshold = st.sidebar.slider("Threshold", 0, 255, hough_params.get("threshold", 50))
min_len = st.sidebar.slider("Min Len", 0, 255, hough_params.get("min_len", 100))
max_gap = st.sidebar.slider("Max Gap", 0, 255, hough_params.get("max_gap", 50))

# Color Parameters
st.sidebar.subheader("Color Filters")
color_params = demo_config.get("color", {})
white_l_min = st.sidebar.slider("White L Min", 0, 255, color_params.get("white_l_min", 200))
yellow_h_min = st.sidebar.slider("Yel H Min", 0, 255, color_params.get("yellow_h_min", 15))
yellow_h_max = st.sidebar.slider("Yel H Max", 0, 255, color_params.get("yellow_h_max", 35))
yellow_s_min = st.sidebar.slider("Yel S Min", 0, 255, color_params.get("yellow_s_min", 100))

# ROI Parameters
st.sidebar.subheader("Region of Interest (ROI)")
roi_params = demo_config.get("roi", {})
top_w = st.sidebar.slider("ROI Top W", 0, 1280, roi_params.get("top_w", 200))
bot_w = st.sidebar.slider("ROI Bot W", 0, 1280, roi_params.get("bot_w", 600))
h_pct = st.sidebar.slider("ROI H %", 0, 100, roi_params.get("h_pct", 40))
bot_off = st.sidebar.slider("ROI Bot Off", 0, 720, roi_params.get("bot_off", 50))

# Lazy load models
@st.cache_resource
def load_models(skip_interval=2):
    vd = VehicleDetector(config.YOLO_MODEL_PATH)
    vd.skip_interval = skip_interval
    return vd, LaneDetector(), SafetyChecker()

if mode == "Prototype Demo":
    vehicle_detector, lane_detector, safety_checker = load_models(skip_interval=2)
else:
    # Disable frame skipping for evaluation
    vehicle_detector, lane_detector, safety_checker = load_models(skip_interval=1)

# Apply updated parameters
lane_detector.canny_low = canny_low
lane_detector.canny_high = canny_high
lane_detector.threshold = threshold
lane_detector.min_len = min_len
lane_detector.max_gap = max_gap
lane_detector.white_l_min = white_l_min
lane_detector.yellow_h_min = yellow_h_min
lane_detector.yellow_h_max = yellow_h_max
lane_detector.yellow_s_min = yellow_s_min
lane_detector.top_w = top_w
lane_detector.bot_w = bot_w
lane_detector.h_pct = h_pct
lane_detector.bot_off = bot_off

if mode == "Prototype Demo":
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Main Output")
        main_placeholder = st.empty()
    with col2:
        st.markdown("### Lane Detection Debug")
        debug_placeholder = st.empty()

    start_button = st.sidebar.button("Start Processing")
    stop_button = st.sidebar.button("Stop Processing")

    if start_button:
        cap = cv2.VideoCapture(video_path)
        segments = demo_config.get("segments", [[0, 1800]])
        sorted_segments = sorted(segments, key=lambda x: x[0])
        
        for start_frame, end_frame in sorted_segments:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            while cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
                ret, frame = cap.read()
                if not ret or stop_button:
                    break
                    
                frame = cv2.resize(frame, (1280, 720))
                
                detections = vehicle_detector.detect(frame)
                left_line, right_line, debug_view = lane_detector.detect(frame)
                lane_info = (left_line, right_line)
                status, divider = safety_checker.assess(detections, lane_info)

                output_frame = draw_results(frame, detections, lane_info, status, divider)

                main_placeholder.image(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB), channels="RGB")
                debug_placeholder.image(cv2.cvtColor(debug_view, cv2.COLOR_BGR2RGB), channels="RGB")
                
                time.sleep(0.01)
                
        cap.release()

elif mode == "Evaluation Metrics":
    st.markdown("### Evaluation Dashboard")
    
    csv_path = os.path.join(current_dir, "data", "annotations", f"ground_truth_dense_{video_id}.csv")
    if not os.path.exists(csv_path):
        st.error(f"Ground Truth CSV not found: {csv_path}")
        st.stop()
        
    # Read CSV
    ground_truth = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader, None)
        frame_idx, label_idx = -1, -1
        if headers:
            for i, h in enumerate(headers):
                if 'frame' in h.lower(): frame_idx = i
                if 'label' in h.lower(): label_idx = i
                
            if frame_idx != -1 and label_idx != -1:
                for row in reader:
                    if len(row) > max(frame_idx, label_idx):
                        try:
                            f_val = row[frame_idx].strip()
                            if f_val:
                                frame_id = int(float(f_val))
                                label = row[label_idx].strip().upper()
                                if label in ["SAFE", "RISKY"]:
                                    ground_truth[frame_id] = label
                        except ValueError:
                            pass
                            
    target_frames = sorted(ground_truth.keys())
    st.write(f"Loaded **{len(target_frames)}** labeled target frames for Video {video_id}.")
    
    start_eval = st.button("Run Evaluation")
    
    if start_eval:
        cap = cv2.VideoCapture(video_path)
        y_true, y_pred = [], []
        
        progress_text = "Running evaluation. Please wait."
        my_bar = st.progress(0, text=progress_text)
        
        col1, col2 = st.columns(2)
        with col1:
            frame_placeholder = st.empty()
        with col2:
            metrics_placeholder = st.empty()
            
        WARMUP_FRAMES = 10
        total_frames = len(target_frames)
        
        for i, target in enumerate(target_frames):
            start_frame = max(0, target - WARMUP_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frames_to_read = target - start_frame + 1
            current_frame = None
            
            for _ in range(frames_to_read):
                ret, current_frame = cap.read()
                if not ret: break
                current_frame = cv2.resize(current_frame, (1280, 720))
                left_line, right_line, _ = lane_detector.detect(current_frame)
            
            if not ret:
                break
                
            detections = vehicle_detector.detect(current_frame)
            lane_info = (left_line, right_line)
            status, _ = safety_checker.assess(detections, lane_info)
            
            final_pred = "SAFE"
            if status in ["RISKY", "WARNING"]:
                final_pred = "RISKY"
                
            final_truth = ground_truth[target]
            y_true.append(final_truth)
            y_pred.append(final_pred)
            
            output_frame = draw_results(current_frame, detections, lane_info, status, right_line)
            frame_placeholder.image(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB), channels="RGB", caption=f"Frame: {target} | True: {final_truth} | Pred: {final_pred}")
            
            my_bar.progress((i + 1) / total_frames, text=f"Evaluating frame {i+1} of {total_frames}...")

        cap.release()
        
        st.success("Evaluation Complete!")
        
        # Calculate and Display Metrics
        labels = ["SAFE", "RISKY"]
        cm = {l: {l2: 0 for l2 in labels} for l in labels}
        for t, p in zip(y_true, y_pred):
            if t in labels and p in labels:
                cm[t][p] += 1
                
        correct = sum([cm[l][l] for l in labels])
        total = len(y_true)
        accuracy = correct / total if total > 0 else 0
        
        st.subheader("Results")
        st.metric(label="Overall Accuracy", value=f"{accuracy:.2%}")
        
        st.write("Confusion Matrix:")
        cm_df = pd.DataFrame(cm).T
        cm_df.index.name = "True Label"
        cm_df.columns.name = "Predicted Label"
        st.dataframe(cm_df)

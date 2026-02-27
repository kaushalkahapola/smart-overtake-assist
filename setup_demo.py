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

def load_config(config_path):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

def save_config(config_path, data):
    with open(config_path, 'w') as f:
        json.dump(data, f, indent=4)
        print(f"Configuration saved to {config_path}")

def get_current_tuning(lane_detector):
    # Reads current trackbar values directly from cv2 window
    return {
        "hough": {
            "canny_low": cv2.getTrackbarPos('Canny Low', 'Lane Tuning'),
            "canny_high": cv2.getTrackbarPos('Canny High', 'Lane Tuning'),
            "threshold": cv2.getTrackbarPos('Threshold', 'Lane Tuning'),
            "min_len": cv2.getTrackbarPos('Min Len', 'Lane Tuning'),
            "max_gap": cv2.getTrackbarPos('Max Gap', 'Lane Tuning')
        },
        "color": {
            "white_l_min": cv2.getTrackbarPos('White L', 'Lane Tuning'),
            "yellow_h_min": cv2.getTrackbarPos('Yel H Min', 'Lane Tuning'),
            "yellow_h_max": cv2.getTrackbarPos('Yel H Max', 'Lane Tuning'),
            "yellow_s_min": cv2.getTrackbarPos('Yel S Min', 'Lane Tuning')
        },
        "roi": {
            "top_w": cv2.getTrackbarPos('ROI Top W', 'Lane Tuning'),
            "bot_w": cv2.getTrackbarPos('ROI Bot W', 'Lane Tuning'),
            "h_pct": cv2.getTrackbarPos('ROI H %', 'Lane Tuning'),
            "bot_off": cv2.getTrackbarPos('ROI Bot Off', 'Lane Tuning')
        }
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 setup_demo.py <video_id: 1 or 2>")
        sys.exit(1)
        
    video_id = sys.argv[1]
    
    # Init config
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "data", "demo_config.json")
    all_configs = load_config(config_path)
    if video_id not in all_configs:
        all_configs[video_id] = {
            "video_file": f"test_{video_id}.mp4",
            "hough": {}, "color": {}, "roi": {},
            "segments": []
        }
    
    video_config = all_configs[video_id]

    print(f"Initializing Setup for DEMO {video_id}...")
    
    # Initialize Modules (Only LaneDetector is strictly needed for tuning, but let's load all to see full output if needed)
    try:
        vehicle_detector = VehicleDetector(config.YOLO_MODEL_PATH)
        lane_detector = LaneDetector()
        safety_checker = SafetyChecker()
        print("Modules loaded successfully.")
    except Exception as e:
        print(f"Error loading modules: {e}")
        return

    video_path = os.path.join(current_dir, "data", "videos", video_config["video_file"])
    if not os.path.exists(video_path):
        print(f"Test video not found at: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # PHASE 1: TUNING
    skip_tuning = False
    if video_config.get("hough"):
        print("\n" + "*"*50)
        print("Existing tuning parameters found for this video.")
        ans = input("Do you want to use existing settings and skip tuning? (y/n): ").strip().lower()
        if ans == 'y':
            skip_tuning = True
        print("*"*50 + "\n")

    if not skip_tuning:
        print("\n" + "="*50)
        print("PHASE 1: TUNING MODE")
        print("Adjust parameters in 'Lane Tuning' window.")
        print("Press 'u' to SAVE the current tuning.")
        print("Press 's' to finish tuning and go to SEGMENT mode.")
        print("="*50 + "\n")

        ret, tuning_frame = cap.read()
        if not ret:
            print("Error: Could not read video frame.")
            return

        tuning_frame = cv2.resize(tuning_frame, (1280, 720))
        tuned_params = None

        while True:
            # Full perception pipeline for the single tuning frame
            detections = vehicle_detector.detect(tuning_frame)
            left_line, right_line, debug_view = lane_detector.detect(tuning_frame)
            
            # We only really need to see the debug view, but let's show main too
            lane_info = (left_line, right_line)
            status, divider = safety_checker.assess(detections, lane_info)
            output_frame = cv2.resize(tuning_frame.copy(), (1280, 720)) # fallback
            
            cv2.imshow("Tuning Preview", debug_view)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('u'):
                tuned_params = get_current_tuning(lane_detector)
                video_config.update(tuned_params)
                all_configs[video_id] = video_config
                save_config(config_path, all_configs)
                print("Tuning parameters updated and saved!")
            elif key == ord('s'):
                if tuned_params is None:
                    # If they forgot to press 'u', automatically grab the current params
                    tuned_params = get_current_tuning(lane_detector)
                    video_config.update(tuned_params)
                    all_configs[video_id] = video_config
                    save_config(config_path, all_configs)
                    print("Tuning parameters implicitly saved!")
                print("Tuning complete. Moving to Phase 2...")
                cv2.destroyWindow("Tuning Preview")
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                print("Quit before completing setup.")
                return
    else:
        # Pre-apply the saved config to the lane detector in case they want to
        # use those settings for the visualizer during the scrubbing phase
        from demo import apply_tuning_defaults
        apply_tuning_defaults(lane_detector, video_config)
        # Advance the OpenCV event loop so the window state propagates (if any)
        cv2.waitKey(1)

    # PHASE 2: SCRUBBING & SEGMENTING
    print("\n" + "="*50)
    print("PHASE 2: SEGMENT PREPARATION")
    print("Controls:")
    print(" [SPACE] : Play / Pause video")
    print(" [a]     : Skip BACK 30 frames (approx 1 sec)")
    print(" [d]     : Skip FORWARD 30 frames (approx 1 sec)")
    print(" [m]     : Mark START of a segment")
    print(" [n]     : Mark END of a segment (Saves the segment)")
    print(" [c]     : Clear all segments from memory")
    print(" [q]     : Quit & Save configuration")
    print("="*50 + "\n")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    paused = True
    current_start = None
    segments = video_config.get("segments", [])

    print(f"Currently saved segments: {segments}")

    # Read first frame
    ret, frame = cap.read()
    if not ret: return
    current_frame_id = 0

    # We need the visualization tools for the scrubber
    from utils.visualization import draw_results

    # To avoid lag while skipping, we'll only run perception when we display
    # but we need to run it every loop so we'll wrap it in the drawing logic
    while True:
        display_frame = frame.copy()
        display_frame = cv2.resize(display_frame, (1280, 720))

        # Run perception on the frame to show what the final demo will look like
        detections = vehicle_detector.detect(display_frame)
        left_line, right_line, _ = lane_detector.detect(display_frame)
        lane_info = (left_line, right_line)
        status, divider = safety_checker.assess(detections, lane_info)
        
        # Draw the actual results
        display_frame = draw_results(display_frame, detections, lane_info, status, divider)

        # Build overlay text
        # Convert frame ID to timestamp
        secs = current_frame_id / fps
        mins = int(secs // 60)
        rem_secs = secs % 60
        time_str = f"{mins:02d}:{rem_secs:05.2f}"
        
        status_text = "PAUSED" if paused else "PLAYING"
        mode_color = (0, 0, 255) if paused else (0, 255, 0)
        
        # Add a dark background for the text overlay so it's readable over the perception output
        overlay_bg = display_frame.copy()
        cv2.rectangle(overlay_bg, (10, 10), (450, 180), (0, 0, 0), -1)
        cv2.rectangle(overlay_bg, (880, 10), (1270, 400), (0, 0, 0), -1)
        display_frame = cv2.addWeighted(display_frame, 0.7, overlay_bg, 0.3, 0)
        
        cv2.putText(display_frame, f"State: {status_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2)
        cv2.putText(display_frame, f"Frame: {current_frame_id} / {total_frames}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Time: {time_str}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if current_start is not None:
            cv2.putText(display_frame, f"Pending start mark: Frame {current_start}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Display existing segments down the right side
        cv2.putText(display_frame, "Recorded Segments:", (900, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        for i, seg in enumerate(segments):
            cv2.putText(display_frame, f"{i+1}: {seg[0]} -> {seg[1]}", (900, 70 + (i*30)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("Video Scrubbing", display_frame)

        # Handle playback
        wait_time = 0 if paused else max(1, int(1000/fps))
        key = cv2.waitKey(wait_time) & 0xFF

        if key == ord(' '):
            paused = not paused
        elif key in [ord('a'), ord('A')]:
            # Skip back 30 frames
            current_frame_id = max(0, current_frame_id - 30)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_id)
            ret, frame = cap.read()
            # Clear history so we don't get weird forecasting lines from jumping backward
            lane_detector.left_history.clear()
            lane_detector.right_history.clear()
        elif key in [ord('d'), ord('D')]:
            # Skip forward 30 frames
            current_frame_id = min(total_frames - 1, current_frame_id + 30)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_id)
            ret, frame = cap.read()
            # Clear history so we don't get weird forecasting lines from jumping forward
            lane_detector.left_history.clear()
            lane_detector.right_history.clear()
        elif key == ord('m'):
            current_start = current_frame_id
            print(f"Segment Start marked at frame {current_start}")
        elif key == ord('n'):
            if current_start is not None:
                if current_frame_id > current_start:
                    segments.append([current_start, current_frame_id])
                    current_start = None
                    print(f"Segment appended! Total segments: {len(segments)}")
                else:
                    print("Error: End frame must be greater than Start frame.")
            else:
                print("Error: Please mark Start frame with 'm' first.")
        elif key == ord('c'):
            segments = []
            current_start = None
            print("Cleared all segments.")
        elif key == ord('q'):
            # Save segments and quit
            video_config["segments"] = segments
            all_configs[video_id] = video_config
            save_config(config_path, all_configs)
            print("Setup finished successfully!")
            break
            
        # If playing, read next frame
        if not paused and key == 255:  # No key pressed
            ret, next_frame = cap.read()
            if ret:
                frame = next_frame
                current_frame_id += 1
            else:
                paused = True  # Auto-pause at end of video

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import cv2
import os
import sys

# Adjust path to import modules if running from src/scripts directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config

def calibrate(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error reading video: {video_path}")
        return

    print("Controls:")
    print("  SPACE: Pause/Resume")
    print("  Mouse: Click Left Edge then Right Edge of a known object (e.g. car)")
    print("  'c': Calculate Focal Length (after clicking)")
    print("  'q': Quit")
    
    paused = False
    clicks = [] # Store x coordinates

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append(x)
            print(f"Clicked at x={x}")
            if len(clicks) > 2:
                clicks.pop(0) # Keep last 2

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", mouse_callback)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # CRITICAL: Resize to match main.py (1280x720)
            # If we don't resize, pixel counts will be different than in the main app
            frame = cv2.resize(frame, (1280, 720))
        
        display = frame.copy()
        
        # Draw clicks
        for x in clicks:
            cv2.line(display, (x, 0), (x, display.shape[0]), (0, 255, 0), 2)
            
        if len(clicks) == 2:
            # Draw line between them
            cv2.line(display, (clicks[0], display.shape[0]//2), (clicks[1], display.shape[0]//2), (0, 0, 255), 2)
            width_px = abs(clicks[1] - clicks[0])
            cv2.putText(display, f"Width: {width_px}px", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Calibration", display)
        
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord(' '):
            paused = not paused
        elif key == ord('q'):
            break
        elif key == ord('c'):
            if len(clicks) == 2:
                width_px = abs(clicks[1] - clicks[0])
                try:
                    real_dist = float(input("Enter real distance to object (meters): "))
                    real_width = float(input("Enter real width of object (meters, e.g. 1.8 for car): "))
                    
                    focal_length = (width_px * real_dist) / real_width
                    print(f"\n[RESULT] Calculated Focal Length: {focal_length:.2f}")
                    print(f"Update src/config.py with: \"FOCAL_LENGTH\": {focal_length:.1f}")
                except ValueError:
                    print("Invalid input.")
            else:
                print("Click 2 points first!")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Default to test video if not provided
    video_path = "../../data/videos/test_1.mp4"
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    
    # Resolve path
    if not os.path.exists(video_path):
         # src/../../data/...
         video_path = os.path.join(current_dir, "..", "..", "data", "videos", "test_1.mp4")
         
    calibrate(video_path)

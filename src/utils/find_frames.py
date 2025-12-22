import cv2
import sys
import os

def find_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error reading video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nLOADED: {os.path.basename(video_path)}")
    print(f"FPS: {fps:.2f} | Total Frames: {total_frames}")
    print("-" * 40)
    print("CONTROLS:")
    print("  [g]   : Go to a specific second (e.g. type '4.5')")
    print("  [a/d] : Previous / Next Frame")
    print("  [z/c] : Jump -30 / +30 Frames (1 second)")
    print("  [q]   : Quit")
    print("-" * 40)

    current_frame = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw Info on Frame
        display = frame.copy()
        # Resize for easier viewing on laptops
        display = cv2.resize(display, (1280, 720))
        
        timestamp = current_frame / fps
        info_text = f"Frame: {current_frame} | Time: {timestamp:.2f}s"
        
        # Black background for text
        cv2.rectangle(display, (0, 0), (400, 50), (0, 0, 0), -1)
        cv2.putText(display, info_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2)

        cv2.imshow("Frame Seeker", display)

        # Wait for key
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        
        elif key == ord('d'): # Next frame
            current_frame = min(current_frame + 1, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            
        elif key == ord('a'): # Prev frame
            current_frame = max(current_frame - 1, 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        elif key == ord('c'): # Jump forward 1s
            current_frame = min(current_frame + 30, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        elif key == ord('z'): # Jump back 1s
            current_frame = max(current_frame - 30, 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            
        elif key == ord('g'): # Go to second
            try:
                target_sec = float(input("\nEnter target seconds: "))
                target_frame = int(target_sec * fps)
                current_frame = max(0, min(target_frame, total_frames - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                print(f"Jumped to Frame {current_frame}")
            except ValueError:
                print("Invalid number.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_video = os.path.join(current_dir, "..", "..", "assets", "videos", "test4.mp4")
    video_path = sys.argv[1] if len(sys.argv) > 1 else default_video
    video_path = os.path.abspath(video_path)
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        sys.exit(1)
        
    find_frame(video_path)

import cv2
import numpy as np

def draw_results(frame, detections, lanes, status):
    output_frame = frame.copy()
    
    # Draw Lanes
    if lanes is not None:
        left_line, right_line = lanes
        try:
            # Draw lines
            cv2.line(output_frame, (int(left_line[0]), int(left_line[1])), (int(left_line[2]), int(left_line[3])), (255, 0, 0), 5)
            cv2.line(output_frame, (int(right_line[0]), int(right_line[1])), (int(right_line[2]), int(right_line[3])), (255, 0, 0), 5)
            
            # Fill lane area transparency
            overlay = output_frame.copy()
            pts = np.array([[
                (left_line[0], left_line[1]), 
                (left_line[2], left_line[3]), 
                (right_line[2], right_line[3]), 
                (right_line[0], right_line[1])
            ]], dtype=np.int32)
            cv2.fillPoly(overlay, pts, (0, 255, 255))
            cv2.addWeighted(overlay, 0.3, output_frame, 0.7, 0, output_frame)
        except Exception as e:
            # Prevent crash on singluar lane detection
            pass

    # Draw Vehicles
    for det in detections:
        x1, y1, x2, y2, cls, conf = det
        
        # Approximate Distance Calculation (Geometric heuristic)
        # Assumes flat road and fixed camera height.
        height, width = frame.shape[:2]
        
        # Heuristic: Horizon is roughly at 50% height
        horizon_y = int(height * 0.5)
        
        # Avoid division by zero or negative
        if y2 > horizon_y + 10: # +10 pixel buffer
            dist = (height * 10) / (y2 - horizon_y) 
            label = f"{dist:.1f}m"
        else:
            label = ">200m" # Distant
            
        color = (0, 255, 0) 
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
        
        # Show Class, Confidence and Dist
        cv2.putText(output_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw Status
    color_status = (0, 255, 0) if "SAFE" in status else (0, 0, 255)
    
    # Status Banner
    cv2.rectangle(output_frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
    cv2.putText(output_frame, f"STATUS: {status}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color_status, 2)

    return output_frame

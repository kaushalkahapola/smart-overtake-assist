import cv2
import numpy as np

def draw_results(frame, detections, lane_info, status, divider_line):
    """Draw lane overlay and vehicle bounding boxes - simplified for 2 lanes"""
    output = frame.copy()
    
    left_line, right_line = lane_info
    
    # Draw lane overlay
    lane_overlay = np.zeros_like(frame)
    if left_line is not None and right_line is not None:
        # Create points for polygon
        # left_line is [x1, y1, x2, y2] (bottom to top approx)
        # right_line is [x1, y1, x2, y2]
        
        # We need 4 points: Left Bottom, Left Top, Right Top, Right Bottom
        # Assuming lines are [x1, y1, x2, y2] where y1 > y2 (bottom to top) usually, 
        # but let's be safe.
        l_p1 = (left_line[0], left_line[1])
        l_p2 = (left_line[2], left_line[3])
        r_p1 = (right_line[0], right_line[1])
        r_p2 = (right_line[2], right_line[3])
        
        # Sort points by Y to identify top/bottom
        # This is a simple heuristic: Assuming typical road view
        # We want the polygon to connect the lines
        
        pts = np.array([l_p1, l_p2, r_p2, r_p1], dtype=np.int32)
        # Draw semi-transparent green filled polygon
        cv2.fillPoly(lane_overlay, [pts], (0, 255, 0))
        
        # Draw lines (Green matching lane5)
        cv2.line(output, l_p1, l_p2, (0, 255, 0), 5)
        cv2.line(output, r_p1, r_p2, (0, 255, 0), 5)

    # Blend with original
    output = cv2.addWeighted(output, 1, lane_overlay, 0.3, 0)
    
    # Draw vehicle bounding boxes
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        
        # Determine color based on position relative to divider
        color = (0, 255, 0)  # Green by default
        
        if divider_line is not None:
            vehicle_x = (x1 + x2) / 2
            vehicle_y = y2
            
            # Linear interpolation for divider x at vehicle_y
            lx1, ly1, lx2, ly2 = divider_line
            if ly2 - ly1 != 0:
                 divider_x = lx1 + (vehicle_y - ly1) * (lx2 - lx1) / (ly2 - ly1)
                 
                 if vehicle_x > divider_x:
                    color = (0, 0, 255)  # Red - in oncoming lane
        
        cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(output, f'{conf:.2f}', (int(x1), int(y1)-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw status
    status_color = (0, 255, 0) if status == "SAFE" else (0, 0, 255)
    cv2.putText(output, f'Status: {status}', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
    
    return output

import cv2
import numpy as np
import config

def draw_results(frame, detections, lane_info, status, divider_line):
    """Draw lane overlay and vehicle bounding boxes with distance info"""
    output = frame.copy()
    
    # Distance estimation params
    params = config.DISTANCE_ESTIMATION_PARAMS
    focal_length = params["FOCAL_LENGTH"]
    class_widths = {2: 1.8, 3: 0.8, 5: 2.5, 7: 2.4}
    default_width = params["KNOWN_WIDTH"]
    
    left_line, right_line = lane_info
    
    # Draw lane overlay
    lane_overlay = np.zeros_like(frame)
    if left_line is not None and right_line is not None:
        l_p1 = (left_line[0], left_line[1])
        l_p2 = (left_line[2], left_line[3])
        r_p1 = (right_line[0], right_line[1])
        r_p2 = (right_line[2], right_line[3])
        
        pts = np.array([l_p1, l_p2, r_p2, r_p1], dtype=np.int32)
        cv2.fillPoly(lane_overlay, [pts], (0, 255, 0))
        
        cv2.line(output, l_p1, l_p2, (0, 255, 0), 5)
        cv2.line(output, r_p1, r_p2, (0, 255, 0), 5)

    # Blend with original
    output = cv2.addWeighted(output, 1, lane_overlay, 0.3, 0)
    
    # Draw vehicle bounding boxes with distance
    for det in detections:
        if len(det) == 7:
            x1, y1, x2, y2, track_id, cls_id, conf = det
        else:
            x1, y1, x2, y2, cls_id, conf = det
            track_id = -1
        
        # Calculate distance
        bbox_width = x2 - x1
        real_w = class_widths.get(cls_id, default_width)
        distance = (focal_length * real_w) / bbox_width if bbox_width > 0 else 0
        
        # Determine color based on position relative to divider
        color = (0, 255, 0)  # Green by default
        
        if divider_line is not None:
            vehicle_x = (x1 + x2) / 2
            vehicle_y = y2
            
            lx1, ly1, lx2, ly2 = divider_line
            if ly2 - ly1 != 0:
                 divider_x = lx1 + (vehicle_y - ly1) * (lx2 - lx1) / (ly2 - ly1)
                 
                 if vehicle_x > divider_x:
                    color = (0, 0, 255)  # Red - in oncoming lane
        
        cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Label: ID + distance
        if track_id != -1:
            label = f'ID:{track_id} {distance:.0f}m'
        else:
            label = f'{distance:.0f}m'
        cv2.putText(output, label, (int(x1), int(y1)-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw status
    if status == "SAFE":
        status_color = (0, 255, 0) # Green
    elif status == "WARNING":
        status_color = (0, 165, 255) # Orange
    else:
        status_color = (0, 0, 255) # Red (RISKY)

    cv2.putText(output, f'Status: {status}', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
    
    return output

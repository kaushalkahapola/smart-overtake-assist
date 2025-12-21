import cv2
import numpy as np

class SafetyChecker:
    def __init__(self):
        self.status = "UNKNOWN"

    def assess(self, detections, lane_info):
        """
        Assess driving safety based on detected vehicles and lane boundaries (2 lanes).
        
        Args:
            detections: List of vehicle detections
            lane_info: (left_line, right_line) - linear segments [x1, y1, x2, y2]
        
        Returns:
            status: "SAFE" or "RISKY"
            divider_line: The right lane segment (boundary of driving lane)
        """
        left_line, right_line = lane_info
        
        # Use right lane as the boundary
        divider_line = right_line
        
        if divider_line is None:
            # No lane detected - can't assess safety
            return "SAFE", None
        
        # Check each detected vehicle
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            
            # Vehicle bottom-center point (closest to camera)
            vehicle_x = (x1 + x2) / 2
            vehicle_y = y2  # Bottom of bounding box
            
            # Calculate divider position at vehicle's y-coordinate
            # Line: x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            lx1, ly1, lx2, ly2 = divider_line
            
            if ly2 - ly1 == 0: # Avoid division by zero (horizontal line)
                divider_x = lx1 
            else:
                divider_x = lx1 + (vehicle_y - ly1) * (lx2 - lx1) / (ly2 - ly1)
            
                divider_x = lx1 + (vehicle_y - ly1) * (lx2 - lx1) / (ly2 - ly1)
            
            # Check overlap with Overtaking Lane (Right Side)
            # Vehicle width
            v_width = x2 - x1
            if v_width <= 0: continue
            
            # Amount of vehicle to the RIGHT of the divider
            # If x2 (right edge) is left of divider, overlap is 0 (negative)
            # If x1 (left edge) is right of divider, overlap is width (ratio > 1)
            overlap_width = x2 - divider_x
            
            overlap_ratio = overlap_width / v_width
            
            # Rule: If > 30% of vehicle is in the Overtaking Lane -> RISKY
            if overlap_ratio > 0.3:
                return "RISKY", divider_line
        
        # All vehicles are in our lane or behind us - SAFE
        return "SAFE", divider_line

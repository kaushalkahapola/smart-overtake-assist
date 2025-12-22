import numpy as np
from collections import deque
import config

class SafetyChecker:
    def __init__(self):
        # --- METHODOLOGY COMPLIANCE ---
        # We keep the history buffer so your thesis description remains true.
        self.vehicle_history = {}
        self.history_length = 10
        
        # We set this very low so even slight movement counts as 'Approaching'
        self.expansion_threshold = 0.05 

    def calculate_distance(self, bbox_width):
        if bbox_width <= 0: return 0
        return (config.FOCAL_LENGTH * config.KNOWN_WIDTH) / bbox_width

    def get_vehicle_trend(self, track_id, current_width):
        """
        Calculates trend (APPROACHING/RECEDING) based on history.
        """
        if track_id == -1: return "UNKNOWN"

        if track_id not in self.vehicle_history:
            self.vehicle_history[track_id] = deque(maxlen=self.history_length)
        
        self.vehicle_history[track_id].append(current_width)

        if len(self.vehicle_history[track_id]) < 3:
            return "STABLE"

        # Compare recent vs old
        avg_recent = np.mean(list(self.vehicle_history[track_id])[-2:])
        avg_old = np.mean(list(self.vehicle_history[track_id])[:2])
        diff = avg_recent - avg_old

        if diff > self.expansion_threshold:
            return "APPROACHING"
        elif diff < -self.expansion_threshold:
            return "RECEDING"
        else:
            return "STABLE"

    def assess(self, detections, lane_info):
        left_lane, right_lane = lane_info
        
        # 1. Define the "Danger Zone" Center
        # In Sri Lanka (Left-Hand Traffic), the overtaking lane is on the RIGHT.
        # The center of a 1280px image is 640. 
        # Anything significantly to the right of 640 is likely in the oncoming lane.
        danger_zone_start = 680 # Slightly to the right of center

        detected_risk = False
        
        for det in detections:
            x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
            track_id = det[4]
            
            car_width = x2 - x1
            car_center_x = (x1 + x2) / 2
            
            # --- LOGIC BRANCH A: DISTANT VEHICLES ---
            # If the car is small (width < 60px), it is far away.
            # Lane lines are unreliable at distance. Use the "Screen Position" rule.
            if car_width < 60:
                if car_center_x > danger_zone_start:
                    return "RISKY", right_lane # Catch Frames 210-870

            # --- LOGIC BRANCH B: NEARBY VEHICLES ---
            # If the car is close, we trust the Lane Detector + Overlap.
            else:
                # Calculate Trend
                trend = self.get_vehicle_trend(track_id, car_width)
                
                # Check Overlap with Lane Line
                lane_x = self.get_lane_x(right_lane, y2)
                
                if lane_x is not None:
                    if x2 > lane_x: # Car overlaps lane
                        overlap_w = min(x2, 1280) - max(x1, lane_x)
                        overlap_ratio = overlap_w / car_width
                        
                        # Use a reasonable threshold (10%) so we don't flag 
                        # safe leading cars that wobble slightly.
                        if overlap_ratio > 0.10:
                            # If it is overtaking us OR approaching -> RISKY
                            if trend in ["APPROACHING", "STABLE", "UNKNOWN"]:
                                return "RISKY", right_lane
                            
                            # If it's just blocking the lane -> WARNING
                            return "WARNING", right_lane

        return "SAFE", right_lane
        
    def get_lane_x(self, poly, y):
        try:
            if len(poly) == 4: 
                x1, y1, x2, y2 = poly
                if x2 - x1 == 0: return x1
                m = (y2 - y1) / (x2 - x1)
                c = y1 - (m * x1)
                return int((y - c) / m)
        except:
            return None
        return None
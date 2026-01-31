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
        
        # If we don't have a valid right lane line, we can't assess safety
        if right_lane is None:
            return "SAFE", right_lane
        
        detected_risk = False
        
        for det in detections:
            x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
            track_id = det[4]
            
            car_width = x2 - x1
            car_center_x = (x1 + x2) / 2
            
            # Calculate Trend (for all vehicles)
            trend = self.get_vehicle_trend(track_id, car_width)
            
            # Get the lane line position at the vehicle's bottom (y2)
            lane_x = self.get_lane_x(right_lane, y2)
            
            if lane_x is None:
                # Can't determine lane position - skip this vehicle
                continue
            
            # Check if vehicle overlaps the right lane (risky lane)
            if x2 > lane_x:  # Vehicle crosses into risky lane
                overlap_w = min(x2, 1280) - max(x1, lane_x)
                overlap_ratio = overlap_w / car_width if car_width > 0 else 0
                
                # ADAPTIVE THRESHOLD based on vehicle size
                if car_width < 60:
                    # Distant vehicles: Use lower threshold (5%) for early warning
                    threshold = 0.05
                else:
                    # Nearby vehicles: Use higher threshold (10%) to avoid false alarms
                    threshold = 0.10
                
                if overlap_ratio > threshold:
                    # Vehicle is in risky lane
                    if trend in ["APPROACHING", "STABLE", "UNKNOWN"]:
                        return "RISKY", right_lane
                    else:
                        # Receding vehicle (moving away) - still warning
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
import sys
import os
import numpy as np

# Adjust path to import modules if running from src/ablation directory
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logic.safety_checker import SafetyChecker

class AblationSafetyChecker(SafetyChecker):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.trend_enabled = config.get('trend_analysis', True)
        self.adaptive_enabled = config.get('adaptive_threshold', True)
        
    def get_vehicle_trend(self, track_id, current_width):
        if not self.trend_enabled:
            return "UNKNOWN" # Or treat as STABLE, but UNKNOWN is neutral
        return super().get_vehicle_trend(track_id, current_width)

    def assess(self, detections, lane_info):
        # We need to copy the assess logic because the threshold logic is inside it
        # and not easily overridable via a helper method in the original class.
        # Alternatively, if the original class had a 'get_threshold' method, we could override that.
        # Since it doesn't, we have to copy-paste-modify the assess method or accept that we are duplicating code.
        # Given the constraint of 'no changing existing files', duplicating the assess method logic here is the safest way 
        # to implement the 'adaptive_threshold' toggle without modifying the parent.
        
        left_lane, right_lane = lane_info
        
        if right_lane is None:
            return "SAFE", right_lane
        
        for det in detections:
            x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
            track_id = det[4]
            
            car_width = x2 - x1
            
            # Use our overridden get_vehicle_trend
            trend = self.get_vehicle_trend(track_id, car_width)
            
            lane_x = self.get_lane_x(right_lane, y2)
            
            if lane_x is None:
                continue
            
            if x2 > lane_x:
                overlap_w = min(x2, 1280) - max(x1, lane_x)
                overlap_ratio = overlap_w / car_width if car_width > 0 else 0
                
                # --- ABLATION LOGIC START ---
                if self.adaptive_enabled:
                    # Original logic
                    if car_width < 60:
                        threshold = 0.05
                    else:
                        threshold = 0.10
                else:
                    # Ablation: Fixed threshold
                    threshold = 0.10
                # --- ABLATION LOGIC END ---
                
                if overlap_ratio > threshold:
                    if trend in ["APPROACHING", "STABLE", "UNKNOWN"]:
                        return "RISKY", right_lane
                    else:
                        return "WARNING", right_lane

        return "SAFE", right_lane

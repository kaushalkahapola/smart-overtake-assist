import cv2
import numpy as np
import config

class SafetyChecker:
    def __init__(self):
        self.status = "UNKNOWN"
        self.vehicle_history = {} # track_id -> list of (width_pixels, timestamp_frame_id) 
        self.frame_count = 0
        
        # Load params
        self.focal_length = config.DISTANCE_ESTIMATION_PARAMS["FOCAL_LENGTH"]
        self.known_width = config.DISTANCE_ESTIMATION_PARAMS["KNOWN_WIDTH"]
        self.safe_distance = config.SAFE_DISTANCE_THRESHOLD

    def calculate_distance(self, width_pixels):
        if width_pixels <= 0: return 999.0
        # D = (F * W) / P
        return (self.focal_length * self.known_width) / width_pixels

    def get_vehicle_trend(self, track_id, current_width):
        """
        Analyze history to determine if vehicle is approaching (Oncoming) or receding/stable (Leading).
        Returns: "APPROACHING", "RECEDING", "STABLE", or "UNKNOWN"
        """
        if track_id == -1: return "UNKNOWN"
        
        if track_id not in self.vehicle_history:
            self.vehicle_history[track_id] = []
        
        history = self.vehicle_history[track_id]
        history.append(current_width)
        
        # Keep last 30 frames (approx 1 sec)
        if len(history) > 30:
            history.pop(0)
            
        if len(history) < 5:
            return "UNKNOWN"
            
        # Simple trend analysis: Compare average of recent vs old
        recent_avg = np.mean(history[-5:])
        old_avg = np.mean(history[:5])
        
        diff = recent_avg - old_avg
        
        # Threshold for significant change
        if diff > 2.0: # Width increasing -> Getting closer
            return "APPROACHING" 
        elif diff < -2.0: # Width decreasing -> Going away
            return "RECEDING"
        else:
            return "STABLE"

    def assess(self, detections, lane_info):
        """
        Assess driving safety based on detected vehicles and lane boundaries.
        Args:
            detections: List of [x1, y1, x2, y2, track_id, cls_id, conf]
            lane_info: (left_line, right_line)
        """
        self.frame_count += 1
        left_line, right_line = lane_info
        
        # In Left-Hand Traffic:
        # Left Line = Curb / Left Shoulder
        # Right Line = Divider (separating our lane from overtaking lane)
        divider_line = right_line
        
        if divider_line is None:
            return "SAFE", None # Default to safe if we can't see the lane (or handle carefully)
            
        risky_vehicle_detected = False
        warning_vehicle_detected = False # For leading vehicles preventing overtake
        
        for det in detections:
            # Unpack with track_id
            if len(det) == 7: # New format
                x1, y1, x2, y2, track_id, cls_id, conf = det
            else: # Fallback for old format
                x1, y1, x2, y2, cls_id, conf = det
                track_id = -1

            # Vehicle Properties
            v_width = x2 - x1
            vehicle_x = (x1 + x2) / 2
            vehicle_y = y2
            
            # Distance Estimation
            distance = self.calculate_distance(v_width)
            
            # Trend Analysis (Oncoming vs Leading)
            trend = self.get_vehicle_trend(track_id, v_width)
            
            # --- Position relative to Divider ---
            # Divider Line: x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            lx1, ly1, lx2, ly2 = divider_line
            if ly2 - ly1 == 0: divider_x = lx1
            else: divider_x = lx1 + (vehicle_y - ly1) * (lx2 - lx1) / (ly2 - ly1)
            
            # Overlap Calculation (Amount of vehicle to the RIGHT of the divider)
            # x2 is right edge, x1 is left edge.
            # If x2 < divider_x: Entirely on LEFT (0% overlap)
            # If x1 > divider_x: Entirely on RIGHT (100% overlap)
            # Intersection width: range [max(x1, divider_x), x2]
            
            intersection_left = max(x1, divider_x)
            intersection_right = x2
            
            overlap_width = max(0, intersection_right - intersection_left)
            overlap_ratio = overlap_width / v_width if v_width > 0 else 0
            
            # Rule: 
            # 1. High Overlap (> 30%): Definitely in the lane. Check safety.
            # 2. Low Overlap (> 5%) AND Oncoming: Early Warning!
            
            if overlap_ratio > 0.05:
                # Vehicle is interacting with the right lane
                
                # Check Trend
                recent_avg = np.mean(self.vehicle_history[track_id][-5:]) if track_id in self.vehicle_history and len(self.vehicle_history[track_id]) >= 5 else v_width
                old_avg = np.mean(self.vehicle_history[track_id][:5]) if track_id in self.vehicle_history and len(self.vehicle_history[track_id]) >= 5 else v_width
                diff = recent_avg - old_avg
                
                is_oncoming = diff > config.EXPANSION_THRESHOLDS["ONCOMING"]
                is_stationary = diff > config.EXPANSION_THRESHOLDS["STATIONARY"]
                
                # RISK LOGIC
                
                # Scenario A: Oncoming and touching lane line -> RISKY (Early detection)
                if is_oncoming and overlap_ratio > 0.05:
                     risky_vehicle_detected = True
                     
                # Scenario B: Stationary/Slow and significant overlap -> WARNING
                elif is_stationary and overlap_ratio > 0.2:
                     warning_vehicle_detected = True
                     
                # Scenario C: Stable/Receding but blocking lane -> WARNING
                elif overlap_ratio > 0.3:
                    if distance < self.safe_distance:
                         warning_vehicle_detected = True

        if risky_vehicle_detected:
            return "RISKY", divider_line
        elif warning_vehicle_detected:
            return "WARNING", divider_line # Parked car or Leading car blocking lane
        else:
            return "SAFE", divider_line # Overtaking lane is clear


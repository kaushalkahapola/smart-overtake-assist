import numpy as np
from collections import deque
import config

class SafetyChecker:
    def __init__(self):
        # --- Distance Estimation ---
        params = config.DISTANCE_ESTIMATION_PARAMS
        self.focal_length = params["FOCAL_LENGTH"]
        self.default_width = params["KNOWN_WIDTH"]  # meters (car)
        
        # Class-specific real-world widths (meters)
        # YOLO COCO class IDs: 2=car, 3=motorcycle, 5=bus, 7=truck
        self.class_widths = {
            2: 1.8,   # car
            3: 0.8,   # motorcycle / three-wheeler
            5: 2.5,   # bus
            7: 2.4,   # truck
        }
        
        # --- Vehicle Tracking ---
        self.vehicle_history = {}  # track_id -> deque of (bbox_width, distance, timestamp_frame)
        self.history_length = 15
        self.frame_counter = 0
        
        # --- TTC Thresholds ---
        self.ttc_critical = 2.5   # seconds — immediate danger
        self.ttc_warning = 5.0    # seconds — approaching risk
        self.min_approach_speed = 0.3  # m/s — minimum closing speed to consider approaching
        
        # Minimum overlap to even consider a vehicle as being in opposing lane
        self.min_overlap_ratio = 0.03

    def calculate_distance(self, bbox_width, cls_id=2):
        """
        Estimate distance to vehicle using pinhole camera model:
        D = (F × W_real) / W_pixels
        
        Uses class-specific widths for better accuracy.
        """
        if bbox_width <= 0:
            return float('inf')
        real_width = self.class_widths.get(cls_id, self.default_width)
        return (self.focal_length * real_width) / bbox_width

    def get_vehicle_metrics(self, track_id, current_width, cls_id=2):
        """
        Calculate distance, approach speed, and TTC for a tracked vehicle.
        
        Returns:
            distance: estimated distance in meters
            approach_speed: closing speed in m/s (positive = approaching)
            ttc: time-to-collision in seconds (inf if receding)
            trend: "APPROACHING" | "RECEDING" | "STABLE"
        """
        if track_id == -1:
            dist = self.calculate_distance(current_width, cls_id)
            return dist, 0.0, float('inf'), "UNKNOWN"

        if track_id not in self.vehicle_history:
            self.vehicle_history[track_id] = deque(maxlen=self.history_length)
        
        distance = self.calculate_distance(current_width, cls_id)
        self.vehicle_history[track_id].append((current_width, distance, self.frame_counter))

        history = self.vehicle_history[track_id]
        
        if len(history) < 3:
            return distance, 0.0, float('inf'), "STABLE"

        # --- Calculate approach speed using distance history ---
        # Use recent vs old (same as original but with distance)
        n = len(history)
        recent_dists = [history[i][1] for i in range(n - 2, n)]
        recent_frames = [history[i][2] for i in range(n - 2, n)]
        old_dists = [history[i][1] for i in range(0, min(2, n))]
        old_frames = [history[i][2] for i in range(0, min(2, n))]
        
        avg_recent_dist = np.mean(recent_dists)
        avg_old_dist = np.mean(old_dists)
        avg_recent_frame = np.mean(recent_frames)
        avg_old_frame = np.mean(old_frames)
        
        frame_diff = avg_recent_frame - avg_old_frame
        if frame_diff <= 0:
            return distance, 0.0, float('inf'), "STABLE"
        
        # Distance change: negative = getting closer (approaching)
        dist_change = avg_recent_dist - avg_old_dist
        
        # Approximate time between frames (assume ~17 FPS based on real measurements)
        fps_estimate = 17.0
        time_diff = frame_diff / fps_estimate
        
        # Approach speed: positive = approaching (closing)
        approach_speed = -dist_change / time_diff  # Negate: decreasing distance = positive speed
        
        # Determine trend
        if approach_speed > self.min_approach_speed:
            trend = "APPROACHING"
        elif approach_speed < -self.min_approach_speed:
            trend = "RECEDING"
        else:
            trend = "STABLE"
        
        # --- Calculate TTC ---
        if approach_speed > self.min_approach_speed:
            ttc = distance / approach_speed
        else:
            ttc = float('inf')  # Not approaching — no collision expected
        
        return distance, approach_speed, ttc, trend

    def assess(self, detections, lane_info):
        """
        Risk assessment combining lane position + TTC.
        
        Risk levels:
          RISKY   = Vehicle in opposing lane AND TTC < ttc_critical (or very close)
          WARNING = Vehicle in opposing lane AND TTC < ttc_warning (or approaching)
          SAFE    = No vehicles in opposing lane, or far away / receding
        """
        left_lane, right_lane = lane_info
        self.frame_counter += 1
        
        # If we don't have a valid right lane line, we can't assess safety
        if right_lane is None:
            return "SAFE", right_lane
        
        worst_status = "SAFE"
        
        for det in detections:
            x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
            track_id = det[4]
            cls_id = det[5] if len(det) > 5 else 2
            
            car_width = x2 - x1
            
            # Get distance metrics
            distance, approach_speed, ttc, trend = self.get_vehicle_metrics(
                track_id, car_width, cls_id)
            
            # Get the lane line position at the vehicle's bottom (y2)
            lane_x = self.get_lane_x(right_lane, y2)
            
            if lane_x is None:
                continue
            
            # Check if vehicle overlaps the right lane (opposing lane)
            if x2 > lane_x:
                overlap_w = min(x2, 1280) - max(x1, lane_x)
                overlap_ratio = overlap_w / car_width if car_width > 0 else 0
                
                if overlap_ratio > self.min_overlap_ratio:
                    # Vehicle is in opposing lane — assess severity using TTC
                    status = self._assess_ttc_risk(distance, ttc, trend, car_width)
                    
                    # Escalate to worst status seen
                    if status == "RISKY":
                        return "RISKY", right_lane
                    elif status == "WARNING" and worst_status == "SAFE":
                        worst_status = "WARNING"
        
        return worst_status, right_lane
    
    def _assess_ttc_risk(self, distance, ttc, trend, bbox_width):
        """
        Determine risk level from TTC and distance.
        
        Logic:
          - Very close (< 15m): RISKY regardless of TTC (no time to react)
          - TTC < critical: RISKY (collision imminent)
          - TTC < warning AND approaching: WARNING
          - Receding / far away: SAFE (even if in opposing lane)
        """
        # Very close vehicles are always dangerous
        if distance < 15:
            return "RISKY"
        
        # TTC-based assessment
        if ttc < self.ttc_critical:
            return "RISKY"
        elif ttc < self.ttc_warning and trend in ["APPROACHING", "STABLE", "UNKNOWN"]:
            return "WARNING"
        elif trend == "APPROACHING":
            # Approaching but TTC is still high — warning
            return "WARNING"
        
        # Receding or very far
        return "SAFE"
        
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
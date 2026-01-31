import config

class BaselineChecker:
    def __init__(self):
        # Naive approach: Fixed distance threshold only
        # No history, No trend analysis, No "Distant Heuristic"
        # 50 meters approx 36 pixels width ((1000 * 1.8) / 50)
        self.width_threshold = 36 

    def assess(self, detections, lane_info):
        left_lane, right_lane = lane_info
        
        # If no lane detected, we can't judge (SAFE default for baseline)
        if right_lane is None:
            return "SAFE", None

        for det in detections:
            x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
            car_width = x2 - x1
            
            # Check if car overlaps the right lane
            lane_x = self.get_lane_x(right_lane, y2)
            
            if lane_x is not None:
                if x2 > lane_x: # Car is in the lane
                    # Calculate simple overlap
                    overlap_w = min(x2, 1280) - max(x1, lane_x)
                    overlap_ratio = overlap_w / car_width
                    
                    # BASELINE RULE:
                    # If it's in the lane (>10% overlap) AND 
                    # it's closer than 50m (width > 36px) -> PANIC!
                    # It ignores whether the car is moving away (Safe) or coming closer (Risky).
                    if overlap_ratio > 0.10 and car_width > self.width_threshold:
                        return "RISKY", right_lane

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
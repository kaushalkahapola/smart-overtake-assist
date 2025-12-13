import numpy as np

class SafetyChecker:
    def __init__(self):
        self.status = "UNKNOWN"

    def assess(self, detections, lanes, frame_width=1280):
        """
        Assess safety based on detections and lane positions.
        - detections: List of [x1, y1, x2, y2, cls, conf]
        - lanes: [left_line, right_line] where line is [x1, y1, x2, y2]
        """
        if lanes is None:
            self.status = "NO LANES"
            return self.status

        # Assuming Left-Hand Traffic (Sri Lanka): Overtaking is on the RIGHT.
        # We need to check if the Right Lane (oncoming) is clear.
        
        # Define the Right Lane Area
        # Ideally, we would detect the far-right boundary, but LaneDetector gives current lane.
        # So "Right Lane" effectively means the lane to the right of the current driving lane?
        # Or does LaneDetector return the driving lane boundaries? Usually yes.
        # So the "Overtaking Lane" is to the RIGHT of the right_line of the current lane.
        
        # Simplification: Let's assume we are looking for vehicles in the right half of the image
        # that are NOT in the current lane, or we check if the current lane IS the overtaking lane (lane change).
        
        # Better approach for Single-Camera Two-Lane Road:
        # The driving lane is bounded by [left_line, right_line].
        # The Overtaking Lane is to the Right of [right_line].
        
        left_line, right_line = lanes
        
        # Check for vehicles in the Overtaking Zone (Right of the Right Line)
        # x_coordinate > right_line_x
        
        risky_vehicle_count = 0
        
        for det in detections:
            x1, y1, x2, y2, cls, conf = det
            car_center_x = (x1 + x2) / 2
            car_bottom_y = y2
            
            # Find x-coordinate of the right lane line at this y-level
            # Line equation x = (y - c) / m
            # We need slope (m) and intercept (c) of the right line
            rx1, ry1, rx2, ry2 = right_line
            
            if rx2 - rx1 == 0: # Vertical line
                lane_x_at_car = rx1
            else:
                m = (ry2 - ry1) / (rx2 - rx1)
                c = ry1 - m * rx1
                if m == 0: m = 0.001
                lane_x_at_car = (car_bottom_y - c) / m
            
            # If car is to the RIGHT of the current lane boundary, it's in the overtaking lane (or off-road)
            # And if it's close enough (y is large), it's a risk.
            if car_center_x > lane_x_at_car:
                risky_vehicle_count += 1
        
        if risky_vehicle_count > 0:
            self.status = "RISKY - ONCOMING TRAFFIC"
        else:
            self.status = "SAFE TO OVERTAKE"
            
        return self.status

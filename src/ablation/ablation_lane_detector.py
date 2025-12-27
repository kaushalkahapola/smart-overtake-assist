import sys
import os
import numpy as np

# Adjust path to import modules if running from src/ablation directory
# But typically this is imported by a runner which sets the path.
# If we want to be safe for standalone testing:
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perception.lane_detector import LaneDetector

class AblationLaneDetector(LaneDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.smoothing_enabled = config.get('temporal_smoothing', True)
        
    def smooth_lines_with_forecasting(self, current_line, history, side='left'):
        # If temporal smoothing is disabled, return current line immediately
        # This effectively gives us the "Raw" line for this frame (after averaging hough lines)
        if not self.smoothing_enabled:
            return current_line
            
        # Otherwise, use the original logic
        return super().smooth_lines_with_forecasting(current_line, history, side)

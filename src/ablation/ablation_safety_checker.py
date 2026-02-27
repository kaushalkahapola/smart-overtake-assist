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
        
    def get_vehicle_metrics(self, track_id, current_width, cls_id=2):
        """Override: disable trend analysis if config says so."""
        distance, approach_speed, ttc, trend = super().get_vehicle_metrics(
            track_id, current_width, cls_id)
        
        if not self.trend_enabled:
            trend = "UNKNOWN"
            ttc = float('inf')
            approach_speed = 0.0
        
        return distance, approach_speed, ttc, trend

    def _assess_ttc_risk(self, distance, ttc, trend, bbox_width):
        """Override: disable adaptive threshold if config says so."""
        if self.adaptive_enabled:
            return super()._assess_ttc_risk(distance, ttc, trend, bbox_width)
        else:
            # Ablation: simple overlap-only logic (no distance/TTC)
            # Just check lane overlap — always RISKY if in opposing lane
            return "RISKY"

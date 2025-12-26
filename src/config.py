import os

# ROI Settings (Region of Interest)
# [x, y] coordinates for a polygon masking the lane area
ROI_VERTICES = [] 

# Safety Thresholds
SAFE_DISTANCE_THRESHOLD = 50.0  # meters (approximate)
TTC_THRESHOLD = 2.5           # seconds

# Model Paths
# Using 'yolov8s.pt' (Small) instead of Nano for better detection of distant bikes
# Resolve path relative to this source file (src/config.py)
# assets is in application/data, src is in application/src
_current_dir = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH = os.path.join(_current_dir, "..", "data", "models", "yolov8n.pt")
TEST_VIDEO_PATH = os.path.join(_current_dir, "..", "data", "videos", "test_1.mp4")

# Lane Detection Defaults
LANE_DETECTION_DEFAULTS = {
    "hough_canny_low": 50,
    "hough_canny_high": 150,
    "hough_rho": 2,
    "hough_theta": 1,
    "hough_threshold": 15,
    "hough_min_line_len": 40,
    "hough_max_line_gap": 20,
    "roi_top_width": 100,
    "roi_bottom_width": 600,
    "roi_height_pct": 0.6,
    "roi_bottom_offset": 50,
    "smooth_factor": 5,
    "white_l_min": 200,
    "yellow_h_min": 15,
    "yellow_h_max": 35,
    "yellow_s_min": 100,
    
    # Forecasting and Stability Parameters
    "forecast_enabled": True,
    "bottom_anchor_threshold": 50,  # max px deviation for bottom point
    "min_lane_separation": 200,     # min px between left and right lanes at bottom
    "max_lane_width": 450,          # max px lane width at bottom (rejects far-right line)
    "lane_width_tolerance": 0.3,    # 30% width change tolerance
    "forecast_weight": 0.7,         # weight for predicted position vs new detection
    "left_lane_max_x_ratio": 0.45   # left lane bottom_x should be < 45% of frame width
}

# Distance Estimation Constants
# F = (P * D) / W
# Assume: Focal Length (pixels) approx 1000 for 720p (Needs calibration)
# Car Width approx 1.8 meters
DISTANCE_ESTIMATION_PARAMS = {
    "FOCAL_LENGTH": 1000.0, 
    "KNOWN_WIDTH": 1.8  # meters
}

# Expansion Thresholds (Width change per frame avg)
# Used to distinguish Oncoming (Fast expansion) from Parked/Slower (Slow expansion)
# These values need tuning based on video resolution and frame rate
# Assuming 30FPS and 720p
EXPANSION_THRESHOLDS = {
    "ONCOMING": 2.0,  # Lowered to catch approach earlier
    "STATIONARY": 0.5 
}

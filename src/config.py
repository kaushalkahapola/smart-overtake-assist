
# ROI Settings (Region of Interest)
# [x, y] coordinates for a polygon masking the lane area
ROI_VERTICES = [] 

# Safety Thresholds
SAFE_DISTANCE_THRESHOLD = 50.0  # meters (approximate)
TTC_THRESHOLD = 2.5           # seconds

# Model Paths
# Using 'yolov8s.pt' (Small) instead of Nano for better detection of distant bikes
YOLO_MODEL_PATH = "../assets/models/yolov8s.pt"

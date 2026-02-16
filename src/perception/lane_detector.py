import cv2
import numpy as np

class LaneDetector:
    """
    Lane Detector based on src/perception/references/lane5.py
    Features: HLS Color Filtering, Spatial Hough Filtering, Smoothing, UI Tuning.
    """
    def __init__(self):
        import config 
        defaults = config.LANE_DETECTION_DEFAULTS

        # --- Hough Params Default ---
        self.hough_canny_low = defaults["hough_canny_low"]
        self.hough_canny_high = defaults["hough_canny_high"]
        self.hough_rho = defaults["hough_rho"]
        self.hough_theta = defaults["hough_theta"]
        self.hough_threshold = defaults["hough_threshold"]
        self.hough_min_line_len = defaults["hough_min_line_len"]
        self.hough_max_line_gap = defaults["hough_max_line_gap"]

        # --- ROI Defaults ---
        self.roi_top_width = defaults["roi_top_width"]
        self.roi_bottom_width = defaults["roi_bottom_width"]
        self.roi_height_pct = defaults["roi_height_pct"]
        self.roi_bottom_offset = defaults["roi_bottom_offset"]

        # --- Smoothing Params ---
        self.smooth_factor = defaults["smooth_factor"]
        self.left_history = []
        self.right_history = []
        
        # Outlier Rejection
        self.left_rejections = 0
        self.right_rejections = 0
        self.max_rejections = 3
        
        # Rejection Thresholds (Legacy - will be replaced by forecasting)
        self.slope_reject_thresh = 0.3 
        self.bottom_x_reject_thresh = 100.0 
        
        # --- Forecasting Parameters ---
        self.forecast_enabled = defaults["forecast_enabled"]
        self.bottom_anchor_threshold = defaults["bottom_anchor_threshold"]
        self.min_lane_separation = defaults["min_lane_separation"]
        self.max_lane_width = defaults["max_lane_width"]
        self.lane_width_tolerance = defaults["lane_width_tolerance"]
        self.forecast_weight = defaults["forecast_weight"]
        self.left_lane_max_x_ratio = defaults["left_lane_max_x_ratio"]
        
        # Lane Width Tracking
        self.lane_width_history = []
        self.expected_lane_width = None

        # --- Color Params ---
        self.white_l_min = defaults["white_l_min"]
        self.yellow_h_min = defaults["yellow_h_min"]
        self.yellow_h_max = defaults["yellow_h_max"]
        self.yellow_s_min = defaults["yellow_s_min"]
        
        self.init_ui()

    def init_ui(self):
        def nothing(x): pass
        cv2.namedWindow('Lane Tuning', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Lane Tuning', 400, 900)
        
        # macOS Fix: Show a placeholder to initialize the window and wait for the event loop
        # Increased height from 600 to 900 to prevent trackbar overlap on macOS
        placeholder = np.zeros((900, 400, 3), dtype=np.uint8)
        cv2.imshow('Lane Tuning', placeholder)
        cv2.waitKey(100) 
        
        # --- Hough Controls ---
        cv2.createTrackbar('Canny Low', 'Lane Tuning', self.hough_canny_low, 255, nothing)
        cv2.createTrackbar('Canny High', 'Lane Tuning', self.hough_canny_high, 255, nothing)
        cv2.createTrackbar('Threshold', 'Lane Tuning', self.hough_threshold, 200, nothing)
        cv2.createTrackbar('Min Len', 'Lane Tuning', self.hough_min_line_len, 200, nothing)
        cv2.createTrackbar('Max Gap', 'Lane Tuning', self.hough_max_line_gap, 200, nothing)
        
        # --- Color Controls ---
        cv2.createTrackbar('White L', 'Lane Tuning', self.white_l_min, 255, nothing)
        cv2.createTrackbar('Yel H Min', 'Lane Tuning', self.yellow_h_min, 179, nothing)
        cv2.createTrackbar('Yel H Max', 'Lane Tuning', self.yellow_h_max, 179, nothing)
        cv2.createTrackbar('Yel S Min', 'Lane Tuning', self.yellow_s_min, 255, nothing)
        
        # --- ROI Controls ---
        cv2.createTrackbar('ROI Top W', 'Lane Tuning', self.roi_top_width, 400, nothing)
        cv2.createTrackbar('ROI Bot W', 'Lane Tuning', self.roi_bottom_width, 800, nothing)
        cv2.createTrackbar('ROI H %', 'Lane Tuning', int(self.roi_height_pct*100), 100, nothing)
        cv2.createTrackbar('ROI Bot Off', 'Lane Tuning', self.roi_bottom_offset, 200, nothing)

    def update_params_from_ui(self):
        self.hough_canny_low = cv2.getTrackbarPos('Canny Low', 'Lane Tuning')
        self.hough_canny_high = cv2.getTrackbarPos('Canny High', 'Lane Tuning')
        self.hough_threshold = max(1, cv2.getTrackbarPos('Threshold', 'Lane Tuning'))
        self.hough_min_line_len = max(1, cv2.getTrackbarPos('Min Len', 'Lane Tuning'))
        self.hough_max_line_gap = max(1, cv2.getTrackbarPos('Max Gap', 'Lane Tuning'))
        
        self.white_l_min = cv2.getTrackbarPos('White L', 'Lane Tuning')
        self.yellow_h_min = cv2.getTrackbarPos('Yel H Min', 'Lane Tuning')
        self.yellow_h_max = max(self.yellow_h_min, cv2.getTrackbarPos('Yel H Max', 'Lane Tuning'))
        self.yellow_s_min = cv2.getTrackbarPos('Yel S Min', 'Lane Tuning')
        
        self.roi_top_width = cv2.getTrackbarPos('ROI Top W', 'Lane Tuning')
        self.roi_bottom_width = cv2.getTrackbarPos('ROI Bot W', 'Lane Tuning')
        self.roi_height_pct = cv2.getTrackbarPos('ROI H %', 'Lane Tuning') / 100.0
        self.roi_bottom_offset = cv2.getTrackbarPos('ROI Bot Off', 'Lane Tuning')

    def isolate_color_edges(self, img):
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        
        # White - High Lightness
        lower_white = np.array([0, self.white_l_min, 0])
        upper_white = np.array([255, 255, 255])
        white_mask = cv2.inRange(hls, lower_white, upper_white)
        
        # Yellow - Specific Hue and Saturation
        lower_yellow = np.array([self.yellow_h_min, 0, self.yellow_s_min])
        upper_yellow = np.array([self.yellow_h_max, 255, 255])
        yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)
        
        combined = cv2.bitwise_or(white_mask, yellow_mask)
        return combined

    def region_of_interest(self, img):
        h, w = img.shape[:2]
        center_x = w // 2
        top_y = int(h * (1 - self.roi_height_pct))
        bottom_y = h - self.roi_bottom_offset
        
        pts = np.array([[
            (center_x - self.roi_bottom_width, bottom_y),
            (center_x + self.roi_bottom_width, bottom_y),
            (center_x + self.roi_top_width, top_y),
            (center_x - self.roi_top_width, top_y)
        ]], dtype=np.int32)
        
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, pts, 255)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image, pts

    def average_slope_bottom_x(self, lines, img_shape):
        left_lines = []
        left_weights = []
        right_lines = []
        right_weights = []
        
        if lines is None:
            return None, None
            
        h, w = img_shape[:2]
        center_x = w // 2
        
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 == x1: continue # Ignore vertical lines
                slope = (y2 - y1) / (x2 - x1)
                
                # Check NaNs or Infs
                if np.isnan(slope) or np.isinf(slope): continue

                if abs(slope) < 1e-3: continue
                # Calculate Bottom X (where line hits bottom of screen)
                # y = m*x + b => x = (y - b)/m
                # b = y1 - m*x1
                bottom_x = (h - (y1 - slope * x1)) / slope
                
                length = np.sqrt((y1 - y2)**2 + (x1 - x2)**2)
                
                # Filter based on slope AND spatial location
                if -0.9 < slope < -0.3: 
                    # Left line: Should be on left side (symmetric with right line)
                    if x1 < center_x and x2 < center_x:
                        left_lines.append((slope, bottom_x))
                        left_weights.append(length)
                elif 0.3 < slope < 0.9:
                    # Right line: Should be on right side
                    if x1 > center_x and x2 > center_x:
                        right_lines.append((slope, bottom_x))
                        right_weights.append(length)
        
        left_lane = None
        right_lane = None
        
        if len(left_lines) > 0:
            # Apply center-proximity bonus: prefer lines closer to center
            # This prevents confusion with far-left lines (curbs, road edges)
            boosted_left_weights = []
            for i, (slope, bottom_x) in enumerate(left_lines):
                # Distance from center (normalized)
                distance_from_center = abs(bottom_x - center_x) / w
                # Proximity bonus: closer to center = higher weight
                # Use exponential decay: lines far from center get penalized
                proximity_bonus = np.exp(-distance_from_center * 2)  # Decay factor = 2
                boosted_weight = left_weights[i] * (1 + proximity_bonus)
                boosted_left_weights.append(boosted_weight)
            
            left_lane = np.dot(boosted_left_weights, left_lines) / np.sum(boosted_left_weights)
            
        if len(right_lines) > 0:
            # INTELLIGENT ALGORITHM: Pick right line based on expected lane width
            if left_lane is not None:
                left_bottom_x = left_lane[1]
                
                # Calculate lane widths for all candidates
                candidate_widths = []
                for slope, bottom_x in right_lines:
                    width = abs(bottom_x - left_bottom_x)
                    candidate_widths.append(width)
                
                # ADAPTIVE SELECTION based on history
                if self.expected_lane_width is not None:
                    # We have historical data - pick candidate closest to expected width
                    best_idx = -1
                    min_deviation = float('inf')
                    
                    for i, width in enumerate(candidate_widths):
                        deviation = abs(width - self.expected_lane_width)
                        
                        # Prefer candidates close to expected width
                        if deviation < min_deviation:
                            min_deviation = deviation
                            best_idx = i
                    
                    # Use the best candidate
                    if best_idx >= 0:
                        right_lane = right_lines[best_idx]
                        
                        # Update expected width (slow adaptation)
                        self.expected_lane_width = 0.9 * self.expected_lane_width + 0.1 * candidate_widths[best_idx]
                else:
                    # NO HISTORY: Pick the NARROWEST lane (most likely correct)
                    # Assumption: single lane is narrower than combined lanes
                    min_width = min(candidate_widths)
                    best_idx = candidate_widths.index(min_width)
                    right_lane = right_lines[best_idx]
                    
                    # Initialize expected width
                    self.expected_lane_width = min_width
            else:
                # No left line - use center proximity
                boosted_right_weights = []
                for i, (slope, bottom_x) in enumerate(right_lines):
                    distance_from_center = abs(bottom_x - center_x) / w
                    proximity_bonus = np.exp(-distance_from_center * 2)
                    boosted_weight = right_weights[i] * (1 + proximity_bonus)
                    boosted_right_weights.append(boosted_weight)
                
                right_lane = np.dot(boosted_right_weights, right_lines) / np.sum(boosted_right_weights)
            
        return left_lane, right_lane

    def forecast_lane_position(self, history):
        """
        Predicts next lane position based on historical trend.
        Returns predicted (slope, bottom_x) or None if insufficient data.
        """
        if len(history) < 3:
            return None
        
        # Extract recent bottom_x positions
        recent_bottom_x = [h[1] for h in history[-3:]]
        recent_slopes = [h[0] for h in history[-3:]]
        
        # Calculate velocity (average change per frame)
        bottom_x_velocity = (recent_bottom_x[-1] - recent_bottom_x[0]) / 2.0
        slope_velocity = (recent_slopes[-1] - recent_slopes[0]) / 2.0
        
        # Predict next position
        predicted_bottom_x = recent_bottom_x[-1] + bottom_x_velocity
        predicted_slope = recent_slopes[-1] + slope_velocity
        
        return (predicted_slope, predicted_bottom_x)
    
    def validate_lane_width(self, left_lane, right_lane):
        """
        Validates that lane width is consistent with history.
        Returns True if valid, False if suspicious.
        """
        if left_lane is None or right_lane is None:
            return True  # Can't validate without both lanes
        
        current_width = abs(right_lane[1] - left_lane[1])  # bottom_x difference
        
        # Check minimum separation
        if current_width < self.min_lane_separation:
            return False
        
        # Track expected width
        if self.expected_lane_width is None:
            self.expected_lane_width = current_width
            return True
        
        # Check if width changed too much
        width_ratio = current_width / self.expected_lane_width
        if width_ratio < (1 - self.lane_width_tolerance) or width_ratio > (1 + self.lane_width_tolerance):
            return False
        
        # Update expected width (slow adaptation)
        self.expected_lane_width = 0.9 * self.expected_lane_width + 0.1 * current_width
        
        return True
    
    def smooth_lines_with_forecasting(self, current_line, history, side='left'):
        """
        Improved smoothing using bottom-point anchor stability and temporal forecasting.
        
        Key Principle: The bottom portion of the lane should remain stable frame-to-frame.
        Only the top portion changes as the vehicle moves forward.
        """
        rejections = self.left_rejections if side == 'left' else self.right_rejections
        
        # No current detection
        if current_line is None:
            if len(history) > 0:
                # Use last known good position
                return np.mean(history, axis=0)
            return None
        
        # First detection - accept it
        if len(history) == 0:
            history.append(current_line)
            if side == 'left': 
                self.left_rejections = 0
            else: 
                self.right_rejections = 0
            return current_line
        
        # --- Forecasting Logic ---
        if self.forecast_enabled and len(history) >= 3:
            predicted = self.forecast_lane_position(history)
            
            if predicted is not None:
                pred_slope, pred_bottom_x = predicted
                curr_slope, curr_bottom_x = current_line
                
                # Check deviation from prediction
                bottom_x_deviation = abs(curr_bottom_x - pred_bottom_x)
                
                if bottom_x_deviation > self.bottom_anchor_threshold:
                    # Suspicious jump detected
                    rejections += 1
                    if side == 'left': 
                        self.left_rejections = rejections
                    else: 
                        self.right_rejections = rejections
                    
                    if rejections > self.max_rejections:
                        # Too many rejections - reset and accept new detection
                        history.clear()
                        history.append(current_line)
                        if side == 'left': 
                            self.left_rejections = 0
                        else: 
                            self.right_rejections = 0
                        return current_line
                    else:
                        # Use weighted blend of prediction and current
                        blended_slope = self.forecast_weight * pred_slope + (1 - self.forecast_weight) * curr_slope
                        blended_bottom_x = self.forecast_weight * pred_bottom_x + (1 - self.forecast_weight) * curr_bottom_x
                        return np.array([blended_slope, blended_bottom_x])
        
        # --- Legacy Smoothing (Fallback) ---
        avg = np.mean(history, axis=0)
        slope_diff = abs(current_line[0] - avg[0])
        bottom_x_diff = abs(current_line[1] - avg[1])
        
        if slope_diff < self.slope_reject_thresh and bottom_x_diff < self.bottom_x_reject_thresh:
            # Valid detection - add to history
            history.append(current_line)
            if len(history) > self.smooth_factor:
                history.pop(0)
            if side == 'left': 
                self.left_rejections = 0
            else: 
                self.right_rejections = 0
        else:
            # Outlier - increment rejections
            rejections += 1
            if side == 'left': 
                self.left_rejections = rejections
            else: 
                self.right_rejections = rejections
            
            if rejections > self.max_rejections:
                # Force reset
                history.clear()
                history.append(current_line)
                if side == 'left': 
                    self.left_rejections = 0
                else: 
                    self.right_rejections = 0
        
        # Return smoothed average
        if len(history) > 0:
            return np.mean(history, axis=0)
        return None

    def make_coordinates(self, image, line_params):
        if line_params is None:
            return None
        slope, bottom_x = line_params
        y1 = image.shape[0]
        y2 = int(image.shape[0] * (1 - self.roi_height_pct)) # Top of ROI
        
        if abs(slope) < 1e-3: return None
        
        # x1 is bottom_x (at y1)
        x1 = int(bottom_x)
        
        # x2 is at y2: x = (y - y1) / m + x1
        x2 = int((y2 - y1) / slope + x1)
        
        # Clamp to prevent overflow
        max_val = 10000
        min_val = -10000
        x1 = int(max(min_val, min(max_val, x1)))
        x2 = int(max(min_val, min(max_val, x2)))
        
        return np.array([x1, y1, x2, y2])

    def create_debug_mosaic(self, result, edges, masked_edges, roi_pts):
        # Exact visualization from lane5.py
        h, w = result.shape[:2]
        scale = 0.5
        small_h, small_w = int(h * scale), int(w * scale)
        
        # Prepare for stacking
        # Edges need to be 3-channel
        edges_color = np.dstack((edges, edges, edges))
        masked_edges_color = np.dstack((masked_edges, masked_edges, masked_edges))
        
        small_result = cv2.resize(result, (small_w, small_h))
        small_edges = cv2.resize(edges_color, (small_w, small_h))
        small_masked = cv2.resize(masked_edges_color, (small_w, small_h))
        
        # Canvas
        stacked = np.zeros((small_h * 2, small_w * 2, 3), dtype=np.uint8)
        
        # Top Left: Result
        stacked[0:small_h, 0:small_w] = small_result
        cv2.putText(stacked, "Result", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # Top Right: ROI Masked Edges
        stacked[0:small_h, small_w:2*small_w] = small_masked
        cv2.putText(stacked, "ROI Masked Edges", (small_w+20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # Bottom Left: Canny Edges (Full) -> Canny+Color
        stacked[small_h:2*small_h, 0:small_w] = small_edges
        cv2.putText(stacked, "Canny+Color", (20, small_h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        return stacked

    def draw_lane_overlay(self, image, left_line, right_line):
        result = image.copy()
        
        # Draw Extrapolated Lane (Green Polygon)
        if left_line is not None and right_line is not None:
            # Create a polygon
            pts = np.array([
                [left_line[0], left_line[1]], # Bottom Left
                [left_line[2], left_line[3]], # Top Left
                [right_line[2], right_line[3]], # Top Right
                [right_line[0], right_line[1]]  # Bottom Right
            ], np.int32)
            
            # Draw semi-transparent green filled polygon
            overlay = result.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            result = cv2.addWeighted(result, 1, overlay, 0.3, 0)
            
            # Draw lines
            cv2.line(result, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 255, 0), 5)
            cv2.line(result, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 255, 0), 5)
        
        elif left_line is not None:
             cv2.line(result, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 255, 0), 5)
        elif right_line is not None:
             cv2.line(result, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 255, 0), 5)
             
        return result

    def detect(self, frame):
        # 1. Update Parameters from UI
        self.update_params_from_ui()
        
        # 3. Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 4. Blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 5. Canny
        edges = cv2.Canny(blur, self.hough_canny_low, self.hough_canny_high)
        
        # 6. Color Thresholding
        color_edges = self.isolate_color_edges(frame)
        
        # 7. Combined Edges
        combined_edges = cv2.bitwise_or(edges, color_edges)
        
        # 8. ROI
        masked_edges, roi_pts = self.region_of_interest(combined_edges)
        
        # 9. Hough
        lines = cv2.HoughLinesP(masked_edges, 
                                rho=self.hough_rho, 
                                theta=self.hough_theta * np.pi/180, 
                                threshold=self.hough_threshold, 
                                minLineLength=self.hough_min_line_len, 
                                maxLineGap=self.hough_max_line_gap)
                                
        # 10. Average & Smooth with Forecasting
        left_raw, right_raw = self.average_slope_bottom_x(lines, frame.shape)
        
        left_smooth = self.smooth_lines_with_forecasting(left_raw, self.left_history, side='left')
        right_smooth = self.smooth_lines_with_forecasting(right_raw, self.right_history, side='right')
        
        # 10.5. Validate Lane Width (Spatial Constraint)
        if not self.validate_lane_width(left_smooth, right_smooth):
            # Invalid lane width - use previous frame's lanes
            if len(self.left_history) > 0 and len(self.right_history) > 0:
                left_smooth = np.mean(self.left_history, axis=0)
                right_smooth = np.mean(self.right_history, axis=0)
        
        # 11. Coordinates
        left_line = self.make_coordinates(frame, left_smooth)
        right_line = self.make_coordinates(frame, right_smooth)
        
        # 12. Visualization (Debug Mosaic)
        # Create Result Image with Green Overlay
        result_img = self.draw_lane_overlay(frame, left_line, right_line)
        
        debug_view = self.create_debug_mosaic(result_img, combined_edges, masked_edges, roi_pts)
        
        return left_line, right_line, debug_view

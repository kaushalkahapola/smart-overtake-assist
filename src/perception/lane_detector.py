import cv2
import numpy as np

class LaneDetector:
    """
    Lane Detector based on src/perception/references/lane5.py
    Features: HLS Color Filtering, Spatial Hough Filtering, Smoothing, UI Tuning.
    """
    def __init__(self):
        # --- Hough Params Default ---
        self.hough_canny_low = 50
        self.hough_canny_high = 150
        self.hough_rho = 2            
        self.hough_theta = 1          
        self.hough_threshold = 15     
        self.hough_min_line_len = 40  
        self.hough_max_line_gap = 20  

        # --- ROI Defaults ---
        self.roi_top_width = 100
        self.roi_bottom_width = 600
        self.roi_height_pct = 0.6
        self.roi_bottom_offset = 50

        # --- Smoothing Params ---
        self.smooth_factor = 5 
        self.left_history = []
        self.right_history = []
        
        # Outlier Rejection
        self.left_rejections = 0
        self.right_rejections = 0
        self.max_rejections = 3
        
        # Rejection Thresholds
        self.slope_reject_thresh = 0.3 
        self.bottom_x_reject_thresh = 100.0 

        # --- Color Params ---
        self.white_l_min = 200
        self.yellow_h_min = 15
        self.yellow_h_max = 35
        self.yellow_s_min = 100
        
        self.init_ui()

    def init_ui(self):
        def nothing(x): pass
        cv2.namedWindow('Lane Tuning', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Lane Tuning', 400, 600)
        
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
                    # Left lane: Should land on let side
                    if x1 < center_x and x2 < center_x:
                        left_lines.append((slope, bottom_x))
                        left_weights.append(length)
                elif 0.3 < slope < 0.9:
                    # Right lane: Should land on right side
                    if x1 > center_x and x2 > center_x:
                        right_lines.append((slope, bottom_x))
                        right_weights.append(length)
        
        left_lane = None
        right_lane = None
        
        if len(left_lines) > 0:
            left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights)
            
        if len(right_lines) > 0:
            right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights)
            
        return left_lane, right_lane

    def smooth_lines(self, current_line, history, side='left'):
        # Determine which rejection counter to use
        rejections = self.left_rejections if side == 'left' else self.right_rejections
        
        if current_line is None:
            if len(history) > 0:
                 return np.mean(history, axis=0)
            return None

        if len(history) == 0:
            history.append(current_line)
            if side == 'left': self.left_rejections = 0
            else: self.right_rejections = 0
            return current_line
        
        avg = np.mean(history, axis=0)
        slope_diff = abs(current_line[0] - avg[0])
        bottom_x_diff = abs(current_line[1] - avg[1])
        
        if slope_diff < self.slope_reject_thresh and bottom_x_diff < self.bottom_x_reject_thresh:
            # Valid detection
            history.append(current_line)
            if len(history) > self.smooth_factor:
                history.pop(0)
            if side == 'left': self.left_rejections = 0
            else: self.right_rejections = 0
        else:
            # Outlier detected
            rejections += 1
            if side == 'left': self.left_rejections = rejections
            else: self.right_rejections = rejections
            
            if rejections > self.max_rejections:
                 history.clear()
                 history.append(current_line)
                 if side == 'left': self.left_rejections = 0
                 else: self.right_rejections = 0

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
                                
        # 10. Average & Smooth
        left_raw, right_raw = self.average_slope_bottom_x(lines, frame.shape)
        
        left_smooth = self.smooth_lines(left_raw, self.left_history, side='left')
        right_smooth = self.smooth_lines(right_raw, self.right_history, side='right')
        
        # 11. Coordinates
        left_line = self.make_coordinates(frame, left_smooth)
        right_line = self.make_coordinates(frame, right_smooth)
        
        # 12. Visualization (Debug Mosaic)
        # Create Result Image with Green Overlay (Lane5 style)
        result_img = self.draw_lane_overlay(frame, left_line, right_line)
        
        debug_view = self.create_debug_mosaic(result_img, combined_edges, masked_edges, roi_pts)
        
        return left_line, right_line, debug_view

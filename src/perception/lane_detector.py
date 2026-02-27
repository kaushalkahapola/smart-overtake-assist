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

        # --- Lane Locking State Machine ---
        # Once a lane is confirmed for N frames, "lock" onto it.
        # In locked state, noise far from the lock position is discarded
        # BEFORE it enters averaging — making it structurally impossible
        # for noise to affect the output.
        self.left_lock = None       # Locked (slope, bottom_x) or None
        self.right_lock = None
        self.left_lock_confidence = 0
        self.right_lock_confidence = 0
        self.lock_threshold = 5     # Frames needed to lock
        self.lock_radius = 80       # px — only lines within this radius of lock survive
        self.lock_drift_rate = 0.1  # How fast lock adapts to gradual lane changes
        self.unlock_threshold = 8   # Consecutive misses before unlocking
        self.left_lock_misses = 0
        self.right_lock_misses = 0

        # --- Color Params ---
        self.white_l_min = defaults["white_l_min"]
        self.yellow_h_min = defaults["yellow_h_min"]
        self.yellow_h_max = defaults["yellow_h_max"]
        self.yellow_s_min = defaults["yellow_s_min"]
        
        self.init_ui()

    def init_ui(self):
        import platform
        system = platform.system()
        
        def nothing(x): pass
        cv2.namedWindow('Lane Tuning', cv2.WINDOW_NORMAL)
        
        if system == 'Darwin':  # macOS
            cv2.resizeWindow('Lane Tuning', 400, 900)
            # macOS Fix: Show a placeholder to initialize the window and wait for the event loop
            # Increased height from 600 to 900 to prevent trackbar overlap on macOS
            placeholder = np.zeros((900, 400, 3), dtype=np.uint8)
            cv2.imshow('Lane Tuning', placeholder)
            cv2.waitKey(100)
        else:  # Linux / Windows
            # Cross-platform Fix: Increase width to 600px to ensure trackbar names fit on Linux.
            # Use a small visible placeholder image (e.g., 50px tall) so the window height
            # doesn't exceed Linux 1080p screens, while keeping labels readable.
            cv2.resizeWindow('Lane Tuning', 600, 600)
            placeholder = np.zeros((50, 600, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Tuning Controls", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Lane Tuning', placeholder)
            cv2.waitKey(10)
        
        
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

    def _reject_outlier_lines(self, lines, weights):
        """
        Median-based outlier rejection using MAD (Median Absolute Deviation).
        Prevents a single noisy Hough line from corrupting the weighted average.
        Returns filtered (lines, weights).
        """
        if len(lines) < 3:
            return lines, weights  # Not enough data for meaningful rejection
        
        slopes = np.array([l[0] for l in lines])
        bottom_xs = np.array([l[1] for l in lines])
        
        # MAD-based rejection for slopes
        median_slope = np.median(slopes)
        mad_slope = max(np.median(np.abs(slopes - median_slope)), 0.1)
        
        # MAD-based rejection for bottom_x
        median_bx = np.median(bottom_xs)
        mad_bx = max(np.median(np.abs(bottom_xs - median_bx)), 30.0)
        
        filtered_lines = []
        filtered_weights = []
        for i, (slope, bottom_x) in enumerate(lines):
            slope_dev = abs(slope - median_slope) / mad_slope
            bx_dev = abs(bottom_x - median_bx) / mad_bx
            # Keep lines within 3× MAD on both metrics
            if slope_dev < 3.0 and bx_dev < 3.0:
                filtered_lines.append((slope, bottom_x))
                filtered_weights.append(weights[i])
        
        # Fallback: if everything was rejected, return originals
        if len(filtered_lines) == 0:
            return lines, weights
        
        return filtered_lines, filtered_weights

    def _cluster_select_lines(self, lines, weights, cluster_gap=40):
        """
        Density-based clustering of Hough lines by bottom_x position.
        Groups lines within cluster_gap pixels, then selects the cluster
        with the highest total line length (weight).
        
        A thick solid line produces many overlapping Hough segments in a
        tight band → highest total weight. Scattered noise dots produce
        isolated short segments → low total weight.
        """
        if len(lines) < 2:
            return lines, weights
        
        # Sort by bottom_x
        indexed = sorted(range(len(lines)), key=lambda i: lines[i][1])
        
        # Greedy clustering by bottom_x proximity
        clusters = []  # list of (indices_list)
        current_cluster = [indexed[0]]
        
        for k in range(1, len(indexed)):
            prev_bx = lines[indexed[k-1]][1]
            curr_bx = lines[indexed[k]][1]
            if abs(curr_bx - prev_bx) <= cluster_gap:
                current_cluster.append(indexed[k])
            else:
                clusters.append(current_cluster)
                current_cluster = [indexed[k]]
        clusters.append(current_cluster)
        
        # Pick cluster with highest total weight (line length)
        best_cluster = max(clusters, key=lambda c: sum(weights[i] for i in c))
        
        sel_lines = [lines[i] for i in best_cluster]
        sel_weights = [weights[i] for i in best_cluster]
        return sel_lines, sel_weights

    def _ema_smooth(self, history):
        """
        Exponentially weighted average of history.
        More recent entries get higher weight → smoother, more responsive output.
        """
        n = len(history)
        if n == 0:
            return None
        if n == 1:
            return np.array(history[0])
        
        alpha = 0.35
        wts = np.array([alpha * ((1 - alpha) ** (n - 1 - i)) for i in range(n)])
        wts /= wts.sum()
        
        slopes = np.array([h[0] for h in history])
        bottom_xs = np.array([h[1] for h in history])
        
        return np.array([np.dot(wts, slopes), np.dot(wts, bottom_xs)])

    def _apply_lane_lock(self, lines, weights, side='left'):
        """
        Pre-filter Hough lines using lane lock position.
        When locked: only lines within lock_radius of the locked bottom_x survive.
        When unlocked: all lines pass through.
        
        This is the key noise immunity mechanism — noise outside the lane
        is discarded BEFORE it can influence any averaging.
        """
        lock = self.left_lock if side == 'left' else self.right_lock
        confidence = self.left_lock_confidence if side == 'left' else self.right_lock_confidence
        
        # Not locked yet — let everything through
        if lock is None or confidence < self.lock_threshold:
            return lines, weights
        
        locked_bottom_x = lock[1]
        
        filtered_lines = []
        filtered_weights = []
        for i, (slope, bottom_x) in enumerate(lines):
            if abs(bottom_x - locked_bottom_x) <= self.lock_radius:
                filtered_lines.append((slope, bottom_x))
                filtered_weights.append(weights[i])
        
        # If lock filtered out EVERYTHING, return originals
        # (avoids total loss of detection — lock will degrade via misses)
        if len(filtered_lines) == 0:
            return lines, weights
        
        return filtered_lines, filtered_weights

    def _update_lane_lock(self, smoothed_line, side='left'):
        """
        Update the lane lock state machine after smoothing.
        
        States:
          UNLOCKED (confidence < lock_threshold): Building confidence
          LOCKED   (confidence >= lock_threshold): Noise filtered, lock drifts
        
        Transitions:
          Detection consistent with lock → confidence++, drift lock position
          Detection missing/inconsistent → misses++
          Too many misses → unlock (confidence = 0)
        """
        if side == 'left':
            lock = self.left_lock
            confidence = self.left_lock_confidence
            misses = self.left_lock_misses
        else:
            lock = self.right_lock
            confidence = self.right_lock_confidence
            misses = self.right_lock_misses
        
        if smoothed_line is None:
            # No detection — increment misses
            misses += 1
            if misses > self.unlock_threshold:
                lock = None
                confidence = 0
                misses = 0
        elif lock is None:
            # First ever lock — initialize
            lock = tuple(smoothed_line)
            confidence = 1
            misses = 0
        else:
            # Check if detection is consistent with current lock
            deviation = abs(smoothed_line[1] - lock[1])
            
            if deviation <= self.lock_radius:
                # Consistent — increase confidence and drift lock
                confidence = min(confidence + 1, self.lock_threshold + 10)
                misses = 0
                
                # Drift lock position (EMA) to follow curves
                new_slope = lock[0] * (1 - self.lock_drift_rate) + smoothed_line[0] * self.lock_drift_rate
                new_bx = lock[1] * (1 - self.lock_drift_rate) + smoothed_line[1] * self.lock_drift_rate
                lock = (new_slope, new_bx)
            else:
                # Inconsistent — this could be noise or a real lane change
                misses += 1
                if misses > self.unlock_threshold:
                    # Real lane change — unlock and re-initialize
                    lock = tuple(smoothed_line)
                    confidence = 1
                    misses = 0
                # Otherwise keep current lock (noise rejection)
        
        # Write back state
        if side == 'left':
            self.left_lock = lock
            self.left_lock_confidence = confidence
            self.left_lock_misses = misses
        else:
            self.right_lock = lock
            self.right_lock_confidence = confidence
            self.right_lock_misses = misses

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
                
                # Use length² as weight — makes long solid lines dominate
                # A 100px solid line gets 25× the weight of a 20px noise segment
                weight = length * length
                
                # Filter based on slope AND spatial location
                # Widened range for two-lane roads with tighter curves (Sri Lankan roads)
                mid_x = (x1 + x2) / 2  # Midpoint for curve-tolerant spatial check
                if -1.2 < slope < -0.25: 
                    # Left line: midpoint should be on left side
                    # (relaxed from both endpoints — allows curved lines crossing center)
                    if mid_x < center_x:
                        left_lines.append((slope, bottom_x))
                        left_weights.append(weight)
                elif 0.25 < slope < 1.2:
                    # Right line: midpoint should be on right side
                    if mid_x > center_x:
                        right_lines.append((slope, bottom_x))
                        right_weights.append(weight)
        
        # --- Lane Lock Pre-filter: discard noise far from locked position ---
        left_lines, left_weights = self._apply_lane_lock(
            left_lines, left_weights, side='left')
        right_lines, right_weights = self._apply_lane_lock(
            right_lines, right_weights, side='right')
        
        # --- Density-based clustering: prefer thick solid lines over noise ---
        left_lines, left_weights = self._cluster_select_lines(left_lines, left_weights)
        right_lines, right_weights = self._cluster_select_lines(right_lines, right_weights)
        
        # --- Median-based outlier rejection within selected cluster ---
        left_lines, left_weights = self._reject_outlier_lines(left_lines, left_weights)
        right_lines, right_weights = self._reject_outlier_lines(right_lines, right_weights)
        
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
        Predicts next lane position using EMA-damped forecasting.
        Uses exponentially weighted recent history instead of raw velocity
        to prevent noise amplification.
        Returns predicted (slope, bottom_x) or None if insufficient data.
        """
        if len(history) < 3:
            return None
        
        # Use up to last 5 frames, weighted by recency (EMA)
        n = min(len(history), 5)
        recent = history[-n:]
        
        # EMA weights: most recent frame gets highest weight
        # weights = [alpha * (1-alpha)^(n-1-i)] for i in range(n)
        alpha = 0.4
        weights = np.array([alpha * ((1 - alpha) ** (n - 1 - i)) for i in range(n)])
        weights /= weights.sum()  # Normalize
        
        recent_slopes = np.array([h[0] for h in recent])
        recent_bottom_x = np.array([h[1] for h in recent])
        
        # EMA-weighted position (this IS the prediction — the weighted
        # average naturally trends toward the latest position)
        ema_slope = np.dot(weights, recent_slopes)
        ema_bottom_x = np.dot(weights, recent_bottom_x)
        
        # Add damped velocity for slight look-ahead
        # Damping factor prevents runaway predictions from noisy velocity
        damping = 0.3
        if n >= 2:
            velocity_bx = (recent_bottom_x[-1] - recent_bottom_x[-2]) * damping
            velocity_slope = (recent_slopes[-1] - recent_slopes[-2]) * damping
        else:
            velocity_bx = 0.0
            velocity_slope = 0.0
        
        predicted_bottom_x = ema_bottom_x + velocity_bx
        predicted_slope = ema_slope + velocity_slope
        
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
                # Use last known good position (EMA-weighted)
                return self._ema_smooth(history)
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
                else:
                    # Detection is consistent with prediction — accept it into history
                    history.append(current_line)
                    if len(history) > self.smooth_factor:
                        history.pop(0)
                    if side == 'left': 
                        self.left_rejections = 0
                    else: 
                        self.right_rejections = 0
        
        # --- Legacy Smoothing (Fallback) ---
        avg = self._ema_smooth(history)
        if avg is None:
            history.append(current_line)
            return current_line
        
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
        
        # Return EMA-weighted smooth average
        if len(history) > 0:
            return self._ema_smooth(history)
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
        
        # 4. Blur (larger kernel for noisy dashcam footage)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # 4.5 Adaptive Contrast Enhancement (CLAHE)
        # Handles uneven lighting (shade + sun patches on Sri Lankan roads)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        blur = clahe.apply(blur)
        
        # 5. Canny
        edges = cv2.Canny(blur, self.hough_canny_low, self.hough_canny_high)
        
        # 6. Color Thresholding
        color_edges = self.isolate_color_edges(frame)
        
        # 6.5 Dilate color mask — connects thick solid lines, making them
        # produce strong continuous edges vs scattered noise dots
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        color_edges = cv2.dilate(color_edges, dilate_kernel, iterations=1)
        
        # 7. Combined Edges
        combined_edges = cv2.bitwise_or(edges, color_edges)
        
        # 7.5 Morphological cleanup — removes noise dots, bridges small gaps
        # Close first (bridge gaps in lines), then Open (remove noise blobs)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Larger for stronger noise removal
        combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, close_kernel)
        combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_OPEN, open_kernel)
        
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
        
        # 10.25. Update Lane Locks (state machine)
        self._update_lane_lock(left_smooth, side='left')
        self._update_lane_lock(right_smooth, side='right')
        
        # 10.5. Validate Lane Width (Spatial Constraint)
        if not self.validate_lane_width(left_smooth, right_smooth):
            # Invalid lane width - use previous frame's lanes
            if len(self.left_history) > 0 and len(self.right_history) > 0:
                left_smooth = self._ema_smooth(self.left_history)
                right_smooth = self._ema_smooth(self.right_history)
        
        # 11. Coordinates
        left_line = self.make_coordinates(frame, left_smooth)
        right_line = self.make_coordinates(frame, right_smooth)
        
        # 12. Visualization (Debug Mosaic)
        # Create Result Image with Green Overlay
        result_img = self.draw_lane_overlay(frame, left_line, right_line)
        
        debug_view = self.create_debug_mosaic(result_img, combined_edges, masked_edges, roi_pts)
        
        return left_line, right_line, debug_view

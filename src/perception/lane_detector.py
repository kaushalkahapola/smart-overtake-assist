import cv2
import numpy as np

class LaneDetector:
    def __init__(self):
        pass

    def canny(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blur, 50, 150)
        return canny

    def region_of_interest(self, image):
        height, width = image.shape[:2]
        # Define a triangle/trapezoid polygon for masking relative to size
        # Bottom-Left, Bottom-Right, Top-Left-ish, Top-Right-ish
        polygons = np.array([
            [
                (int(width * 0.1), height),          # Bottom-Left
                (int(width * 0.9), height),          # Bottom-Right
                (int(width * 0.55), int(height * 0.6)), # Top-Right (Apex)
                (int(width * 0.45), int(height * 0.6))  # Top-Left (Apex)
            ]
        ])
        
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygons, 255)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    def detect(self, frame):
        """
        Detects lanes in the given frame.
        Returns the averaged lines for left and right lanes.
        """
        canny_image = self.canny(frame)
        cropped_image = self.region_of_interest(canny_image)
        
        # Hough Transform
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        
        averaged_lines = self.average_slope_intercept(frame, lines)
        
        # Return both the lines AND the debug image (edges)
        return averaged_lines, cropped_image

    def average_slope_intercept(self, image, lines):
        left_fit = []
        right_fit = []
        if lines is None:
            return None
            
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            
            # Filter vertical-ish lines
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
                
        if not left_fit or not right_fit:
            return None # Could handle partial detection (only left or only right) better

        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        
        left_line = self.make_coordinates(image, left_fit_average)
        right_line = self.make_coordinates(image, right_fit_average)
        
        return np.array([left_line, right_line])

    def make_coordinates(self, image, line_parameters):
        try:
            slope, intercept = line_parameters
        except TypeError:
            return np.array([0, 0, 0, 0]) # Fallback
            
        y1 = image.shape[0]
        y2 = int(y1 * (3/5))
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])

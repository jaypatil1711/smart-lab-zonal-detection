import cv2
import numpy as np

class MotionDetector:
    def __init__(self, threshold=25):
        self.background = None
        self.threshold = threshold

    def update_background(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
        if self.background is None:
            self.background = gray_frame
        else:
            self.background = cv2.addWeighted(self.background, 0.5, gray_frame, 0.5, 0)

    def detect_motion(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        if self.background is None:
            self.update_background(frame)
            return False

        frame_delta = cv2.absdiff(self.background, gray_frame)
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            return True  # Motion detected

        self.update_background(frame)
        return False  # No motion detected
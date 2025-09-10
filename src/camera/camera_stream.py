import cv2

class CameraStream:
    def __init__(self, camera_index=0):
        self.camera = cv2.VideoCapture(camera_index)
        if not self.camera.isOpened():
            raise Exception("Could not open video device")
        self.brightness = 1.0

    def get_frame(self):
        ret, frame = self.camera.read()
        if not ret:
            raise Exception("Could not read frame from camera")
        # Apply brightness adjustment
        return cv2.convertScaleAbs(frame, alpha=self.brightness, beta=0)
        
    def adjust_brightness(self, detected):
        # Adjust brightness based on detection status
        target = 1.0 if detected else 0.3  # Bright when detected, very dull when not
        # Smooth transition
        self.brightness += (target - self.brightness) * 0.15  # Faster transition
        self.brightness = max(0.3, min(1.0, self.brightness))  # Clamp between 0.3 and 1.0

    def release(self):
        self.camera.release()
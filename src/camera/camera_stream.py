import cv2

class CameraStream:
    def __init__(self, camera_index=0):
        self.camera = cv2.VideoCapture(camera_index)
        if not self.camera.isOpened():
            raise Exception("Could not open video device")

    def get_frame(self):
        ret, frame = self.camera.read()
        if not ret:
            raise Exception("Could not read frame from camera")
        return frame

    def release(self):
        self.camera.release()
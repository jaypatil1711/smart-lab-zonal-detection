def initialize_camera():
    import cv2

    # Initialize the camera
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise Exception("Could not open video device")

    return camera

def release_camera(camera):
    # Release the camera resources
    if camera.isOpened():
        camera.release()

def get_frame(camera):
    # Capture a frame from the camera
    ret, frame = camera.read()
    if not ret:
        raise Exception("Could not read frame from camera")
    return frame
import cv2
from camera.camera_stream import CameraStream
from detection.zone_detection import ZoneDetector
import time

def main():
    # Initialize the camera and zone detector
    camera = CameraStream()
    detector = ZoneDetector()

    # Define zones: (zone_id, x1, y1, x2, y2)
    zones = [
        (1, 50, 50, 200, 200),    # Zone 1
        (2, 250, 50, 400, 200),   # Zone 2
        (3, 150, 250, 300, 400)   # Zone 3
    ]

    try:
        while True:
            # Capture the current frame
            frame = camera.get_frame()

            # Detect motion in zones and update energy management
            detected_zones = detector.detect_motion_in_zone(frame, zones)

            # Display energy stats
            stats = detector.get_energy_stats()
            cv2.putText(frame, f"Active Zones: {stats['active_zones']}/{stats['total_zones']}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow("Smart Lab Zonal Detection", frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Small delay to prevent high CPU usage
            time.sleep(0.01)

    finally:
        # Cleanup
        detector.cleanup()
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
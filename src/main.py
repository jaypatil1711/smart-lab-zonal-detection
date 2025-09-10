
import cv2
from camera.camera_stream import CameraStream
from detection.yolo_detection import YOLODetector
from utils.energy_manager import EnergyManager
import time

def main():
    # Initialize the camera and YOLO detector
    camera = CameraStream()
    yolo_detector = YOLODetector()
    energy_manager = EnergyManager()
    
    # Track previous state for teacher detection
    previous_teacher_present = False

    print("Smart Lab Teacher Detection System Started")
    print("Press 'q' to quit, 'r' to reset zone coordinates")
    
    # Get zone information
    zone_info = yolo_detector.get_zone_info()
    print(f"Detection Zone: {zone_info['description']}")
    print(f"Zone Coordinates: {zone_info['coordinates']}")
    print(f"Target Classes: {zone_info['yolo_settings']['target_classes']}")
    print("System is now monitoring for teachers/instructors...")

    try:
        while True:
            # Capture the current frame
            frame = camera.get_frame()

            # Detect teachers in zone using YOLO
            teacher_detected, detections, annotated_frame = yolo_detector.detect_objects_in_zone(frame)
            
            # Update energy management
            energy_manager.update_zone_status(1, teacher_detected)
            
            # Log teacher arrival/departure
            if teacher_detected and not previous_teacher_present:
                print("👨‍🏫 Teacher arrived!")
            elif not teacher_detected and previous_teacher_present:
                print("👨‍🏫 Teacher left!")
                
            previous_teacher_present = teacher_detected
            
            # Adjust camera brightness based on teacher detection
            camera.adjust_brightness(teacher_detected)

            # Display teacher detection information
            if detections:
                cv2.putText(annotated_frame, f"Teachers Detected: {len(detections)}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                for i, detection in enumerate(detections):
                    cv2.putText(annotated_frame, 
                              f"Teacher: {detection['confidence']:.2f}",
                              (10, 90 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Display zone status
            stats = energy_manager.get_energy_stats()
            cv2.putText(annotated_frame, f"Zone Status: {'TEACHER PRESENT' if teacher_detected else 'NO TEACHER'}", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow("Smart Lab Teacher Detection", annotated_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset zone coordinates (you can modify these values)
                yolo_detector.update_zone_coordinates(50, 50, 600, 400)
                print("Zone coordinates reset")

            # Small delay to prevent high CPU usage
            time.sleep(0.01)

    finally:
        # Cleanup
        yolo_detector.cleanup()
        energy_manager.cleanup()
        camera.release()
        cv2.destroyAllWindows()
        print("System shutdown complete")

if __name__ == "__main__":
    main()
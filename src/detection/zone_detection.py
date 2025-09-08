import cv2
from utils.energy_manager import EnergyManager
import numpy as np

class ZoneDetector:
    def __init__(self, config_file='energy_config.json'):
        # Initialize the energy manager
        self.energy_manager = EnergyManager(config_file)
        self.motion_threshold = 25  # Adjustable threshold for motion detection
        self.background = None

    def detect_motion_in_zone(self, frame, zones):
        """
        Analyze the frame to detect motion within predefined zones.
        zones: list of tuples (zone_id, x1, y1, x2, y2)
        """
        detected_zones = []
        
        # Convert frame to grayscale and apply blur
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
        
        # Initialize or update background model
        if self.background is None:
            self.background = blurred_frame
            return detected_zones
        
        # Calculate frame delta
        frame_delta = cv2.absdiff(self.background, blurred_frame)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Update background model
        self.background = cv2.addWeighted(self.background, 0.8, blurred_frame, 0.2, 0)
        
        # Check each zone for motion
        for zone in zones:
            zone_id, x1, y1, x2, y2 = zone
            zone_frame = thresh[y1:y2, x1:x2]
            
            # Calculate the percentage of motion pixels in the zone
            motion_pixels = np.count_nonzero(zone_frame)
            zone_area = (x2 - x1) * (y2 - y1)
            motion_percentage = (motion_pixels / zone_area) * 100
            
            # Determine if zone is occupied
            is_occupied = motion_percentage > self.motion_threshold
            
            if is_occupied:
                detected_zones.append(zone)
            
            # Update energy management system
            self.energy_manager.update_zone_status(zone_id, is_occupied)
            
            # Draw zone rectangle with status
            color = (0, 255, 0) if is_occupied else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Zone {zone_id}: {'Occupied' if is_occupied else 'Empty'}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return detected_zones
    
    def get_energy_stats(self):
        """Get current energy usage statistics"""
        return self.energy_manager.get_energy_stats()
    
    def cleanup(self):
        """Cleanup resources"""
        self.energy_manager.cleanup()
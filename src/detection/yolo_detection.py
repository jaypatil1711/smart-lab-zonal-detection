import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Any
import json

class YOLODetector:
    def __init__(self, config_file='energy_config.json'):
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize YOLO model
        yolo_config = self.config.get('yolo_settings', {})
        model_path = yolo_config.get('model_path', 'yolov8n.pt')
        
        try:
            self.model = YOLO(model_path)
            print(f"YOLO model loaded successfully: {model_path}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            # Fallback to default model
            self.model = YOLO('yolov8n.pt')
        
        # YOLO settings
        self.confidence_threshold = yolo_config.get('confidence_threshold', 0.5)
        self.target_classes = yolo_config.get('target_classes', ['person'])
        self.iou_threshold = yolo_config.get('iou_threshold', 0.45)
        
        # Zone configuration
        self.zone_config = self.config['zones']['1']  # Single zone
        self.zone_coords = (
            self.zone_config['x1'],
            self.zone_config['y1'], 
            self.zone_config['x2'],
            self.zone_config['y2']
        )

    def _load_config(self, config_file: str) -> dict:
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file {config_file} not found, using default settings")
            return {
                'yolo_settings': {
                    'model_path': 'yolov8n.pt',
                    'confidence_threshold': 0.5,
                    'target_classes': ['person'],
                    'iou_threshold': 0.45
                },
                'zones': {
                    '1': {'x1': 50, 'y1': 50, 'x2': 600, 'y2': 400}
                }
            }

    def detect_objects_in_zone(self, frame: np.ndarray) -> Tuple[bool, List[Dict], np.ndarray]:
        """
        Detect objects in the defined zone using YOLO
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (is_occupied, detections, annotated_frame)
        """
        # Run YOLO detection
        results = self.model(frame, conf=self.confidence_threshold, iou=self.iou_threshold)
        
        # Extract zone coordinates
        x1, y1, x2, y2 = self.zone_coords
        
        # Create annotated frame
        annotated_frame = frame.copy()
        
        # Draw zone rectangle
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, "Detection Zone", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        detections = []
        is_occupied = False
        
        # Process detections
        if results and len(results) > 0:
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class name
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        
                        # Check if it's a target class
                        if class_name in self.target_classes:
                            # Get bounding box coordinates
                            x1_box, y1_box, x2_box, y2_box = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0])
                            
                            # Check if detection is within zone
                            if self._is_detection_in_zone(x1_box, y1_box, x2_box, y2_box):
                                is_occupied = True
                                
                                # Store detection info
                                detection_info = {
                                    'class': class_name,
                                    'confidence': confidence,
                                    'bbox': (int(x1_box), int(y1_box), int(x2_box), int(y2_box))
                                }
                                detections.append(detection_info)
                                
                                # Draw detection on frame
                                cv2.rectangle(annotated_frame, 
                                            (int(x1_box), int(y1_box)), 
                                            (int(x2_box), int(y2_box)), 
                                            (0, 0, 255), 2)
                                cv2.putText(annotated_frame, 
                                          f"{class_name}: {confidence:.2f}",
                                          (int(x1_box), int(y1_box)-10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Update zone status display
        status_color = (0, 0, 255) if is_occupied else (0, 255, 0)
        status_text = "TEACHER DETECTED" if is_occupied else "NO TEACHER"
        cv2.putText(annotated_frame, f"Zone Status: {status_text}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        return is_occupied, detections, annotated_frame

    def _is_detection_in_zone(self, x1_box: float, y1_box: float, x2_box: float, y2_box: float) -> bool:
        """
        Check if a detection bounding box intersects with the zone
        
        Args:
            x1_box, y1_box, x2_box, y2_box: Detection bounding box coordinates
            
        Returns:
            True if detection intersects with zone
        """
        zone_x1, zone_y1, zone_x2, zone_y2 = self.zone_coords
        
        # Check for intersection
        return not (x2_box < zone_x1 or x1_box > zone_x2 or 
                   y2_box < zone_y1 or y1_box > zone_y2)

    def get_zone_info(self) -> Dict[str, Any]:
        """Get zone configuration information"""
        return {
            'zone_id': 1,
            'coordinates': self.zone_coords,
            'description': self.zone_config.get('description', 'Main Lab Area'),
            'yolo_settings': {
                'confidence_threshold': self.confidence_threshold,
                'target_classes': self.target_classes,
                'iou_threshold': self.iou_threshold
            }
        }

    def update_zone_coordinates(self, x1: int, y1: int, x2: int, y2: int):
        """Update zone coordinates dynamically"""
        self.zone_coords = (x1, y1, x2, y2)
        print(f"Zone coordinates updated to: ({x1}, {y1}, {x2}, {y2})")

    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'model'):
            del self.model

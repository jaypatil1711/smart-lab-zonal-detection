import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Any
import json
import time
from collections import defaultdict, deque
import torch

class AdvancedAIDetector:
    """
    Advanced AI-powered detection system with multiple ML models
    """
    
    def __init__(self, config_file='ai_config.json'):
        self.config = self._load_config(config_file)
        
        # Initialize multiple AI models
        self.yolo_model = self._load_yolo_model()
        self.pose_model = self._load_pose_model()
        self.face_model = self._load_face_model()
        
        # Tracking and analytics
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        self.person_ids = {}
        self.activity_classifier = self._load_activity_classifier()
        
        # Performance metrics
        self.detection_stats = {
            'total_detections': 0,
            'unique_persons': set(),
            'activities_detected': defaultdict(int),
            'confidence_scores': deque(maxlen=100)
        }
        
    def _load_config(self, config_file: str) -> dict:
        """Load AI configuration"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Default AI configuration
            config = {
                'models': {
                    'yolo': {
                        'path': 'yolov8n.pt',
                        'confidence': 0.5,
                        'iou_threshold': 0.45
                    },
                    'pose': {
                        'path': 'yolov8n-pose.pt',
                        'confidence': 0.6
                    },
                    'face': {
                        'path': 'yolov8n-face.pt',
                        'confidence': 0.7
                    }
                },
                'tracking': {
                    'max_disappeared': 30,
                    'max_distance': 50
                },
                'analytics': {
                    'activity_threshold': 0.8,
                    'pose_confidence': 0.6
                }
            }
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)
            return config
    
    def _load_yolo_model(self):
        """Load YOLO object detection model"""
        try:
            model_path = self.config['models']['yolo']['path']
            model = YOLO(model_path)
            print(f"✅ YOLO model loaded: {model_path}")
            return model
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            return None
    
    def _load_pose_model(self):
        """Load pose estimation model"""
        try:
            model_path = self.config['models']['pose']['path']
            model = YOLO(model_path)
            print(f"✅ Pose model loaded: {model_path}")
            return model
        except Exception as e:
            print(f"⚠️ Pose model not available: {e}")
            return None
    
    def _load_face_model(self):
        """Load face detection model"""
        try:
            model_path = self.config['models']['face']['path']
            model = YOLO(model_path)
            print(f"✅ Face model loaded: {model_path}")
            return model
        except Exception as e:
            print(f"⚠️ Face model not available: {e}")
            return None
    
    def _load_activity_classifier(self):
        """Load activity classification model"""
        # This would be a custom trained model for lab activities
        # For now, we'll use rule-based classification
        return {
            'teaching': {'pose_keypoints': [0, 1, 2], 'threshold': 0.8},
            'sitting': {'pose_keypoints': [11, 12], 'threshold': 0.7},
            'standing': {'pose_keypoints': [11, 12], 'threshold': 0.6},
            'group_work': {'multiple_persons': True, 'distance_threshold': 100}
        }
    
    def detect_and_analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Advanced AI detection and analysis pipeline
        """
        results = {
            'detections': [],
            'poses': [],
            'faces': [],
            'activities': [],
            'tracking': {},
            'analytics': {}
        }
        
        # 1. Object Detection
        if self.yolo_model:
            detections = self._detect_objects(frame)
            results['detections'] = detections
        
        # 2. Pose Estimation
        if self.pose_model:
            poses = self._estimate_poses(frame)
            results['poses'] = poses
        
        # 3. Face Detection
        if self.face_model:
            faces = self._detect_faces(frame)
            results['faces'] = faces
        
        # 4. Activity Classification
        activities = self._classify_activities(results)
        results['activities'] = activities
        
        # 5. Object Tracking
        tracking = self._track_objects(results['detections'])
        results['tracking'] = tracking
        
        # 6. Analytics
        analytics = self._compute_analytics(results)
        results['analytics'] = analytics
        
        # Update statistics
        self._update_stats(results)
        
        return results
    
    def _detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Enhanced object detection with confidence scoring"""
        if not self.yolo_model:
            return []
        
        results = self.yolo_model(frame, 
                                conf=self.config['models']['yolo']['confidence'],
                                iou=self.config['models']['yolo']['iou_threshold'])
        
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.yolo_model.names[class_id]
                    
                    detection = {
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': confidence,
                        'class': class_name,
                        'class_id': class_id,
                        'timestamp': time.time()
                    }
                    detections.append(detection)
        
        return detections
    
    def _estimate_poses(self, frame: np.ndarray) -> List[Dict]:
        """Estimate human poses"""
        if not self.pose_model:
            return []
        
        results = self.pose_model(frame, 
                                conf=self.config['models']['pose']['confidence'])
        
        poses = []
        for result in results:
            if result.keypoints is not None:
                for keypoints in result.keypoints:
                    pose_data = {
                        'keypoints': keypoints.xy[0].cpu().numpy().tolist(),
                        'confidence': keypoints.conf[0].cpu().numpy().tolist(),
                        'timestamp': time.time()
                    }
                    poses.append(pose_data)
        
        return poses
    
    def _detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """Detect and analyze faces"""
        if not self.face_model:
            return []
        
        results = self.face_model(frame, 
                                conf=self.config['models']['face']['confidence'])
        
        faces = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    
                    face_data = {
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': confidence,
                        'timestamp': time.time()
                    }
                    faces.append(face_data)
        
        return faces
    
    def _classify_activities(self, results: Dict) -> List[Dict]:
        """Classify activities based on poses and detections"""
        activities = []
        
        # Rule-based activity classification
        poses = results.get('poses', [])
        detections = results.get('detections', [])
        
        # Count people
        person_count = len([d for d in detections if d['class'] == 'person'])
        
        # Classify based on pose keypoints
        for pose in poses:
            if pose['confidence'] and len(pose['confidence']) > 0:
                avg_confidence = np.mean(pose['confidence'])
                
                if avg_confidence > self.config['analytics']['pose_confidence']:
                    # Simple activity classification
                    if person_count > 1:
                        activities.append({
                            'activity': 'group_work',
                            'confidence': 0.8,
                            'participants': person_count
                        })
                    else:
                        activities.append({
                            'activity': 'individual_work',
                            'confidence': 0.7,
                            'participants': person_count
                        })
        
        return activities
    
    def _track_objects(self, detections: List[Dict]) -> Dict:
        """Track objects across frames"""
        tracking_data = {}
        
        for detection in detections:
            if detection['class'] == 'person':
                # Simple tracking based on position
                bbox = detection['bbox']
                center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                
                # Find closest existing track
                closest_id = None
                min_distance = float('inf')
                
                for track_id, history in self.track_history.items():
                    if history:
                        last_center = history[-1]
                        distance = np.sqrt((center[0] - last_center[0])**2 + 
                                         (center[1] - last_center[1])**2)
                        if distance < min_distance and distance < self.config['tracking']['max_distance']:
                            min_distance = distance
                            closest_id = track_id
                
                if closest_id is not None:
                    self.track_history[closest_id].append(center)
                    tracking_data[closest_id] = {
                        'center': center,
                        'bbox': bbox,
                        'confidence': detection['confidence'],
                        'track_length': len(self.track_history[closest_id])
                    }
                else:
                    # New track
                    new_id = len(self.track_history) + 1
                    self.track_history[new_id].append(center)
                    tracking_data[new_id] = {
                        'center': center,
                        'bbox': bbox,
                        'confidence': detection['confidence'],
                        'track_length': 1
                    }
        
        return tracking_data
    
    def _compute_analytics(self, results: Dict) -> Dict:
        """Compute advanced analytics"""
        analytics = {
            'occupancy_rate': 0,
            'activity_diversity': 0,
            'average_confidence': 0,
            'detection_quality': 'good'
        }
        
        detections = results.get('detections', [])
        activities = results.get('activities', [])
        
        if detections:
            confidences = [d['confidence'] for d in detections]
            analytics['average_confidence'] = np.mean(confidences)
            
            if analytics['average_confidence'] > 0.8:
                analytics['detection_quality'] = 'excellent'
            elif analytics['average_confidence'] > 0.6:
                analytics['detection_quality'] = 'good'
            else:
                analytics['detection_quality'] = 'poor'
        
        if activities:
            unique_activities = set(a['activity'] for a in activities)
            analytics['activity_diversity'] = len(unique_activities)
        
        return analytics
    
    def _update_stats(self, results: Dict):
        """Update detection statistics"""
        self.detection_stats['total_detections'] += len(results.get('detections', []))
        
        tracking = results.get('tracking', {})
        for track_id in tracking.keys():
            self.detection_stats['unique_persons'].add(track_id)
        
        activities = results.get('activities', [])
        for activity in activities:
            self.detection_stats['activities_detected'][activity['activity']] += 1
        
        detections = results.get('detections', [])
        for detection in detections:
            self.detection_stats['confidence_scores'].append(detection['confidence'])
    
    def get_ai_insights(self) -> Dict[str, Any]:
        """Get AI-powered insights"""
        insights = {
            'total_detections': self.detection_stats['total_detections'],
            'unique_persons_today': len(self.detection_stats['unique_persons']),
            'most_common_activity': max(self.detection_stats['activities_detected'], 
                                       key=self.detection_stats['activities_detected'].get) if self.detection_stats['activities_detected'] else 'none',
            'average_confidence': np.mean(self.detection_stats['confidence_scores']) if self.detection_stats['confidence_scores'] else 0,
            'system_performance': 'excellent' if np.mean(self.detection_stats['confidence_scores']) > 0.8 else 'good'
        }
        
        return insights
    
    def cleanup(self):
        """Cleanup AI models"""
        if hasattr(self, 'yolo_model') and self.yolo_model:
            del self.yolo_model
        if hasattr(self, 'pose_model') and self.pose_model:
            del self.pose_model
        if hasattr(self, 'face_model') and self.face_model:
            del self.face_model


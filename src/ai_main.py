import cv2
import time
import json
from datetime import datetime
from camera.camera_stream import CameraStream
from detection.yolo_detection import YOLODetector
from utils.energy_manager import EnergyManager
from ai.advanced_detector import AdvancedAIDetector
from ai.predictive_analytics import PredictiveAnalytics
from ai.nlp_interface import NLPInterface
import threading
import queue

class SmartLabAI:
    """
    Main AI-powered Smart Lab system
    """
    
    def __init__(self, config_file='ai_config.json'):
        self.config = self._load_config(config_file)
        
        # Initialize core components
        self.camera = CameraStream()
        self.yolo_detector = YOLODetector()
        self.energy_manager = EnergyManager()
        
        # Initialize AI components
        self.ai_detector = AdvancedAIDetector()
        self.analytics = PredictiveAnalytics()
        self.nlp = NLPInterface()
        
        # System state
        self.is_running = False
        self.previous_teacher_present = False
        self.ai_insights = {}
        
        # Data collection
        self.data_buffer = []
        self.insights_update_interval = 30  # seconds
        self.last_insights_update = time.time()
        
        # Voice command processing
        self.command_queue = queue.Queue()
        self.voice_enabled = self.config.get('voice', {}).get('enabled', True)
        
        print("🤖 Smart Lab AI System Initialized")
        print("🧠 AI Features: Advanced Detection, Predictive Analytics, Voice Control")
        
    def _load_config(self, config_file: str) -> dict:
        """Load AI configuration"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'voice': {'enabled': True},
                'ai_features': {
                    'advanced_detection': True,
                    'predictive_analytics': True,
                    'voice_control': True
                }
            }
    
    def start(self):
        """Start the AI-powered Smart Lab system"""
        self.is_running = True
        
        # Start voice interface if enabled
        if self.voice_enabled:
            self.nlp.start_voice_listening()
            print("🎤 Voice interface activated")
        
        # Start background AI processing
        threading.Thread(target=self._ai_processing_loop, daemon=True).start()
        
        # Start insights update loop
        threading.Thread(target=self._insights_update_loop, daemon=True).start()
        
        print("🚀 Smart Lab AI System Started")
        print("Press 'q' to quit, 'r' to reset zones, 'v' to toggle voice, 'i' to show AI insights")
        
        self._main_loop()
    
    def _main_loop(self):
        """Main detection and display loop"""
        try:
            while self.is_running:
                # Capture frame
                frame = self.camera.get_frame()
                
                # Basic YOLO detection
                teacher_detected, detections, annotated_frame = self.yolo_detector.detect_objects_in_zone(frame)
                
                # Update energy management
                self.energy_manager.update_zone_status(1, teacher_detected)
                
                # Log teacher arrival/departure
                if teacher_detected and not self.previous_teacher_present:
                    print("👨‍🏫 Teacher arrived!")
                    self._handle_teacher_arrival()
                elif not teacher_detected and self.previous_teacher_present:
                    print("👨‍🏫 Teacher left!")
                    self._handle_teacher_departure()
                
                self.previous_teacher_present = teacher_detected
                
                # Adjust camera brightness
                self.camera.adjust_brightness(teacher_detected)
                
                # Enhanced display with AI information
                annotated_frame = self._add_ai_overlay(annotated_frame, teacher_detected, detections)
                
                # Display frame
                cv2.imshow("🤖 Smart Lab AI System", annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.yolo_detector.update_zone_coordinates(50, 50, 600, 400)
                    print("🔄 Zone coordinates reset")
                elif key == ord('v'):
                    self._toggle_voice_interface()
                elif key == ord('i'):
                    self._show_ai_insights()
                
                # Small delay
                time.sleep(0.01)
                
        finally:
            self._cleanup()
    
    def _ai_processing_loop(self):
        """Background AI processing loop"""
        while self.is_running:
            try:
                # Capture frame for AI analysis
                frame = self.camera.get_frame()
                
                # Advanced AI detection and analysis
                ai_results = self.ai_detector.detect_and_analyze(frame)
                
                # Store data for analytics
                self._store_ai_data(ai_results)
                
                # Update AI insights
                self.ai_insights = self._compute_ai_insights(ai_results)
                
                time.sleep(1)  # Process every second
                
            except Exception as e:
                print(f"❌ AI processing error: {e}")
                time.sleep(5)
    
    def _insights_update_loop(self):
        """Update AI insights periodically"""
        while self.is_running:
            try:
                current_time = time.time()
                if current_time - self.last_insights_update >= self.insights_update_interval:
                    self._update_analytics_insights()
                    self.last_insights_update = current_time
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                print(f"❌ Insights update error: {e}")
                time.sleep(10)
    
    def _add_ai_overlay(self, frame, teacher_detected, detections):
        """Add AI information overlay to frame"""
        # Basic detection info
        if detections:
            cv2.putText(frame, f"AI Detections: {len(detections)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            for i, detection in enumerate(detections):
                cv2.putText(frame, 
                          f"Confidence: {detection['confidence']:.2f}",
                          (10, 90 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # AI insights overlay
        if self.ai_insights:
            y_offset = 150
            
            # System performance
            performance = self.ai_insights.get('system_performance', 'good')
            color = (0, 255, 0) if performance == 'excellent' else (0, 255, 255) if performance == 'good' else (0, 0, 255)
            cv2.putText(frame, f"AI Performance: {performance.upper()}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Unique persons detected
            unique_persons = self.ai_insights.get('unique_persons_today', 0)
            cv2.putText(frame, f"Unique Persons Today: {unique_persons}", 
                       (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Most common activity
            common_activity = self.ai_insights.get('most_common_activity', 'none')
            cv2.putText(frame, f"Common Activity: {common_activity}", 
                       (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Voice interface status
        voice_status = "ON" if self.voice_enabled else "OFF"
        color = (0, 255, 0) if self.voice_enabled else (0, 0, 255)
        cv2.putText(frame, f"Voice: {voice_status}", 
                   (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # AI system status
        cv2.putText(frame, "🤖 AI ENABLED", 
                   (frame.shape[1] - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame
    
    def _store_ai_data(self, ai_results):
        """Store AI detection data for analytics"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'occupancy_count': len(ai_results.get('detections', [])),
            'confidence_score': np.mean([d['confidence'] for d in ai_results.get('detections', [])]) if ai_results.get('detections') else 0,
            'activity_type': ai_results.get('activities', [{}])[0].get('activity', 'unknown') if ai_results.get('activities') else 'unknown',
            'zone_id': 1
        }
        
        # Store in analytics system
        self.analytics.store_detection_data(data)
        
        # Store energy data
        energy_data = {
            'active_zones': 1 if data['occupancy_count'] > 0 else 0,
            'energy_consumption': 75 if data['occupancy_count'] > 0 else 25,
            'occupancy_rate': data['occupancy_count'] / 5.0  # Assuming max 5 people
        }
        self.analytics.store_energy_data(energy_data)
    
    def _compute_ai_insights(self, ai_results):
        """Compute AI insights from detection results"""
        insights = self.ai_detector.get_ai_insights()
        
        # Add predictive analytics
        try:
            analytics_insights = self.analytics.get_ai_insights()
            insights.update(analytics_insights)
        except Exception as e:
            print(f"⚠️ Analytics error: {e}")
        
        return insights
    
    def _update_analytics_insights(self):
        """Update analytics insights"""
        try:
            # Get recent data for anomaly detection
            recent_data = self.data_buffer[-10:] if len(self.data_buffer) >= 10 else self.data_buffer
            
            if recent_data:
                anomalies = self.analytics.detect_anomalies(recent_data)
                if anomalies['total_anomalies'] > 0:
                    print(f"🚨 Anomaly detected: {anomalies['total_anomalies']} anomalies found")
        
        except Exception as e:
            print(f"⚠️ Analytics update error: {e}")
    
    def _handle_teacher_arrival(self):
        """Handle teacher arrival event"""
        if self.voice_enabled:
            self.nlp.speak("Teacher has arrived in the lab")
        
        # Store event data
        event_data = {
            'event': 'teacher_arrival',
            'timestamp': datetime.now().isoformat(),
            'ai_confidence': 0.9
        }
        self.data_buffer.append(event_data)
    
    def _handle_teacher_departure(self):
        """Handle teacher departure event"""
        if self.voice_enabled:
            self.nlp.speak("Teacher has left the lab")
        
        # Store event data
        event_data = {
            'event': 'teacher_departure',
            'timestamp': datetime.now().isoformat(),
            'ai_confidence': 0.9
        }
        self.data_buffer.append(event_data)
    
    def _toggle_voice_interface(self):
        """Toggle voice interface on/off"""
        self.voice_enabled = not self.voice_enabled
        
        if self.voice_enabled:
            self.nlp.start_voice_listening()
            print("🎤 Voice interface activated")
        else:
            self.nlp.stop_voice_listening()
            print("🔇 Voice interface deactivated")
    
    def _show_ai_insights(self):
        """Display AI insights in console"""
        print("\n" + "="*50)
        print("🤖 AI INSIGHTS REPORT")
        print("="*50)
        
        if self.ai_insights:
            print(f"📊 Total Detections: {self.ai_insights.get('total_detections', 0)}")
            print(f"👥 Unique Persons Today: {self.ai_insights.get('unique_persons_today', 0)}")
            print(f"🎯 Most Common Activity: {self.ai_insights.get('most_common_activity', 'none')}")
            print(f"📈 Average Confidence: {self.ai_insights.get('average_confidence', 0):.2f}")
            print(f"⚡ System Performance: {self.ai_insights.get('system_performance', 'good')}")
            
            # Predictive analytics
            if 'occupancy_prediction' in self.ai_insights:
                predictions = self.ai_insights['occupancy_prediction']['predictions']
                if predictions:
                    next_hour = predictions[0]
                    print(f"🔮 Next Hour Prediction: {next_hour['predicted_occupancy']} people")
            
            if 'energy_optimization' in self.ai_insights:
                optimization = self.ai_insights['energy_optimization']
                print(f"💡 Energy Savings Potential: {optimization.get('total_savings_potential', 0):.1f} watts")
        
        print("="*50 + "\n")
    
    def _cleanup(self):
        """Cleanup all resources"""
        self.is_running = False
        
        # Cleanup AI components
        self.ai_detector.cleanup()
        self.nlp.cleanup()
        
        # Cleanup core components
        self.yolo_detector.cleanup()
        self.energy_manager.cleanup()
        self.camera.release()
        cv2.destroyAllWindows()
        
        print("🤖 AI System shutdown complete")

def main():
    """Main function to run the AI-powered Smart Lab"""
    try:
        # Initialize and start the AI system
        smart_lab = SmartLabAI()
        smart_lab.start()
        
    except KeyboardInterrupt:
        print("\n🛑 System interrupted by user")
    except Exception as e:
        print(f"❌ System error: {e}")
    finally:
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()


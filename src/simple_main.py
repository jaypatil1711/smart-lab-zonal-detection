import cv2
import time
import json
from datetime import datetime
from camera.camera_stream import CameraStream
from detection.yolo_detection import YOLODetector
from utils.energy_manager import EnergyManager
from simple_ai_client import SimpleAIClient
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleSmartLabSystem:
    """
    Simplified Smart Lab system using cloud API with API key
    No async, no Docker, no complex setup needed!
    """
    
    def __init__(self, api_key: str, service: str = "google_vision"):
        # Initialize core components
        self.camera = CameraStream()
        self.yolo_detector = YOLODetector()
        self.energy_manager = EnergyManager()
        
        # Simple AI client with API key
        self.ai_client = SimpleAIClient(api_key, service)
        
        # System state
        self.is_running = False
        self.previous_teacher_present = False
        
        # Simple performance tracking
        self.performance_stats = {
            'total_frames': 0,
            'api_calls': 0,
            'average_response_time': 0,
            'last_update': time.time()
        }
        
        print("🚀 Simple Smart Lab System Initialized")
        print("⚡ Features: Cloud AI API, Simple setup, No Docker needed!")
        
    def start(self):
        """Start the simplified Smart Lab system"""
        self.is_running = True
        
        print("🎯 Simple Smart Lab System Started")
        print("Press 'q' to quit, 'r' to reset zones, 'i' to show insights, 'p' to show performance")
        
        self._main_loop()
    
    def _main_loop(self):
        """Main detection and display loop"""
        try:
            while self.is_running:
                # Capture frame
                frame = self.camera.get_frame()
                
                # Basic YOLO detection for immediate display
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
                
                # Enhanced display with API information
                annotated_frame = self._add_api_overlay(annotated_frame, teacher_detected, detections)
                
                # Display frame
                cv2.imshow("⚡ Simple Smart Lab System", annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.yolo_detector.update_zone_coordinates(50, 50, 600, 400)
                    print("🔄 Zone coordinates reset")
                elif key == ord('i'):
                    self._show_insights()
                elif key == ord('p'):
                    self._show_performance_stats()
                elif key == ord('c'):
                    # Test cloud API detection
                    self._test_cloud_detection(frame)
                
                # Update performance stats
                self.performance_stats['total_frames'] += 1
                
                # Small delay
                time.sleep(0.01)
                
        finally:
            self._cleanup()
    
    def _test_cloud_detection(self, frame):
        """Test cloud API detection (called with 'c' key)"""
        try:
            print("🔍 Testing cloud API detection...")
            start_time = time.time()
            
            result = self.ai_client.detect_objects(frame, confidence_threshold=0.5)
            
            processing_time = time.time() - start_time
            
            print(f"✅ Cloud detection completed in {processing_time:.3f}s")
            print(f"📊 Found {len(result['detections'])} objects")
            
            for i, detection in enumerate(result['detections']):
                print(f"  {i+1}. {detection['class']} (confidence: {detection['confidence']:.2f})")
            
            # Update performance stats
            self.performance_stats['api_calls'] += 1
            current_avg = self.performance_stats['average_response_time']
            total_calls = self.performance_stats['api_calls']
            self.performance_stats['average_response_time'] = (
                (current_avg * (total_calls - 1) + processing_time) / total_calls
            )
            
        except Exception as e:
            print(f"❌ Cloud detection error: {e}")
    
    def _add_api_overlay(self, frame, teacher_detected, detections):
        """Add API information overlay to frame"""
        # Basic detection info
        if detections:
            cv2.putText(frame, f"Local Detections: {len(detections)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # API performance overlay
        y_offset = 150
        
        # API status
        cv2.putText(frame, "☁️ CLOUD AI API", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Performance metrics
        avg_response_time = self.performance_stats['average_response_time']
        cv2.putText(frame, f"Avg Response: {avg_response_time:.3f}s", 
                   (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # API calls count
        api_calls = self.performance_stats['api_calls']
        cv2.putText(frame, f"API Calls: {api_calls}", 
                   (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Press 'c' for cloud detection", 
                   (10, y_offset + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    
    def _handle_teacher_arrival(self):
        """Handle teacher arrival event"""
        # Send voice command to API
        try:
            result = self.ai_client.process_voice_command("Teacher has arrived in the lab")
            print(f"🎤 Voice response: {result['response']}")
        except Exception as e:
            print(f"❌ Voice command error: {e}")
    
    def _handle_teacher_departure(self):
        """Handle teacher departure event"""
        # Send voice command to API
        try:
            result = self.ai_client.process_voice_command("Teacher has left the lab")
            print(f"🎤 Voice response: {result['response']}")
        except Exception as e:
            print(f"❌ Voice command error: {e}")
    
    def _show_insights(self):
        """Display insights in console"""
        print("\n" + "="*60)
        print("🤖 AI INSIGHTS REPORT (Cloud API)")
        print("="*60)
        
        try:
            insights = self.ai_client.get_insights()
            
            print(f"📊 Total Detections: {insights.get('total_detections', 0)}")
            print(f"👥 Unique Persons Today: {insights.get('unique_persons_today', 0)}")
            print(f"🎯 Most Common Activity: {insights.get('most_common_activity', 'none')}")
            print(f"📈 Average Confidence: {insights.get('average_confidence', 0):.2f}")
            print(f"⚡ System Performance: {insights.get('system_performance', 'good')}")
            
        except Exception as e:
            print(f"❌ Error getting insights: {e}")
        
        print("="*60 + "\n")
    
    def _show_performance_stats(self):
        """Display performance statistics"""
        print("\n" + "="*50)
        print("⚡ PERFORMANCE STATISTICS")
        print("="*50)
        
        stats = self.performance_stats
        print(f"📊 Total Frames Processed: {stats['total_frames']}")
        print(f"🔗 API Calls Made: {stats['api_calls']}")
        print(f"⚡ Average Response Time: {stats['average_response_time']:.3f}s")
        
        # Calculate FPS
        if stats['total_frames'] > 0:
            time_elapsed = time.time() - stats['last_update']
            fps = stats['total_frames'] / time_elapsed if time_elapsed > 0 else 0
            print(f"🎬 Average FPS: {fps:.1f}")
        
        print("="*50 + "\n")
    
    def _cleanup(self):
        """Cleanup all resources"""
        self.is_running = False
        
        # Cleanup core components
        self.yolo_detector.cleanup()
        self.energy_manager.cleanup()
        self.camera.release()
        cv2.destroyAllWindows()
        
        print("⚡ Simple System shutdown complete")

def main():
    """Main function to run the simplified Smart Lab"""
    try:
        # Get API key from user
        api_key = input("Enter your API key: ").strip()
        if not api_key:
            print("❌ API key is required!")
            return
        
        # Choose service
        print("\nAvailable services:")
        print("1. Google Cloud Vision API")
        print("2. OpenAI Vision API") 
        print("3. Hugging Face Inference API")
        
        choice = input("Choose service (1-3): ").strip()
        service_map = {
            "1": "google_vision",
            "2": "openai_vision", 
            "3": "huggingface"
        }
        service = service_map.get(choice, "google_vision")
        
        # Initialize and start the system
        smart_lab = SimpleSmartLabSystem(api_key, service)
        smart_lab.start()
        
    except KeyboardInterrupt:
        print("\n🛑 System interrupted by user")
    except Exception as e:
        print(f"❌ System error: {e}")
    finally:
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()

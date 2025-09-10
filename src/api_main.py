import cv2
import asyncio
import time
import json
from datetime import datetime
from camera.camera_stream import CameraStream
from detection.yolo_detection import YOLODetector
from utils.energy_manager import EnergyManager
from api.ai_client import AIClient, APIConfig
import threading
import queue
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmartLabAPISystem:
    """
    High-performance Smart Lab system with API backend
    """
    
    def __init__(self, api_config: APIConfig = None):
        self.api_config = api_config or APIConfig()
        
        # Initialize core components
        self.camera = CameraStream()
        self.yolo_detector = YOLODetector()
        self.energy_manager = EnergyManager()
        
        # API client
        self.ai_client = None
        
        # System state
        self.is_running = False
        self.previous_teacher_present = False
        self.ai_insights = {}
        
        # Performance tracking
        self.performance_stats = {
            'total_frames': 0,
            'api_calls': 0,
            'cache_hits': 0,
            'average_response_time': 0,
            'last_update': time.time()
        }
        
        # Data collection
        self.data_buffer = []
        self.insights_update_interval = 30  # seconds
        self.last_insights_update = time.time()
        
        # API processing queue
        self.api_queue = queue.Queue(maxsize=100)
        self.api_worker_thread = None
        
        print("🚀 Smart Lab API System Initialized")
        print("⚡ Features: High-performance API backend, Real-time processing, WebSocket streaming")
        
    async def start(self):
        """Start the API-powered Smart Lab system"""
        self.is_running = True
        
        # Initialize API client
        self.ai_client = AIClient(self.api_config)
        await self.ai_client.connect()
        
        # Connect to WebSocket for real-time updates
        await self.ai_client.connect_websocket()
        
        # Register WebSocket message handlers
        self.ai_client.on_message("detection_update", self._handle_detection_update)
        self.ai_client.on_message("voice_response", self._handle_voice_response)
        
        # Start API worker thread
        self.api_worker_thread = threading.Thread(target=self._api_worker, daemon=True)
        self.api_worker_thread.start()
        
        # Start background tasks
        asyncio.create_task(self._background_ai_processing())
        asyncio.create_task(self._insights_update_loop())
        asyncio.create_task(self._performance_monitor())
        
        print("🎯 Smart Lab API System Started")
        print("Press 'q' to quit, 'r' to reset zones, 'i' to show insights, 'p' to show performance")
        
        await self._main_loop()
    
    async def _main_loop(self):
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
                    await self._handle_teacher_arrival()
                elif not teacher_detected and self.previous_teacher_present:
                    print("👨‍🏫 Teacher left!")
                    await self._handle_teacher_departure()
                
                self.previous_teacher_present = teacher_detected
                
                # Adjust camera brightness
                self.camera.adjust_brightness(teacher_detected)
                
                # Enhanced display with API information
                annotated_frame = self._add_api_overlay(annotated_frame, teacher_detected, detections)
                
                # Display frame
                cv2.imshow("⚡ Smart Lab API System", annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.yolo_detector.update_zone_coordinates(50, 50, 600, 400)
                    print("🔄 Zone coordinates reset")
                elif key == ord('i'):
                    await self._show_ai_insights()
                elif key == ord('p'):
                    self._show_performance_stats()
                
                # Queue frame for API processing
                if not self.api_queue.full():
                    self.api_queue.put(frame.copy())
                
                # Update performance stats
                self.performance_stats['total_frames'] += 1
                
                # Small delay
                await asyncio.sleep(0.01)
                
        finally:
            await self._cleanup()
    
    def _api_worker(self):
        """Background API worker thread"""
        while self.is_running:
            try:
                if not self.api_queue.empty():
                    frame = self.api_queue.get(timeout=1)
                    
                    # Process frame with API
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        result = loop.run_until_complete(
                            self.ai_client.detect_objects(frame, zone_id=1, confidence_threshold=0.5)
                        )
                        
                        # Update performance stats
                        self.performance_stats['api_calls'] += 1
                        processing_time = result.get('processing_time', 0)
                        
                        # Update average response time
                        current_avg = self.performance_stats['average_response_time']
                        total_calls = self.performance_stats['api_calls']
                        self.performance_stats['average_response_time'] = (
                            (current_avg * (total_calls - 1) + processing_time) / total_calls
                        )
                        
                        logger.info(f"⚡ API detection processed in {processing_time:.3f}s")
                        
                    except Exception as e:
                        logger.error(f"❌ API processing error: {e}")
                    finally:
                        loop.close()
                
                time.sleep(0.1)  # Small delay to prevent high CPU usage
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ API worker error: {e}")
                time.sleep(1)
    
    async def _background_ai_processing(self):
        """Background AI processing loop"""
        while self.is_running:
            try:
                # Get AI insights periodically
                if time.time() - self.last_insights_update >= self.insights_update_interval:
                    insights = await self.ai_client.get_ai_insights()
                    self.ai_insights = insights
                    self.last_insights_update = time.time()
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"❌ Background AI processing error: {e}")
                await asyncio.sleep(10)
    
    async def _insights_update_loop(self):
        """Update AI insights periodically"""
        while self.is_running:
            try:
                # Get predictions
                predictions = await self.ai_client.get_predictions(hours_ahead=4)
                
                # Store insights
                self.ai_insights['predictions'] = predictions
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Insights update error: {e}")
                await asyncio.sleep(60)
    
    async def _performance_monitor(self):
        """Monitor system performance"""
        while self.is_running:
            try:
                # Get cache stats
                cache_stats = await self.ai_client.get_cache_stats()
                
                # Update performance stats
                self.performance_stats['cache_hits'] = cache_stats.get('cache_size', 0)
                self.performance_stats['last_update'] = time.time()
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"❌ Performance monitor error: {e}")
                await asyncio.sleep(30)
    
    def _add_api_overlay(self, frame, teacher_detected, detections):
        """Add API information overlay to frame"""
        # Basic detection info
        if detections:
            cv2.putText(frame, f"API Detections: {len(detections)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            for i, detection in enumerate(detections):
                cv2.putText(frame, 
                          f"Confidence: {detection['confidence']:.2f}",
                          (10, 90 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # API performance overlay
        y_offset = 150
        
        # API status
        cv2.putText(frame, "⚡ API BACKEND", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Performance metrics
        avg_response_time = self.performance_stats['average_response_time']
        cv2.putText(frame, f"Avg Response: {avg_response_time:.3f}s", 
                   (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # API calls count
        api_calls = self.performance_stats['api_calls']
        cv2.putText(frame, f"API Calls: {api_calls}", 
                   (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Cache hits
        cache_hits = self.performance_stats['cache_hits']
        cv2.putText(frame, f"Cache Hits: {cache_hits}", 
                   (10, y_offset + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # WebSocket status
        ws_status = "CONNECTED" if self.ai_client and self.ai_client.websocket_connected else "DISCONNECTED"
        color = (0, 255, 0) if ws_status == "CONNECTED" else (0, 0, 255)
        cv2.putText(frame, f"WebSocket: {ws_status}", 
                   (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    async def _handle_detection_update(self, data):
        """Handle detection update from WebSocket"""
        logger.info(f"📡 Received detection update: {data}")
    
    async def _handle_voice_response(self, data):
        """Handle voice response from WebSocket"""
        logger.info(f"🎤 Received voice response: {data}")
    
    async def _handle_teacher_arrival(self):
        """Handle teacher arrival event"""
        # Send voice command to API
        try:
            result = await self.ai_client.process_voice_command("Teacher has arrived in the lab")
            logger.info(f"🎤 Voice response: {result['response']}")
        except Exception as e:
            logger.error(f"❌ Voice command error: {e}")
        
        # Store event data
        event_data = {
            'event': 'teacher_arrival',
            'timestamp': datetime.now().isoformat(),
            'api_processed': True
        }
        self.data_buffer.append(event_data)
    
    async def _handle_teacher_departure(self):
        """Handle teacher departure event"""
        # Send voice command to API
        try:
            result = await self.ai_client.process_voice_command("Teacher has left the lab")
            logger.info(f"🎤 Voice response: {result['response']}")
        except Exception as e:
            logger.error(f"❌ Voice command error: {e}")
        
        # Store event data
        event_data = {
            'event': 'teacher_departure',
            'timestamp': datetime.now().isoformat(),
            'api_processed': True
        }
        self.data_buffer.append(event_data)
    
    async def _show_ai_insights(self):
        """Display AI insights in console"""
        print("\n" + "="*60)
        print("🤖 AI INSIGHTS REPORT (API-Powered)")
        print("="*60)
        
        try:
            insights = await self.ai_client.get_ai_insights()
            
            if insights:
                print(f"📊 Total Detections: {insights.get('insights', {}).get('total_detections', 0)}")
                print(f"👥 Unique Persons Today: {insights.get('insights', {}).get('unique_persons_today', 0)}")
                print(f"🎯 Most Common Activity: {insights.get('insights', {}).get('most_common_activity', 'none')}")
                print(f"📈 Average Confidence: {insights.get('insights', {}).get('average_confidence', 0):.2f}")
                print(f"⚡ System Performance: {insights.get('insights', {}).get('system_performance', 'good')}")
                
                # Performance metrics
                perf_metrics = insights.get('performance_metrics', {})
                print(f"🎯 Detection Accuracy: {perf_metrics.get('detection_accuracy', 0):.2f}")
                print(f"🔮 Prediction Accuracy: {perf_metrics.get('prediction_accuracy', 0):.2f}")
                print(f"⚡ Response Time: {perf_metrics.get('response_time', 0):.3f}s")
                print(f"💾 Cache Hit Rate: {perf_metrics.get('cache_hit_rate', 0):.2f}")
            
            # Get predictions
            predictions = await self.ai_client.get_predictions(hours_ahead=4)
            if predictions and predictions.get('predictions'):
                print(f"\n🔮 Next Hour Prediction: {predictions['predictions'][0]['predicted_occupancy']} people")
                print(f"📊 Model Type: {predictions.get('model_type', 'unknown')}")
                print(f"🎯 Accuracy: {predictions.get('accuracy', 0):.2f}")
            
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
        print(f"💾 Cache Hits: {stats['cache_hits']}")
        print(f"⚡ Average Response Time: {stats['average_response_time']:.3f}s")
        
        # Calculate FPS
        if stats['total_frames'] > 0:
            time_elapsed = time.time() - stats['last_update']
            fps = stats['total_frames'] / time_elapsed if time_elapsed > 0 else 0
            print(f"🎬 Average FPS: {fps:.1f}")
        
        # API efficiency
        if stats['api_calls'] > 0:
            efficiency = stats['cache_hits'] / stats['api_calls'] * 100
            print(f"📈 API Efficiency: {efficiency:.1f}%")
        
        print("="*50 + "\n")
    
    async def _cleanup(self):
        """Cleanup all resources"""
        self.is_running = False
        
        # Cleanup API client
        if self.ai_client:
            await self.ai_client.disconnect()
        
        # Cleanup core components
        self.yolo_detector.cleanup()
        self.energy_manager.cleanup()
        self.camera.release()
        cv2.destroyAllWindows()
        
        print("⚡ API System shutdown complete")

async def main():
    """Main function to run the API-powered Smart Lab"""
    try:
        # Configure API
        api_config = APIConfig(
            base_url="http://localhost:8000",
            websocket_url="ws://localhost:8000/ws",
            timeout=30,
            cache_enabled=True
        )
        
        # Initialize and start the API system
        smart_lab = SmartLabAPISystem(api_config)
        await smart_lab.start()
        
    except KeyboardInterrupt:
        print("\n🛑 System interrupted by user")
    except Exception as e:
        print(f"❌ System error: {e}")
    finally:
        print("👋 Goodbye!")

if __name__ == "__main__":
    asyncio.run(main())

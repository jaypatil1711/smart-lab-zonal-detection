import asyncio
import aiohttp
import json
import base64
import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import logging
from dataclasses import dataclass
import websockets
import threading
import time

logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """API configuration"""
    base_url: str = "http://localhost:8000"
    websocket_url: str = "ws://localhost:8000/ws"
    timeout: int = 30
    max_retries: int = 3
    cache_enabled: bool = True

class AIClient:
    """High-performance AI API client"""
    
    def __init__(self, config: APIConfig = None):
        self.config = config or APIConfig()
        self.session = None
        self.websocket = None
        self.websocket_connected = False
        self.message_handlers = {}
        self.cache = {}
        self.cache_timestamps = {}
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
    
    async def connect(self):
        """Connect to API"""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
            logger.info(f"✅ Connected to AI API at {self.config.base_url}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to API: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from API"""
        if self.session:
            await self.session.close()
        if self.websocket:
            await self.websocket.close()
        logger.info("❌ Disconnected from AI API")
    
    async def connect_websocket(self):
        """Connect to WebSocket for real-time updates"""
        try:
            self.websocket = await websockets.connect(self.config.websocket_url)
            self.websocket_connected = True
            
            # Start listening for messages
            asyncio.create_task(self._websocket_listener())
            
            logger.info("✅ WebSocket connected")
        except Exception as e:
            logger.error(f"❌ WebSocket connection failed: {e}")
            self.websocket_connected = False
    
    async def _websocket_listener(self):
        """Listen for WebSocket messages"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                message_type = data.get("type")
                
                if message_type in self.message_handlers:
                    handler = self.message_handlers[message_type]
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                        
        except websockets.exceptions.ConnectionClosed:
            logger.warning("⚠️ WebSocket connection closed")
            self.websocket_connected = False
        except Exception as e:
            logger.error(f"❌ WebSocket error: {e}")
    
    def on_message(self, message_type: str, handler: Callable):
        """Register message handler for WebSocket"""
        self.message_handlers[message_type] = handler
    
    async def _make_request(self, method: str, endpoint: str, data: Dict = None, params: Dict = None) -> Dict:
        """Make HTTP request with retry logic"""
        url = f"{self.config.base_url}{endpoint}"
        
        for attempt in range(self.config.max_retries):
            try:
                async with self.session.request(
                    method, url, json=data, params=params
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_text}")
                        
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    logger.error(f"❌ Request failed after {self.config.max_retries} attempts: {e}")
                    raise
                else:
                    logger.warning(f"⚠️ Request attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
    
    def _get_cache_key(self, endpoint: str, params: Dict = None) -> str:
        """Generate cache key"""
        key_data = f"{endpoint}_{json.dumps(params or {}, sort_keys=True)}"
        return str(hash(key_data))
    
    def _is_cache_valid(self, key: str, ttl: int = 60) -> bool:
        """Check if cache entry is valid"""
        if not self.config.cache_enabled:
            return False
        
        if key not in self.cache:
            return False
        
        timestamp = self.cache_timestamps.get(key, 0)
        return time.time() - timestamp < ttl
    
    def _set_cache(self, key: str, value: Any):
        """Set cache entry"""
        if self.config.cache_enabled:
            self.cache[key] = value
            self.cache_timestamps[key] = time.time()
    
    async def detect_objects(self, image: np.ndarray, zone_id: int = 1, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """Detect objects in image"""
        try:
            # Convert image to base64
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Check cache
            cache_key = self._get_cache_key("/api/detect", {
                "zone_id": zone_id,
                "confidence_threshold": confidence_threshold,
                "image_hash": str(hash(image_base64))
            })
            
            if self._is_cache_valid(cache_key, ttl=1):  # 1 second cache for detections
                logger.info("🚀 Cache hit for detection")
                return self.cache[cache_key]
            
            # Make API request
            data = {
                "image_data": image_base64,
                "zone_id": zone_id,
                "confidence_threshold": confidence_threshold
            }
            
            result = await self._make_request("POST", "/api/detect", data=data)
            
            # Cache result
            self._set_cache(cache_key, result)
            
            logger.info(f"✅ Detection completed in {result.get('processing_time', 0):.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Detection failed: {e}")
            raise
    
    async def get_predictions(self, hours_ahead: int = 4, zone_id: int = 1) -> Dict[str, Any]:
        """Get AI predictions"""
        try:
            # Check cache
            cache_key = self._get_cache_key("/api/predict", {
                "hours_ahead": hours_ahead,
                "zone_id": zone_id
            })
            
            if self._is_cache_valid(cache_key, ttl=300):  # 5 minutes cache for predictions
                logger.info("🚀 Cache hit for predictions")
                return self.cache[cache_key]
            
            # Make API request
            data = {
                "hours_ahead": hours_ahead,
                "zone_id": zone_id
            }
            
            result = await self._make_request("POST", "/api/predict", data=data)
            
            # Cache result
            self._set_cache(cache_key, result)
            
            logger.info("✅ Predictions retrieved")
            return result
            
        except Exception as e:
            logger.error(f"❌ Predictions failed: {e}")
            raise
    
    async def process_voice_command(self, command: str, user_id: str = None) -> Dict[str, Any]:
        """Process voice command"""
        try:
            # Check cache
            cache_key = self._get_cache_key("/api/voice", {"command": command})
            
            if self._is_cache_valid(cache_key, ttl=60):  # 1 minute cache for voice
                logger.info("🚀 Cache hit for voice command")
                return self.cache[cache_key]
            
            # Make API request
            data = {
                "command": command,
                "user_id": user_id
            }
            
            result = await self._make_request("POST", "/api/voice", data=data)
            
            # Cache result
            self._set_cache(cache_key, result)
            
            logger.info(f"✅ Voice command processed: {result.get('intent', 'unknown')}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Voice command failed: {e}")
            raise
    
    async def get_ai_insights(self) -> Dict[str, Any]:
        """Get AI insights"""
        try:
            # Check cache
            cache_key = self._get_cache_key("/api/insights")
            
            if self._is_cache_valid(cache_key, ttl=30):  # 30 seconds cache for insights
                logger.info("🚀 Cache hit for insights")
                return self.cache[cache_key]
            
            # Make API request
            result = await self._make_request("GET", "/api/insights")
            
            # Cache result
            self._set_cache(cache_key, result)
            
            logger.info("✅ AI insights retrieved")
            return result
            
        except Exception as e:
            logger.error(f"❌ AI insights failed: {e}")
            raise
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            result = await self._make_request("GET", "/api/cache/stats")
            return result
        except Exception as e:
            logger.error(f"❌ Cache stats failed: {e}")
            raise
    
    async def clear_cache(self, cache_type: str = None) -> Dict[str, Any]:
        """Clear cache"""
        try:
            params = {"cache_type": cache_type} if cache_type else {}
            result = await self._make_request("POST", "/api/cache/clear", params=params)
            
            # Clear local cache too
            if cache_type:
                keys_to_remove = [k for k in self.cache.keys() if k.startswith(cache_type)]
                for key in keys_to_remove:
                    del self.cache[key]
                    del self.cache_timestamps[key]
            else:
                self.cache.clear()
                self.cache_timestamps.clear()
            
            logger.info(f"✅ Cache cleared for {cache_type or 'all types'}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Clear cache failed: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        try:
            result = await self._make_request("GET", "/health")
            return result
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            raise
    
    async def ping_websocket(self):
        """Ping WebSocket connection"""
        if self.websocket_connected and self.websocket:
            try:
                await self.websocket.send(json.dumps({"type": "ping"}))
                logger.info("📡 WebSocket ping sent")
            except Exception as e:
                logger.error(f"❌ WebSocket ping failed: {e}")
                self.websocket_connected = False

class AIClientSync:
    """Synchronous wrapper for AI client"""
    
    def __init__(self, config: APIConfig = None):
        self.config = config or APIConfig()
        self.client = None
        self.loop = None
        self.thread = None
    
    def start(self):
        """Start the async client in a separate thread"""
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.client = AIClient(self.config)
            self.loop.run_until_complete(self.client.connect())
            self.loop.run_forever()
        
        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()
        
        # Wait for client to be ready
        time.sleep(1)
    
    def stop(self):
        """Stop the async client"""
        if self.loop and self.client:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=5)
    
    def detect_objects(self, image: np.ndarray, zone_id: int = 1, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """Detect objects (sync wrapper)"""
        if not self.client:
            raise Exception("Client not started. Call start() first.")
        
        future = asyncio.run_coroutine_threadsafe(
            self.client.detect_objects(image, zone_id, confidence_threshold),
            self.loop
        )
        return future.result(timeout=self.config.timeout)
    
    def get_predictions(self, hours_ahead: int = 4, zone_id: int = 1) -> Dict[str, Any]:
        """Get predictions (sync wrapper)"""
        if not self.client:
            raise Exception("Client not started. Call start() first.")
        
        future = asyncio.run_coroutine_threadsafe(
            self.client.get_predictions(hours_ahead, zone_id),
            self.loop
        )
        return future.result(timeout=self.config.timeout)
    
    def process_voice_command(self, command: str, user_id: str = None) -> Dict[str, Any]:
        """Process voice command (sync wrapper)"""
        if not self.client:
            raise Exception("Client not started. Call start() first.")
        
        future = asyncio.run_coroutine_threadsafe(
            self.client.process_voice_command(command, user_id),
            self.loop
        )
        return future.result(timeout=self.config.timeout)
    
    def get_ai_insights(self) -> Dict[str, Any]:
        """Get AI insights (sync wrapper)"""
        if not self.client:
            raise Exception("Client not started. Call start() first.")
        
        future = asyncio.run_coroutine_threadsafe(
            self.client.get_ai_insights(),
            self.loop
        )
        return future.result(timeout=self.config.timeout)

# Example usage
async def main():
    """Example usage of AI client"""
    config = APIConfig(
        base_url="http://localhost:8000",
        websocket_url="ws://localhost:8000/ws",
        timeout=30
    )
    
    async with AIClient(config) as client:
        # Connect to WebSocket
        await client.connect_websocket()
        
        # Register message handlers
        client.on_message("detection_update", lambda data: print(f"Detection update: {data}"))
        client.on_message("voice_response", lambda data: print(f"Voice response: {data}"))
        
        # Test detection
        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        result = await client.detect_objects(test_image)
        print(f"Detection result: {result}")
        
        # Test predictions
        predictions = await client.get_predictions()
        print(f"Predictions: {predictions}")
        
        # Test voice command
        voice_result = await client.process_voice_command("Hey Lab, what's the status?")
        print(f"Voice result: {voice_result}")
        
        # Test insights
        insights = await client.get_ai_insights()
        print(f"Insights: {insights}")

if __name__ == "__main__":
    asyncio.run(main())


#!/usr/bin/env python3
"""
API-Backed Smart Lab System Setup Script
This script sets up the high-performance API backend for maximum responsiveness
"""

import os
import sys
import subprocess
import json
import asyncio
import time
from pathlib import Path

def install_requirements():
    """Install API requirements"""
    print("📦 Installing API requirements...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_api.txt"])
        print("✅ API requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "src/api",
        "models",
        "data",
        "logs",
        "ai_data",
        "cache"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_api_config():
    """Create API configuration"""
    print("⚙️ Creating API configuration...")
    
    config = {
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "reload": True,
            "log_level": "info"
        },
        "client": {
            "base_url": "http://localhost:8000",
            "websocket_url": "ws://localhost:8000/ws",
            "timeout": 30,
            "max_retries": 3,
            "cache_enabled": True
        },
        "models": {
            "yolo": {
                "path": "yolov8n.pt",
                "confidence": 0.5,
                "iou_threshold": 0.45,
                "target_classes": ["person", "chair", "desk", "laptop", "book"]
            }
        },
        "cache": {
            "redis_enabled": False,
            "memory_cache_ttl": {
                "detection": 1,
                "prediction": 300,
                "insights": 30,
                "voice": 60
            }
        },
        "performance": {
            "max_workers": 4,
            "queue_size": 100,
            "batch_size": 1,
            "processing_interval": 0.1
        }
    }
    
    with open("api_config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    print("✅ API configuration created")

def create_startup_scripts():
    """Create startup scripts"""
    print("🚀 Creating startup scripts...")
    
    # API server startup script
    api_server_script = """#!/usr/bin/env python3
import uvicorn
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    print("🚀 Starting Smart Lab AI API Server...")
    print("📡 API Documentation: http://localhost:8000/docs")
    print("🔗 API Base URL: http://localhost:8000")
    print("📡 WebSocket URL: ws://localhost:8000/ws")
    
    uvicorn.run(
        "src.api.ai_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
"""
    
    with open("start_api_server.py", "w") as f:
        f.write(api_server_script)
    
    # API client startup script
    api_client_script = """#!/usr/bin/env python3
import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from api_main import main

if __name__ == "__main__":
    print("⚡ Starting Smart Lab API Client...")
    print("🔗 Connecting to API server at http://localhost:8000")
    print("📡 WebSocket streaming enabled")
    print("Press 'q' to quit, 'i' to show insights, 'p' to show performance")
    
    asyncio.run(main())
"""
    
    with open("start_api_client.py", "w") as f:
        f.write(api_client_script)
    
    print("✅ Startup scripts created")

def test_api_setup():
    """Test API setup"""
    print("🧪 Testing API setup...")
    
    try:
        # Test imports
        import fastapi
        import uvicorn
        import aiohttp
        import websockets
        import redis
        
        print("✅ All API libraries imported successfully")
        
        # Test FastAPI app creation
        from fastapi import FastAPI
        app = FastAPI()
        print("✅ FastAPI app created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ API setup test failed: {e}")
        return False

def create_docker_config():
    """Create Docker configuration for easy deployment"""
    print("🐳 Creating Docker configuration...")
    
    dockerfile = """FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the API server
CMD ["python", "start_api_server.py"]
"""
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile)
    
    # Docker Compose
    docker_compose = """version: '3.8'

services:
  ai-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - PYTHONPATH=/app
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
    volumes:
      - redis_data:/data

volumes:
  redis_data:
"""
    
    with open("docker-compose.yml", "w") as f:
        f.write(docker_compose)
    
    print("✅ Docker configuration created")

def create_performance_test():
    """Create performance test script"""
    print("⚡ Creating performance test...")
    
    test_script = """#!/usr/bin/env python3
import asyncio
import aiohttp
import time
import numpy as np
import base64
import cv2
from api.ai_client import AIClient, APIConfig

async def performance_test():
    \"\"\"Test API performance\"\"\"
    print("⚡ Starting API Performance Test...")
    
    config = APIConfig(
        base_url="http://localhost:8000",
        websocket_url="ws://localhost:8000/ws",
        timeout=30
    )
    
    async with AIClient(config) as client:
        # Test detection performance
        print("\\n🔍 Testing Detection Performance...")
        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        
        start_time = time.time()
        for i in range(10):
            result = await client.detect_objects(test_image)
            print(f"Detection {i+1}: {result.get('processing_time', 0):.3f}s")
        
        detection_time = time.time() - start_time
        print(f"Average detection time: {detection_time/10:.3f}s")
        
        # Test prediction performance
        print("\\n🔮 Testing Prediction Performance...")
        start_time = time.time()
        for i in range(5):
            result = await client.get_predictions()
            print(f"Prediction {i+1}: Retrieved")
        
        prediction_time = time.time() - start_time
        print(f"Average prediction time: {prediction_time/5:.3f}s")
        
        # Test voice command performance
        print("\\n🎤 Testing Voice Command Performance...")
        start_time = time.time()
        for i in range(5):
            result = await client.process_voice_command(f"Test command {i+1}")
            print(f"Voice {i+1}: {result.get('intent', 'unknown')}")
        
        voice_time = time.time() - start_time
        print(f"Average voice processing time: {voice_time/5:.3f}s")
        
        # Test cache performance
        print("\\n💾 Testing Cache Performance...")
        cache_stats = await client.get_cache_stats()
        print(f"Cache stats: {cache_stats}")
        
        print("\\n✅ Performance test completed!")

if __name__ == "__main__":
    asyncio.run(performance_test())
"""
    
    with open("test_api_performance.py", "w") as f:
        f.write(test_script)
    
    print("✅ Performance test created")

def main():
    """Main setup function"""
    print("⚡ Smart Lab API System Setup")
    print("=" * 50)
    
    # Step 1: Install requirements
    if not install_requirements():
        print("❌ Setup failed at requirements installation")
        return False
    
    # Step 2: Create directories
    create_directories()
    
    # Step 3: Create configuration
    create_api_config()
    
    # Step 4: Create startup scripts
    create_startup_scripts()
    
    # Step 5: Create Docker config
    create_docker_config()
    
    # Step 6: Create performance test
    create_performance_test()
    
    # Step 7: Test setup
    if not test_api_setup():
        print("❌ Setup failed at API test")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 API System Setup Complete!")
    print("=" * 50)
    print("📋 Next Steps:")
    print("1. Start API Server: python start_api_server.py")
    print("2. Start API Client: python start_api_client.py")
    print("3. Test Performance: python test_api_performance.py")
    print("4. View API Docs: http://localhost:8000/docs")
    print("5. WebSocket Test: ws://localhost:8000/ws")
    print("\n🚀 Your Smart Lab is now API-powered!")
    print("⚡ Features: High-performance backend, Real-time streaming, Caching")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


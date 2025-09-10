#!/usr/bin/env python3
"""
AI/ML Setup Script for Smart Lab System
This script sets up the AI/ML environment and downloads required models
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def install_requirements():
    """Install AI/ML requirements"""
    print("📦 Installing AI/ML requirements...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_ai.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "src/ai",
        "models",
        "data",
        "logs",
        "ai_data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def download_models():
    """Download AI models"""
    print("🤖 Downloading AI models...")
    
    try:
        from ultralytics import YOLO
        
        # Download YOLO models
        models = [
            "yolov8n.pt",      # Object detection
            "yolov8n-pose.pt", # Pose estimation
            "yolov8n-face.pt"  # Face detection
        ]
        
        for model_name in models:
            print(f"📥 Downloading {model_name}...")
            model = YOLO(model_name)
            print(f"✅ {model_name} downloaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading models: {e}")
        return False

def create_ai_config():
    """Create AI configuration file"""
    print("⚙️ Creating AI configuration...")
    
    config = {
        "models": {
            "yolo": {
                "path": "yolov8n.pt",
                "confidence": 0.5,
                "iou_threshold": 0.45,
                "target_classes": ["person", "chair", "desk", "laptop", "book"]
            },
            "pose": {
                "path": "yolov8n-pose.pt",
                "confidence": 0.6,
                "keypoints": 17
            },
            "face": {
                "path": "yolov8n-face.pt",
                "confidence": 0.7
            }
        },
        "tracking": {
            "max_disappeared": 30,
            "max_distance": 50,
            "tracking_method": "centroid"
        },
        "analytics": {
            "activity_threshold": 0.8,
            "pose_confidence": 0.6,
            "face_recognition": False,
            "emotion_detection": False
        },
        "ml_features": {
            "occupancy_prediction": True,
            "energy_optimization": True,
            "anomaly_detection": True,
            "activity_classification": True
        },
        "voice": {
            "enabled": True,
            "language": "en-US",
            "voice_rate": 200,
            "voice_volume": 0.8
        },
        "commands": {
            "wake_words": ["hey lab", "smart lab", "lab assistant"],
            "timeout": 5
        },
        "data_collection": {
            "save_detections": True,
            "save_poses": True,
            "save_activities": True,
            "database_path": "ai_data/ai_data.db"
        }
    }
    
    with open("ai_config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    print("✅ AI configuration created")

def test_installation():
    """Test AI installation"""
    print("🧪 Testing AI installation...")
    
    try:
        # Test imports
        import cv2
        import numpy as np
        import torch
        import sklearn
        import pandas as pd
        from ultralytics import YOLO
        
        print("✅ All AI libraries imported successfully")
        
        # Test YOLO model loading
        model = YOLO("yolov8n.pt")
        print("✅ YOLO model loaded successfully")
        
        # Test basic functionality
        import numpy as np
        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model(test_image)
        print("✅ YOLO inference test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False

def create_startup_script():
    """Create startup script for AI system"""
    print("🚀 Creating startup script...")
    
    startup_script = """#!/usr/bin/env python3
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ai_main import main

if __name__ == "__main__":
    print("🤖 Starting Smart Lab AI System...")
    main()
"""
    
    with open("start_ai.py", "w") as f:
        f.write(startup_script)
    
    print("✅ Startup script created: start_ai.py")

def main():
    """Main setup function"""
    print("🤖 Smart Lab AI/ML Setup")
    print("=" * 40)
    
    # Step 1: Install requirements
    if not install_requirements():
        print("❌ Setup failed at requirements installation")
        return False
    
    # Step 2: Create directories
    create_directories()
    
    # Step 3: Download models
    if not download_models():
        print("⚠️ Model download failed, but continuing...")
    
    # Step 4: Create configuration
    create_ai_config()
    
    # Step 5: Test installation
    if not test_installation():
        print("❌ Setup failed at installation test")
        return False
    
    # Step 6: Create startup script
    create_startup_script()
    
    print("\n" + "=" * 40)
    print("🎉 AI/ML Setup Complete!")
    print("=" * 40)
    print("📋 Next Steps:")
    print("1. Run: python start_ai.py")
    print("2. Say 'Hey Lab' to activate voice commands")
    print("3. Press 'i' to view AI insights")
    print("4. Press 'v' to toggle voice interface")
    print("5. Press 'q' to quit")
    print("\n🤖 Your Smart Lab is now AI-powered!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


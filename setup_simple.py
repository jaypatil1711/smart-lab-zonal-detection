#!/usr/bin/env python3
"""
Simple setup script for Smart Lab system with cloud API
No Docker, no async complexity - just install and run!
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_simple.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False
    return True

def create_env_file():
    """Create .env file for API keys"""
    env_file = ".env"
    if not os.path.exists(env_file):
        print("🔑 Creating .env file for API keys...")
        with open(env_file, "w") as f:
            f.write("# Smart Lab API Configuration\n")
            f.write("# Add your API keys here\n\n")
            f.write("# Google Cloud Vision API\n")
            f.write("GOOGLE_API_KEY=your_google_api_key_here\n\n")
            f.write("# OpenAI API\n")
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n\n")
            f.write("# Hugging Face API\n")
            f.write("HUGGINGFACE_API_KEY=your_huggingface_api_key_here\n")
        print("✅ .env file created! Please add your API keys.")
    else:
        print("✅ .env file already exists!")

def check_yolo_model():
    """Check if YOLO model exists"""
    model_path = "yolov8n.pt"
    if not os.path.exists(model_path):
        print("🤖 Downloading YOLO model...")
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')
            print("✅ YOLO model downloaded!")
        except Exception as e:
            print(f"⚠️ Could not download YOLO model: {e}")
            print("   You can download it manually from: https://github.com/ultralytics/ultralytics")
    else:
        print("✅ YOLO model found!")

def main():
    """Main setup function"""
    print("🚀 Setting up Simple Smart Lab System")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed!")
        return
    
    # Create .env file
    create_env_file()
    
    # Check YOLO model
    check_yolo_model()
    
    print("\n" + "=" * 50)
    print("✅ Setup complete!")
    print("\n📋 Next steps:")
    print("1. Add your API keys to the .env file")
    print("2. Run: python src/simple_main.py")
    print("\n🎯 Available API services:")
    print("   • Google Cloud Vision API (free tier: 1000 requests/month)")
    print("   • OpenAI Vision API (paid)")
    print("   • Hugging Face Inference API (free tier available)")
    print("\n💡 No Docker, no async complexity - just simple HTTP requests!")

if __name__ == "__main__":
    main()

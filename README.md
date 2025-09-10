# 🚀 Smart Lab Zonal Detection System

**Simple, Fast, and Easy to Use!** No Docker, no async complexity - just add your API key and run!

## ✨ Features

- 🎯 **Real-time Object Detection** - Detect people in lab zones
- ☁️ **Cloud AI Integration** - Use powerful cloud APIs with just an API key
- ⚡ **Simple Setup** - No Docker, no complex configuration
- 🔋 **Energy Management** - Smart energy saving based on occupancy
- 📊 **Performance Monitoring** - Track system performance
- 🎤 **Voice Commands** - Simple voice command processing

## 🚀 Quick Start

### 1. Install Dependencies
```bash
python setup_simple.py
```

### 2. Get Free API Key
Choose one of these **FREE** options:

| Service | Free Tier | Setup |
|---------|-----------|-------|
| **Google Cloud Vision** | 1,000 requests/month | API key only |
| **Hugging Face** | 50-100 calls/day | Access token only |

**Recommended: Google Cloud Vision API**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable Vision API
3. Create API key
4. Add to `.env` file: `GOOGLE_API_KEY=your_key_here`

### 3. Run the System
```bash
python src/simple_main.py
```

## 🎮 Controls

- **`q`** - Quit
- **`r`** - Reset zones  
- **`i`** - Show insights
- **`p`** - Show performance
- **`c`** - Test cloud detection

## 📁 Project Structure

```
zonal-detection/
├── src/
│   ├── simple_main.py          # Main system (simple, no async!)
│   ├── simple_ai_client.py     # Cloud API client
│   ├── camera/                 # Camera handling
│   ├── detection/              # Local YOLO detection
│   └── utils/                  # Utilities
├── requirements.txt            # Simple dependencies
├── setup_simple.py            # Easy setup script
├── README_SIMPLE.md           # Detailed documentation
└── .env                       # API keys (create this)
```

## 🔧 How It Works

### Simple Architecture
```
Camera → Local YOLO → Display
   ↓
Cloud API (on demand) → Results
```

### Key Benefits
- **No Server Management** - Cloud handles everything
- **No Model Downloads** - Use cloud models
- **No Docker** - Simple Python installation
- **No Async Complexity** - Synchronous code
- **Pay Per Use** - Only pay for API calls

## 💡 Usage Examples

### Basic Detection
```python
from simple_ai_client import SimpleAIClient

client = SimpleAIClient(api_key="your_key", service="google_vision")
result = client.detect_objects(image)
print(f"Found {len(result['detections'])} objects")
```

### Voice Commands
```python
response = client.process_voice_command("What's the lab status?")
print(response['response'])
```

## 🆓 Free API Options

### Google Cloud Vision API (Recommended)
- **Free Tier:** 1,000 requests/month
- **Perfect for:** Object detection, person detection
- **Setup:** Google account + API key

### Hugging Face Inference API
- **Free Tier:** 50-100 API calls/day
- **Perfect for:** Custom models
- **Setup:** Free account + access token

## 🎯 Perfect For

- **Prototyping** - Quick setup and testing
- **Small Projects** - No infrastructure needed
- **Learning** - Simple, understandable code
- **Cost-Conscious** - Pay only for usage
- **Quick Demos** - Get running in minutes

## 📚 Documentation

- **[Simple Setup Guide](README_SIMPLE.md)** - Detailed setup instructions
- **[API Documentation](README_SIMPLE.md#api-integration)** - How to use cloud APIs
- **[Troubleshooting](README_SIMPLE.md#troubleshooting)** - Common issues and solutions

## 🚀 Next Steps

1. **Start Simple** - Use this version first
2. **Get API Key** - Choose Google Vision or Hugging Face
3. **Run System** - `python src/simple_main.py`
4. **Customize** - Add your own features

---

**Ready to get started?** Run `python setup_simple.py` and you'll be up and running in minutes! 🎉
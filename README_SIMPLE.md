# 🚀 Simple Smart Lab System

**No Docker, No Async, No Complexity!** Just a simple API key and you're ready to go.

## ✨ Why This Approach?

- **🔑 API Key Only** - No complex server setup
- **⚡ Simple HTTP Requests** - No async/await complexity  
- **🐳 No Docker** - Just install and run
- **☁️ Cloud AI** - Use powerful cloud APIs
- **💰 Cost Effective** - Pay only for what you use

## 🎯 Supported AI Services

| Service | Free Tier | Best For | Setup |
|---------|-----------|----------|-------|
| **Google Cloud Vision** | 1000 requests/month | Object detection | API key only |
| **Hugging Face** | Free tier available | Custom models | API key only |
| **OpenAI Vision** | Paid | Advanced analysis | API key only |

## 🚀 Quick Start

### 1. Install Dependencies
```bash
python setup_simple.py
```

### 2. Get API Key
Choose one service and get your API key:

**Google Cloud Vision (Recommended):**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable Vision API
3. Create credentials (API key)
4. Add to `.env` file: `GOOGLE_API_KEY=your_key_here`

**Hugging Face (Free):**
1. Go to [Hugging Face](https://huggingface.co/settings/tokens)
2. Create access token
3. Add to `.env` file: `HUGGINGFACE_API_KEY=your_token_here`

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

## 📁 File Structure

```
zonal-detection/
├── src/
│   ├── simple_main.py          # Main system (no async!)
│   ├── simple_ai_client.py     # Simple API client
│   ├── camera/                 # Camera handling
│   ├── detection/              # Local YOLO detection
│   └── utils/                  # Utilities
├── requirements_simple.txt     # Simple dependencies
├── setup_simple.py            # Easy setup script
└── .env                       # API keys (create this)
```

## 🔧 How It Works

### Simple Architecture
```
Camera → Local YOLO → Display
   ↓
Cloud API (on demand) → Results
```

### API Integration
```python
# Simple HTTP request - no async needed!
client = SimpleAIClient(api_key="your_key", service="google_vision")
result = client.detect_objects(image)
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

### Get Insights
```python
insights = client.get_insights()
print(f"Total detections: {insights['total_detections']}")
```

## 🔄 Migration from Complex Version

If you're migrating from the async version:

1. **Replace** `api_main.py` with `simple_main.py`
2. **Replace** `ai_client.py` with `simple_ai_client.py`
3. **Remove** `ai_api.py` (no longer needed)
4. **Add** your API key to `.env` file
5. **Run** `python src/simple_main.py`

## 🆚 Comparison

| Feature | Complex Version | Simple Version |
|---------|----------------|----------------|
| Setup | Docker + async | API key only |
| Dependencies | 20+ packages | 5 packages |
| Server | Local FastAPI | Cloud API |
| Models | Local YOLO | Cloud models |
| Caching | Redis + local | Cloud handles |
| WebSocket | Custom | Not needed |
| Maintenance | High | Low |

## 🎯 Perfect For

- **Prototyping** - Quick setup and testing
- **Small Projects** - No infrastructure needed
- **Learning** - Simple, understandable code
- **Cost-Conscious** - Pay only for usage
- **Quick Demos** - Get running in minutes

## 🚀 Next Steps

1. **Start Simple** - Use this version first
2. **Scale Up** - Move to complex version if needed
3. **Customize** - Add your own features
4. **Deploy** - Easy to deploy anywhere

---

**Ready to get started?** Run `python setup_simple.py` and you'll be up and running in minutes! 🎉

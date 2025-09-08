# Smart Lab Zonal Detection System

A smart laboratory management system that uses computer vision and IoT to detect occupancy in different zones and manage energy efficiency.

## Features

### Motion Detection & Zonal Analysis
- Real-time motion detection using computer vision
- Multiple zone monitoring with independent settings
- Adjustable motion sensitivity thresholds
- Background subtraction for accurate detection
- Visual feedback with zone status overlay
- Support for both Raspberry Pi and USB cameras

### Energy Management
- Automated device control based on zone occupancy
- Individual timeout settings for each zone
- Working hours scheduling system (8 AM - 6 PM by default)
- Real-time power state management
- GPIO-based device control for each zone
- Energy usage statistics and reporting

### Smart Automation
- Configurable zone-specific timeouts
- Automatic device deactivation in unoccupied zones
- Working hours enforcement
- Background monitoring thread for reliability
- Automatic state recovery after system restart

### System Features
- Real-time monitoring dashboard
- Live occupancy status display
- Zone activity visualization
- Easy-to-use configuration system
- Modular and extensible architecture
- Clean shutdown and resource management

### Configuration & Customization
- JSON-based configuration file
- Adjustable zone parameters
- Customizable timeout settings
- Configurable GPIO pin assignments
- Flexible working hours definition
- Adjustable motion detection parameters

## Project Structure

```
zonal-detection
├── src
│   ├── main.py                # Entry point of the application
│   ├── camera
│   │   ├── camera_stream.py    # Handles camera streaming
│   │   └── camera_utils.py     # Utility functions for camera operations
│   ├── detection
│   │   ├── zone_detection.py    # Implements zonal detection logic
│   │   └── motion_detection.py   # Functions for detecting motion
│   └── utils
│       └── helpers.py          # Helper functions for logging and configuration
├── requirements.txt            # Required Python libraries
├── README.md                   # Project documentation
└── .gitignore                  # Files to ignore in version control
```

## Setup Instructions

1. **Install Required Libraries**: Make sure you have Python installed on your Raspberry Pi. Install the required libraries by running:
   ```
   pip install -r requirements.txt
   ```

2. **Connect the Raspberry Pi Camera**: Ensure that the Raspberry Pi camera is properly connected and enabled in the Raspberry Pi configuration settings.

3. **Run the Application**: Execute the main script to start the zonal detection system:
   ```
   python src/main.py
   ```

## Usage

- The application will start capturing video from the Raspberry Pi camera.
- It will analyze the frames for motion detection within predefined zones.
- Adjust the zone parameters in `zone_detection.py` as needed for your specific use case.

## Main Components

- **Camera Stream**: Captures video frames from the Raspberry Pi camera.
- **Zonal Detection**: Analyzes frames to detect motion in specified zones.
- **Motion Detection**: Implements algorithms for detecting motion in video frames.

## License

This project is open-source and available for modification and distribution under the MIT License.
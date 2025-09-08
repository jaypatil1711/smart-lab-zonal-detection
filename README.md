# Zonal Detection with Raspberry Pi Camera

This project implements a zonal detection system using a Raspberry Pi camera. The system captures video frames and analyzes them to detect motion within predefined zones.

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
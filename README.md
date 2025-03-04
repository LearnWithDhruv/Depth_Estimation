# Distance Estimation Using Depth Mapping

## Overview

This script estimates depth from an Orbbec Astra 2 depth camera using a deep learning model (MiDaS) and calculates real-world distances between selected points. It also includes a calibration mode to improve depth accuracy.

## Features

- Captures video using a webcam.
- Uses the MiDaS model to estimate depth.
- Allows users to select two points on the screen.
- Calculates real-world distances between selected points.
- Supports calibration for improved accuracy.
- Displays real-time depth and distance on the screen.

## Libraries Used

- `cv2` – Captures and displays video.
- `torch` – Loads and runs the MiDaS depth model.
- `numpy` – Performs numerical calculations.
- `PIL (Image)` – Processes frames before passing them to the model.
- `json` – Loads and saves calibration data.
- `time` – Calculates FPS (Frames Per Second).
- `os` – Checks for existing calibration files.

## How Depth Estimation Works

1. The script loads the MiDaS model to predict a depth map from a 2D image.
2. The depth map is a grayscale representation where brighter pixels indicate closer objects and darker pixels indicate farther objects.
3. The user selects two points on the screen by clicking.
4. The script retrieves depth values at those points and converts them into real-world distances using calibration data.
5. In calibration mode, users can add reference points to improve depth accuracy.

## Calibration and Distance Calculation

### Calibration Mode:

- Maps raw depth values to real-world distances.
- Uses user-provided reference points.
- Allows the user to set a real-world distance and save reference data.

### Distance Calculation:

1. Extracts depth values at selected points.
2. Applies calibration to convert raw depth to real-world depth.
3. Computes the pixel distance between two points.
4. Uses focal length and depth values to estimate real-world X-Y distance.
5. Applies the 3D distance formula to find the final distance.

## Data Storage and Display

- **On-Screen Display:** Shows FPS, mode (calibration/measurement), and distance.
- **Calibration Data (depth_calibration.json):** Stores depth-to-distance mappings for improved accuracy.

## How the Script Works (Step-by-Step)

1. Loads or creates a calibration file (`depth_calibration.json`).
2. Loads the MiDaS depth estimation model.
3. Starts the webcam feed.
4. Processes every 5th frame to estimate depth.
5. Displays the depth map using a color gradient.
6. Users can select two points by clicking on the screen.
7. If calibration mode is enabled:
   - The user sets a real-world distance.
   - Pressing `Enter` adds a calibration point.
8. If measurement mode is enabled:
   - The script calculates the real-world distance between two points.
9. Pressing `Q` exits the program.

## Key Features and Controls

### Video Feed:

- **"RGB Feed"** window: Shows live webcam feed.
- **"Depth Map"** window: Displays estimated depth.

### User Controls:

- **Left Click:** Select two points for measurement.
- **'C' Key:** Toggle calibration mode.
- **'+' / '-' Key:** Adjust real-world distance in calibration mode.
- **'Enter' Key:** Save a calibration point.
- **'Q' Key:** Quit the program.

## Improvements Made

- Added calibration mode to improve real-world accuracy.
- Optimized depth processing by running every 5 frames.
- Real-time depth visualization with a heatmap-style depth map.
- Improved UI messages to guide the user.

## Conclusion

This script enables accurate depth measurement using AI and an Orbbec Astra 2 depth camera with multiple calibration points. By continuously adding calibration data, accuracy improves over time. This tool is valuable for applications in robotics, augmented reality, and distance estimation.

---

### Author:

[Your Name]

### License:

[Specify License Here]

### Contact:

[Your Contact Information]

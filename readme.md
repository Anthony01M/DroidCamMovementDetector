# DroidCamMovementDetector

## YOLO Object Detection with Alarm System

This project implements a real-time object detection system using YOLO (You Only Look Once) and OpenCV. It captures video from a DroidCam, processes the frames to detect objects, and triggers an alarm if a person is detected.

## Features

- Real-time object detection using YOLOv3.
- Alarm system that plays a sound when a person is detected.
- Adjustable frame size and FPS settings.
- Full-screen mode for the video display.
- Toggle alarm mode and stop alarm sound with keyboard inputs.

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/Anthony01M/DroidCamMovementDetector.git
    cd DroidCamMovementDetector
    ```

2. **Install the required dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Download YOLOv3 weights and configuration files**:
    - Download the YOLOv3 weights from [here](https://pjreddie.com/media/files/yolov3.weights).
    - Download the YOLOv3 configuration file from [here](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg).
    - Place these files in the `models/yolo` directory.

4. **Download COCO names file**:
    - Download the COCO names file from [here](https://github.com/pjreddie/darknet/blob/master/data/coco.names).
    - Place this file in the `models/yolo` directory.

5. **Download Alert Sound**:
    - Download an alert sound file (e.g., `alarm.wav`) from [here](https://uppbeat.io/sfx/tag/emergency)
    - Place this file in the `alert` directory.

6. **Install DroidCam Client**:
    - Download and install the DroidCam Application on your Android device from [here](https://play.google.com/store/apps/details?id=com.dev47apps.droidcam).
    
7. **Fix Configuration**:
    - Open the `config.yml` file and update every filed as needed.

## Usage

1. **Run the script**:
    ```sh
    python main.py
    ```

2. **Keyboard Controls**:
    - Press `t` to toggle alarm mode.
    - Press `g` to stop the alarm sound.
    - Press `f2` to toggle full-screen mode.
    - Press `q` to quit the application.

## Disclaimer

This project is intended for educational purposes only. It should not be used for any malicious activities or to invade anyone's privacy. The authors are not responsible for any misuse of this software.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
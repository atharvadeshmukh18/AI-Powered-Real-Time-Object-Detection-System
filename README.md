# AI Object Detection System

A real-time object detection project using YOLOv8, OpenCV, and an optional voice assistant.

## Overview

This project detects objects from a webcam or video file using the Ultralytics YOLOv8 model. It also includes a voice assistant for spoken commands and optional Streamlit dashboard support.

## Features

- Real-time object detection with `yolov8n.pt`
- Voice assistant using `SpeechRecognition` and `pyttsx3`
- Keyboard controls for detection, screenshots, and voice toggle
- Built-in confidence threshold adjustment
- Optional Streamlit dashboard for browser-based monitoring
- Automatic screenshot saving when important objects appear

## Requirements

- Python 3.9+
- `ultralytics`
- `opencv-python`
- `numpy`
- `SpeechRecognition`
- `pyttsx3`
- `pyaudio`
- `streamlit` (optional)
- `streamlit-webrtc` (optional)
- `Pillow`
- `requests`
- `python-dotenv`
- `loguru`

## Setup

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies and prepare the project:

```powershell
python setup.py
```

This installs the required packages, downloads the `yolov8n.pt` weights, and verifies camera and audio support.

## Run

Run the application with the default webcam:

```powershell
python app.py
```

CLI options:

```powershell
python app.py --source 1
python app.py --source video.mp4
python app.py --conf 0.5
python app.py --no-voice
python app.py --width 1280 --height 720
```

## Voice Assistant

The voice assistant listens for natural commands and responds with TTS. Supported commands include:

- "what do you see"
- "describe the scene"
- "how many persons"
- "count cars"
- "take a screenshot"
- "start detection"
- "stop detection"
- "increase confidence"
- "decrease confidence"
- "help"

Press `V` while the app window is focused to toggle voice on and off.

## Dashboard

If you want a browser-based interface, run:

```powershell
streamlit run dashboard.py
```

## Project Structure

- `app.py` — main application entry point
- `assistant.py` — speech recognition and TTS voice assistant
- `detection.py` — YOLOv8 detection and drawing logic
- `dashboard.py` — optional Streamlit dashboard
- `config.py` — central configuration values
- `setup.py` — automated setup script
- `requirements.txt` — dependency list

## Notes

- `yolov8n.pt` is the YOLOv8 Nano model weights file used for fast inference.
- If `pyaudio` fails to install on Windows, install it from a compatible wheel.
- When using a video file source, the app loops the video automatically.

## License

This project is open for modification and reuse.

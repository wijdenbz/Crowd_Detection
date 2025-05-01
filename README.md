# Crowd Detection with YOLO Models

A Streamlit application for real-time crowd detection and analysis using YOLOv11 models.

## Features

- Real-time video processing with YOLOv11 models
- Multiple model options (yolo11n, yolo11s, yolo11m, yolo11l, yolo11x)
- Performance metrics tracking
- Interactive visualization of detection results
- FPS and object count monitoring

## Installation

1. Clone the repository:
```bash
git clone https://github.com/wijdenbz/Crowd_Detection.git
cd Crowd-Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Upload a video file through the web interface
3. Select a YOLO model and confidence threshold
4. Click "Start Processing" to begin detection

## Project Structure

- `app.py`: Main Streamlit application
- `evaluation.py`: Evaluation metrics and processing functions
- `requirements.txt`: Project dependencies

## Requirements

- Python 3.8+
- Streamlit
- OpenCV
- PyTorch
- Ultralytics YOLO

## License

MIT License
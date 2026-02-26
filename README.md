# Vi-SAFE

A video violence detection system combining YOLOv8 object detection with Temporal Segment Networks (TSN) for the RWF-2000 dataset. Supports both **pre-recorded video analysis** and **real-time camera detection**.

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Download Model Weights

https://github.com/ultralytics/assets/releases

### Data Structure
```
data/
├── train/
│   ├── Fight/      # Violence videos
│   └── NonFight/   # Normal videos  
└── val/
    ├── Fight/
    └── NonFight/
```

### Training
```bash
# Simple version
cd app_simple
python main.py --rwf2000_root /path/to/data --epochs 50

# Full version with advanced features
cd app
python main.py --rwf2000_root /path/to/data --epochs 100

# Standalone training script
python run_training.py --rwf2000_root /path/to/data --epochs 50
```

## Inference

### Single Video
```bash
python inference.py path/to/video.mp4 --checkpoint checkpoints/best_model.pt
```

### Annotated Video Output
```bash
python visualize_inference.py path/to/video.mp4 --output_path output_video.mp4
```

### 🎥 Real-Time Camera Detection
```bash
# Webcam (default camera 0)
python realtime_inference.py

# Second camera
python realtime_inference.py --source 1

# RTSP / IP camera stream
python realtime_inference.py --source rtsp://192.168.1.10:554/stream

# With recording
python realtime_inference.py --record output.mp4

# Faster inference (update prediction every 4 frames)
python realtime_inference.py --inference_interval 4

# Custom alert threshold
python realtime_inference.py --alert_threshold 0.8
```

**Controls (in the OpenCV window):**
- `q` or `ESC` — Quit
- `s` — Take screenshot
- `r` — Toggle recording

### 🌐 Streamlit Web App (Upload + Real-Time)
```bash
streamlit run streamlit_app.py
```

The web app offers two modes:
- **📹 Upload Video** — Upload and analyze a pre-recorded video
- **🎥 Real-Time Camera** — Connect to a webcam or IP camera for live detection

## Model Architecture

- **YOLOv8**: Object detection and spatial encoding
- **TSN**: Temporal feature extraction (ResNet-50 backbone)
- **Fusion**: Multi-modal feature concatenation → FC classifier
- **Classes**: Fight / NonFight

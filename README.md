# Vi-SAFE

A video violence detection system combining YOLOv8 object detection with Temporal Segment Networks (TSN) for the RWF-2000 dataset.

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
```

## Model Architecture

- **YOLOv8**: Object detection and spatial encoding
- **TSN**: Temporal feature extraction
- **Fusion**: Multi-modal feature combination

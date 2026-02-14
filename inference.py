#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vi-SAFE Inference Script
Run violence detection on a single video using the trained model.
"""

import os
import cv2
import torch
import argparse
import numpy as np
import torchvision.transforms as T
from PIL import Image

# Import model definitions from our training script
# This ensures we use the EXACT same architecture definition
try:
    from run_training import YOLOEncoder, TSNWithYOLO
except ImportError:
    # If run_training.py is not in path or has issues, we might need to redefine or fix path
    import sys
    sys.path.append(os.getcwd())
    from run_training import YOLOEncoder, TSNWithYOLO

# Constants
CLASSES = ['NonFight', 'Fight']  # 0: NonFight, 1: Fight

def load_video(video_path, num_segments=12):
    """
    Load and preprocess a video for the model.
    Samples 'num_segments' frames uniformly.
    Returns: tensor of shape [1, T, 3, H, W]
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video has 0 frames: {video_path}")

    # Sample frames
    if total_frames < num_segments:
        indices = np.linspace(0, total_frames - 1, num_segments).astype(int)
    else:
        indices = np.linspace(0, total_frames - 1, num_segments).astype(int)
        
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            # Fallback: black frame
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            
        frames.append(frame)
    cap.release()
    
    # Transform
    # Same validation transform as training: ToTensor (scale 0-1)
    transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor()
    ])
    
    frames_t = [transform(f) for f in frames]
    frames_tensor = torch.stack(frames_t, dim=0) # [T, 3, H, W]
    
    # Add batch dimension: [1, T, 3, H, W]
    return frames_tensor.unsqueeze(0)

def predict(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Initialize Models
    print("Initializing models...")
    # Matches training config
    yolo_encoder = YOLOEncoder(
        yolo_weights_path=args.yolo_weights,
        device=device,
        freeze_yolo=True # Inference only
    ).to(device)
    
    model = TSNWithYOLO(
        mmaction_cfg_path=args.mmaction_cfg,
        yolo_encoding_dim=128,
        device=device
    ).to(device)
    
    # 2. Load Checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    yolo_encoder.load_state_dict(checkpoint['yolo_encoder_state_dict'])
    
    model.eval()
    yolo_encoder.train(False) # Ensure generic eval mode (YOLOwrapper handles internal eval)
    
    # 3. Process Video
    print(f"Processing video: {args.video_path}")
    frames = load_video(args.video_path, num_segments=args.num_segments)
    frames = frames.to(device) # [B, T, C, H, W]
    
    # 4. Inference
    with torch.no_grad():
        # Get YOLO features
        # frames shape: [1, T, 3, H, W] -> loop over batch 1
        yolo_enc_batch = []
        for b_idx in range(frames.shape[0]):
            yolo_enc = yolo_encoder(frames[b_idx]) # [T, 128]
            yolo_enc_batch.append(yolo_enc)
        yolo_enc_batch = torch.stack(yolo_enc_batch, dim=0) # [1, T, 128]
        
        # Forward pass
        logits = model(frames, yolo_enc_batch) # [1, 2]
        probs = torch.softmax(logits, dim=1)
        
        # Get result
        score, pred_idx = torch.max(probs, dim=1)
        pred_idx = pred_idx.item()
        score = score.item()
        
    label = CLASSES[pred_idx]
    
    # 5. Output
    print("\n" + "="*30)
    print("PREDICTION RESULTS")
    print("="*30)
    print(f"Video: {os.path.basename(args.video_path)}")
    print(f"Class: {label}")
    print(f"Confidence: {score:.4f}")
    print("="*30 + "\n")
    
    # Breakdown
    print(f"Probability Distribution:")
    print(f"  NonFight: {probs[0][0].item():.4f}")
    print(f"  Fight:    {probs[0][1].item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vi-SAFE Inference")
    parser.add_argument('video_path', type=str, help="Path to input video file")
    parser.add_argument('--checkpoint', type=str, default="checkpoints/best_model.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument('--mmaction_cfg', type=str,
                        default="./mmaction2/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py",
                        help="Path to mmaction2 TSN config configuration file")
    parser.add_argument('--yolo_weights', type=str, default="yolov8n.pt",
                        help="Path to YOLO weights")
    parser.add_argument('--num_segments', type=int, default=12, help="Number of frames to sample")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    predict(args)

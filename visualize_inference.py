#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vi-SAFE Visualization Script
Run violence detection and save an annotated video.
"""

import os
import cv2
import torch
import argparse
import numpy as np
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

# Import model definitions
try:
    from run_training import YOLOEncoder, TSNWithYOLO
except ImportError:
    import sys
    sys.path.append(os.getcwd())
    from run_training import YOLOEncoder, TSNWithYOLO

CLASSES = ['NonFight', 'Fight']
COLORS = [(0, 255, 0), (0, 0, 255)] # Green, Red

def visualize(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Models
    print("Loading models...")
    yolo_encoder = YOLOEncoder(
        yolo_weights_path=args.yolo_weights,
        device=device,
        freeze_yolo=True
    ).to(device)
    
    model = TSNWithYOLO(
        mmaction_cfg_path=args.mmaction_cfg,
        yolo_encoding_dim=128,
        device=device
    ).to(device)
    
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    yolo_encoder.load_state_dict(checkpoint['yolo_encoder_state_dict'])
    
    model.eval()
    yolo_encoder.train(False)
    
    # 2. Open Video
    print(f"Processing video: {args.video_path}")
    cap = cv2.VideoCapture(args.video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))
    
    transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor()
    ])
    
    # Buffer for TSN (needs num_segments frames)
    # For real-time visualization, we can use a sliding window or just accumulate enough frames
    # Here, we will simplify: We will run inference on a buffer every N frames, 
    # and use the last prediction for annotation until the next update.
    
    buffer = []
    current_pred = "Analyzing..."
    current_color = (255, 255, 255)
    current_prob = 0.0
    
    # Inference interval (e.g., every 8 frames)
    interval = args.num_segments
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Add to buffer
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buffer.append(rgb_frame)
        
        # Keep buffer size manageable (e.g., max num_segments)
        if len(buffer) > interval:
            buffer.pop(0)
            
        # Run inference when buffer is full (every 'interval' frames or sliding window?)
        # Let's do sliding window prediction every 'interval' frames to be efficient
        if len(buffer) == interval and frame_idx % interval == 0:
            # Prepare tensor
            frames_t = []
            for buf_frame in buffer:
                 # Resize to model input size (224x224)
                 resized = cv2.resize(buf_frame, (224, 224))
                 frames_t.append(transform(resized))
            
            frames_tensor = torch.stack(frames_t).unsqueeze(0).to(device) # [1, T, 3, H, W]
            
            with torch.no_grad():
                yolo_enc_batch = []
                for b_i in range(frames_tensor.shape[0]):
                    yolo_enc = yolo_encoder(frames_tensor[b_i])
                    yolo_enc_batch.append(yolo_enc)
                yolo_enc_batch = torch.stack(yolo_enc_batch, dim=0)
                
                logits = model(frames_tensor, yolo_enc_batch)
                probs = torch.softmax(logits, dim=1)
                score, pred_idx = torch.max(probs, dim=1)
                
                pred_idx = pred_idx.item()
                current_prob = score.item()
                current_pred = CLASSES[pred_idx]
                current_color = COLORS[pred_idx]
        
        # Annotate Frame
        # Draw box or text
        # (Top-left corner)
        cv2.rectangle(frame, (0, 0), (280, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"{current_pred} ({current_prob:.2f})", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, current_color, 2)
        
        out.write(frame)
        frame_idx += 1
        
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames", end='\r')
            
    cap.release()
    out.release()
    print(f"\nDone! Saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str)
    parser.add_argument('--output_path', type=str, default='output_video.mp4')
    parser.add_argument('--checkpoint', type=str, default="checkpoints/best_model.pt")
    parser.add_argument('--mmaction_cfg', type=str,
                        default="./mmaction2/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py")
    parser.add_argument('--yolo_weights', type=str, default="yolov8n.pt")
    parser.add_argument('--num_segments', type=int, default=12)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    visualize(args)

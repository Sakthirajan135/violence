#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vi-SAFE Real-Time Inference Script
Run violence detection on a live camera feed (webcam / RTSP / IP camera).

Usage:
    python realtime_inference.py                         # Webcam (default camera 0)
    python realtime_inference.py --source 1              # Second camera
    python realtime_inference.py --source rtsp://...     # RTSP stream
    python realtime_inference.py --source http://...     # IP camera HTTP stream
"""

import os
import sys
import cv2
import time
import torch
import argparse
import numpy as np
import torchvision.transforms as T
from collections import deque

# Import model definitions from training script
sys.path.insert(0, os.path.dirname(__file__))
from run_training import YOLOEncoder, TSNWithYOLO

# Constants
CLASSES = ['NonFight', 'Fight']
COLORS_BGR = {
    'NonFight': (83, 200, 0),    # Green in BGR
    'Fight': (68, 23, 255),      # Red in BGR
    'Analyzing': (255, 255, 255) # White
}
ALERT_OVERLAY_BGR = (0, 0, 200)  # Dark red tint for fight alert


def load_models(checkpoint_path, mmaction_cfg, yolo_weights, device):
    """Load and return the trained models."""
    print(f"[Vi-SAFE] Loading models on {device}...")

    yolo_encoder = YOLOEncoder(
        yolo_weights_path=yolo_weights,
        device=device,
        freeze_yolo=True
    ).to(device)

    model = TSNWithYOLO(
        mmaction_cfg_path=mmaction_cfg,
        yolo_encoding_dim=128,
        device=device
    ).to(device)

    print(f"[Vi-SAFE] Loading checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    yolo_encoder.load_state_dict(checkpoint['yolo_encoder_state_dict'])

    model.eval()
    yolo_encoder.train(False)

    epoch = checkpoint.get('epoch', 'N/A')
    val_acc = checkpoint.get('val_acc', 'N/A')
    print(f"[Vi-SAFE] Model loaded — Epoch: {epoch}, Val Acc: {val_acc}")

    return model, yolo_encoder


def preprocess_frame(frame, size=(224, 224)):
    """Convert a BGR frame to a normalized tensor."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, size)
    transform = T.Compose([T.ToPILImage(), T.ToTensor()])
    return transform(resized)


def run_inference_on_buffer(model, yolo_encoder, frame_buffer, device):
    """
    Run the YOLO+TSN model on a buffer of preprocessed frame tensors.
    Returns: (label, confidence, nonfight_prob, fight_prob)
    """
    frames_tensor = torch.stack(list(frame_buffer)).unsqueeze(0).to(device)  # [1, T, 3, H, W]

    with torch.no_grad():
        yolo_enc = yolo_encoder(frames_tensor[0])          # [T, 128]
        yolo_enc_batch = yolo_enc.unsqueeze(0)             # [1, T, 128]
        logits = model(frames_tensor, yolo_enc_batch)      # [1, 2]
        probs = torch.softmax(logits, dim=1)
        score, pred_idx = torch.max(probs, dim=1)

    pred_idx = pred_idx.item()
    return CLASSES[pred_idx], score.item(), probs[0][0].item(), probs[0][1].item()


def draw_overlay(frame, label, confidence, fps, nonfight_prob, fight_prob,
                 buffer_fill_pct, recording=False):
    """Draw a rich HUD overlay on the frame."""
    h, w = frame.shape[:2]

    # ── Fight alert: red tint on entire frame ──
    if label == 'Fight' and confidence > 0.6:
        overlay = frame.copy()
        alpha = min(0.3, confidence * 0.4)
        cv2.rectangle(overlay, (0, 0), (w, h), ALERT_OVERLAY_BGR, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # ── Top status bar ──
    bar_h = 70
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Title
    cv2.putText(frame, "Vi-SAFE REALTIME", (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1, cv2.LINE_AA)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    # Recording indicator
    if recording:
        cv2.circle(frame, (w - 25, 25), 8, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 65, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # ── Prediction box (bottom-left) ──
    box_w, box_h = 320, 120
    box_x, box_y = 10, h - box_h - 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    color = COLORS_BGR.get(label, COLORS_BGR['Analyzing'])

    # Border accent
    cv2.rectangle(frame, (box_x, box_y), (box_x + 4, box_y + box_h), color, -1)

    # Label
    cv2.putText(frame, label, (box_x + 15, box_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

    # Confidence bar
    conf_bar_y = box_y + 55
    conf_bar_w = int((box_w - 30) * confidence)
    cv2.rectangle(frame, (box_x + 15, conf_bar_y), (box_x + 15 + box_w - 30, conf_bar_y + 12),
                  (60, 60, 60), -1)
    cv2.rectangle(frame, (box_x + 15, conf_bar_y), (box_x + 15 + conf_bar_w, conf_bar_y + 12),
                  color, -1)
    cv2.putText(frame, f"{confidence:.1%}", (box_x + 15 + box_w - 25, conf_bar_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

    # Probabilities
    prob_y = box_y + 85
    cv2.putText(frame, f"NonFight: {nonfight_prob:.1%}", (box_x + 15, prob_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (83, 200, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Fight: {fight_prob:.1%}", (box_x + 170, prob_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (68, 23, 255), 1, cv2.LINE_AA)

    # Buffer fill indicator
    cv2.putText(frame, f"Buffer: {buffer_fill_pct:.0%}", (box_x + 15, prob_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)

    return frame


def realtime_detect(args):
    """Main real-time detection loop."""
    # Determine source
    source = args.source
    if source.isdigit():
        source = int(source)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[Vi-SAFE] Device: {device}")

    # Load models
    model, yolo_encoder = load_models(
        args.checkpoint, args.mmaction_cfg, args.yolo_weights, device
    )

    # Open video source
    print(f"[Vi-SAFE] Opening source: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[Vi-SAFE] ERROR: Cannot open video source: {source}")
        sys.exit(1)

    # Get source properties
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Vi-SAFE] Source: {src_w}x{src_h} @ {src_fps:.1f} FPS")

    # Sliding window buffer for temporal segments
    num_segments = args.num_segments
    frame_buffer = deque(maxlen=num_segments)

    # Inference throttle: run inference every N captured frames
    inference_interval = args.inference_interval

    # State
    current_label = "Analyzing..."
    current_conf = 0.0
    nonfight_prob = 0.5
    fight_prob = 0.5
    frame_count = 0
    fps_timer = time.time()
    fps_display = 0.0
    fps_frame_count = 0

    # Optional recording
    writer = None
    recording = False
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        record_fps = src_fps if src_fps > 0 else 30.0
        writer = cv2.VideoWriter(args.record, fourcc, record_fps, (src_w, src_h))
        recording = True
        print(f"[Vi-SAFE] Recording to: {args.record}")

    # Optional alert logging
    alert_log = []

    print(f"[Vi-SAFE] Starting real-time detection (press 'q' to quit, 's' to screenshot)")
    print(f"[Vi-SAFE] Buffer size: {num_segments} frames, Inference every {inference_interval} frames")
    print("=" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, int):
                    # Camera disconnected
                    print("[Vi-SAFE] Camera disconnected!")
                    break
                else:
                    # Stream ended, try to reconnect
                    print("[Vi-SAFE] Stream ended or lost. Attempting reconnect...")
                    cap.release()
                    time.sleep(2)
                    cap = cv2.VideoCapture(source)
                    continue

            frame_count += 1
            fps_frame_count += 1

            # FPS calculation (update every 0.5 seconds)
            elapsed = time.time() - fps_timer
            if elapsed >= 0.5:
                fps_display = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_timer = time.time()

            # Preprocess and add to buffer
            frame_tensor = preprocess_frame(frame)
            frame_buffer.append(frame_tensor)

            buffer_fill_pct = len(frame_buffer) / num_segments

            # Run inference when buffer is full and at the right interval
            if len(frame_buffer) == num_segments and frame_count % inference_interval == 0:
                current_label, current_conf, nonfight_prob, fight_prob = \
                    run_inference_on_buffer(model, yolo_encoder, frame_buffer, device)

                # Log fight alerts
                if current_label == 'Fight' and current_conf >= args.alert_threshold:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    alert_msg = f"[ALERT] {timestamp} — Fight detected (conf: {current_conf:.2%})"
                    print(f"\033[91m{alert_msg}\033[0m")  # Red console output
                    alert_log.append(alert_msg)

            # Draw overlay
            display_frame = draw_overlay(
                frame.copy(), current_label, current_conf,
                fps_display, nonfight_prob, fight_prob,
                buffer_fill_pct, recording
            )

            # Show
            cv2.imshow("Vi-SAFE Real-Time Violence Detection", display_frame)

            # Record
            if writer is not None:
                writer.write(display_frame)

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                print("[Vi-SAFE] Quitting...")
                break
            elif key == ord('s'):
                # Screenshot
                ss_path = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(ss_path, display_frame)
                print(f"[Vi-SAFE] Screenshot saved: {ss_path}")
            elif key == ord('r'):
                # Toggle recording
                if writer is None:
                    rec_path = f"recording_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(rec_path, fourcc, src_fps if src_fps > 0 else 30.0,
                                             (src_w, src_h))
                    recording = True
                    print(f"[Vi-SAFE] Recording started: {rec_path}")
                else:
                    writer.release()
                    writer = None
                    recording = False
                    print("[Vi-SAFE] Recording stopped.")

    except KeyboardInterrupt:
        print("\n[Vi-SAFE] Interrupted by user.")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

        # Summary
        print("\n" + "=" * 60)
        print("[Vi-SAFE] Session Summary")
        print(f"  Total frames processed: {frame_count}")
        print(f"  Total alerts: {len(alert_log)}")
        if alert_log:
            print("  Alert Log:")
            for a in alert_log:
                print(f"    {a}")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vi-SAFE Real-Time Violence Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python realtime_inference.py                              # Webcam (camera 0)
  python realtime_inference.py --source 1                   # Second camera
  python realtime_inference.py --source rtsp://192.168.1.10:554/stream  # RTSP
  python realtime_inference.py --record output.mp4          # Record output
  python realtime_inference.py --inference_interval 4       # Faster updates
        """
    )
    parser.add_argument('--source', type=str, default='0',
                        help="Video source: camera index (0,1,...) or URL (rtsp://, http://)")
    parser.add_argument('--checkpoint', type=str, default="checkpoints/best_model.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument('--mmaction_cfg', type=str,
                        default="./mmaction2/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py",
                        help="Path to mmaction2 TSN config")
    parser.add_argument('--yolo_weights', type=str, default="yolov8n.pt",
                        help="Path to YOLO weights")
    parser.add_argument('--num_segments', type=int, default=12,
                        help="Number of frames in the temporal buffer (sliding window)")
    parser.add_argument('--inference_interval', type=int, default=6,
                        help="Run inference every N frames (lower = more responsive, higher = faster FPS)")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device: cuda or cpu")
    parser.add_argument('--alert_threshold', type=float, default=0.7,
                        help="Minimum confidence to trigger fight alert (0.0-1.0)")
    parser.add_argument('--record', type=str, default=None,
                        help="Path to save recording (e.g., output.mp4)")

    args = parser.parse_args()
    realtime_detect(args)

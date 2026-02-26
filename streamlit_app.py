#!/usr/bin/env python3
"""
Vi-SAFE Streamlit Application
Violence detection web interface with video upload AND real-time camera support.
"""

import os
import cv2
import time
import torch
import tempfile
import numpy as np
import streamlit as st
import torchvision.transforms as T
from PIL import Image
from collections import deque
from datetime import datetime


def log(msg, level="INFO"):
    """Print a timestamped log message to the terminal."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = {
        "INFO": "\033[94m[Vi-SAFE]\033[0m",
        "SUCCESS": "\033[92m[Vi-SAFE ✓]\033[0m",
        "WARNING": "\033[93m[Vi-SAFE ⚠]\033[0m",
        "ALERT": "\033[91m[Vi-SAFE 🚨]\033[0m",
        "RESULT": "\033[96m[Vi-SAFE →]\033[0m",
    }.get(level, "[Vi-SAFE]")
    print(f"{prefix} [{ts}] {msg}", flush=True)

# Import model definitions
import sys
sys.path.insert(0, os.path.dirname(__file__))
from run_training import YOLOEncoder, TSNWithYOLO

# Constants
CLASSES = ['NonFight', 'Fight']
COLORS = {'NonFight': '#00C853', 'Fight': '#FF1744'}
EMOJI = {'NonFight': '✅', 'Fight': '🚨'}

# ─── Page Config ───
st.set_page_config(
    page_title="Vi-SAFE | Violence Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: #a0aec0;
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }

    .result-card {
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1rem;
    }

    .result-fight {
        background: linear-gradient(135deg, #2d1117 0%, #3d1520 100%);
        border-left: 4px solid #FF1744;
    }

    .result-nonfight {
        background: linear-gradient(135deg, #0d2818 0%, #0d3320 100%);
        border-left: 4px solid #00C853;
    }

    .result-label {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }

    .result-confidence {
        font-size: 1.2rem;
        font-weight: 500;
        opacity: 0.8;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #e2e8f0;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }

    .frame-gallery {
        display: flex;
        gap: 8px;
        overflow-x: auto;
        padding: 1rem 0;
    }

    .sidebar-info {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stProgress > div > div {
        background: linear-gradient(90deg, #00C853, #00E676);
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1e, #1a1a2e);
    }

    .realtime-status {
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .status-live {
        background: linear-gradient(135deg, #1a2e1a 0%, #0d3320 100%);
        border-left: 4px solid #00C853;
    }

    .status-stopped {
        background: linear-gradient(135deg, #2e1a1a 0%, #3d1520 100%);
        border-left: 4px solid #FF1744;
    }

    .alert-card {
        background: linear-gradient(135deg, #3d1520 0%, #2d1117 100%);
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        border-left: 3px solid #FF1744;
        font-size: 0.85rem;
    }

    .mode-tabs {
        background: linear-gradient(135deg, #0f0f1e, #1a1a2e);
        padding: 0.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .live-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #FF1744;
        animation: pulse 1.5s infinite;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models(checkpoint_path, mmaction_cfg, yolo_weights, device_str):
    """Load and cache the trained models."""
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

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

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    yolo_encoder.load_state_dict(checkpoint['yolo_encoder_state_dict'])

    model.eval()
    yolo_encoder.train(False)

    train_info = {
        'epoch': checkpoint.get('epoch', 'N/A'),
        'val_acc': checkpoint.get('val_acc', 'N/A')
    }

    val_acc_display = f"{train_info['val_acc']:.4f}" if isinstance(train_info['val_acc'], float) else str(train_info['val_acc'])
    log(f"Models loaded successfully | Device: {device} | Epoch: {train_info['epoch']} | Val Acc: {val_acc_display}", "SUCCESS")

    return model, yolo_encoder, device, train_info


def process_video(video_path, num_segments=12):
    """Extract frames and video info from a video file."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    if total_frames <= 0:
        cap.release()
        return None, None

    indices = np.linspace(0, total_frames - 1, num_segments).astype(int)

    frames_raw = []
    frames_display = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            frames_display.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
        frames_raw.append(frame)

    cap.release()

    transform = T.Compose([T.ToPILImage(), T.ToTensor()])
    frames_t = [transform(f) for f in frames_raw]
    frames_tensor = torch.stack(frames_t, dim=0).unsqueeze(0)

    video_info = {
        'total_frames': total_frames,
        'fps': fps,
        'width': width,
        'height': height,
        'duration': duration,
        'display_frames': frames_display
    }

    return frames_tensor, video_info


def run_inference(model, yolo_encoder, frames, device):
    """Run violence detection inference on a batch of frames."""
    frames = frames.to(device)

    with torch.no_grad():
        yolo_enc_batch = []
        for b_idx in range(frames.shape[0]):
            yolo_enc = yolo_encoder(frames[b_idx])
            yolo_enc_batch.append(yolo_enc)
        yolo_enc_batch = torch.stack(yolo_enc_batch, dim=0)

        logits = model(frames, yolo_enc_batch)
        probs = torch.softmax(logits, dim=1)

        score, pred_idx = torch.max(probs, dim=1)
        pred_idx = pred_idx.item()
        score = score.item()

    return {
        'label': CLASSES[pred_idx],
        'confidence': score,
        'nonfight_prob': probs[0][0].item(),
        'fight_prob': probs[0][1].item()
    }


def preprocess_frame_for_model(frame_rgb, size=(224, 224)):
    """Convert an RGB frame to a tensor for model input."""
    resized = cv2.resize(frame_rgb, size)
    transform = T.Compose([T.ToPILImage(), T.ToTensor()])
    return transform(resized)


def run_realtime_mode(model, yolo_encoder, device, num_segments, camera_source,
                      inference_interval, alert_threshold):
    """Run real-time camera violence detection within Streamlit."""
    log(f"Real-time mode started | Source: {camera_source} | Buffer: {num_segments} | Interval: {inference_interval} | Threshold: {alert_threshold}", "INFO")

    # Parse camera source
    if camera_source.isdigit():
        source = int(camera_source)
    else:
        source = camera_source

    # Layout
    st.markdown("### 🎥 Live Camera Feed")

    col_video, col_stats = st.columns([2, 1])

    with col_stats:
        st.markdown("### 📊 Live Statistics")
        status_placeholder = st.empty()
        result_placeholder = st.empty()
        prob_placeholder = st.empty()
        metrics_placeholder = st.empty()
        alert_placeholder = st.empty()

    with col_video:
        frame_placeholder = st.empty()

    # Controls
    stop_button = st.button("⏹️ Stop Camera", type="primary")

    # Initialize state
    if 'realtime_alerts' not in st.session_state:
        st.session_state.realtime_alerts = []

    # Open camera
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        log(f"Failed to open camera source: {camera_source}", "WARNING")
        st.error(f"❌ Cannot open camera source: `{camera_source}`")
        st.info("💡 Make sure your camera is connected and not being used by another application.")
        return

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)

    # Show live status
    with col_stats:
        status_placeholder.markdown(f"""
        <div class="realtime-status status-live">
            <div><span class="live-dot"></span> <strong>LIVE</strong></div>
            <p style="margin: 0.3rem 0 0 0; font-size: 0.85rem; color: #a0aec0;">
                Source: {camera_source} &bull; {src_w}×{src_h} @ {src_fps:.0f}fps
            </p>
        </div>
        """, unsafe_allow_html=True)

    frame_buffer = deque(maxlen=num_segments)
    frame_count = 0
    current_label = "Analyzing..."
    current_conf = 0.0
    nonfight_prob = 0.5
    fight_prob = 0.5
    total_fights = 0
    session_start = time.time()

    try:
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Lost camera feed. Retrying...")
                time.sleep(1)
                cap.release()
                cap = cv2.VideoCapture(source)
                continue

            frame_count += 1

            # Convert to RGB for display and processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Preprocess and add to buffer
            frame_tensor = preprocess_frame_for_model(frame_rgb)
            frame_buffer.append(frame_tensor)

            buffer_fill = len(frame_buffer) / num_segments

            # Run inference
            if len(frame_buffer) == num_segments and frame_count % inference_interval == 0:
                frames_tensor = torch.stack(list(frame_buffer)).unsqueeze(0).to(device)

                with torch.no_grad():
                    yolo_enc = yolo_encoder(frames_tensor[0])
                    yolo_enc_batch = yolo_enc.unsqueeze(0)
                    logits = model(frames_tensor, yolo_enc_batch)
                    probs = torch.softmax(logits, dim=1)
                    score, pred_idx = torch.max(probs, dim=1)

                current_label = CLASSES[pred_idx.item()]
                current_conf = score.item()
                nonfight_prob = probs[0][0].item()
                fight_prob = probs[0][1].item()

                # Log every inference to terminal
                log(f"[LIVE] Frame #{frame_count:>6d} | {current_label:<8s} | Conf: {current_conf:.2%} | NonFight: {nonfight_prob:.2%} | Fight: {fight_prob:.2%}", "RESULT")

                # Alert detection
                if current_label == 'Fight' and current_conf >= alert_threshold:
                    total_fights += 1
                    timestamp = time.strftime("%H:%M:%S")
                    alert = f"🚨 {timestamp} — Fight detected ({current_conf:.1%})"
                    log(f"FIGHT ALERT #{total_fights} at {timestamp} — Confidence: {current_conf:.2%}", "ALERT")
                    st.session_state.realtime_alerts.insert(0, alert)
                    # Keep only last 20 alerts
                    st.session_state.realtime_alerts = st.session_state.realtime_alerts[:20]

            # Update display (every 3 frames for performance)
            if frame_count % 3 == 0:
                # Add overlay text to the frame for display
                display_img = Image.fromarray(frame_rgb)
                frame_placeholder.image(display_img, width='stretch')

                # Update result card
                if current_label in COLORS:
                    css_class = 'result-fight' if current_label == 'Fight' else 'result-nonfight'
                    color = COLORS[current_label]
                    emoji = EMOJI[current_label]
                    result_placeholder.markdown(f"""
                    <div class="result-card {css_class}">
                        <div style="font-size: 2.5rem;">{emoji}</div>
                        <div class="result-label" style="color: {color};">{current_label}</div>
                        <div class="result-confidence">Confidence: {current_conf:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    result_placeholder.info("⏳ Filling buffer...")

                # Update probability bars
                prob_placeholder.markdown(f"""
                **Probability Distribution**
                - 🟢 NonFight: `{nonfight_prob:.1%}`
                - 🔴 Fight: `{fight_prob:.1%}`
                """)

                # Update metrics
                elapsed = time.time() - session_start
                metrics_placeholder.markdown(f"""
                <div class="metric-card" style="margin-bottom: 0.5rem;">
                    <div class="metric-value">{frame_count}</div>
                    <div class="metric-label">Frames Processed</div>
                </div>
                <div class="metric-card" style="margin-bottom: 0.5rem;">
                    <div class="metric-value">{elapsed:.0f}s</div>
                    <div class="metric-label">Session Duration</div>
                </div>
                <div class="metric-card" style="margin-bottom: 0.5rem;">
                    <div class="metric-value">{total_fights}</div>
                    <div class="metric-label">Fight Alerts</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{buffer_fill:.0%}</div>
                    <div class="metric-label">Buffer Fill</div>
                </div>
                """, unsafe_allow_html=True)

                # Update alerts
                if st.session_state.realtime_alerts:
                    alerts_html = "".join([
                        f'<div class="alert-card">{a}</div>'
                        for a in st.session_state.realtime_alerts[:5]
                    ])
                    alert_placeholder.markdown(f"""
                    <div style="margin-top: 1rem;">
                        <strong>🔔 Recent Alerts</strong>
                        {alerts_html}
                    </div>
                    """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Camera error: {e}")

    finally:
        cap.release()
        session_duration = time.time() - session_start
        log("="*60, "INFO")
        log("REAL-TIME SESSION ENDED", "INFO")
        log(f"  Duration       : {session_duration:.1f}s", "INFO")
        log(f"  Frames processed: {frame_count}", "INFO")
        log(f"  Total alerts   : {total_fights}", "INFO")
        log("="*60, "INFO")
        with col_stats:
            status_placeholder.markdown("""
            <div class="realtime-status status-stopped">
                <strong>⏹️ STOPPED</strong>
                <p style="margin: 0.3rem 0 0 0; font-size: 0.85rem; color: #a0aec0;">
                    Camera feed ended
                </p>
            </div>
            """, unsafe_allow_html=True)


# ─── Main App ───
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ Vi-SAFE — Violence Detection System</h1>
        <p>Real-time and video-based violence detection using YOLOv8 + TSN deep learning fusion</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        checkpoint_path = st.text_input(
            "Model Checkpoint",
            value="checkpoints/best_model.pt",
            help="Path to the trained model checkpoint"
        )

        mmaction_cfg = st.text_input(
            "MMAction Config",
            value="./mmaction2/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py"
        )

        yolo_weights = st.text_input(
            "YOLO Weights",
            value="yolov8n.pt"
        )

        device_str = st.selectbox(
            "Device",
            ["cuda", "cpu"],
            index=0 if torch.cuda.is_available() else 1
        )

        num_segments = st.slider(
            "Frame Segments",
            min_value=4, max_value=16, value=12,
            help="Number of frames to sample from the video / buffer size for real-time"
        )

        st.markdown("---")

        st.markdown("### 📊 Model Architecture")
        st.markdown("""
        <div class="sidebar-info">
            <p><strong>Backbone:</strong> ResNet-50 (TSN)</p>
            <p><strong>Encoder:</strong> YOLOv8 Nano</p>
            <p><strong>Fusion:</strong> Feature Concatenation</p>
            <p><strong>Classes:</strong> Fight / NonFight</p>
        </div>
        """, unsafe_allow_html=True)

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        st.error(f"❌ Checkpoint not found: `{checkpoint_path}`. Please train the model first.")
        st.info("Run: `python run_training.py` to train the model.")
        return

    # Load models
    with st.spinner("🔄 Loading models... (first time may take a moment)"):
        try:
            model, yolo_encoder, device, train_info = load_models(
                checkpoint_path, mmaction_cfg, yolo_weights, device_str
            )
            val_acc_str = f"{train_info['val_acc']:.4f}" if isinstance(train_info['val_acc'], float) else str(train_info['val_acc'])
            st.sidebar.markdown(f"""
            <div class="sidebar-info">
                <p><strong>✅ Model Loaded</strong></p>
                <p>Trained Epoch: {train_info['epoch']}</p>
                <p>Val Accuracy: {val_acc_str}</p>
                <p>Device: {device}</p>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Failed to load models: {e}")
            return

    # ─── Mode Selection ───
    mode = st.radio(
        "Detection Mode",
        ["📹 Upload Video", "🎥 Real-Time Camera"],
        horizontal=True,
        help="Choose between uploading a video file or using a live camera feed"
    )

    st.markdown("---")

    if mode == "🎥 Real-Time Camera":
        # ─── Real-Time Camera Mode ───
        st.markdown("""
        <div class="sidebar-info" style="margin-bottom: 1.5rem;">
            <p>📷 <strong>Real-Time Mode</strong> — Connect a webcam or IP camera for live violence detection.</p>
            <p style="font-size: 0.85rem; color: #a0aec0;">
                Tip: Use camera index <code>0</code> for default webcam, or paste an RTSP/HTTP URL for IP cameras.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col_src, col_interval, col_thresh = st.columns(3)
        with col_src:
            camera_source = st.text_input(
                "Camera Source",
                value="0",
                help="Camera index (0, 1, ...) or RTSP/HTTP URL"
            )
        with col_interval:
            inference_interval = st.slider(
                "Inference Interval",
                min_value=1, max_value=30, value=6,
                help="Run model every N frames (lower = more responsive, higher = faster)"
            )
        with col_thresh:
            alert_threshold = st.slider(
                "Alert Threshold",
                min_value=0.5, max_value=1.0, value=0.7, step=0.05,
                help="Minimum confidence to trigger a fight alert"
            )

        start_camera = st.button("▶️ Start Camera", type="primary")

        if start_camera:
            run_realtime_mode(
                model, yolo_encoder, device, num_segments,
                camera_source, inference_interval, alert_threshold
            )

    else:
        # ─── Upload Video Mode ───
        col_upload, col_info = st.columns([2, 1])

        with col_upload:
            st.markdown("### 📹 Upload Video")
            uploaded_file = st.file_uploader(
                "Choose a video file",
                type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
                help="Upload a video to analyze for violence detection"
            )

        with col_info:
            st.markdown("### 📋 Supported Formats")
            st.markdown("""
            - **MP4** (.mp4)
            - **AVI** (.avi)
            - **MOV** (.mov)
            - **MKV** (.mkv)
            - **WebM** (.webm)
            """)

        if uploaded_file is not None:
            log(f"Video uploaded: '{uploaded_file.name}' ({uploaded_file.size / 1024:.1f} KB)", "INFO")

            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                st.markdown("---")
                col_video, col_result = st.columns([1, 1])

                with col_video:
                    st.markdown("### 🎬 Input Video")
                    st.video(uploaded_file)

                with st.spinner("🔍 Analyzing video for violence..."):
                    log(f"Processing video: '{uploaded_file.name}' with {num_segments} segments...", "INFO")
                    frames, video_info = process_video(tmp_path, num_segments=num_segments)

                    if frames is None:
                        log(f"Failed to read video: '{uploaded_file.name}'", "WARNING")
                        st.error("❌ Could not read the video. The file may be corrupted.")
                        return

                    log(f"Video info: {video_info['total_frames']} frames | {video_info['fps']:.1f} FPS | {video_info['width']}x{video_info['height']} | {video_info['duration']:.1f}s", "INFO")

                    inference_start = time.time()
                    result = run_inference(model, yolo_encoder, frames, device)
                    inference_time = time.time() - inference_start

                    log("="*60, "RESULT")
                    log(f"VIDEO RESULT: '{uploaded_file.name}'", "RESULT")
                    log(f"  Prediction : {result['label']}", "RESULT")
                    log(f"  Confidence : {result['confidence']:.2%}", "RESULT")
                    log(f"  NonFight   : {result['nonfight_prob']:.2%}", "RESULT")
                    log(f"  Fight      : {result['fight_prob']:.2%}", "RESULT")
                    log(f"  Inference  : {inference_time:.2f}s", "RESULT")
                    log("="*60, "RESULT")

                # ─── Results ───
                with col_result:
                    st.markdown("### 🎯 Prediction Result")

                    label = result['label']
                    confidence = result['confidence']
                    css_class = 'result-fight' if label == 'Fight' else 'result-nonfight'
                    color = COLORS[label]
                    emoji = EMOJI[label]

                    st.markdown(f"""
                    <div class="result-card {css_class}">
                        <div style="font-size: 3rem;">{emoji}</div>
                        <div class="result-label" style="color: {color};">{label}</div>
                        <div class="result-confidence">Confidence: {confidence:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Probability bars
                    st.markdown("**Probability Distribution**")
                    col_nf, col_f = st.columns(2)
                    with col_nf:
                        st.metric("NonFight", f"{result['nonfight_prob']:.1%}")
                        st.progress(result['nonfight_prob'])
                    with col_f:
                        st.metric("Fight", f"{result['fight_prob']:.1%}")
                        st.progress(result['fight_prob'])

                # ─── Video Info ───
                st.markdown("---")
                st.markdown("### 📊 Video Details")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{video_info['total_frames']}</div>
                        <div class="metric-label">Total Frames</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{video_info['fps']:.1f}</div>
                        <div class="metric-label">FPS</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{video_info['width']}×{video_info['height']}</div>
                        <div class="metric-label">Resolution</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{video_info['duration']:.1f}s</div>
                        <div class="metric-label">Duration</div>
                    </div>
                    """, unsafe_allow_html=True)

                # ─── Sampled Frames ───
                if video_info['display_frames']:
                    st.markdown("### 🖼️ Sampled Frames")
                    st.caption(f"{num_segments} frames uniformly sampled for analysis")

                    frame_cols = st.columns(min(len(video_info['display_frames']), 8))
                    for i, frame in enumerate(video_info['display_frames'][:8]):
                        with frame_cols[i]:
                            frame_img = Image.fromarray(frame)
                            st.image(frame_img, caption=f"Frame {i+1}", width='stretch')

            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        else:
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; padding: 3rem 0;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🎬</div>
                <h3 style="color: #e2e8f0;">Upload a video to get started</h3>
                <p style="color: #718096;">Drop a video file above or click "Browse files" to select one</p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

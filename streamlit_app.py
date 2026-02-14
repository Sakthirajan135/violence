#!/usr/bin/env python3
"""
Vi-SAFE Streamlit Application
Violence detection web interface using the trained TSN + YOLOv8 model.
"""

import os
import cv2
import torch
import tempfile
import numpy as np
import streamlit as st
import torchvision.transforms as T
from PIL import Image

# Import model definitions
import sys
sys.path.insert(0, os.path.dirname(__file__))
from run_training import YOLOEncoder, TSNWithYOLO

# Constants
CLASSES = ['NonFight', 'Fight']
COLORS = {'NonFight': '#00C853', 'Fight': '#FF1744'}  # Green, Red
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
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models(checkpoint_path, mmaction_cfg, yolo_weights, device_str):
    """Load and cache the trained models."""
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    # Initialize models
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

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    yolo_encoder.load_state_dict(checkpoint['yolo_encoder_state_dict'])

    model.eval()
    yolo_encoder.train(False)

    train_info = {
        'epoch': checkpoint.get('epoch', 'N/A'),
        'val_acc': checkpoint.get('val_acc', 'N/A')
    }

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

    # Sample frames
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

    # Transform
    transform = T.Compose([T.ToPILImage(), T.ToTensor()])
    frames_t = [transform(f) for f in frames_raw]
    frames_tensor = torch.stack(frames_t, dim=0).unsqueeze(0)  # [1, T, 3, H, W]

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
    """Run violence detection inference."""
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


# ─── Main App ───
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ Vi-SAFE — Violence Detection System</h1>
        <p>Upload a video to detect violence using YOLOv8 + TSN deep learning fusion</p>
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
            help="Number of frames to sample from the video"
        )

        st.markdown("---")

        # Model info
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

    # ─── Upload Section ───
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
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            # Show video
            st.markdown("---")
            col_video, col_result = st.columns([1, 1])

            with col_video:
                st.markdown("### 🎬 Input Video")
                st.video(uploaded_file)

            # Process video
            with st.spinner("🔍 Analyzing video for violence..."):
                frames, video_info = process_video(tmp_path, num_segments=num_segments)

                if frames is None:
                    st.error("❌ Could not read the video. The file may be corrupted.")
                    return

                result = run_inference(model, yolo_encoder, frames, device)

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
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    else:
        # Demo section when no video is uploaded
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

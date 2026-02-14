#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vi-SAFE Training Script V2 — Enhanced for Higher Accuracy
Violence detection using YOLOv8 + TSN with:
  - Attention-based feature fusion
  - Richer YOLO features (bbox + confidence + class)
  - Label smoothing
  - Learning rate warmup + cosine annealing
  - Early stopping
  - Gradient clipping
  - Mixup augmentation
  - Temporal jittering in frame sampling
"""

import os
import sys
import cv2
import glob
import math
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T

# Register mmaction modules
from mmaction.utils import register_all_modules
register_all_modules(init_default_scope=True)

from mmengine import Config
from mmaction.registry import MODELS
from ultralytics import YOLO

import random as py_random


# ============== DATASET WITH TEMPORAL JITTERING ==============
class RWF2000DatasetV2(Dataset):
    """
    RWF-2000 dataset loader with temporal jittering for better generalization.
    
    Key improvement: Instead of fixed frame indices, adds random jitter
    during training to see slightly different frames each epoch.
    """

    def __init__(
            self,
            root_dir,
            mode='train',
            num_segments=16,
            transform=None,
            random_seed=42,
            temporal_jitter=True
    ):
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.num_segments = num_segments
        self.transform = transform
        self.temporal_jitter = temporal_jitter and (mode == 'train')

        self.pos_folder_name = 'Fight'
        self.neg_folder_name = 'NonFight'

        fight_dir = os.path.join(self.root_dir, self.mode, self.pos_folder_name)
        nonfight_dir = os.path.join(self.root_dir, self.mode, self.neg_folder_name)

        fight_videos = glob.glob(os.path.join(fight_dir, '*.mp4')) + \
                       glob.glob(os.path.join(fight_dir, '*.avi'))
        nonfight_videos = glob.glob(os.path.join(nonfight_dir, '*.mp4')) + \
                          glob.glob(os.path.join(nonfight_dir, '*.avi'))

        self.video_list = []
        self.labels = []

        for vf in fight_videos:
            self.video_list.append(vf)
            self.labels.append(1)

        for vnf in nonfight_videos:
            self.video_list.append(vnf)
            self.labels.append(0)

        print(f"[RWF2000DatasetV2] Mode: {self.mode}")
        print(f"  Found {len(fight_videos)} fight videos in: {fight_dir}")
        print(f"  Found {len(nonfight_videos)} nonfight videos in: {nonfight_dir}")
        print(f"  => Total {len(self.video_list)} videos.\n")

        if len(self.video_list) == 0:
            raise RuntimeError(
                f"No videos found for mode={self.mode} under root={self.root_dir}. "
                f"Check your directory structure or path."
            )

        data = list(zip(self.video_list, self.labels))
        py_random.Random(random_seed).shuffle(data)
        self.video_list, self.labels = zip(*data)

    def __len__(self):
        return len(self.video_list)

    def _get_indices(self, total_frames):
        """Get frame indices with optional temporal jittering."""
        if total_frames <= self.num_segments:
            indices = np.linspace(0, total_frames - 1, self.num_segments).astype(int)
        else:
            # Divide video into segments, pick one frame per segment
            segment_length = total_frames / self.num_segments
            indices = []
            for i in range(self.num_segments):
                start = int(i * segment_length)
                end = int((i + 1) * segment_length)
                if self.temporal_jitter:
                    # Random frame within segment
                    idx = random.randint(start, max(start, end - 1))
                else:
                    # Center frame of segment
                    idx = (start + end) // 2
                indices.append(idx)
            indices = np.array(indices)
        return indices

    def __getitem__(self, idx):
        video_path = self.video_list[idx]
        label = self.labels[idx]

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            raise RuntimeError(
                f"Video {video_path} has 0 frames or is unreadable."
            )

        indices = self._get_indices(total_frames)
        frames = []

        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame = cv2.resize(frame, (224, 224))

            if self.transform:
                frame = self.transform(frame)
            else:
                frame = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0

            frames.append(frame)

        cap.release()
        frames_tensor = torch.stack(frames, dim=0)  # [T, 3, 224, 224]
        return frames_tensor, torch.tensor(label, dtype=torch.long)


# ============== ENHANCED YOLO ENCODER ==============
class YOLOEncoderV2(nn.Module):
    """
    Enhanced YOLO-based encoder that extracts richer features:
    - Bounding box coordinates (x, y, w, h) for ALL detected persons
    - Confidence scores
    - Number of detections
    - Statistical features (std of positions for motion estimation)
    
    This gives 12 features per frame instead of just 4.
    """

    def __init__(self, yolo_weights_path='yolov8n.pt', device='cuda', freeze_yolo=True):
        super().__init__()
        self.device = device

        self._yolo_wrapper = [YOLO(yolo_weights_path, task='detect')]
        self.yolo_model.to(self.device)

        if freeze_yolo:
            for param in self.yolo_model.model.parameters():
                param.requires_grad = False
            self.yolo_model.model.eval()

        # Enhanced MLP for richer 12-dim input features
        self.mlp = nn.Sequential(
            nn.Linear(12, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )

    @property
    def yolo_model(self):
        return self._yolo_wrapper[0]

    def train(self, mode=True):
        """Custom train method to handle both MLP and the wrapped YOLO model."""
        super().train(mode)
        if hasattr(self, '_yolo_wrapper'):
            self.yolo_model.model.eval()

    def _extract_features(self, result):
        """Extract rich features from a single YOLO result."""
        bboxes = result.boxes
        
        if bboxes is not None and len(bboxes) > 0:
            # Filter for person class (class 0 in COCO)
            classes = bboxes.cls
            person_mask = (classes == 0)
            
            if person_mask.sum() > 0:
                person_boxes = bboxes[person_mask]
                xyxy = person_boxes.xyxy
                confs = person_boxes.conf
                
                xc = (xyxy[:, 0] + xyxy[:, 2]) / 2.0 / 224.0  # Normalized
                yc = (xyxy[:, 1] + xyxy[:, 3]) / 2.0 / 224.0
                w = (xyxy[:, 2] - xyxy[:, 0]) / 224.0
                h = (xyxy[:, 3] - xyxy[:, 1]) / 224.0
                
                features = torch.tensor([
                    xc.mean().item(),           # Mean x center
                    yc.mean().item(),           # Mean y center
                    w.mean().item(),            # Mean width
                    h.mean().item(),            # Mean height
                    confs.mean().item(),        # Mean confidence
                    confs.max().item(),         # Max confidence
                    float(len(person_boxes)),   # Number of persons detected
                    xc.std().item() if len(xc) > 1 else 0.0,  # Spread of persons (x)
                    yc.std().item() if len(yc) > 1 else 0.0,  # Spread of persons (y)
                    w.std().item() if len(w) > 1 else 0.0,     # Variation in widths
                    h.std().item() if len(h) > 1 else 0.0,     # Variation in heights
                    (w * h).sum().item(),       # Total area covered by persons
                ], device=self.device)
            else:
                features = torch.zeros(12, device=self.device)
        else:
            features = torch.zeros(12, device=self.device)
        
        return features

    def forward(self, frames):
        if len(frames.shape) != 4:
            raise ValueError(f"Expect frames shape [T,3,H,W], got {frames.shape}")

        frames_batch = frames.to(self.device, non_blocking=True)

        with torch.no_grad():
            results = self.yolo_model.predict(frames_batch, verbose=False)

        yolo_encodings = []
        for r in results:
            feat = self._extract_features(r)
            yolo_encodings.append(feat)

        yolo_encodings = torch.stack(yolo_encodings, dim=0)
        yolo_encodings = self.mlp(yolo_encodings)
        return yolo_encodings


# ============== ATTENTION FUSION MODULE ==============
class CrossAttentionFusion(nn.Module):
    """
    Cross-attention mechanism to fuse TSN and YOLO features.
    Instead of simple concatenation, this learns which YOLO features
    are most relevant for the current TSN features and vice versa.
    """
    
    def __init__(self, tsn_dim, yolo_dim, num_heads=4, dropout=0.3):
        super().__init__()
        self.tsn_dim = tsn_dim
        self.yolo_dim = yolo_dim
        
        # Project both to same dimension
        self.hidden_dim = 256
        self.tsn_proj = nn.Linear(tsn_dim, self.hidden_dim)
        self.yolo_proj = nn.Linear(yolo_dim, self.hidden_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norm
        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, tsn_feat, yolo_feat):
        """
        Args:
            tsn_feat: [B, tsn_dim]
            yolo_feat: [B, yolo_dim]
        Returns:
            fused: [B, hidden_dim]
        """
        # Project
        tsn_proj = self.tsn_proj(tsn_feat).unsqueeze(1)   # [B, 1, hidden]
        yolo_proj = self.yolo_proj(yolo_feat).unsqueeze(1)  # [B, 1, hidden]
        
        # Concatenate as sequence
        combined = torch.cat([tsn_proj, yolo_proj], dim=1)  # [B, 2, hidden]
        
        # Self-attention over the two features
        attn_out, _ = self.attention(combined, combined, combined)
        combined = self.ln1(combined + attn_out)
        
        # FFN
        ffn_out = self.ffn(combined)
        combined = self.ln2(combined + ffn_out)
        
        # Pool (mean of the two attended features)
        fused = combined.mean(dim=1)  # [B, hidden]
        return fused


# ============== TSN WITH YOLO V2 ==============
class TSNWithYOLOV2(nn.Module):
    """
    Enhanced TSN model with attention-based YOLO feature fusion.
    
    Key improvements:
    - Cross-attention fusion instead of simple concatenation
    - Deeper classification head
    - Residual connection
    """

    def __init__(
            self,
            mmaction_cfg_path,
            yolo_encoding_dim=128,
            device='cuda',
            freeze_tsn_backbone=False
    ):
        super().__init__()
        self.device = device

        config = Config.fromfile(mmaction_cfg_path)
        if 'train_cfg' not in config.model:
            config.model['train_cfg'] = config.get('train_cfg', None)
        if 'test_cfg' not in config.model:
            config.model['test_cfg'] = config.get('test_cfg', None)

        self.tsn = MODELS.build(config.model)
        self.tsn.to(self.device)

        if freeze_tsn_backbone:
            for param in self.tsn.backbone.parameters():
                param.requires_grad = False

        original_fc = self.tsn.cls_head.fc_cls
        in_features = original_fc.in_features  # 2048
        
        # Attention-based fusion
        self.fusion = CrossAttentionFusion(
            tsn_dim=in_features,
            yolo_dim=yolo_encoding_dim,
            num_heads=4,
            dropout=0.3
        )
        
        # Deeper classification head
        hidden_dim = self.fusion.hidden_dim  # 256
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)  # Binary: Fight / NonFight
        )

    def forward(self, frames, yolo_encodings):
        B, T, C, H, W = frames.shape
        frames_reshape = frames.view(-1, C, H, W)
        feat = self.tsn.backbone(frames_reshape)
        feat = self.tsn.cls_head.avg_pool(feat)
        feat = feat.view(feat.size(0), -1)

        feat = feat.view(B, T, -1)
        tsn_feat = feat.mean(dim=1)  # [B, 2048]

        yolo_feat = yolo_encodings.mean(dim=1)  # [B, 128]
        
        # Attention fusion
        fused_feat = self.fusion(tsn_feat, yolo_feat)  # [B, 256]
        
        # Classification
        logits = self.classifier(fused_feat)
        return logits


# ============== MIXUP AUGMENTATION ==============
def mixup_data(x, y, alpha=0.2):
    """
    Mixup: creates virtual training examples by linearly interpolating
    between pairs. This significantly improves generalization on small datasets.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss: weighted combination of losses for both mixed labels."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============== LEARNING RATE WARMUP ==============
class WarmupCosineScheduler:
    """
    Linear warmup for first N steps, then cosine annealing.
    This stabilizes early training when learning from pretrained weights.
    """
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            alpha = epoch / self.warmup_epochs
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = base_lr * alpha
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    
    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


# ============== TRAINING ==============
def train(args):
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"\n{'='*60}")
    print(f"Vi-SAFE V2 — Enhanced Training Configuration")
    print(f"{'='*60}")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Batch Size:       {args.batch_size}")
    print(f"  Learning Rate:    {args.lr}")
    print(f"  Weight Decay:     {args.weight_decay}")
    print(f"  Num Segments:     {args.num_segments}")
    print(f"  Label Smoothing:  {args.label_smoothing}")
    print(f"  Mixup Alpha:      {args.mixup_alpha}")
    print(f"  Warmup Epochs:    {args.warmup_epochs}")
    print(f"  Early Stop After: {args.patience} epochs")
    print(f"  Grad Clip Norm:   {args.grad_clip}")
    print(f"{'='*60}\n")

    # Data transforms (enhanced augmentation)
    train_transform = T.Compose([
        T.ToPILImage(),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.RandomResizedCrop(224, scale=(0.75, 1.0)),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
        T.RandomGrayscale(p=0.15),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])
    val_transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor()
    ])

    # Datasets with temporal jittering
    train_dataset = RWF2000DatasetV2(
        root_dir=args.data_root,
        mode='train',
        num_segments=args.num_segments,
        transform=train_transform,
        temporal_jitter=True
    )
    val_dataset = RWF2000DatasetV2(
        root_dir=args.data_root,
        mode='val',
        num_segments=args.num_segments,
        transform=val_transform,
        temporal_jitter=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Initialize models
    print("Loading Enhanced YOLO encoder (V2)...")
    yolo_encoder = YOLOEncoderV2(
        yolo_weights_path=args.yolo_weights,
        device=device,
        freeze_yolo=True  # Always freeze YOLO — it's already great at detection
    ).to(device)

    print("Loading Enhanced TSN model (V2 with attention fusion)...")
    model = TSNWithYOLOV2(
        mmaction_cfg_path=args.mmaction_cfg,
        yolo_encoding_dim=128,
        device=device,
        freeze_tsn_backbone=args.freeze_tsn_backbone
    ).to(device)

    # ---- Optimizer with differential learning rates ----
    # Lower LR for pretrained backbone, higher LR for new layers
    backbone_params = list(model.tsn.backbone.parameters())
    new_params = (
        list(model.fusion.parameters()) +
        list(model.classifier.parameters()) +
        list(yolo_encoder.mlp.parameters())
    )
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},  # 10x lower for pretrained
        {'params': new_params, 'lr': args.lr},
    ], weight_decay=args.weight_decay)
    
    # Warmup + cosine annealing scheduler
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        min_lr=1e-6
    )
    
    # Label smoothing cross entropy
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_val_acc = 0.0
    epochs_without_improvement = 0

    print("\n" + "=" * 60)
    print("Starting Enhanced Training V2...")
    print("=" * 60 + "\n")

    for epoch in range(args.epochs):
        scheduler.step(epoch)
        current_lr = scheduler.get_last_lr()
        
        # Training
        model.train()
        yolo_encoder.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
        for frames, labels in pbar_train:
            frames = frames.to(device)
            labels = labels.to(device)

            # --- Mixup augmentation ---
            use_mixup = args.mixup_alpha > 0 and random.random() < 0.5  # 50% chance
            if use_mixup:
                frames, labels_a, labels_b, lam = mixup_data(frames, labels, args.mixup_alpha)
            
            # YOLO encoding
            yolo_enc_batch = []
            for b_idx in range(frames.shape[0]):
                yolo_enc = yolo_encoder(frames[b_idx])
                yolo_enc_batch.append(yolo_enc)
            yolo_enc_batch = torch.stack(yolo_enc_batch, dim=0)

            logits = model(frames, yolo_enc_batch)
            
            if use_mixup:
                loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            else:
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(yolo_encoder.parameters()),
                    args.grad_clip
                )
            
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            _, preds = torch.max(logits, dim=1)
            if not use_mixup:
                train_correct += (preds == labels).sum().item()
            else:
                train_correct += (lam * (preds == labels_a).float() + 
                                  (1 - lam) * (preds == labels_b).float()).sum().item()
            train_total += labels.size(0)

            pbar_train.set_postfix({
                "loss": f"{train_loss / train_total:.4f}",
                "acc": f"{train_correct / train_total:.4f}"
            })

        train_acc = train_correct / train_total
        avg_train_loss = train_loss / train_total

        # Validation
        model.eval()
        yolo_encoder.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]")
            for frames, labels in pbar_val:
                frames = frames.to(device)
                labels = labels.to(device)

                yolo_enc_batch = []
                for b_idx in range(frames.shape[0]):
                    yolo_enc = yolo_encoder(frames[b_idx])
                    yolo_enc_batch.append(yolo_enc)
                yolo_enc_batch = torch.stack(yolo_enc_batch, dim=0)

                logits = model(frames, yolo_enc_batch)
                loss = criterion(logits, labels)

                val_loss += loss.item() * labels.size(0)
                _, preds = torch.max(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                pbar_val.set_postfix({
                    "loss": f"{val_loss / val_total:.4f}",
                    "acc": f"{val_correct / val_total:.4f}"
                })

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / val_total

        print(f"\n[Epoch {epoch + 1}/{args.epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} || "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"LR: {current_lr[0]:.2e} / {current_lr[1]:.2e}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            save_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "yolo_encoder_state_dict": yolo_encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "version": "v2"
            }, save_path)
            print(f"  >>> NEW BEST model saved at epoch {epoch + 1}, val_acc={val_acc:.4f} <<<")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement}/{args.patience} epochs "
                  f"(best: {best_val_acc:.4f})")
        
        # Early stopping
        if epochs_without_improvement >= args.patience:
            print(f"\n{'='*60}")
            print(f"EARLY STOPPING at epoch {epoch + 1}!")
            print(f"No improvement for {args.patience} consecutive epochs.")
            print(f"{'='*60}")
            break

    print(f"\n{'='*60}")
    print(f"Training finished!")
    print(f"  Best Val Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")
    print(f"  Checkpoint saved:  {os.path.join(args.output_dir, 'best_model.pt')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vi-SAFE V2 Enhanced Training")

    # Data paths
    parser.add_argument('--data_root', type=str, default="./data",
                        help="Root directory of the dataset")
    parser.add_argument('--mmaction_cfg', type=str,
                        default="./mmaction2/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py",
                        help="Path to mmaction2 TSN config file")
    parser.add_argument('--yolo_weights', type=str, default="yolov8n.pt",
                        help="YOLOv8 model weights path")
    parser.add_argument('--output_dir', type=str, default="./checkpoints",
                        help="Output directory for model checkpoints")

    # Training hyperparameters  
    parser.add_argument('--epochs', type=int, default=25, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size")
    parser.add_argument('--num_segments', type=int, default=16, help="Number of frame segments (increased from 12)")
    parser.add_argument('--lr', type=float, default=5e-4, help="Learning rate (lower for stability)")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay (increased for regularization)")
    
    # V2 enhancements
    parser.add_argument('--label_smoothing', type=float, default=0.1, help="Label smoothing factor")
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help="Mixup alpha (0 to disable)")
    parser.add_argument('--warmup_epochs', type=int, default=3, help="Number of warmup epochs")
    parser.add_argument('--patience', type=int, default=8, help="Early stopping patience")
    parser.add_argument('--grad_clip', type=float, default=1.0, help="Gradient clipping max norm")

    # Training strategy
    parser.add_argument('--freeze_tsn_backbone', action='store_true',
                        help="Freeze TSN backbone")

    # Other
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use")
    parser.add_argument('--num_workers', type=int, default=0, help="DataLoader workers")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train(args)

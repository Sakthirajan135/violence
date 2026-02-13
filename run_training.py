#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vi-SAFE Training Script
Violence detection using YOLOv8 + TSN

This is a standalone script to run training on your local machine.
"""

import os
import sys
import cv2
import glob
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T

# Register mmaction modules
from mmaction.utils import register_all_modules
register_all_modules(init_default_scope=True)

from mmengine import Config
from mmaction.registry import MODELS
from ultralytics import YOLO

# Use alias to avoid conflict with built-in random()
import random as py_random


# ============== DATASET ==============
class RWF2000Dataset(Dataset):
    """
    RWF-2000 dataset loader for violence detection.
    
    Expected structure:
        root_dir/
          train/
            Fight/
              *.avi
            NonFight/
              *.avi
          val/
            Fight/
            NonFight/
    """

    def __init__(
            self,
            root_dir,
            mode='train',
            num_segments=8,
            transform=None,
            random_seed=42
    ):
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.num_segments = num_segments
        self.transform = transform

        self.pos_folder_name = 'Fight'
        self.neg_folder_name = 'NonFight'

        fight_dir = os.path.join(self.root_dir, self.mode, self.pos_folder_name)
        nonfight_dir = os.path.join(self.root_dir, self.mode, self.neg_folder_name)

        # Search for both .mp4 and .avi files
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

        print(f"[RWF2000Dataset] Mode: {self.mode}")
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
        if total_frames < self.num_segments:
            indices = np.linspace(0, total_frames - 1, self.num_segments).astype(int)
        else:
            indices = np.linspace(0, total_frames - 1, self.num_segments).astype(int)
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


# ============== YOLO ENCODER ==============
class YOLOEncoder(nn.Module):
    """YOLO-based encoder for bounding box features."""

    def __init__(self, yolo_weights_path='yolov8n.pt', device='cuda', freeze_yolo=True):
        super().__init__()
        self.device = device

        # Load YOLO with detection task
        # IMPORTANT: Store in a list to prevent nn.Module from registering it as a submodule
        # This prevents nn.Module.train(bool) from calling YOLO.train(trainer), which causes a crash.
        self._yolo_wrapper = [YOLO(yolo_weights_path, task='detect')]
        self.yolo_model.to(self.device)

        if freeze_yolo:
            for param in self.yolo_model.model.parameters():
                param.requires_grad = False
            self.yolo_model.model.eval()

        self.mlp = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    @property
    def yolo_model(self):
        return self._yolo_wrapper[0]

    def train(self, mode=True):
        """Custom train method to handle both MLP and the wrapped YOLO model."""
        # 1. Handle normal submodules (like self.mlp)
        super().train(mode)
        
        # 2. Handle the wrapped YOLO model manually
        # IMPORTANT: We MUST keep YOLO in eval mode because:
        # a) We use .predict() which expects eval mode for consistent output
        # b) We use torch.no_grad() so we can't train YOLO weights anyway
        if hasattr(self, '_yolo_wrapper'):
            self.yolo_model.model.eval()

    def forward(self, frames):
        if len(frames.shape) != 4:
            raise ValueError(f"Expect frames shape [T,3,H,W], got {frames.shape}")
        
        frames_batch = frames.to(self.device, non_blocking=True)

        with torch.no_grad():
            # Use the properly wrapped model
            results = self.yolo_model.predict(frames_batch, verbose=False)

        yolo_encodings = []
        for r in results:
            bboxes = r.boxes
            if bboxes is not None and len(bboxes) > 0:
                xyxy = bboxes.xyxy
                xc = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
                yc = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
                w = (xyxy[:, 2] - xyxy[:, 0])
                h = (xyxy[:, 3] - xyxy[:, 1])

                encoding_vec = torch.tensor(
                    [xc.mean().item(), yc.mean().item(), w.mean().item(), h.mean().item()],
                    device=self.device
                )
            else:
                encoding_vec = torch.zeros(4, device=self.device)

            yolo_encodings.append(encoding_vec)

        yolo_encodings = torch.stack(yolo_encodings, dim=0)
        yolo_encodings = self.mlp(yolo_encodings)
        return yolo_encodings


# ============== TSN WITH YOLO ==============
class TSNWithYOLO(nn.Module):
    """TSN model with YOLO feature fusion."""

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
        in_features = original_fc.in_features
        out_features = original_fc.out_features

        self.fusion_fc = nn.Linear(in_features + yolo_encoding_dim, out_features)

        with torch.no_grad():
            self.fusion_fc.weight[:, :in_features] = original_fc.weight
            self.fusion_fc.bias[:] = original_fc.bias

    def forward(self, frames, yolo_encodings):
        B, T, C, H, W = frames.shape
        frames_reshape = frames.view(-1, C, H, W)
        feat = self.tsn.backbone(frames_reshape)
        feat = self.tsn.cls_head.avg_pool(feat)
        feat = feat.view(feat.size(0), -1)

        feat = feat.view(B, T, -1)
        tsn_feat = feat.mean(dim=1)

        yolo_feat = yolo_encodings.mean(dim=1)
        fused_feat = torch.cat([tsn_feat, yolo_feat], dim=1)
        logits = self.fusion_fc(fused_feat)
        return logits


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

    # Data transforms
    train_transform = T.Compose([
        T.ToPILImage(),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        T.ToTensor()
    ])
    val_transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor()
    ])

    # Datasets
    train_dataset = RWF2000Dataset(
        root_dir=args.data_root,
        mode='train',
        num_segments=args.num_segments,
        transform=train_transform
    )
    val_dataset = RWF2000Dataset(
        root_dir=args.data_root,
        mode='val',
        num_segments=args.num_segments,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Initialize models
    print("Loading YOLO encoder...")
    yolo_encoder = YOLOEncoder(
        yolo_weights_path=args.yolo_weights,
        device=device,
        freeze_yolo=args.freeze_yolo
    ).to(device)

    print("Loading TSN model...")
    model = TSNWithYOLO(
        mmaction_cfg_path=args.mmaction_cfg,
        yolo_encoding_dim=128,
        device=device,
        freeze_tsn_backbone=args.freeze_tsn_backbone
    ).to(device)

    # Optimizer
    if args.train_all_params:
        params = list(model.parameters()) + list(yolo_encoder.parameters())
    else:
        params = list(model.fusion_fc.parameters()) + list(yolo_encoder.mlp.parameters())

    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")

    for epoch in range(args.epochs):
        # Training
        model.train()
        yolo_encoder.train(not args.freeze_yolo)
        train_loss, train_correct, train_total = 0.0, 0, 0

        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
        for frames, labels in pbar_train:
            frames = frames.to(device)
            labels = labels.to(device)

            yolo_enc_batch = []
            for b_idx in range(frames.shape[0]):
                yolo_enc = yolo_encoder(frames[b_idx])
                yolo_enc_batch.append(yolo_enc)
            yolo_enc_batch = torch.stack(yolo_enc_batch, dim=0)

            logits = model(frames, yolo_enc_batch)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            _, preds = torch.max(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            pbar_train.set_postfix({
                "loss": f"{train_loss / train_total:.4f}",
                "acc": f"{train_correct / train_total:.4f}"
            })

        train_acc = train_correct / train_total
        avg_train_loss = train_loss / train_total
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

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
                    "loss": f"{val_loss / val_total:.4f}"
                })

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / val_total

        print(f"[Epoch {epoch + 1}/{args.epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} || "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "yolo_encoder_state_dict": yolo_encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc
            }, save_path)
            print(f"  >>> Best model saved at epoch {epoch + 1}, val_acc={val_acc:.4f}")

    print(f"\n{'='*50}")
    print(f"Training finished. Best Val Acc = {best_val_acc:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vi-SAFE Training")
    
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
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size")
    parser.add_argument('--num_segments', type=int, default=8, help="Number of frame segments")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay")

    # Training strategy
    parser.add_argument('--train_all_params', action='store_true', default=True,
                        help="Train all parameters")
    parser.add_argument('--freeze_yolo', action='store_true',
                        help="Freeze YOLO encoder")
    parser.add_argument('--freeze_tsn_backbone', action='store_true',
                        help="Freeze TSN backbone")

    # Other
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use")
    parser.add_argument('--num_workers', type=int, default=0, help="DataLoader workers")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train(args)

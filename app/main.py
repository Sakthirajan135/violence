# /home/shengguang/PycharmProjects/yolo_minimum/app/main.py

import os
import cv2
import glob
import wandb
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from ultralytics import YOLO  # ultralytics==8.0.105
import torchvision.transforms as T

from mmaction.utils import register_all_modules
register_all_modules(init_default_scope=True)

from app.dataset_rwf2000 import RWF2000Dataset
from app.encoder_yolo import YOLOEncoder
from app.yolo_tsn import TSNWithYOLO


class EarlyStopping:
    """
    A simple early stopping utility class.
    If validation metric ('metric') does not improve by at least 'delta'
    for 'patience' consecutive epochs, training is halted early.
    """

    def __init__(self, patience=5, delta=0.0, mode='max'):
        """
        :param patience: How many epochs to wait after last time the monitored metric improved.
        :param delta:    Minimum change in the monitored metric to qualify as an improvement.
        :param mode:     'max' or 'min' to decide improvement direction.
                         'max' is typical for accuracy; 'min' for loss.
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode

        if mode == 'max':
            self.best_metric = -np.inf
        else:  # 'min'
            self.best_metric = np.inf

        self.counter = 0
        self.should_stop = False

    def step(self, metric):
        """
        :param metric: The current epoch's validation metric (accuracy or loss)
        :return: True if should early stop, else False.
        """
        if self.mode == 'max':
            # We want metric to increase
            improvement = metric - self.best_metric
            if improvement > self.delta:
                self.best_metric = metric
                self.counter = 0
            else:
                self.counter += 1
        else:
            # We want metric to decrease
            improvement = self.best_metric - metric
            if improvement > self.delta:
                self.best_metric = metric
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

        return self.should_stop


def train(args):
    # 设置随机种子 (保证可复现)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 初始化 wandb
    wandb.init(project="RWF-2000-TSN-with-YOLO",
               name=args.wandb_name,
               config=vars(args))

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 数据增广 (可根据需要随意增加/替换)
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

    # 数据集与 DataLoader
    train_dataset = RWF2000Dataset(
        root_dir=args.rwf2000_root,
        mode='train',
        num_segments=args.num_segments,
        transform=train_transform
    )
    val_dataset = RWF2000Dataset(
        root_dir=args.rwf2000_root,
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

    # 初始化 YOLOEncoder + TSNWithYOLO
    yolo_encoder = YOLOEncoder(
        yolo_weights_path=args.yolo_weights,
        device=device,
        freeze_yolo=args.freeze_yolo
    ).to(device)

    model = TSNWithYOLO(
        mmaction_cfg_path=args.mmaction_cfg,
        yolo_encoding_dim=128,
        device=device,
        freeze_tsn_backbone=args.freeze_tsn_backbone
    ).to(device)

    # 优化器 (可选：只训练融合层 or 训练全部参数)
    if args.train_all_params:
        params = list(model.parameters()) + list(yolo_encoder.parameters())
    else:
        # 只训练 fusion_fc + YOLO MLP
        params = list(model.fusion_fc.parameters()) + list(yolo_encoder.mlp.parameters())

    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    # T_max = args.epochs, 使余弦退火的一个周期等于总epoch数量
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    if args.early_stop_patience > 0:
        early_stopper = EarlyStopping(patience=args.early_stop_patience,
                                      delta=0.0,
                                      mode='max')
    else:
        early_stopper = None

    for epoch in range(args.epochs):
        # ------------------- 训练 -------------------
        model.train()
        yolo_encoder.train(not args.freeze_yolo)  # 如果freeze_yolo=True则保持eval
        train_loss, train_correct, train_total = 0.0, 0, 0

        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
        for frames, labels in pbar_train:
            frames = frames.to(device)  # [B, T, 3, H, W]
            labels = labels.to(device)  # [B]

            # 1) YOLO 编码
            yolo_enc_batch = []
            for b_idx in range(frames.shape[0]):
                yolo_enc = yolo_encoder(frames[b_idx])  # [T, 128]
                yolo_enc_batch.append(yolo_enc)
            yolo_enc_batch = torch.stack(yolo_enc_batch, dim=0)  # [B, T, 128]

            # 2) 模型前向
            logits = model(frames, yolo_enc_batch)

            # 3) 损失
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item() * labels.size(0)
            _, preds = torch.max(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            pbar_train.set_postfix({
                "loss": f"{train_loss / train_total:.4f}",
                "acc": f"{train_correct / train_total:.4f}"
            })

            # wandb log
            wandb.log({
                "train/loss_step": loss.item()
            })

        train_acc = train_correct / train_total
        avg_train_loss = train_loss / train_total

        # 学习率调度器步进 (余弦退火)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ------------------- 验证 -------------------
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

        # wandb log
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": avg_train_loss,
            "train/acc": train_acc,
            "val/loss": avg_val_loss,
            "val/acc": val_acc,
            "lr": current_lr
        })

        print(f"[Epoch {epoch + 1}/{args.epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} || "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")

        # ------------------- 保存最佳模型 -------------------
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

        # -------------- Early Stopping Check --------------
        if early_stopper is not None:
            if early_stopper.step(val_acc):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    print(f"Training finished. Best Val Acc = {best_val_acc:.4f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据 & 路径
    parser.add_argument('--rwf2000_root', type=str,
                        help="RWF-2000 数据集路径")
    parser.add_argument('--mmaction_cfg', type=str,
                        help="mmaction2 TSN 配置文件路径")
    parser.add_argument('--yolo_weights', type=str,
                        help="YOLOv8 模型权重文件路径")
    parser.add_argument('--output_dir', type=str, default="./checkpoints",
                        help="模型保存路径")

    # 训练超参数
    parser.add_argument('--epochs', type=int, default=100, help="训练轮数")
    parser.add_argument('--batch_size', type=int, default=8, help="batch size")
    parser.add_argument('--num_segments', type=int, default=8, help="TSN 帧段数")
    parser.add_argument('--lr', type=float, default=1e-3, help="初始学习率")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="权重衰减系数")

    # 训练策略
    parser.add_argument('--train_all_params', default=True, action='store_true',
                        help="是否训练所有参数（YOLO+TSN），默认只训练YOLO的MLP和TSN的fusion_fc")
    parser.add_argument('--freeze_yolo', action='store_true',
                        help="是否冻结 YOLO 编码器（不训练YOLO）")
    parser.add_argument('--freeze_tsn_backbone', action='store_true',
                        help="是否冻结 TSN backbone，仅训练 fusion_fc")

    # 其他
    parser.add_argument('--seed', type=int, default=42, help="随机种子")
    parser.add_argument('--device', type=str, default='cuda', help="训练使用的设备")
    parser.add_argument('--num_workers', type=int, default=8, help="DataLoader 的 num_workers")
    parser.add_argument('--wandb_name', type=str, default='TSN-YOLO-RWF2000',
                        help="wandb run name")

    parser.add_argument('--early_stop_patience', type=int, default=20,
                        help="Patience for early stopping. If set to 0, early stopping is disabled.")

    args = parser.parse_args()

    # 若输出目录不存在，创建
    os.makedirs(args.output_dir, exist_ok=True)

    train(args)

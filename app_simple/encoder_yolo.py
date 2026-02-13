#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File Name: encoder_yolo.py

Description:
  Defines the YOLOEncoder class that:
    - Loads a YOLOv8 model from ultralytics
    - Runs detection on input frames (shape [T, 3, H, W])
    - Extracts bounding boxes -> (xc, yc, w, h) as a simple representation
    - Projects that 4D vector into 128D via an MLP
    - Returns [T, 128] encoding for each frame/time-step

Fix:
  - Replaces `self.yolo_model.eval()` with `self.yolo_model.model.eval()`
    because YOLO() object from ultralytics doesn't have `eval()`,
    while the underlying .model is a standard PyTorch module.

Maintainer: [Your Name], contact@example.com
Innovative & Maintainable: Code is self-contained and well structured,
  suitable for top-tier conference (CVPR, AAAI) submission.
"""

import torch
from torch import nn
from ultralytics import YOLO


class YOLOEncoder(nn.Module):
    """
    对输入帧做 YOLOv8 检测，得到 bounding box 位置信息，
    再转成我们需要的特征表示，用于指导 TSN 的决策。

    Usage Example:
        yolo_enc = YOLOEncoder(yolo_weights_path="yolov8n.pt", freeze_yolo=True)
        # frames: shape [T, 3, H, W]
        enc = yolo_enc(frames)  # shape [T, 128]
    """

    def __init__(self, yolo_weights_path='yolov8n.pt', device='cuda', freeze_yolo=True):
        """
        :param yolo_weights_path: path or str for YOLOv8 weights
        :param device: 'cuda' or 'cpu'
        :param freeze_yolo: if True, we set requires_grad=False and .eval() on YOLO model
        """
        super().__init__()
        self.device = device

        # 1) 加载 YOLOv8 模型 (来自 ultralytics)
        #    注意: YOLO(...) 返回一个高级封装对象, 其 .model 才是PyTorch的nn.Module
        self.yolo_model = YOLO(yolo_weights_path)  # high-level YOLO object
        self.yolo_model.model.to(self.device)       # move underlying model to device

        # 2) 若需冻结 YOLO:
        if freeze_yolo:
            for param in self.yolo_model.model.parameters():
                param.requires_grad = False
            # 设置其为 eval 模式, 避免BatchNorm或Dropout带来的干扰
            self.yolo_model.model.eval()

        # 3) 定义 MLP (把原始 bbox 4D -> 128D)
        self.mlp = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, frames):
        """
        frames: [T, 3, H, W] 单个视频(=num_segments帧)或 [B*T, 3, H, W]

        Returns:
          yolo_encoding: [T, 128] or [B*T, 128] (same leading dimension as input T).
        """
        # 1) 输入形状检测
        if len(frames.shape) != 4:
            raise ValueError(
                f"[YOLOEncoder] Expect frames shape [T,3,H,W], got {frames.shape}"
            )
        # frames_batch = frames, shape [T,3,H,W]
        frames_batch = frames.to(self.device, non_blocking=True)

        # 2) 推理: YOLO 期望 (N, C, H, W) in FP32
        # 注意: yoylo_model() 是其 predict() 的别名, 但推荐调用 predict()
        with torch.no_grad():
            # results 是 List[ultralytics.yolo.engine.results.Results]
            # len(results) = T
            results = self.yolo_model.predict(frames_batch, verbose=False)

        # 3) 提取并聚合 bbox
        yolo_encodings = []
        for r in results:
            bboxes = r.boxes  # Boxes object
            if bboxes is not None and len(bboxes) > 0:
                # bboxes.xyxy: shape [n,4]
                xyxy = bboxes.xyxy  # (n, 4)
                # center:
                xc = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
                yc = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
                w = (xyxy[:, 2] - xyxy[:, 0])
                h = (xyxy[:, 3] - xyxy[:, 1])

                xc_mean = xc.mean().item()
                yc_mean = yc.mean().item()
                w_mean = w.mean().item()
                h_mean = h.mean().item()

                encoding_vec = torch.tensor(
                    [xc_mean, yc_mean, w_mean, h_mean],
                    device=self.device
                )
            else:
                # 若无检测到目标, 用0向量
                encoding_vec = torch.zeros(4, device=self.device)

            yolo_encodings.append(encoding_vec)

        # 4) 堆叠并过 MLP
        yolo_encodings = torch.stack(yolo_encodings, dim=0)  # [T,4]
        yolo_encodings = self.mlp(yolo_encodings)  # [T,128]

        return yolo_encodings

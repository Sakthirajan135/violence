#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File Name: dataset_rwf2000.py
Description:
  - Defines the RWF2000Dataset class for loading and sampling frames
    from the RWF-2000 dataset (fight/non-fight).
  - This version is updated to search BOTH *.mp4 and *.avi (or more),
    to adapt to the directory structure like:
        RWF-2000/
          train/
            Fight/
              *.avi
            NonFight/
              *.avi
          val/
            Fight/
            NonFight/

Usage Example:
  from dataset_rwf2000 import RWF2000Dataset
  dataset = RWF2000Dataset(
      root_dir="/path/to/RWF-2000",
      mode='train',  # or 'val'
      ...
  )
  frames, label = dataset[0]

Maintainer: [Your Name], contact@example.com
Innovative & Maintainable: This file aims to be self-contained, easy to
  modify or extend, and aligned with top-tier conference code quality.
"""

import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

# 关键：使用别名来避免内置 random() 冲突
import random as py_random


class RWF2000Dataset(Dataset):
    """
    修订说明：
       原本仅搜 *.mp4，现增加对 *.avi 的搜索。
       数据集结构示意：
         RWF-2000/
           train/
             Fight/
               *.avi
             NonFight/
               *.avi
           val/
             Fight/
             NonFight/
    其余逻辑保持一致，只要文件路径存在，就会找到视频并正常加载。
    """

    def __init__(
            self,
            root_dir,
            mode='train',
            num_segments=8,
            transform=None,
            random_seed=42
    ):
        """
        :param root_dir: RWF-2000 数据集根目录, e.g. "/home/xxx/RWF-2000"
        :param mode: 'train' 或 'val'，若您有 'test' 也可再扩展
        :param num_segments: 抽取多少帧段
        :param transform: 对每帧的图像增广或预处理函数
        :param random_seed: 用于打乱的随机种子
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.num_segments = num_segments
        self.transform = transform

        # 新增一个映射: 现目录是Fight、NonFight
        self.pos_folder_name = 'Fight'
        self.neg_folder_name = 'NonFight'

        # 收集所有视频路径
        fight_dir = os.path.join(self.root_dir, self.mode, self.pos_folder_name)
        nonfight_dir = os.path.join(self.root_dir, self.mode, self.neg_folder_name)

        # 同时搜索 *.mp4 与 *.avi (若还有 .mov/.mkv 可继续扩展)
        fight_glob_mp4 = os.path.join(fight_dir, '*.mp4')
        fight_glob_avi = os.path.join(fight_dir, '*.avi')
        nonfight_glob_mp4 = os.path.join(nonfight_dir, '*.mp4')
        nonfight_glob_avi = os.path.join(nonfight_dir, '*.avi')

        fight_videos_mp4 = glob.glob(fight_glob_mp4)
        fight_videos_avi = glob.glob(fight_glob_avi)
        nonfight_videos_mp4 = glob.glob(nonfight_glob_mp4)
        nonfight_videos_avi = glob.glob(nonfight_glob_avi)

        # 合并
        fight_videos = fight_videos_mp4 + fight_videos_avi
        nonfight_videos = nonfight_videos_mp4 + nonfight_videos_avi

        self.video_list = []
        self.labels = []

        # label=1 for Fight
        for vf in fight_videos:
            self.video_list.append(vf)
            self.labels.append(1)

        # label=0 for NonFight
        for vnf in nonfight_videos:
            self.video_list.append(vnf)
            self.labels.append(0)

        # 打印下找到的视频数
        print("[RWF2000Dataset] Mode:", self.mode)
        print(f"  Found {len(fight_videos)} fight videos in: {fight_dir}")
        print(f"  Found {len(nonfight_videos)} nonfight videos in: {nonfight_dir}")
        print(f"  => Total {len(self.video_list)} videos.\n")

        # 如果没有找到任何视频，则 data 为空 => zip(*data) 会报 ValueError
        if len(self.video_list) == 0:
            raise RuntimeError(
                f"No videos found for mode={self.mode} under root={self.root_dir}. "
                f"Check your directory structure or path."
            )

        data = list(zip(self.video_list, self.labels))
        # 使用 py_random.Random(...)，避免覆盖内置 random
        py_random.Random(random_seed).shuffle(data)

        # 解压
        self.video_list, self.labels = zip(*data)

    def __len__(self):
        return len(self.video_list)

    def _get_indices(self, total_frames):
        """
        返回要采样的帧序列下标，根据 self.num_segments。
        这里默认等间隔采样，若帧数不够则重复取。
        若需要其他策略(随机段采样、多段融合等)可自行扩展。
        """
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
                f"Video {video_path} has 0 frames or is unreadable. Please check dataset integrity."
            )

        indices = self._get_indices(total_frames)
        frames = []

        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                # 如果读取失败，就用一张黑图占位
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                # BGR -> RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 统一resize到 224x224
            frame = cv2.resize(frame, (224, 224))

            # 若有 transform 则应用
            if self.transform:
                frame = self.transform(frame)
            else:
                # 默认：转tensor、归一化
                frame = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0

            frames.append(frame)

        cap.release()

        # frames: list of [3, 224, 224], length = num_segments
        frames_tensor = torch.stack(frames, dim=0)  # [T, 3, 224, 224]

        return frames_tensor, torch.tensor(label, dtype=torch.long)

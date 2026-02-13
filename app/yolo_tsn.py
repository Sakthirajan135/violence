

import torch
from torch import nn
from mmengine import Config

# We now import from the registry rather than from mmaction.models
from mmaction.registry import MODELS


class TSNWithYOLO(nn.Module):
    def __init__(
            self,
            mmaction_cfg_path,
            yolo_encoding_dim=128,
            device='cuda',
            freeze_tsn_backbone=False
    ):
        """
        :param mmaction_cfg_path: path to a single-file config for TSN in MMAction2 style
        :param yolo_encoding_dim: dimension of YOLO's encoded features to fuse
        :param device: 'cuda' or 'cpu'
        :param freeze_tsn_backbone: if True, sets requires_grad=False on the TSN backbone
        """
        super().__init__()
        self.device = device

        # 1) Parse the config
        config = Config.fromfile(mmaction_cfg_path)
        # Insert train_cfg/test_cfg into model config if needed
        # (some config variants already contain them, so we do it safely)
        if 'train_cfg' not in config.model:
            config.model['train_cfg'] = config.get('train_cfg', None)
        if 'test_cfg' not in config.model:
            config.model['test_cfg'] = config.get('test_cfg', None)

        # 2) Build TSN from the registry
        self.tsn = MODELS.build(config.model)
        self.tsn.to(self.device)

        # 3) Optionally freeze TSN backbone
        if freeze_tsn_backbone:
            for param in self.tsn.backbone.parameters():
                param.requires_grad = False

        # 4) Create a fusion layer that appends YOLO encoding to TSN's final features
        #    We assume TSN has self.tsn.cls_head.fc_cls = final FC
        original_fc = self.tsn.cls_head.fc_cls
        in_features = original_fc.in_features   # e.g. 2048 for ResNet50
        out_features = original_fc.out_features

        self.fusion_fc = nn.Linear(in_features + yolo_encoding_dim, out_features)

        # Copy original fc_cls weights & bias into the corresponding portion of fusion_fc
        with torch.no_grad():
            self.fusion_fc.weight[:, :in_features] = original_fc.weight
            self.fusion_fc.bias[:] = original_fc.bias

    def forward(self, frames, yolo_encodings):
        """
        frames:        shape [B, T, 3, H, W]
        yolo_encodings shape [B, T, yolo_encoding_dim], e.g. 128
        Returns:       shape [B, num_classes]
        """
        B, T, C, H, W = frames.shape

        # 1) Run TSN backbone
        #    Flatten temporal dimension => [B*T, 3, H, W]
        frames_reshape = frames.view(-1, C, H, W)   # => [B*T, 3, H, W]
        feat = self.tsn.backbone(frames_reshape)    # => [B*T, feat_dim, ...]
        # Global average pool (assuming TSN's cls_head has avg_pool)
        feat = self.tsn.cls_head.avg_pool(feat)     # => [B*T, feat_dim, 1, 1]
        feat = feat.view(feat.size(0), -1)          # => [B*T, feat_dim]

        # 2) Temporal fusion for TSN
        feat = feat.view(B, T, -1)   # => [B, T, feat_dim]
        tsn_feat = feat.mean(dim=1)  # => [B, feat_dim]

        # 3) YOLO enc: shape [B, T, yolo_dim], do temporal average => [B, yolo_dim]
        yolo_feat = yolo_encodings.mean(dim=1)

        # 4) Concat TSN + YOLO => [B, feat_dim + yolo_dim]
        fused_feat = torch.cat([tsn_feat, yolo_feat], dim=1)

        # 5) Classify
        logits = self.fusion_fc(fused_feat)  # => [B, num_classes]
        return logits

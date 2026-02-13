import torch
from torch import nn
import torch.nn.functional as F
from ultralytics import YOLO


class BboxTransformerAggregator(nn.Module):

    def __init__(
        self,
        input_dim=13,
        hidden_dim=16,
        aggregator_out_dim=128,
        num_heads=2,
        use_cls_token=False
    ):
        """
        :param input_dim:        dimension of the per-bbox input (e.g. 5 + class_embed_dim)
        :param hidden_dim:       dimension for the transformer's internal d_model
                                 (must be divisible by num_heads)
        :param aggregator_out_dim: dimension of final output (e.g. 128)
        :param num_heads:        multi-head attention heads (hidden_dim must be divisible by this)
        :param use_cls_token:    whether to add a learnable [CLS] token for pooling
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.aggregator_out_dim = aggregator_out_dim
        self.num_heads = num_heads
        self.use_cls_token = use_cls_token

        # 1) Project input_dim -> hidden_dim
        self.in_proj = nn.Linear(input_dim, hidden_dim)

        # 2) Single-layer multi-head self-attention block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # (Optional) a learnable [CLS] token
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # 3) Final projection to aggregator_out_dim
        self.out_proj = nn.Linear(hidden_dim, aggregator_out_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.constant_(self.in_proj.bias, 0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0)
        if self.use_cls_token:
            nn.init.zeros_(self.cls_token)

    def forward(self, bbox_embeddings):
        """
        :param bbox_embeddings: [N, input_dim], N is # of boxes for this frame
        :return: a single vector [aggregator_out_dim]
        """
        if bbox_embeddings.ndim != 2:
            raise ValueError(
                "BboxTransformerAggregator input must be [N, input_dim]. Got shape: "
                f"{bbox_embeddings.shape}"
            )

        # 1) Project to hidden_dim => [N, hidden_dim]
        x = self.in_proj(bbox_embeddings)

        # 2) (Optional) prepend [CLS] token
        if self.use_cls_token:
            # shape [1,1,hidden_dim] -> [1, hidden_dim]
            cls_tok = self.cls_token.squeeze(0)
            # concat => [N+1, hidden_dim]
            x = torch.cat([cls_tok, x], dim=0)

        # 3) Transformer encoder expects shape [batch, seq, d_model]
        x = x.unsqueeze(0)  # => [1, (N or N+1), hidden_dim]
        x = self.transformer_encoder(x)  # => [1, seq_len, hidden_dim]

        # 4) Pooling
        if self.use_cls_token:
            # Use the [CLS] token output
            x = x[:, 0, :]  # => [1, hidden_dim]
        else:
            x = x.mean(dim=1)  # => [1, hidden_dim]

        # 5) Final projection => aggregator_out_dim
        x = self.out_proj(x)  # => [1, aggregator_out_dim]
        return x.squeeze(0)   # => [aggregator_out_dim]


class YOLOEncoder(nn.Module):

    def __init__(
        self,
        yolo_weights_path='yolov8n.pt',
        device='cuda',
        freeze_yolo=True,
        conf_thresh=0.3,
        top_k=5,
        class_embed_dim=8,
        use_attention=True,
        use_frame_residual=True
    ):
        """
        :param yolo_weights_path: YOLOv8 weights
        :param device: 'cuda' or 'cpu'
        :param freeze_yolo: if True, YOLO backbone is frozen
        :param conf_thresh: min confidence threshold for bboxes
        :param top_k: keep at most top_k bboxes by conf for each frame
        :param class_embed_dim: dimension for class ID embedding
        :param use_attention: whether to use single-layer transformer aggregator
                              or a simpler MLP aggregator
        :param use_frame_residual: if True, also encode the raw frame with a small CNN
                                   and sum it with YOLO aggregator output (i.e. "residual").
        """
        super().__init__()
        self.device = device
        self.conf_thresh = conf_thresh
        self.top_k = top_k
        self.use_attention = use_attention
        self.use_frame_residual = use_frame_residual
        self.freeze_yolo = freeze_yolo

        # Load YOLOv8 model
        self.yolo_model = YOLO(yolo_weights_path)
        self.yolo_model.model.to(self.device)

        if freeze_yolo:
            for param in self.yolo_model.model.parameters():
                param.requires_grad = False
            self.yolo_model.model.eval()

        # Class embedding
        # For YOLOv8 default 80 classes (COCO)
        self.num_classes = 80
        self.class_embed_dim = class_embed_dim
        self.class_embedding = nn.Embedding(self.num_classes, class_embed_dim)

        # Bbox base dimension
        # (xc, yc, w, h, conf) => 5 + class_embed_dim
        self.bbox_base_dim = 5 + self.class_embed_dim  # e.g. 13 if class_embed_dim=8
        aggregator_out_dim = 128

        # Aggregator => self.mlp
        # The code in main.py expects yolo_encoder.mlp to exist
        # so we unify both aggregator types under self.mlp
        if self.use_attention:
            # Transformer-based aggregator
            hidden_dim = 16
            num_heads = 2
            self.mlp = BboxTransformerAggregator(
                input_dim=self.bbox_base_dim,
                hidden_dim=hidden_dim,  # must be divisible by num_heads
                aggregator_out_dim=aggregator_out_dim,
                num_heads=num_heads,
                use_cls_token=False
            )
        else:
            # Simpler MLP aggregator => output is shape [k, 128], so we average across k
            self.mlp = nn.Sequential(
                nn.Linear(self.bbox_base_dim, 64),
                nn.ReLU(),
                nn.Linear(64, aggregator_out_dim)
            )

        # Frame Residual CNN
        # 如果希望把整帧信息也加入，当YOLO没检测到目标时也有备用特征。
        if self.use_frame_residual:
            # 简单CNN => 将 [3, H, W] -> [128]
            self.frame_residual_cnn = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),  # -> [16, 1, 1]
                nn.Flatten(),                  # -> [16]
                nn.Linear(16, aggregator_out_dim)  # -> [128]
            )
        else:
            print('No frame residual')
            self.frame_residual_cnn = None

    def forward(self, frames):
        """
        :param frames: [T, 3, H, W]
        :return:       [T, 128] YOLO-based embeddings (plus optional frame residual)
        """
        if len(frames.shape) != 4:
            raise ValueError(
                f"[YOLOEncoder] Expect frames shape [T,3,H,W], got {frames.shape}"
            )

        # 有时 transforms 不一定给 float32；我们确保一下，同时 clone 避免潜在 inplace 冲突
        frames_batch = frames.to(self.device, non_blocking=True).clone().float()

        # YOLO Inference
        # 如果 YOLO 要参与梯度计算，就不能用 no_grad；
        # 若 freeze_yolo=True，就用 no_grad() 包裹
        if self.freeze_yolo:
            with torch.no_grad():
                results = self.yolo_model.predict(frames_batch, verbose=False)
        else:
            results = self.yolo_model.predict(frames_batch, verbose=False)

        # Prepare container
        yolo_encodings = []

        # Build frame embeddings
        for t_idx, r in enumerate(results):
            # --(a) YOLO aggregator part--
            bboxes = r.boxes  # Boxes object from ultralytics
            if bboxes is not None and len(bboxes) > 0:
                # Filter by confidence
                confs = bboxes.conf  # shape [n]
                keep_mask = confs >= self.conf_thresh
                kept_indices = torch.nonzero(keep_mask).squeeze()
                if kept_indices.numel() > 0:
                    # Keep only boxes above threshold
                    confs = confs[keep_mask]
                    xyxy = bboxes.xyxy[keep_mask]
                    cls_ids = bboxes.cls[keep_mask]

                    # Sort by conf desc, keep top_k
                    sorted_inds = torch.argsort(confs, descending=True)
                    sorted_inds = sorted_inds[:self.top_k]
                    confs = confs[sorted_inds]
                    xyxy = xyxy[sorted_inds]
                    cls_ids = cls_ids[sorted_inds]

                    # Convert xyxy -> (xc, yc, w, h)
                    xc = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
                    yc = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
                    w = (xyxy[:, 2] - xyxy[:, 0]).clamp(min=0)
                    h = (xyxy[:, 3] - xyxy[:, 1]).clamp(min=0)

                    conf_vec = confs.unsqueeze(1)  # => [k,1]

                    # Class IDs -> embedding
                    cls_ids = cls_ids.long().clamp_(0, self.num_classes - 1)
                    cls_embed = self.class_embedding(cls_ids)  # => [k, class_embed_dim]

                    # Base feat => [k, 5 + class_embed_dim]
                    xcw = torch.stack([xc, yc, w, h], dim=1)  # => [k,4]
                    base_feat = torch.cat([xcw, conf_vec], dim=1)  # => [k,5]
                    full_feat = torch.cat([base_feat, cls_embed], dim=1)  # => [k, 5+class_embed_dim]

                    # aggregator => returns [128] or [k,128]
                    if self.use_attention:
                        aggregator_out = self.mlp(full_feat)  # => [128]
                    else:
                        # MLP aggregator => returns [k, 128], average across k
                        agg_output = self.mlp(full_feat)  # => [k,128]
                        aggregator_out = agg_output.mean(dim=0)  # => [128]

                else:
                    # No boxes above threshold
                    aggregator_out = torch.zeros((128,), device=self.device)
            else:
                # No bboxes => zero vector
                aggregator_out = torch.zeros((128,), device=self.device)

            # Frame residual part
            if self.use_frame_residual and self.frame_residual_cnn is not None:
                # frames_batch[t_idx] => shape [3,H,W]
                # pass through small CNN => shape [128]
                raw_frame = frames_batch[t_idx].unsqueeze(0)  # => [1,3,H,W]
                frame_feat = self.frame_residual_cnn(raw_frame)  # => [1,128]
                frame_feat = frame_feat.squeeze(0)              # => [128]

                # sum as a "residual" or fallback
                fused_out = aggregator_out + frame_feat
            else:
                print('No frame residual')
                fused_out = aggregator_out

            yolo_encodings.append(fused_out)

        # Stack => shape [T, 128]
        yolo_encodings = torch.stack(yolo_encodings, dim=0)

        # print(f"[YOLOEncoder] yolo_encodings shape = {yolo_encodings.shape}")
        return yolo_encodings

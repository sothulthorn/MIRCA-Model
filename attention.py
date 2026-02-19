"""
Attention modules used in the MIRCA model.

- CoordinateAttention (CA)
- EfficientMultiScaleAttention (EMA)
"""

import torch
import torch.nn as nn


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention (CA).

    Decomposes 2-D global pooling into two direction-aware 1-D pooling
    operations (horizontal / vertical), preserving precise spatial
    location information for position-sensitive attention.

    Reference: Hou et al., "Coordinate Attention for Efficient Mobile
    Network Design", CVPR 2021.
    """

    def __init__(self, channels, reduction=32):
        super().__init__()
        mid = max(8, channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_squeeze = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(mid)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid, channels, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape

        # Direction-aware 1-D pooling
        x_h = self.pool_h(x)                           # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)      # (B, C, W, 1)

        # Concatenate, squeeze, activate
        y = torch.cat([x_h, x_w], dim=2)               # (B, C, H+W, 1)
        y = self.act(self.bn(self.conv_squeeze(y)))     # (B, mid, H+W, 1)

        # Split into H and W components
        y_h, y_w = torch.split(y, [H, W], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)                  # (B, mid, 1, W)

        # Generate direction-aware attention maps
        a_h = torch.sigmoid(self.conv_h(y_h))           # (B, C, H, 1)
        a_w = torch.sigmoid(self.conv_w(y_w))           # (B, C, 1, W)

        return x * a_h * a_w


class EfficientMultiScaleAttention(nn.Module):
    """
    Efficient Multi-Scale Attention (EMA).

    Splits channels into G groups and applies three parallel branches:
      - Two 1x1 branches with 1-D pooling along H and W respectively,
        fused via element-wise multiplication.
      - One 3x3 branch for local spatial-channel interactions.

    The two branch outputs are combined through global-average-pooling
    followed by softmax-weighted aggregation.

    Reference: Ouyang et al., "Efficient Multi-Scale Attention Module
    with Cross-Spatial Learning", ICASSP 2023.
    """

    def __init__(self, channels, groups=4):
        super().__init__()
        self.groups = groups
        self.gc = channels // groups

        # Shared 1x1 conv for the two 1-D pooling branches
        self.conv1x1 = nn.Conv2d(self.gc, self.gc, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.gc)

        # 3x3 conv branch
        self.conv3x3 = nn.Conv2d(self.gc, self.gc, kernel_size=3,
                                 padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.gc)

    def forward(self, x):
        B, C, H, W = x.shape
        G, gc = self.groups, self.gc

        # Reshape into groups: (B*G, gc, H, W)
        x_g = x.reshape(B * G, gc, H, W)

        # ---- Branches 1 & 2: 1-D pooling + shared 1x1 conv ----
        x_h = x_g.mean(dim=3, keepdim=True)            # (B*G, gc, H, 1)
        x_w_t = x_g.mean(dim=2, keepdim=True)           # (B*G, gc, 1, W)
        x_w_t = x_w_t.permute(0, 1, 3, 2)               # (B*G, gc, W, 1)

        x_cat = torch.cat([x_h, x_w_t], dim=2)          # (B*G, gc, H+W, 1)
        x_cat = self.bn1(self.conv1x1(x_cat))

        feat_h, feat_w = torch.split(x_cat, [H, W], dim=2)
        a_h = torch.sigmoid(feat_h)                      # (B*G, gc, H, 1)
        a_w = torch.sigmoid(feat_w.permute(0, 1, 3, 2)) # (B*G, gc, 1, W)
        attn_1x1 = a_h * a_w                             # (B*G, gc, H, W)

        # ---- Branch 3: 3x3 conv ----
        attn_3x3 = torch.sigmoid(self.bn3(self.conv3x3(x_g)))

        # ---- Aggregate via channel-wise weighting ----
        w1 = (x_g * attn_1x1).mean(dim=[2, 3], keepdim=True)
        w3 = (x_g * attn_3x3).mean(dim=[2, 3], keepdim=True)

        weights = torch.softmax(torch.stack([w1, w3], dim=-1), dim=-1)
        out = x_g * attn_1x1 * weights[..., 0] + x_g * attn_3x3 * weights[..., 1]

        return out.reshape(B, C, H, W)

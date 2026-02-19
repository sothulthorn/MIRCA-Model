"""
MIRCA model: Multi-source Information Residual CNN-Attention.

Architecture:
    1. FusionModule  — learnable pixel-level weighted fusion of two CWT images.
    2. Stem          — Conv7x7 -> BN -> ReLU -> MaxPool.
    3. Four stages   — MIRCA residual blocks with CA + EMA attention.
    4. Head          — Global Average Pooling -> FC -> num_classes.
"""

import torch
import torch.nn as nn

from config import Config
from attention import CoordinateAttention, EfficientMultiScaleAttention


class FusionModule(nn.Module):
    """
    Pixel-level weighted fusion of two grayscale images.
    F3 = W * F1 + (1 - W) * F2, where W is a learnable scalar.
    """

    def __init__(self):
        super().__init__()
        self.raw_weight = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5

    def forward(self, img1, img2):
        w = torch.sigmoid(self.raw_weight)
        return w * img1 + (1.0 - w) * img2

    @property
    def fusion_weight(self):
        return torch.sigmoid(self.raw_weight).item()


class MIRCABlock(nn.Module):
    """
    Single MIRCA residual block.

    Main path:
        Conv3x3 -> BN -> ReLU -> CA -> Conv3x3 -> BN -> EMA
    Shortcut:
        Identity  (or Conv1x1+BN with stride for downsampling)
    Output:
        ReLU(main + shortcut)
    """

    def __init__(self, in_channels, out_channels, stride=1,
                 ca_reduction=32, ema_groups=4):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.ca = CoordinateAttention(out_channels, reduction=ca_reduction)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.ema = EfficientMultiScaleAttention(out_channels, groups=ema_groups)

        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.ca(out)
        out = self.bn2(self.conv2(out))
        out = self.ema(out)

        return self.relu(out + identity)


class MIRCA(nn.Module):
    """
    Full MIRCA model.

    Args:
        cfg: Config object with stage_channels, stage_blocks, etc.
    """

    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = Config()
        self.cfg = cfg
        ch = cfg.stage_channels
        bl = cfg.stage_blocks

        self.fusion = FusionModule()

        self.stem = nn.Sequential(
            nn.Conv2d(cfg.in_channels, ch[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.stage1 = self._make_stage(ch[0], ch[0], bl[0], 1, cfg)
        self.stage2 = self._make_stage(ch[0], ch[1], bl[1], 2, cfg)
        self.stage3 = self._make_stage(ch[1], ch[2], bl[2], 2, cfg)
        self.stage4 = self._make_stage(ch[2], ch[3], bl[3], 2, cfg)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(ch[3], cfg.num_classes)

        self._initialize_weights()

    def _make_stage(self, in_ch, out_ch, n_blocks, stride, cfg):
        layers = [MIRCABlock(in_ch, out_ch, stride,
                             cfg.ca_reduction, cfg.ema_groups)]
        for _ in range(1, n_blocks):
            layers.append(MIRCABlock(out_ch, out_ch, 1,
                                     cfg.ca_reduction, cfg.ema_groups))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, vib_img, cur_img):
        """
        Args:
            vib_img: (B, 1, H, W) vibration CWT image.
            cur_img: (B, 1, H, W) current CWT image.
        Returns:
            logits:  (B, num_classes).
        """
        x = self.fusion(vib_img, cur_img)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x).flatten(1)
        return self.fc(x)

    def extract_features(self, vib_img, cur_img):
        """Return penultimate-layer features for t-SNE visualization."""
        x = self.fusion(vib_img, cur_img)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.global_pool(x).flatten(1)

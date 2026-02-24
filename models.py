"""
MIRCA Model: Multi-Source Information Residual CNN-Attention.

Implements:
- Coordinate Attention (CA) module
- Efficient Multi-Scale Attention (EMA) module
- MIRCA Block (residual block with CA + EMA)
- Learnable Fusion Module
- Full MIRCA model with ablation variants
"""
import torch
import torch.nn as nn


# ==============================================================================
# Coordinate Attention (CA) Module - Figure 5a
# ==============================================================================
class CoordinateAttention(nn.Module):
    """
    Coordinate Attention decomposes 2D global pooling into two 1D direction-aware
    pooling operations (horizontal and vertical), preserving spatial location info.
    """

    def __init__(self, channels, reduction=32):
        super().__init__()
        mid_channels = max(8, channels // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))   # (B, C, H, 1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))    # (B, C, 1, W)

        # Shared 1x1 convolution on concatenated features
        self.conv_reduce = nn.Conv2d(channels, mid_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(mid_channels)
        self.act = nn.ReLU(inplace=True)

        # Separate 1x1 convolutions for h and w attention maps
        self.conv_h = nn.Conv2d(mid_channels, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, channels, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape

        # Direction-aware pooling
        x_h = self.pool_h(x)                                # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)           # (B, C, W, 1)

        # Concatenate along spatial dimension and shared transform
        y = torch.cat([x_h, x_w], dim=2)                    # (B, C, H+W, 1)
        y = self.act(self.bn(self.conv_reduce(y)))           # (B, mid_C, H+W, 1)

        # Split back into h and w components
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)                       # (B, mid_C, 1, W)

        # Generate attention maps
        att_h = self.conv_h(x_h).sigmoid()                   # (B, C, H, 1)
        att_w = self.conv_w(x_w).sigmoid()                   # (B, C, 1, W)

        return x * att_h * att_w


# ==============================================================================
# Efficient Multi-Scale Attention (EMA) Module - Figure 5b
# ==============================================================================
class EfficientMultiScaleAttention(nn.Module):
    """
    EMA divides input into channel groups, uses parallel 1x1 and 3x3 branches
    to capture multi-scale spatial information with cross-spatial learning.
    """

    def __init__(self, channels, groups=8):
        super().__init__()
        self.groups = groups
        gc = channels // groups  # channels per group

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.agp = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=-1)

        # 1x1 branch: shared conv on direction-pooled features
        self.conv1x1 = nn.Conv2d(gc, gc, kernel_size=1, bias=False)
        # 3x3 branch: local context
        self.conv3x3 = nn.Conv2d(gc, gc, kernel_size=3, padding=1, bias=False)
        self.gn = nn.GroupNorm(1, gc)

    def forward(self, x):
        b, c, h, w = x.size()
        g = self.groups
        gc = c // g

        # Group channels: (B, C, H, W) -> (B*G, C//G, H, W)
        x = x.reshape(b * g, gc, h, w)

        # --- 1x1 branch: direction-aware pooling ---
        x_h = self.pool_h(x)                                 # (B*G, gc, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)            # (B*G, gc, W, 1)

        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))     # (B*G, gc, H+W, 1)
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        # Apply sigmoid attention from 1x1 branches to input
        x1 = self.gn(x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())

        # --- 3x3 branch: local spatial context ---
        x2 = self.conv3x3(x)

        # --- Cross-spatial learning via matmul ---
        # Global average pooling -> softmax -> channel attention vectors
        x11 = self.softmax(
            self.agp(x1).reshape(b * g, 1, gc)               # (B*G, 1, gc)
        ).contiguous()
        x12 = x2.reshape(b * g, gc, -1).contiguous()         # (B*G, gc, H*W)

        x21 = self.softmax(
            self.agp(x2).reshape(b * g, 1, gc)               # (B*G, 1, gc)
        ).contiguous()
        x22 = x1.reshape(b * g, gc, -1).contiguous()         # (B*G, gc, H*W)

        # Cross-attention: combine spatial maps from both branches
        # Use element-wise multiply + reduce instead of bmm to avoid
        # CUBLAS strided batched GEMM issues on some GPU/driver combos
        weights = (
            (x11.transpose(1, 2) * x12).sum(dim=1, keepdim=True) +
            (x21.transpose(1, 2) * x22).sum(dim=1, keepdim=True)
        ).reshape(b * g, 1, h, w)                             # (B*G, 1, H*W)

        out = (x * weights.sigmoid()).reshape(b, c, h, w)
        return out


# ==============================================================================
# MIRCA Block - Figure 4
# ==============================================================================
class MIRCABlock(nn.Module):
    """
    Residual block with optional CA and EMA attention modules.

    Main branch: Conv -> BN -> ReLU -> [CA] -> Conv -> BN -> [EMA]
    Shortcut: identity or 1x1 Conv+BN for dimension change
    Output: ReLU(main + shortcut)
    """

    def __init__(self, in_channels, out_channels, stride=1,
                 use_ca=True, use_ema=True, ema_groups=8):
        super().__init__()

        # Main branch
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.ca = CoordinateAttention(out_channels) if use_ca else nn.Identity()

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Ensure EMA groups divides channels evenly
        actual_groups = ema_groups
        while out_channels % actual_groups != 0 and actual_groups > 1:
            actual_groups //= 2
        self.ema = (
            EfficientMultiScaleAttention(out_channels, groups=actual_groups)
            if use_ema else nn.Identity()
        )

        # Shortcut branch (Figure 4b: downsampling with 1x1 conv)
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

        out = self.relu(out + identity)
        return out


# ==============================================================================
# Learnable Fusion Module - Equation (2)
# ==============================================================================
class LearnableFusion(nn.Module):
    """
    Pixel-level weighted fusion: F3 = W * F1 + (1 - W) * F2
    W is a learnable parameter optimized during training.
    """

    def __init__(self, init_weight=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(init_weight))

    def forward(self, f1, f2):
        w = torch.sigmoid(self.weight)  # Constrain to [0, 1]
        return w * f1 + (1.0 - w) * f2


# ==============================================================================
# Full MIRCA Model - Figure 2 & 3
# ==============================================================================
class MIRCA(nn.Module):
    """
    Multi-Source Information Residual CNN-Attention model.

    Architecture: ResNet-34 backbone with CA + EMA attention in each block,
    preceded by a learnable multi-source fusion module.

    Args:
        num_classes: Number of fault categories (default: 8)
        stage_layers: Number of blocks per stage (default: [3, 4, 6, 3])
        stage_channels: Channels per stage (default: [64, 128, 256, 512])
        use_ca: Whether to use Coordinate Attention
        use_ema: Whether to use EMA attention
        ema_groups: Group count for EMA module
        use_fusion: Whether to use learnable fusion (False for single-source)
    """

    def __init__(self, num_classes=8,
                 stage_layers=None, stage_channels=None,
                 use_ca=True, use_ema=True, ema_groups=8,
                 use_fusion=True):
        super().__init__()

        if stage_layers is None:
            stage_layers = [3, 4, 6, 3]
        if stage_channels is None:
            stage_channels = [64, 128, 256, 512]

        self.use_fusion = use_fusion

        # Multi-source fusion module
        if use_fusion:
            self.fusion = LearnableFusion(init_weight=0.5)

        # Initial convolution layer (7x7, stride 2)
        self.stem = nn.Sequential(
            nn.Conv2d(1, stage_channels[0], kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(stage_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Build 4 stages
        self.stages = nn.ModuleList()
        in_ch = stage_channels[0]
        for stage_idx, (num_blocks, out_ch) in enumerate(
            zip(stage_layers, stage_channels)
        ):
            blocks = []
            for block_idx in range(num_blocks):
                stride = 2 if (stage_idx > 0 and block_idx == 0) else 1
                blocks.append(
                    MIRCABlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        stride=stride,
                        use_ca=use_ca,
                        use_ema=use_ema,
                        ema_groups=ema_groups,
                    )
                )
                in_ch = out_ch
            self.stages.append(nn.Sequential(*blocks))

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(stage_channels[-1], num_classes)

        # Weight initialization
        self._initialize_weights()

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

    def forward(self, vib_img, cur_img=None):
        """
        Args:
            vib_img: Vibration CWT grayscale image (B, 1, H, W)
            cur_img: Current CWT grayscale image (B, 1, H, W), optional
        Returns:
            logits: (B, num_classes)
        """
        # Fusion
        if self.use_fusion and cur_img is not None:
            x = self.fusion(vib_img, cur_img)
        else:
            x = vib_img

        # Feature extraction
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)

        # Classification
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        logits = self.fc(features)
        return logits

    def extract_features(self, vib_img, cur_img=None):
        """Extract features before the FC layer (for t-SNE visualization)."""
        if self.use_fusion and cur_img is not None:
            x = self.fusion(vib_img, cur_img)
        else:
            x = vib_img

        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)

        x = self.avgpool(x)
        return torch.flatten(x, 1)


# ==============================================================================
# Factory function for ablation variants
# ==============================================================================
def build_model(variant="MIRCA", num_classes=8, ema_groups=8, use_fusion=True):
    """
    Build model for ablation study.

    Args:
        variant: One of "Baseline", "Baseline+EMA", "Baseline+CA", "MIRCA"
        num_classes: Number of output classes
        ema_groups: EMA group parameter
        use_fusion: Whether to use multi-source fusion

    Returns:
        MIRCA model instance
    """
    configs = {
        "Baseline":     {"use_ca": False, "use_ema": False},
        "Baseline+EMA": {"use_ca": False, "use_ema": True},
        "Baseline+CA":  {"use_ca": True,  "use_ema": False},
        "MIRCA":        {"use_ca": True,  "use_ema": True},
    }

    if variant not in configs:
        raise ValueError(f"Unknown variant: {variant}. "
                         f"Choose from {list(configs.keys())}")

    cfg = configs[variant]
    return MIRCA(
        num_classes=num_classes,
        use_ca=cfg["use_ca"],
        use_ema=cfg["use_ema"],
        ema_groups=ema_groups,
        use_fusion=use_fusion,
    )


if __name__ == "__main__":
    # Quick sanity check
    for variant in ["Baseline", "Baseline+EMA", "Baseline+CA", "MIRCA"]:
        model = build_model(variant, num_classes=8)
        params = sum(p.numel() for p in model.parameters())
        # Test forward pass
        vib = torch.randn(2, 1, 224, 224)
        cur = torch.randn(2, 1, 224, 224)
        out = model(vib, cur)
        print(f"{variant:15s} | params: {params/1e6:.2f}M | output: {out.shape}")

"""
Shared-encoder CNN with two task-specific heads.

The encoder can be either:
  - "cnn":      original 4-block ConvNet
  - "resnet18": pretrained ResNet18 adapted to 1-channel input
Selected via config.ENCODER_TYPE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import config


# ---------------------------------------------------------------------------
# Original CNN encoder
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class CNNEncoder(nn.Module):
    """
    4-block CNN encoder (your original SharedEncoder, renamed).
    Input : (B, 1, 224, 224)
    Output: (B, C) where C = config.ENCODER_CHANNELS[-1]
    """

    def __init__(self):
        super().__init__()
        ch = config.ENCODER_CHANNELS
        self.blocks = nn.Sequential(
            ConvBlock(ch[0], ch[1]),   # 224 -> 112
            ConvBlock(ch[1], ch[2]),   # 112 -> 56
            ConvBlock(ch[2], ch[3]),   # 56  -> 28
            ConvBlock(ch[3], ch[4]),   # 28  -> 14
        )
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.feat_dim = ch[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = self.gap(x)
        return torch.flatten(x, start_dim=1)


# ---------------------------------------------------------------------------
# Pretrained ResNet18 encoder
# ---------------------------------------------------------------------------

class ResNetEncoder(nn.Module):
    """
    Pretrained ResNet18 adapted for 1-channel grayscale input.
    Input : (B, 1, 224, 224)
    Output: (B, 512)
    """

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        self._freeze_bn = freeze_backbone

        old_conv = self.backbone.conv1
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        self.backbone.conv1 = new_conv

        self.backbone.fc = nn.Identity()
        self.feat_dim = 512

        # Optionally freeze the pretrained backbone (keep conv1 trainable).
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self.backbone.conv1.parameters():
                p.requires_grad = True

            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.momentum = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def train(self, mode: bool = True):
        super().train(mode)
        if self._freeze_bn:
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def build_encoder() -> nn.Module:
    """Return the encoder selected in config.ENCODER_TYPE."""
    name = getattr(config, "ENCODER_TYPE", "cnn").lower()
    if name == "resnet18":
        return ResNetEncoder(
            pretrained=getattr(config, "PRETRAINED", True),
            freeze_backbone=getattr(config, "FREEZE_BACKBONE", False),
        )
    return CNNEncoder()


# ---------------------------------------------------------------------------
# Task head and multi-task model
# ---------------------------------------------------------------------------

class TaskHead(nn.Module):
    """Small MLP head: Dropout -> Linear -> ReLU -> Dropout -> Linear."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(config.DROPOUT_HEAD),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT_MID),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BUSMultiTaskCNN(nn.Module):
    """
    Shared encoder + two heads. The encoder is chosen by config.ENCODER_TYPE.
    Returns (logits_a, logits_b) where:
      logits_a: (B, 3)   -- for CrossEntropyLoss
      logits_b: (B, 1)   -- for BCEWithLogitsLoss
    """

    def __init__(self):
        super().__init__()
        self.encoder = build_encoder()
        feat_dim = self.encoder.feat_dim

        self.head_a = TaskHead(
            in_dim=feat_dim,
            hidden_dim=config.HEAD_HIDDEN,
            out_dim=config.NUM_CLASSES_A,
        )
        self.head_b = TaskHead(
            in_dim=feat_dim,
            hidden_dim=config.HEAD_HIDDEN,
            out_dim=config.NUM_CLASSES_B,
        )

    def forward(self, x: torch.Tensor):
        features = self.encoder(x)
        logits_a = self.head_a(features)
        logits_b = self.head_b(features)
        return logits_a, logits_b

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        self.eval()
        logits_a, logits_b = self.forward(x)
        probs_a = F.softmax(logits_a, dim=1)
        prob_b = torch.sigmoid(logits_b).squeeze(1)
        return probs_a, prob_b
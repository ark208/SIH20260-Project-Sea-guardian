"""
Project DRISHTI - PyTorch DeepLabV3+ Model
Reference: https://github.com/jfzhang95/pytorch-deeplab-xception.git
Adapted for single-channel (SAR sigma0 / dual-pol) input and 3-class segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: tuple = (6, 12, 18), out_channels: int = 256):
        super(ASPP, self).__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        ]
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 3):
        super(DeepLabV3Plus, self).__init__()

        # Encoder Stage 1 (Low-level feature extractor)
        self.entry_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Encoder Stage 2 (High-level feature extractor with downsampling)
        self.mid_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # ASPP Module
        self.aspp = ASPP(in_channels=256, atrous_rates=(6, 12, 18), out_channels=256)

        # Low-level projection
        self.low_level_project = nn.Sequential(
            nn.Conv2d(64, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # Decoder Stage
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        
        # Low-level features (1/2 size)
        low_feat = self.entry_conv(x)
        
        # High-level features (1/8 size)
        high_feat = self.mid_conv(low_feat)
        
        # ASPP features
        aspp_feat = self.aspp(high_feat)
        
        # Upsample ASPP to low-level feature size (1/2)
        aspp_upsampled = F.interpolate(aspp_feat, size=low_feat.shape[-2:], mode='bilinear', align_corners=False)
        low_projected = self.low_level_project(low_feat)
        
        # Concatenate & Decode
        fused = torch.cat([aspp_upsampled, low_projected], dim=1)
        logits_half = self.decoder(fused)
        
        # Final upsample to original resolution
        out = F.interpolate(logits_half, size=input_size, mode='bilinear', align_corners=False)
        return out

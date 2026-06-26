"""U-Net Generator for frame interpolation.

Input: concat(key_first, key_last, time_map) = 7 channels
Output: interpolated frame at time t (3 channels)
"""

import torch
import torch.nn as nn


class DownBlock(nn.Module):
    """Encoder block: Conv -> BatchNorm -> LeakyReLU -> Conv -> BatchNorm -> LeakyReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = self.block(x)
        pooled = self.pool(features)
        return pooled, features


class UpBlock(nn.Module):
    """Decoder block: Upsample -> Concat skip -> Conv -> BatchNorm -> ReLU."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatches from odd dimensions
        if x.shape != skip.shape:
            x = nn.functional.interpolate(
                x, size=skip.shape[2:], mode='bilinear', align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class UNetGenerator(nn.Module):
    """U-Net generator for frame interpolation.

    Takes concatenated key_first (3ch) + key_last (3ch) + time_map (1ch) = 7 channels.
    Outputs a 3-channel RGB frame at the specified time position.
    """

    def __init__(self, in_channels=7, out_channels=3, base_features=64):
        super().__init__()
        bf = base_features

        # Encoder
        self.down1 = DownBlock(in_channels, bf)       # 64
        self.down2 = DownBlock(bf, bf * 2)            # 128
        self.down3 = DownBlock(bf * 2, bf * 4)        # 256
        self.down4 = DownBlock(bf * 4, bf * 8)        # 512

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bf * 8, bf * 16, 3, padding=1),
            nn.BatchNorm2d(bf * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(bf * 16, bf * 16, 3, padding=1),
            nn.BatchNorm2d(bf * 16),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.up4 = UpBlock(bf * 16, bf * 8, bf * 8)
        self.up3 = UpBlock(bf * 8, bf * 4, bf * 4)
        self.up2 = UpBlock(bf * 4, bf * 2, bf * 2)
        self.up1 = UpBlock(bf * 2, bf, bf)

        # Output
        self.out_conv = nn.Sequential(
            nn.Conv2d(bf, out_channels, 1),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, key_first, key_last, t):
        """Forward pass.

        Args:
            key_first: (B, 3, H, W) first keyframe, normalized to [-1, 1]
            key_last: (B, 3, H, W) last keyframe, normalized to [-1, 1]
            t: (B, 1) time values in [0, 1]

        Returns:
            (B, 3, H, W) generated frame in [-1, 1]
        """
        B, _, H, W = key_first.shape

        # Create time map: spatial tensor filled with t value
        time_map = t.view(B, 1, 1, 1).expand(B, 1, H, W)

        # Concatenate inputs
        x = torch.cat([key_first, key_last, time_map], dim=1)  # (B, 7, H, W)

        # Encoder
        x, s1 = self.down1(x)
        x, s2 = self.down2(x)
        x, s3 = self.down3(x)
        x, s4 = self.down4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections
        x = self.up4(x, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)

        return self.out_conv(x)

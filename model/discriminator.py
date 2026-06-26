"""PatchGAN Discriminator for conditional frame interpolation.

Classifies whether a frame is real or generated, conditioned on
the two keyframes. Outputs a grid of real/fake predictions (patches).
"""

import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """Conditional PatchGAN discriminator.

    Input: concat(frame, key_first, key_last) = 9 channels
    Output: (B, 1, H/16, W/16) patch predictions
    """

    def __init__(self, in_channels=9, base_features=64, n_layers=3):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, base_features, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf = base_features
        for i in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers += [
                nn.Conv2d(nf_prev, nf, 4, stride=2, padding=1),
                nn.BatchNorm2d(nf),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # Second-to-last layer
        nf_prev = nf
        nf = min(nf * 2, 512)
        layers += [
            nn.Conv2d(nf_prev, nf, 4, stride=1, padding=1),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Final prediction layer
        layers += [nn.Conv2d(nf, 1, 4, stride=1, padding=1)]

        self.model = nn.Sequential(*layers)

    def forward(self, frame, key_first, key_last):
        """Forward pass.

        Args:
            frame: (B, 3, H, W) real or generated frame
            key_first: (B, 3, H, W) first keyframe
            key_last: (B, 3, H, W) last keyframe

        Returns:
            (B, 1, H', W') patch predictions (logits)
        """
        x = torch.cat([frame, key_first, key_last], dim=1)
        return self.model(x)

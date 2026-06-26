"""GAN model for generating inbetween frames from keyframe pairs.

Architecture:
- Generator: U-Net encoder-decoder with skip connections
  Input: concat(key_first, key_last, time_map) = 7 channels
  Output: interpolated frame at time t (3 channels)

- Discriminator: PatchGAN (conditional)
  Input: concat(frame, key_first, key_last) = 9 channels
  Output: patch-level real/fake predictions

The time parameter t in (0, 1) indicates position between keyframes:
  t=0 -> key_first, t=1 -> key_last, t=0.5 -> middle inbetween
"""

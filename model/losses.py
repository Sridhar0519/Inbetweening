"""Loss functions for GAN training.

Combines:
- L1 reconstruction loss (pixel-level accuracy)
- Adversarial loss (realistic outputs)
- Perceptual loss (VGG feature matching for sharpness)
"""

from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models


# Path to local VGG19 weights (pre-downloaded for offline use)
_VGG19_LOCAL_PATH = Path(__file__).parent.parent / "weights" / "vgg19-dcbb9e9d.pth"


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG19 feature maps.

    Compares intermediate feature representations rather than raw pixels,
    encouraging perceptually similar outputs.
    """

    def __init__(self):
        super().__init__()
        # Load VGG19 from local weights if available (offline server)
        if _VGG19_LOCAL_PATH.exists():
            vgg = models.vgg19(weights=None)
            vgg.load_state_dict(torch.load(_VGG19_LOCAL_PATH, map_location='cpu', weights_only=True))
        else:
            vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        # Use features up to relu3_4 (layer index 16)
        self.feature_extractor = nn.Sequential(*list(vgg.features[:16]))
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # VGG normalization
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalize(self, x):
        """Convert from [-1, 1] to VGG normalized range."""
        x = (x + 1) / 2  # [-1,1] -> [0,1]
        return (x - self.mean) / self.std

    def forward(self, generated, target):
        gen_features = self.feature_extractor(self._normalize(generated))
        target_features = self.feature_extractor(self._normalize(target))
        return nn.functional.l1_loss(gen_features, target_features)


class GANLoss(nn.Module):
    """Least-squares GAN loss (LSGAN) -- more stable than vanilla BCE."""

    def __init__(self):
        super().__init__()

    def forward(self, prediction, is_real):
        target = torch.ones_like(prediction) if is_real else torch.zeros_like(prediction)
        return nn.functional.mse_loss(prediction, target)


class CombinedLoss(nn.Module):
    """Combined generator loss: L1 + Adversarial + Perceptual."""

    def __init__(self, lambda_l1=10.0, lambda_perc=1.0, lambda_adv=1.0):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_perc = lambda_perc
        self.lambda_adv = lambda_adv
        self.l1_loss = nn.L1Loss()
        # Skip loading VGG19 entirely if perceptual loss is disabled
        self.perceptual_loss = VGGPerceptualLoss() if lambda_perc > 0 else None
        self.gan_loss = GANLoss()

    def forward(self, generated, target, disc_prediction):
        """Compute combined generator loss.

        Args:
            generated: (B, 3, H, W) generated frame
            target: (B, 3, H, W) ground truth frame
            disc_prediction: discriminator output on generated frame

        Returns:
            total_loss, loss_dict with individual components
        """
        loss_l1 = self.l1_loss(generated, target)
        loss_adv = self.gan_loss(disc_prediction, is_real=True)

        if self.perceptual_loss is not None:
            loss_perc = self.perceptual_loss(generated, target)
        else:
            loss_perc = torch.tensor(0.0, device=generated.device)

        total = (
            self.lambda_l1 * loss_l1
            + self.lambda_perc * loss_perc
            + self.lambda_adv * loss_adv
        )

        return total, {
            'l1': loss_l1.item(),
            'perceptual': loss_perc.item(),
            'adversarial': loss_adv.item(),
            'total': total.item(),
        }

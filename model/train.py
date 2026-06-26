"""Training script for the inbetween frame generation GAN.

Usage:
    python -m model.train --dataset ./output --epochs 100 --batch-size 8
    python -m model.train --dataset ./output --resume training_output/checkpoints/latest.pt
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import save_image
from tqdm import tqdm

from .generator import UNetGenerator
from .discriminator import PatchDiscriminator
from .losses import CombinedLoss, GANLoss
from .dataset import InbetweenDataset

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def denormalize(tensor):
    """Convert from [-1, 1] to [0, 1] for saving."""
    return (tensor + 1) / 2


def save_samples(generator, val_loader, device, epoch, output_dir):
    """Save visual samples from the validation set."""
    generator.eval()
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        batch = next(iter(val_loader))
        key_first = batch['key_first'].to(device)
        key_last = batch['key_last'].to(device)
        t = batch['t'].to(device)
        target = batch['target'].to(device)

        generated = generator(key_first, key_last, t)

        # Save grid: key_first | generated | target | key_last
        n = min(4, key_first.size(0))
        comparison = torch.cat([
            denormalize(key_first[:n]),
            denormalize(generated[:n]),
            denormalize(target[:n]),
            denormalize(key_last[:n]),
        ], dim=0)
        save_image(comparison, samples_dir / f"epoch_{epoch:04d}.png", nrow=n)

    generator.train()


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Using device: %s", device)

    if device.type == 'cuda':
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info("VRAM: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)
    else:
        # Optimize CPU threading
        import os
        num_cores = os.cpu_count() or 4
        torch.set_num_threads(num_cores)
        torch.set_num_interop_threads(max(1, num_cores // 2))
        logger.info("CPU threads: %d compute, %d interop", num_cores, max(1, num_cores // 2))

    # Dataset
    dataset = InbetweenDataset(args.dataset, image_size=args.image_size)
    logger.info("Dataset: %d samples", len(dataset))

    if len(dataset) == 0:
        logger.error("No training samples found. Run the dataset creator first.")
        sys.exit(1)

    # Optionally limit dataset size for faster training
    if args.max_samples and len(dataset) > args.max_samples:
        dataset, _ = random_split(
            dataset, [args.max_samples, len(dataset) - args.max_samples],
            generator=torch.Generator().manual_seed(42)
        )
        logger.info("Limited to %d samples (--max-samples)", args.max_samples)

    # Train/val split (90/10)
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    use_cuda = device.type == 'cuda'
    num_workers = args.num_workers if use_cuda else 0
    pin_memory = use_cuda

    if not use_cuda:
        logger.info("CPU mode: setting num_workers=0, pin_memory=False for stability.")
        logger.info("Tip: use --image-size 128 --batch-size 4 for faster CPU training.")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    logger.info("Train: %d, Val: %d", train_size, val_size)

    # Models
    bf = args.base_features
    generator = UNetGenerator(in_channels=7, out_channels=3, base_features=bf).to(device)
    discriminator = PatchDiscriminator(in_channels=9, base_features=bf).to(device)

    param_count_g = sum(p.numel() for p in generator.parameters()) / 1e6
    param_count_d = sum(p.numel() for p in discriminator.parameters()) / 1e6
    logger.info("Generator: %.2fM params, Discriminator: %.2fM params", param_count_g, param_count_d)

    # Losses
    criterion_g = CombinedLoss(
        lambda_l1=args.lambda_l1,
        lambda_perc=args.lambda_perc,
        lambda_adv=args.lambda_adv,
    ).to(device)
    criterion_d = GANLoss().to(device)

    # Optimizers
    opt_g = Adam(generator.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    # Schedulers
    scheduler_g = CosineAnnealingLR(opt_g, T_max=args.epochs, eta_min=1e-6)
    scheduler_d = CosineAnnealingLR(opt_d, T_max=args.epochs, eta_min=1e-6)

    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        opt_g.load_state_dict(checkpoint['opt_g'])
        opt_d.load_state_dict(checkpoint['opt_d'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info("Resumed from epoch %d", start_epoch)

    # Training loop
    logger.info("Starting training for %d epochs...", args.epochs)

    for epoch in range(start_epoch, args.epochs):
        generator.train()
        discriminator.train()

        epoch_losses = {
            'd_real': 0, 'd_fake': 0,
            'g_total': 0, 'g_l1': 0, 'g_perc': 0, 'g_adv': 0,
        }
        num_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            unit="batch",
            leave=True,
        )
        for batch in pbar:
            key_first = batch['key_first'].to(device)
            key_last = batch['key_last'].to(device)
            target = batch['target'].to(device)
            t = batch['t'].to(device)

            # --- Train Discriminator ---
            opt_d.zero_grad()

            with torch.no_grad():
                fake = generator(key_first, key_last, t)

            # Real
            pred_real = discriminator(target, key_first, key_last)
            loss_d_real = criterion_d(pred_real, is_real=True)

            # Fake
            pred_fake = discriminator(fake.detach(), key_first, key_last)
            loss_d_fake = criterion_d(pred_fake, is_real=False)

            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            opt_d.step()

            # --- Train Generator ---
            opt_g.zero_grad()

            fake = generator(key_first, key_last, t)
            pred_fake = discriminator(fake, key_first, key_last)

            loss_g, loss_dict = criterion_g(fake, target, pred_fake)
            loss_g.backward()
            opt_g.step()

            # Accumulate
            epoch_losses['d_real'] += loss_d_real.item()
            epoch_losses['d_fake'] += loss_d_fake.item()
            epoch_losses['g_total'] += loss_dict['total']
            epoch_losses['g_l1'] += loss_dict['l1']
            epoch_losses['g_perc'] += loss_dict['perceptual']
            epoch_losses['g_adv'] += loss_dict['adversarial']
            num_batches += 1

            # Update progress bar with current losses
            pbar.set_postfix({
                'D': f"{(loss_d_real.item() + loss_d_fake.item()) * 0.5:.3f}",
                'G': f"{loss_dict['total']:.3f}",
                'L1': f"{loss_dict['l1']:.3f}",
            })

        pbar.close()

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= max(num_batches, 1)

        scheduler_g.step()
        scheduler_d.step()

        # Logging
        logger.info(
            "Epoch [%d/%d] "
            "D_real: %.4f D_fake: %.4f | "
            "G_total: %.4f (L1: %.4f, Perc: %.4f, Adv: %.4f)",
            epoch + 1, args.epochs,
            epoch_losses['d_real'], epoch_losses['d_fake'],
            epoch_losses['g_total'],
            epoch_losses['g_l1'], epoch_losses['g_perc'], epoch_losses['g_adv'],
        )

        # Save samples
        if (epoch + 1) % args.sample_every == 0:
            save_samples(generator, val_loader, device, epoch + 1, output_dir)

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            checkpoint = {
                'epoch': epoch,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'opt_g': opt_g.state_dict(),
                'opt_d': opt_d.state_dict(),
            }
            torch.save(checkpoint, checkpoints_dir / f"epoch_{epoch + 1:04d}.pt")
            torch.save(checkpoint, checkpoints_dir / "latest.pt")
            logger.info("  Saved checkpoint at epoch %d", epoch + 1)

    logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Train the inbetween frame generation GAN.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m model.train --dataset ./output --epochs 100
  python -m model.train --dataset ./output --resume training_output/checkpoints/latest.pt
  python -m model.train --dataset ./output --batch-size 4 --image-size 128  # low VRAM
        """,
    )
    parser.add_argument('--dataset', '-d', type=str, required=True,
                        help='Path to dataset root (output from main.py)')
    parser.add_argument('--output', '-o', type=str, default='./training_output',
                        help='Output directory for checkpoints and samples')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (default: 8, reduce if OOM)')
    parser.add_argument('--image-size', type=int, default=256,
                        help='Training image size (default: 256)')
    parser.add_argument('--lr-g', type=float, default=2e-4,
                        help='Generator learning rate (default: 2e-4)')
    parser.add_argument('--lr-d', type=float, default=2e-4,
                        help='Discriminator learning rate (default: 2e-4)')
    parser.add_argument('--lambda-l1', type=float, default=10.0,
                        help='L1 loss weight (default: 10.0)')
    parser.add_argument('--lambda-perc', type=float, default=1.0,
                        help='Perceptual loss weight (default: 1.0)')
    parser.add_argument('--lambda-adv', type=float, default=1.0,
                        help='Adversarial loss weight (default: 1.0)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers (default: 4)')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--sample-every', type=int, default=5,
                        help='Save visual samples every N epochs (default: 5)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit dataset to N samples (faster training)')
    parser.add_argument('--base-features', type=int, default=64,
                        help='Base feature count for G/D (default: 64, use 32 for CPU)')
    parser.add_argument('--fast-cpu', action='store_true',
                        help='Auto-optimize for CPU: image-size=64, batch-size=4, '
                             'lambda-perc=0, base-features=32, max-samples=10000')
    args = parser.parse_args()

    # --fast-cpu overrides defaults (explicit CLI args still take priority)
    if args.fast_cpu:
        defaults = {
            'image_size': 64, 'batch_size': 4, 'lambda_perc': 0.0,
            'base_features': 32, 'max_samples': 10000, 'num_workers': 0,
        }
        for key, val in defaults.items():
            # Only override if user didn't explicitly set it
            if f'--{key.replace("_", "-")}' not in sys.argv:
                setattr(args, key, val)
        logger.info("Fast-CPU mode: %s", {k: getattr(args, k) for k in defaults})

    train(args)


if __name__ == '__main__':
    main()

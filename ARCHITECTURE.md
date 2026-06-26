# Inbetween Frame Generation GAN — Architecture Documentation

## Table of Contents

1. [Libraries & Why These Ones](#1-libraries--why-these-ones)
2. [Generator — U-Net](#2-generator--u-net)
3. [Discriminator — PatchGAN](#3-discriminator--patchgan)
4. [Loss Functions](#4-loss-functions)
5. [Dataset & Training Pipeline](#5-dataset--training-pipeline)
6. [Summary](#6-summary-why-this-specific-combination)

---

## 1. Libraries & Why These Ones

### PyTorch (`torch`, `torch.nn`)

The core deep learning framework — defines layers, tensors, autograd (automatic differentiation), and GPU acceleration.

| | PyTorch | TensorFlow/Keras | JAX |
|---|---|---|---|
| Debugging | Easy — standard Python, print anywhere | Harder — graph mode, eager mode is slower | Functional style, harder mental model |
| Research adoption | ~80% of new papers | Declining in research | Growing but niche |
| Dynamic graphs | Native — build graph on-the-fly | Added later (eager mode), not as natural | Requires `jit` for performance |
| GAN training | Natural — alternating G/D updates trivially | Awkward with `tf.function` | Possible but verbose |
| Community/examples | Largest for generative models | Largest for deployment/production | Smallest |

**Bottom line:** For a GAN research/personal project, PyTorch is the clear choice. TensorFlow would be better for deploying to mobile/edge. JAX if you needed massive TPU parallelism.

### torchvision (`transforms`, `models`, `save_image`)

Computer vision utilities built on PyTorch. Three specific uses in this project:

**1. `transforms` (dataset.py)** — Image preprocessing pipeline:

```python
transforms.Compose([
    transforms.Resize((256, 256)),               # Uniform size for batching
    transforms.ToTensor(),                        # PIL Image → (C,H,W) float tensor [0,1]
    transforms.Normalize([0.5]*3, [0.5]*3),       # [0,1] → [-1,1] range
])
```

Why [-1, 1]? The generator uses `Tanh()` output which naturally outputs [-1, 1]. Matching the target range makes training stable.

**2. `models.vgg19` (losses.py)** — Pretrained VGG19 for perceptual loss (see Section 4).

**3. `save_image` (train.py)** — Saves tensor grids as PNG for visual monitoring during training.

**Alternative considered:** `albumentations` for data augmentation (faster, more augmentation options). Not needed here since we're not augmenting — our frames must stay spatially consistent as triplets.

### PIL/Pillow (`Image`)

Loads and saves images from disk. It's the standard Python image library that `torchvision.transforms` expects as input. OpenCV (`cv2`) is faster for video but Pillow is simpler for single-image load/save and integrates natively with torchvision.

### tqdm

Progress bars for the training loop. No real alternative needed — it's the universal Python progress bar.

---

## 2. Generator — U-Net

> Source: `model/generator.py`

### Architecture Diagram

```
Input: [key_first(3ch) + key_last(3ch) + time_map(1ch)] = 7 channels
                              │
                    ┌─────────▼──────────┐
                    │   Encoder (Down)    │
                    │                     │
                    │  down1: 7→64       ─┤── skip₁ (64ch)
                    │  down2: 64→128     ─┤── skip₂ (128ch)
                    │  down3: 128→256    ─┤── skip₃ (256ch)
                    │  down4: 256→512    ─┤── skip₄ (512ch)
                    │                     │
                    │  Bottleneck: 512→1024│
                    │                     │
                    │   Decoder (Up)      │
                    │                     │
                    │  up4: 1024+512→512  │◄── skip₄
                    │  up3: 512+256→256   │◄── skip₃
                    │  up2: 256+128→128   │◄── skip₂
                    │  up1: 128+64→64     │◄── skip₁
                    │                     │
                    │  1×1 Conv → Tanh    │
                    └─────────┬──────────┘
                              │
                    Output: 3 channels (RGB)
```

### Component Details

#### DownBlock (Encoder block)

```
Conv2d(3×3) → BatchNorm → LeakyReLU(0.2) → Conv2d(3×3) → BatchNorm → LeakyReLU(0.2) → MaxPool(2×2)
```

- **Conv2d(3×3, padding=1):** 3×3 kernel preserves spatial dimensions. `padding=1` ensures output H,W = input H,W before pooling.
- **BatchNorm2d:** Normalizes activations per-channel across the batch. Stabilizes training, allows higher learning rates, acts as mild regularization.
- **LeakyReLU(0.2):** Unlike ReLU (kills negative values → dead neurons), LeakyReLU lets negative values through at 0.2× slope. Critical in GANs to prevent gradient death. The 0.2 slope is the standard from the DCGAN paper.
- **MaxPool2d(2):** Halves spatial dimensions (256→128→64→32→16). Extracts dominant features.

#### UpBlock (Decoder block)

```
ConvTranspose2d(2×2, stride=2) → Concat(skip) → Conv2d(3×3) → BatchNorm → ReLU → Conv2d(3×3) → BatchNorm → ReLU
```

- **ConvTranspose2d:** "Learnable upsampling" — doubles spatial dimensions. Unlike bilinear upsampling, it has learnable parameters so the network decides *how* to fill in detail.
- **Skip connections (the `cat`):** This is the U-Net's superpower. The encoder captures "what" (semantics), the decoder needs "where" (spatial detail). Skip connections pass fine spatial details directly from encoder to decoder, preventing the bottleneck from being an information bottleneck.
- **ReLU (not LeakyReLU):** In the decoder, we're reconstructing — we want clean positive activations.

#### Time Map

```python
time_map = t.view(B, 1, 1, 1).expand(B, 1, H, W)
```

A scalar `t=0.3` becomes a full (1, 256, 256) tensor filled with 0.3, concatenated as the 7th input channel. This tells the network *where in time* to generate.

**Why a spatial map instead of injecting t elsewhere?** The network can learn location-dependent temporal behavior — e.g., a character's arm at position (100, 50) might move differently than the static background at (200, 200). A spatial map lets early convolutions combine spatial + temporal info immediately.

#### Tanh Output

Squashes output to [-1, 1], matching the target normalization. Sigmoid (output [0,1]) would work too with different normalization, but Tanh centered at 0 is standard for image GANs — it keeps gradients healthier.

### Why U-Net Over Alternatives?

| Architecture | Pros | Cons | Why not for our case |
|---|---|---|---|
| **U-Net** ✅ | Skip connections preserve fine detail; proven for image-to-image; efficient | Not the absolute SOTA for video | Best balance of quality, simplicity, and trainability |
| **ResNet-based generator** | Good gradient flow via residual connections | No multi-scale skip connections; loses fine spatial detail | Worse at preserving high-frequency details (edges, textures) |
| **Plain encoder-decoder** (no skips) | Simpler | Loses all spatial detail at bottleneck; produces blurry outputs | Tested extensively — provably worse for image translation (pix2pix paper) |
| **Transformer/ViT** | Captures global context; SOTA on some benchmarks | Huge memory (O(n²)); needs massive datasets (millions); slow training | Our dataset is likely thousands of samples, not millions. Would severely overfit. |
| **RAFT/optical flow-based** | Physically principled; captures motion explicitly | Two-frame only; can't handle occlusions well; needs flow estimation | More complex pipeline; our GAN learns implicit motion |
| **Diffusion model** | Current SOTA image quality | 10-100× slower inference; 10× more training needed; huge memory | Impractical for a personal project on Colab free tier |

---

## 3. Discriminator — PatchGAN

> Source: `model/discriminator.py`

### Architecture Diagram

```
Input: [frame(3ch) + key_first(3ch) + key_last(3ch)] = 9 channels
    │
    Conv2d(4×4, stride=2) → LeakyReLU          64ch, /2
    Conv2d(4×4, stride=2) → BN → LeakyReLU     128ch, /4
    Conv2d(4×4, stride=2) → BN → LeakyReLU     256ch, /8
    Conv2d(4×4, stride=1) → BN → LeakyReLU     512ch
    Conv2d(4×4, stride=1) → output              1ch
    │
    Output: (B, 1, ~16, ~16) grid of real/fake scores
```

### Why "Patch" Instead of Classifying the Whole Image?

A regular discriminator outputs a single real/fake score. PatchGAN outputs a **grid** — each cell classifies a ~70×70 pixel **patch** of the input.

Benefits:

- **Forces local realism:** Each patch must look real independently → catches local artifacts (blurry edges, color bleeding) better.
- **Fewer parameters:** ~2.8M vs potentially 50M+ for a full-image discriminator → faster, less overfitting.
- **Works at any resolution:** Since it's fully convolutional, same weights work for any input size.
- **Acts like a texture/style loss:** Empirically shown (pix2pix paper) to sharpen results, complementing the L1 loss which handles structure.

### Why Conditional (9 Channels, Not 3)?

The discriminator sees `[frame, key_first, key_last]` — not just the frame alone. This makes it **conditional**: it can check if the generated frame is *consistent with these specific keyframes*. Without conditioning, the discriminator just checks "is this a real-looking anime frame?" which isn't useful — it needs to check "does this frame plausibly exist *between these two keyframes*?"

### Architecture Choices

- **4×4 kernels:** Standard for PatchGAN. Slightly larger receptive field than 3×3, proven in DCGAN/pix2pix.
- **stride=2 for downsampling** (instead of MaxPool): GANs learn better spatial features when downsampling is learned via strided convolutions rather than fixed pooling.
- **No BatchNorm in first layer:** Standard practice — BN in the first layer of D destabilizes early training.

### Alternatives Not Chosen

| Discriminator | Why not |
|---|---|
| **Full-image discriminator** | Loses local detail awareness; more parameters; harder to train |
| **Multi-scale discriminator** (pix2pixHD) | Better for high-res (512+); overkill for 256×256; adds complexity |
| **Spectral normalization** (SNGAN) | Could be added as an improvement; BN works fine for our scale |
| **No discriminator** (pure regression) | Outputs would be blurry — adversarial loss is what creates sharpness |

---

## 4. Loss Functions

> Source: `model/losses.py`

The generator is trained with three losses combined:

```
Total_G_Loss = 10.0 × L1 + 1.0 × Perceptual + 1.0 × Adversarial
```

### L1 Reconstruction Loss — `lambda_l1 = 10.0`

```python
L1 = mean(|generated - target|)
```

**Purpose:** Pixel-level accuracy. Ensures the generated frame has roughly correct colors and structure.

**Why L1, not L2 (MSE)?** L2 penalizes large errors quadratically → the network hedges by averaging → **blurry** outputs. L1 penalizes linearly → less blurring, preserves edges better. This is well-established in pix2pix research.

**Why weight 10.0?** L1 is the dominant "structure" signal. Without heavy L1, the GAN generates sharp but structurally wrong frames. The 10:1:1 ratio (L1:perc:adv) ensures structural correctness first, then sharpness.

### Perceptual Loss (VGG) — `lambda_perc = 1.0`

```python
VGG_loss = L1(VGG_features(generated), VGG_features(target))
```

How it works:

1. Takes a pretrained VGG19 (trained on ImageNet — "knows" what objects/textures look like)
2. Extracts features up to `relu3_4` (layer 16) — mid-level features capturing textures, edges, shapes
3. Compares the feature maps of generated vs target using L1

**Why this is important:** Two images can look very similar to humans but have high pixel-level L1. For example, a frame shifted by 1 pixel has huge L1 but looks identical. Perceptual loss compares *features* (edges, textures, patterns) instead of raw pixels — it says "do these images *look* similar?" rather than "are these pixels identical?"

**Why VGG19 specifically?**

- Pretrained on 1.2M diverse images → excellent general-purpose feature extractor
- Well-studied in style transfer and super-resolution literature
- `relu3_4` specifically captures mid-level features (texture + structure). Earlier layers = edges only, later layers = semantic (too abstract).

**Alternatives not chosen:**

| Loss | Why not |
|---|---|
| **LPIPS** (Learned Perceptual Image Patch Similarity) | Slightly better perceptual metric but adds another dependency; VGG perceptual loss is simpler and nearly as good |
| **SSIM loss** | Good structural comparison but gradient is noisy; doesn't capture texture quality |
| **No perceptual loss** | Results lose texture detail and look "flat" |
| **CLIP-based loss** | Too semantic (understands meaning, not pixel relationships); overkill for frame interpolation |

### LSGAN Adversarial Loss — `lambda_adv = 1.0`

```python
D_loss = MSE(D(real), 1) + MSE(D(fake), 0)   # Train discriminator
G_loss = MSE(D(fake), 1)                       # Train generator (fool D)
```

**Why Least Squares (LSGAN) instead of vanilla GAN (BCE)?**

| | Vanilla GAN (BCE) | LSGAN (MSE) ✅ | WGAN-GP |
|---|---|---|---|
| Gradient signal | Saturates when D is confident → vanishing gradients | Always provides gradient (quadratic penalty for being far from target) | Continuous gradient via Wasserstein distance |
| Mode collapse | Common | Less common | Least common |
| Training stability | Fragile | Stable | Most stable but slower |
| Complexity | Simple | Simple ✅ | Needs gradient penalty computation; 2-5× more D updates |

**LSGAN is the sweet spot:** Almost as stable as WGAN-GP, much simpler to implement, and proven in image-to-image tasks.

---

## 5. Dataset & Training Pipeline

### InbetweenDataset (`model/dataset.py`)

**Directory structure expected:**

```
output/
  video_name/
    shot_NNN/
      segment_NNN/
        key_first.png
        key_last.png
        inbetweens/
          frame_0001.png
          frame_0002.png
          ...
```

**The `t` value computation:**

```python
t = (i + 1) / (n + 1)
```

For a segment with 5 inbetween frames, the values are: `t = 1/6, 2/6, 3/6, 4/6, 5/6`. This means `t=0` is exactly the first keyframe and `t=1` is exactly the last — the network never trains on those exact values, only on positions between them. This is correct because at `t=0` and `t=1` the answer is trivially the keyframe itself.

**Each sample returns:**

| Key | Shape | Description |
|---|---|---|
| `key_first` | (3, 256, 256) | First keyframe, normalized to [-1, 1] |
| `key_last` | (3, 256, 256) | Last keyframe, normalized to [-1, 1] |
| `target` | (3, 256, 256) | Ground truth inbetween frame |
| `t` | (1,) | Temporal position in [0, 1] |

### Training Loop (`model/train.py`)

Key choices:

- **Adam optimizer with betas=(0.5, 0.999):** Standard for GANs. The default `beta1=0.9` causes momentum to overshoot in the adversarial game. `0.5` is less aggressive → more stable.
- **CosineAnnealingLR scheduler:** Slowly decays LR following a cosine curve to `eta_min=1e-6`. Smoother than step decay; avoids sudden jumps that can destabilize GANs.
- **Alternating D/G training (1:1 ratio):** Each batch: freeze G → train D → freeze D → train G. Standard GAN protocol. The 1:1 ratio works with LSGAN; WGAN would need 5 D steps per G step.
- **90/10 train/val split with seed=42:** Reproducible split for consistent evaluation.

**DataLoader settings:**

```python
DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
```

- `pin_memory=True` — Pre-allocates tensors in page-locked memory → faster CPU→GPU transfer.
- `drop_last=True` — Drops incomplete final batch. BatchNorm breaks with batch_size=1.
- `num_workers=4` — 4 parallel processes load and preprocess images while GPU trains → no I/O bottleneck.

### Checkpointing & Resumption

Checkpoints save every N epochs to `training_output/checkpoints/` containing:

- Generator state dict
- Discriminator state dict
- Both optimizer state dicts
- Current epoch number

A `latest.pt` symlink always points to the most recent checkpoint for easy resumption.

---

## 6. Summary: Why This Specific Combination

```
U-Net Generator + PatchGAN Discriminator + (L1 + VGG Perceptual + LSGAN) Loss
```

This is essentially the **pix2pix** architecture adapted for temporal frame interpolation (with the `t` time conditioning added). pix2pix is the most battle-tested image-to-image translation framework for projects at this scale because:

1. **Trainable on small-medium datasets** (thousands of samples) — unlike transformers/diffusion which need millions.
2. **Trainable on a single T4 GPU** in hours — unlike diffusion models needing days on A100s.
3. **Fast inference** — single forward pass, no iterative denoising.
4. **Well-understood** — extensive literature on tuning, failure modes, and fixes.
5. **Good enough quality** for anime frame interpolation — the style is forgiving of minor imperfections.

### What Would Change for a Production/SOTA System

| Current | Upgrade to | When it's worth it |
|---|---|---|
| U-Net | Flow-based model (RIFE, IFRNet) | When you need explicit motion modeling and have compute |
| Single-scale PatchGAN | Multi-scale discriminator | When training at 512+ resolution |
| LSGAN | WGAN-GP or hinge loss | If training becomes unstable at scale |
| VGG perceptual | LPIPS | When perceptual quality is the primary metric |
| Entire GAN pipeline | Diffusion model (FILM, etc.) | When you have 10-100× more compute and data |

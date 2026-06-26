# InBetween Frame Model Training

Generate an animation inbetweening dataset from videos, then train the GAN model.

This repo is optimized for sharing on GitHub and running training through Docker.

## Quick Start (Docker Training)

Prerequisites:

- Docker Engine 24+
- Docker Compose v2 (`docker compose`)

1. Clone the repository.
2. Place your prepared dataset in `./output` (same structure produced by `main.py`).
3. Build the training image:

```bash
docker compose build --no-cache train
```

> Use `--no-cache` to ensure a clean build. The image installs CPU-only PyTorch and training dependencies (no system packages required).

4. Start training:

```bash
docker compose run --rm train
```

Training artifacts are written to `./training_output` on your host.
### GPU Training Image

If your peer has an NVIDIA GPU and the NVIDIA Container Toolkit installed, build the GPU image instead:

```bash
docker compose build --no-cache train-gpu
```

Then run training with the GPU service (use `--gpus all` at runtime):

```bash
docker compose run --rm --gpus all train-gpu
```

The dataset should be mounted by your peer into `./output` on the host (or via `DATASET_DIR` override).

#### Push the GPU image to a Docker registry

Build and tag the image:

```bash
docker compose build --no-cache train-gpu
docker tag shot-segregation-train:gpu <your-docker-repo>/shot-segregation-train:gpu
```

Push it to your repository:

```bash
docker push <your-docker-repo>/shot-segregation-train:gpu
```

Then your peer can pull and run it with the same mounted dataset/output volumes.
Optional path overrides:

```bash
cp .env.example .env
# edit DATASET_DIR / TRAINING_DIR as needed
```

### Run with custom training options

Pass any `model.train` arguments after the service name (these override the defaults in `CMD`):

```bash
docker compose run --rm train \
  --epochs 50 \
  --batch-size 2 \
  --image-size 128 \
  --num-workers 0 \
  --save-every 5 \
  --sample-every 5
```

Default CMD runs `--fast-cpu` mode (image-size 64, batch-size 4, base-features 32, no perceptual loss) which is safe for CPU-only environments.

### Monitor training logs

```bash
# Run in background
docker compose run -d --name training train

# Follow logs live
docker logs -f training

# Or read log files written to disk
tail -f training_output/logs/$(ls -t training_output/logs/ | head -1)
```

## Repository Layout

- `main.py`: dataset creation pipeline (video -> shots -> keyframes -> segments)
- `model/train.py`: GAN training entrypoint
- `docker-compose.yml`: one-command training workflow
- `Dockerfile`: CPU-only training image (Python 3.11-slim, no system package installs)

## Create Dataset (Optional)

If your collaborators need to generate the dataset first:

```bash
python main.py --input ./videos --output ./output
```

Validate produced samples:

```bash
python -m src.validate_dataset ./output
```

## Local Python Training (No Docker)

```bash
bash setup_and_train.sh --dataset ./output --output ./training_output
```

## GitHub Sharing Notes

This project intentionally excludes heavy/generated files from git:

- datasets (`output/`)
- checkpoints and logs (`training_output/`, `test_training/`, `*.pt`)
- local environments (`venv/`)
- downloaded wheel cache (`wheels/`)
- local media archives (`*.mkv`, `*.mp4`, etc.)

Recommended before first push:

```bash
bash github_preflight.sh
git init
git add .
git status
```

If `git status` shows huge binaries, remove them from tracking before commit.

## Training Data Structure

Expected dataset layout:

```text
output/
  video_name/
    shot_001/
      segment_001/
        key_first.png
        key_last.png
        inbetweens/
          frame_0001.png
          ...
```

## Important Model Note

Perceptual loss uses VGG19. If `weights/vgg19-dcbb9e9d.pth` is not present, torchvision will download weights automatically when perceptual loss is enabled (`--lambda-perc > 0`).

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
docker compose build train
```

4. Start training:

```bash
docker compose run --rm train
```

Training artifacts are written to `./training_output` on your host.

Optional path overrides:

```bash
cp .env.example .env
# edit DATASET_DIR / TRAINING_DIR as needed
```

### Run with custom training options

Pass any `model.train` arguments after the service name:

```bash
docker compose run --rm train \
  --epochs 50 \
  --batch-size 2 \
  --image-size 128 \
  --save-every 5 \
  --sample-every 5
```

## Repository Layout

- `main.py`: dataset creation pipeline (video -> shots -> keyframes -> segments)
- `model/train.py`: GAN training entrypoint
- `docker-compose.yml`: one-command training workflow
- `Dockerfile`: CPU training image

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

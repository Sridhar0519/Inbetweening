# Shot Segregation GPU Docker Training

This guide explains how to use the GPU-enabled Docker image after it has been pulled.

## What is included

- project code and Python dependencies
- GPU-capable PyTorch installation
- training entrypoint `python -u -m model.train`

## What is NOT included

- dataset files
- training output and checkpoints

> The dataset must be provided by the peer and mounted into the container at runtime.

## Run training with GPU access

After your friend pulls the image, run it with GPU access and mounted dataset/output folders:

```bash
docker run --rm --gpus all \
  -v /path/to/dataset:/workspace/output:ro \
  -v /path/to/training_output:/workspace/training_output \
  -v /path/to/weights:/workspace/weights:ro \
  sridhar0519/shot-segregation-train:gpu \
  --dataset /workspace/output \
  --output /workspace/training_output \
  --epochs 100 \
  --batch-size 4 \
  --image-size 128 \
  --num-workers 2
```

### Recommended host layout

- dataset: `/path/to/dataset`
- output/checkpoints: `/path/to/training_output`
- weights: `/path/to/weights`

If the dataset or output paths differ, update the host paths in the `docker run` command.

## Default mount paths inside container

- `/workspace/output` → dataset (read-only)
- `/workspace/training_output` → training output and checkpoints
- `/workspace/weights` → local weights directory

## Overriding training arguments

Any arguments after the image name are forwarded to `model.train`.

Example:

```bash
docker run --rm --gpus all \
  -v /path/to/dataset:/workspace/output:ro \
  -v /path/to/training_output:/workspace/training_output \
  -v /path/to/weights:/workspace/weights:ro \
  sridhar0519/shot-segregation-train:gpu \
  --epochs 50 \
  --batch-size 4 \
  --image-size 128 \
  --num-workers 2 \
  --save-every 5 \
  --sample-every 5
```

## Notes for peers

- Ensure NVIDIA Container Toolkit is installed and working
- Ensure host GPU drivers are compatible with CUDA 12.1
- The container will use GPU only when launched with `--gpus all`
- If GPU flags are omitted, training falls back to CPU

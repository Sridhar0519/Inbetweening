FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

COPY requirements_train.txt /tmp/requirements_train.txt

# CPU-only PyTorch + training dependencies.
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && python -m pip install -r /tmp/requirements_train.txt

COPY model /workspace/model

ENTRYPOINT ["python", "-u", "-m", "model.train"]
CMD ["--dataset", "/workspace/output", "--output", "/workspace/training_output", "--epochs", "100", "--batch-size", "4", "--image-size", "128", "--num-workers", "0", "--fast-cpu"]

# PyTorch Object Detection with Faster R-CNN

This repository contains an implementation of object detection using Faster R-CNN with MobileNetV3 backbone on the Pascal VOC dataset. The implementation uses PyTorch and supports model checkpointing, training resumption, and evaluation metrics.

## Features

- Custom VOC dataset implementation
- Faster R-CNN with MobileNetV3 backbone
- Training and evaluation pipeline
- Model checkpointing and resumption
- Mean Average Precision (mAP) evaluation
- Learning rate scheduling

## Requirements

```
torch
torchvision
torchmetrics
huggingface_hub
tqdm
```

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Dataset

The project uses the Pascal VOC dataset. The dataset will be automatically downloaded when running the training script. The VOC classes included are:

```python
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
```

## Project Structure

```
.
├── data/                  # Dataset directory
├── checkpoints/          # Model checkpoints
├── main.py              # Main training script
└── README.md
```

## Usage

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Run the training:
```bash
python main.py
```

The script will:
- Download the VOC dataset if not present
- Initialize the model
- Train for the specified number of epochs
- Save checkpoints during training
- Evaluate model performance using mAP

## Training Configuration

Key training parameters (can be modified in `main.py`):
- Batch size: 1
- Learning rate: 0.005
- Epochs: 10
- Optimizer: SGD with momentum (0.9)
- Learning rate scheduler: StepLR (step_size=3, gamma=0.1)

## Model Checkpoints

The training script saves two types of checkpoints in the `checkpoints` directory:
- `best_model.pth`: Model with the highest validation mAP
- `last_checkpoint.pth`: Latest model state (for training resumption)

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Scheduler state dict
- Current epoch
- Best mAP score

## Evaluation

The model is evaluated using Mean Average Precision (mAP) on the validation set after each epoch. The evaluation metrics are computed using the `torchmetrics.detection.MeanAveragePrecision` class.

## Acknowledgments

- Implementation based on PyTorch's object detection tutorial
- Dataset: Pascal VOC

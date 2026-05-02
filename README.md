# Diffusion-Based Multi-Frame Super-Resolution with Implicit Alignment

This repository contains the official implementation for diffusion-based multi-frame super-resolution with implicit alignment and FOMAML meta-learning.

## Overview

We propose a conditional diffusion model for multi-frame super-resolution that leverages:
- **Implicit Alignment**: A learnable window-based attention module that aligns burst frames without explicit optical flow
- **Extra Feature Extraction**: A FrameEncoder + ImplicitAligner pipeline that extracts rich multi-frame features as additional conditioning
- **Composite Loss**: A five-term training objective combining diffusion loss, color consistency, VGG perceptual loss, Sobel edge loss, and MS-SSIM
- **FOMAML Meta-Learning**: Cross-subject first-order MAML for improved generalization across diverse scenes

## Requirements

- Python 3.8+
- PyTorch >= 2.0
- CUDA-capable GPU (recommended: 24GB+ VRAM)

### Installation

```bash
pip install -r requirements.txt
pip install -e .
```

Key dependencies: `torch`, `torchvision`, `einops`, `lpips`, `pytorch-msssim`, `tensorboard`, `scikit-image`, `opencv-python`

## Dataset and Pretrained Weights

The dataset and pretrained weights are available on Google Drive:

[https://drive.google.com/drive/folders/1K1u0Lut5eNGW1jyjhmUVge4uiW_qld4W?usp=sharing](https://drive.google.com/drive/folders/1K1u0Lut5eNGW1jyjhmUVge4uiW_qld4W?usp=sharing)

After downloading:
- Extract the dataset into the `data/` directory (see structure below)
- Place the pretrained checkpoint (e.g., `checkpoint.pt`) in the project root

## Dataset Preparation

Prepare your dataset in the following structure:

```
data/
  train/
    LR_aligned/
      scene_001/
        frame_00.png
        frame_01.png
        ...
        frame_19.png
        gt.png          # last image is the ground truth
      scene_002/
        ...
  test/
    LR_aligned/
      scene_001/
        ...
```

Each scene folder should contain at least 6 PNG images. The last image in sorted order is treated as the ground truth (GT). All preceding images are low-resolution (LR) burst frames.

## Quick Start: Smoke Test

Run the demo script to verify the pipeline works without any dataset:

```bash
python demo_sr.py
```

This creates synthetic data (HR=64, LR=16) and tests the full pipeline including training loss computation and DDIM sampling.

## Training

### Train from Scratch

```bash
python train.py
```

Key parameters (modify in `train.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset_root` | `data` | Path to dataset |
| `image_size` | `160` | Training patch size |
| `batch_size` | `4` | Batch size |
| `burst_size` | `20` | Number of burst frames for conditioning |
| `extra_burst_size` | `20` | Number of frames for extra feature extraction |
| `diffusion_steps` | `500` | Total diffusion timesteps |
| `epochs` | `10000` | Total training epochs |
| `use_meta_learning` | `True` | Enable FOMAML meta-learning |

The training script:
1. Trains a UNet-based diffusion model conditioned on burst frames
2. Uses FrameEncoder + ImplicitAligner for extra feature extraction
3. Applies composite loss (diffusion + color + perceptual + texture + MS-SSIM)
4. Optionally uses FOMAML meta-learning for cross-subject generalization
5. Logs to TensorBoard under `runs/vscode_tensorboard/`
6. Saves checkpoints every 100 epochs to `outdoor.pt`

### Resume Training

Set `resume_path` in `train.py` to your checkpoint path:

```python
resume_path = "outdoor.pt"
```

### Monitor Training

```bash
tensorboard --logdir runs/
```

## Evaluation

### Evaluate with Pretrained Model

```bash
python evaluate.py --weights checkpoint.pt --dataset-root data --split test
```

Available arguments:

```
--weights          Path to model checkpoint (default: checkpoint.pt)
--dataset-root     Path to dataset root (default: data)
--split            Dataset split to evaluate (default: test)
--image-size       Image size (default: 160)
--batch-size       Batch size (default: 1)
--burst-size       Number of burst frames (default: 20)
--extra-burst-size Number of extra burst frames (default: 20)
--diffusion-steps  Diffusion steps (default: 500)
--seed             Random seed (default: 1234)
--max-batches      Limit number of batches (-1 for all)
--out-dir          Output directory for images
```

The evaluation script outputs:
- Per-sample PSNR and L1 metrics
- Reference, output, and GT images saved to `eval_outputs/`

## Using Pretrained Models

1. Place your pretrained checkpoint (e.g., `outdoor.pt`) in the project root
2. Run evaluation:

```bash
python evaluate.py --weights outdoor.pt --dataset-root outdoor
```

## Model Architecture

- **Backbone**: Modified UNet from guided-diffusion with burst conditioning
- **Channels**: 64 (lightweight) or 256 (full)
- **Attention**: Multi-head attention at resolutions 40x40 and 20x20
- **Noise Schedule**: Sigmoid schedule with 500 diffusion steps
- **Sampling**: DDIM with 50 steps (configurable)
- **Conditioning**: Burst frames processed through SimpleConditionFeature + extra features from FrameEncoder/ImplicitAligner

## Project Structure

```
.
├── train.py                      # Main training script with meta-learning
├── evaluate.py                   # Evaluation script with metric computation
├── composite_loss.py             # Composite loss + FOMAML meta-learning
├── demo_sr.py                    # Quick smoke test (no dataset needed)
├── guided_diffusion/             # Core diffusion framework
│   ├── gaussian_diffusion.py     # Diffusion process (forward/reverse)
│   ├── unet.py                   # UNet model with burst conditioning
│   ├── script_util.py            # Model creation utilities
│   ├── respace.py                # Timestep respacing (DDIM)
│   ├── train_util.py             # Training loop utilities
│   ├── image_datasets.py         # Dataset loading
│   └── ...                       # Other utilities
├── model/                        # Feature extraction and loss modules
│   ├── condition_module.py       # Burst feature extraction (EDA-based)
│   └── mssim.py                  # MS-SSIM loss implementation
├── setup.py                      # Package setup
├── requirements.txt              # Python dependencies
└── README.md
```

## Acknowledgements

This implementation builds upon:
- [guided-diffusion](https://github.com/openai/guided-diffusion) (OpenAI)

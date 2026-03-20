# Duality AI Offroad Semantic Scene Segmentation

## Overview

This project implements a semantic segmentation model for off-road desert environments as part of the Duality AI Offroad Autonomy Segmentation Challenge. The model is trained on synthetic data generated from Duality AI's Falcon digital twin simulation platform and evaluated on unseen desert environment images.

---

## Model Architecture

The model uses a two-component architecture:

- **Backbone**: DINOv2 ViT-B/14 (`dinov2_vitb14`) pretrained by Meta AI via `frozen`
- **Segmentation Head**: Custom ConvNeXt-style decoder (`ConvNeXtHead`)
- **Input Resolution**: 476*266
- **Output Classes**: 11

### ConvNeXtHead Design

| Layer | Details |
|-------|---------|
| Stem | Conv2d(384, 128, kernel=7) + BatchNorm2d |
| Block | Depthwise Conv2d(128, 128, kernel=7) + BatchNorm2d + GELU + Pointwise Conv + BatchNorm2d |
| Classifier | Conv2d(128, num_classes, kernel=1) |

The backbone features are reshaped from token format (B, N, C) to spatial format (B, C, H, W) before being passed to the segmentation head.

---

## Dataset

- **Source**: Duality AI FalconEditor — synthetic desert environment digital twins
- **Train Set**: 2857 images
- **Validation Set**: 317 images
- **Format**: RGB color images + segmentation masks

### Class Definitions

| Class ID | Class Name |
|----------|------------|
| 0 | Background |
| 100 | Trees |
| 200 | Lush Bushes |
| 300 | Dry Grass |
| 500 | Dry Bushes |
| 550 | Ground Clutter |
| 600 | Flowers |
| 700 | Logs |
| 800 | Rocks |
| 7100 | Landscape |
| 10000 | Sky |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Random Seed | 42 |
| Batch Size | 4 |
| Image Size | 518 x 518 |
| Learning Rate | 3e-4 |
| Optimizer | AdamW |
| Weight Decay | 1e-4 |
| LR Schedule | CosineAnnealingLR |
| Epochs | 50 |
| Loss Function | 0.4 x CrossEntropy + 0.6 x Dice |

### Class Weights

Rare and visually similar classes were assigned higher weights to address class imbalance:

| Class | Weight |
|-------|--------|
| Background | 0.5 |
| Trees | 1.0 |
| Lush Bushes | 1.0 |
| Dry Grass | 1.0 |
| Dry Bushes | 1.0 |
| Ground Clutter | 3.0 |
| Flowers | 2.0 |
| Logs | 5.0 |
| Rocks | 5.0 |
| Landscape | 1.0 |
| Sky | 0.5 |

### Data Augmentation

- Horizontal Flip (p=0.5)
- Random Brightness and Contrast (p=0.3)
- Gaussian Noise (p=0.15)
- Normalization (ImageNet mean/std)

---

## Results

| Metric | Value |
|--------|-------|
| Baseline Val IoU | 0.2924 |
| Final Val IoU | 0.725 |
| Improvement | +0.433 |

### Per-Class IoU

| Class | IoU |
|-------|-----|
| Sky | 0.96 |
| Trees | 0.78 |
| Dry Grass | 0.72 |
| Landscape | 0.81 |
| Flowers | 0.40|
| Lush Bushes | 0.68 |
| Dry Bushes | 0.69 |
| Rocks | 0.66 |
| Ground Clutter | 0.58 |
| Logs | 0.52 |
| Background | 0.0000 |

---

## Failure Cases

- **Logs (IoU: 0.52)**: Low recall due to visual similarity with dry branches and ground clutter. The class occupies very few pixels in most images, making it difficult to segment accurately.
- **Rocks (IoU: 0.66)**: Confused with landscape and dry ground due to similar color and texture in synthetic desert environments.
- **Ground Clutter (IoU: 0.58)**: Highly varied appearance leads to frequent misclassification as landscape or dry grass.
- **Background (IoU: 0.0000)**: Background class has zero or near-zero pixel representation in the dataset, making it impossible to learn.

---

## Setup and Reproduction

### Requirements

```
torch
torchvision
albumentations
segmentation-models-pytorch
timm
Pillow
numpy
tqdm
matplotlib
seaborn
scikit-learn
```

Install all dependencies:

```bash
pip install torch torchvision albumentations segmentation-models-pytorch timm pillow numpy tqdm matplotlib seaborn scikit-learn
```

### Directory Structure

```
project/
├── train_segmentation.py        # Baseline training script
├── train_upgraded.py            # Final upgraded training script
├── test_segmentation.py         # Testing/inference script
├── visualize.py                 # Visualization script
├── Code-duality.ipynb           # Full experiment notebook
├── config.py                    # Training configuration
├── segmentation_head.pth        # Baseline model weights
├── segmentation_head_upgraded.pth  # Final model weights
├── README.md
├── 111.png                      # Sample prediction output
├── Ai-predicted image.jpeg
└── ai-predicted-2.jpeg
```

### Training

```bash
conda activate EDU
python train_deeplabv3.py
```

### Generating Visualizations

```bash
python visualize_all.py
```

### Expected Outputs

- Best model saved to `deeplabv3_results/deeplabv3_best.pth`
- Training curves saved to `deeplabv3_results/training_curves.png`
- Per-class IoU chart saved to `deeplabv3_results/per_class_iou.png`
- All visualizations saved to `visualizations/`

---

## Key Technical Decisions

**Why DeepLabV3+ with ResNet50**: Provides strong multi-scale feature extraction through ASPP (Atrous Spatial Pyramid Pooling) which is particularly effective for outdoor scene segmentation with objects at varying scales.

**Why Combined Loss**: CrossEntropy alone does not optimize directly for IoU. Adding Dice loss (weighted at 0.6) pushes the model to optimize overlap directly, leading to better segmentation boundaries.

**Why Class Weights**: The desert dataset has extreme class imbalance. Logs and Rocks occupy less than 1 percent of pixels in most images. Without higher weights, the model ignores these classes entirely.

**Why Random Seed 42**: Ensures full reproducibility of training, data shuffling, and augmentation across runs.

---

## Team

Duality AI Hackathon — Offroad Segmentation Track
Harshit Mendiratta - AI Engineer
Ayush Rai- AI Engineer
Rakshit Dahiya- Documentation , Designing and Pitching

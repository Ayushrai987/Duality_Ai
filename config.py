# Training Configuration — Creative Codex | Xen-O-Thon 2026

# Model
BACKBONE = "dinov2_vitb14"      # DINOv2 ViT-Base/14
BACKBONE_FROZEN = True
EMBED_DIM = 768
NUM_CLASSES = 10
INPUT_HEIGHT = 266
INPUT_WIDTH = 476

# Training
EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0

# Optimizer & Scheduler
OPTIMIZER = "AdamW"
LR_SCHEDULER = "CosineAnnealingLR"

# Loss Function
LOSS = "Dice + CrossEntropy (50/50)"

# Augmentations
AUGMENTATIONS = [
    "RandomResizedCrop (scale 0.5-1.0)",
    "HorizontalFlip (p=0.5)",
    "VerticalFlip (p=0.2)",
    "RandomBrightnessContrast (p=0.4)",
    "HueSaturationValue (p=0.3)",
    "GaussNoise (p=0.2)",
    "GridDistortion (p=0.2)",
]

# Platform
PLATFORM = "Google Colab (T4/A100 GPU)"

# Results
BASELINE_IOU = 0.29
FINAL_IOU = 0.725

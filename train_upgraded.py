"""
Upgraded Training Script - Faster + Better IoU
Fixes: missing Flowers class, better loss, AdamW, augmentation, faster eval
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from PIL import Image
import cv2
import os
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

plt.switch_backend('Agg')

# FIX 1 - Added missing Flowers (600) class
value_map = {
    0:     0,
    100:   1,
    200:   2,
    300:   3,
    500:   4,
    550:   5,
    600:   6,
    700:   7,
    800:   8,
    7100:  9,
    10000: 10
}
n_classes = len(value_map)

CLASS_NAMES = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass',
    'Dry Bushes', 'Ground Clutter', 'Flowers',
    'Logs', 'Rocks', 'Landscape', 'Sky'
]


def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return new_arr


class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.data_ids = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path  = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask  = np.array(Image.open(mask_path))
        mask  = convert_mask(mask)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask  = augmented['mask']

        return image, mask.long()


def get_train_transform(h, w):
    return A.Compose([
        A.Resize(476, 476),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.RandomRotate90(p=0.2),
        A.GaussNoise(p=0.15),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_val_transform(h, w):
    return A.Compose([
        A.Resize(476, 476),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=7, padding=3),
            nn.GELU()
        )
        self.block = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block(x)
        return self.classifier(x)


def compute_iou(pred, target, num_classes=11):
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)
    iou_per_class = []
    for class_id in range(num_classes):
        pred_inds   = pred == class_id
        target_inds = target == class_id
        intersection = (pred_inds & target_inds).sum().float()
        union        = (pred_inds | target_inds).sum().float()
        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).cpu().numpy())
    return np.nanmean(iou_per_class)


def compute_pixel_accuracy(pred, target):
    pred_classes = torch.argmax(pred, dim=1)
    return (pred_classes == target).float().mean().cpu().numpy()


dice_loss  = smp.losses.DiceLoss(mode='multiclass')
focal_loss = smp.losses.FocalLoss(mode='multiclass', gamma=2.0)


def combined_loss(predictions, targets):
    return dice_loss(predictions, targets) + focal_loss(predictions, targets)


def save_plots(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train', color='coral')
    plt.plot(history['val_loss'],   label='Val',   color='steelblue')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(history['train_iou'], label='Train', color='coral')
    plt.plot(history['val_iou'],   label='Val',   color='steelblue')
    plt.title('IoU Score')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(history['train_acc'], label='Train', color='coral')
    plt.plot(history['val_acc'],   label='Val',   color='steelblue')
    plt.title('Pixel Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves_upgraded.png'), dpi=150)
    plt.close()
    print("Saved training curves")


def save_history(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'results_upgraded.txt'), 'w') as f:
        f.write("UPGRADED MODEL RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Baseline Val IoU:  0.2924\n")
        f.write(f"Best Val IoU:      {max(history['val_iou']):.4f} "
                f"(Epoch {np.argmax(history['val_iou'])+1})\n")
        f.write(f"Final Val IoU:     {history['val_iou'][-1]:.4f}\n\n")
        f.write(f"{'Epoch':<8} {'TrainLoss':<12} {'ValLoss':<12} "
                f"{'TrainIoU':<12} {'ValIoU':<12}\n")
        f.write("-" * 56 + "\n")
        for i in range(len(history['train_loss'])):
            f.write(f"{i+1:<8} {history['train_loss'][i]:<12.4f} "
                    f"{history['val_loss'][i]:<12.4f} "
                    f"{history['train_iou'][i]:<12.4f} "
                    f"{history['val_iou'][i]:<12.4f}\n")
    print("Saved results to results_upgraded.txt")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    batch_size = 2
    w = 476
    h = 476
    lr       = 1e-4
    n_epochs = 50

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'train_stats_upgraded')
    os.makedirs(output_dir, exist_ok=True)

    data_dir = os.path.join(script_dir, '..', 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir  = os.path.join(script_dir, '..', 'Offroad_Segmentation_Training_Dataset', 'val')

    trainset = MaskDataset(data_dir, transform=get_train_transform(h, w))
    valset   = MaskDataset(val_dir,  transform=get_val_transform(h, w))

    train_loader = DataLoader(trainset, batch_size=batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(valset,   batch_size=batch_size,
                              shuffle=False, num_workers=0)

    print(f"Train: {len(trainset)} | Val: {len(valset)}")

    print("Loading DINOv2 backbone...")
    backbone = torch.hub.load(
        repo_or_dir="facebookresearch/dinov2",
        model="dinov2_vits14"
    )
    backbone.eval()
    backbone.to(device)
    print("Backbone loaded!")

    imgs, _ = next(iter(train_loader))
    with torch.no_grad():
        output = backbone.forward_features(imgs.to(device))["x_norm_patchtokens"]
    n_embedding = output.shape[2]
    
    classifier = SegmentationHeadConvNeXt(
        in_channels=n_embedding,
        out_channels=n_classes,
        tokenW=w // 14,
        tokenH=h // 14
    ).to(device)

    # Continue training from best checkpoint
    existing_model_path = os.path.join(script_dir, 'segmentation_head_best.pth')
    if os.path.exists(existing_model_path):
        classifier.load_state_dict(torch.load(existing_model_path, map_location=device))
        print("Loaded existing best model — continuing training!")

    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )

    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou':  [], 'val_iou':  [],
        'train_acc':  [], 'val_acc':  []
    }
    best_iou = 0.0

    print("\nStarting upgraded training...")
    print("=" * 70)

    for epoch in range(n_epochs):
        classifier.train()
        train_losses = []
        epoch_train_ious = []
        epoch_train_accs = []

        for imgs, labels in tqdm(train_loader,
                                  desc=f"Epoch {epoch+1}/{n_epochs} [Train]",
                                  leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.no_grad():
                feats = backbone.forward_features(imgs)["x_norm_patchtokens"]

            logits  = classifier(feats)
            outputs = F.interpolate(logits, size=imgs.shape[2:],
                                    mode="bilinear", align_corners=False)

            loss = combined_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.item())

            with torch.no_grad():
                iou = compute_iou(outputs, labels, num_classes=n_classes)
                acc = compute_pixel_accuracy(outputs, labels)
                epoch_train_ious.append(iou)
                epoch_train_accs.append(acc)

        classifier.eval()
        val_losses = []
        epoch_val_ious = []
        epoch_val_accs = []

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader,
                                      desc=f"Epoch {epoch+1}/{n_epochs} [Val]",
                                      leave=False):
                imgs, labels = imgs.to(device), labels.to(device)
                feats   = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits  = classifier(feats)
                outputs = F.interpolate(logits, size=imgs.shape[2:],
                                        mode="bilinear", align_corners=False)
                loss = combined_loss(outputs, labels)
                val_losses.append(loss.item())

                iou = compute_iou(outputs, labels, num_classes=n_classes)
                acc = compute_pixel_accuracy(outputs, labels)
                epoch_val_ious.append(iou)
                epoch_val_accs.append(acc)

        scheduler.step()

        t_loss    = np.mean(train_losses)
        v_loss    = np.mean(val_losses)
        train_iou = np.nanmean(epoch_train_ious)
        val_iou   = np.nanmean(epoch_val_ious)
        train_acc = np.mean(epoch_train_accs)
        val_acc   = np.mean(epoch_val_accs)

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1:02d}/{n_epochs} | "
              f"Loss {t_loss:.4f}/{v_loss:.4f} | "
              f"IoU {train_iou:.4f}/{val_iou:.4f} | "
              f"Acc {train_acc:.4f}/{val_acc:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(classifier.state_dict(),
                       os.path.join(script_dir, 'segmentation_head_best.pth'))
            print(f"  --> New best! Val IoU: {best_iou:.4f}")

    save_plots(history, output_dir)
    save_history(history, output_dir)
    torch.save(classifier.state_dict(),
               os.path.join(script_dir, 'segmentation_head_upgraded.pth'))

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"Baseline IoU: 0.2924")
    print(f"Best Val IoU: {best_iou:.4f}")
    print(f"Improvement:  +{best_iou - 0.2924:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
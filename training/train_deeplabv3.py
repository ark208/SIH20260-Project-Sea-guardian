import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""
Project DRISHTI - DeepLabV3+ Training & Fine-Tuning Pipeline
Trains on Zenodo Sentinel-1 SAR Oil Spill Dataset with Focal + Dice Loss.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from core.sar_vision.deeplabv3_model import DeepLabV3Plus
from training.dataset import SAROilSpillDataset

class CombinedFocalDiceLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.5):
        super(CombinedFocalDiceLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Cross Entropy
        ce_loss = self.ce(pred, target)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()

        # Multi-class Dice Loss
        pred_softmax = torch.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        dice_total = 0.0

        for c in range(num_classes):
            p_c = pred_softmax[:, c, :, :]
            t_c = (target == c).float()
            intersection = (p_c * t_c).sum()
            union = p_c.sum() + t_c.sum() + 1e-6
            dice_c = 1.0 - (2.0 * intersection / union)
            dice_total += dice_c

        dice_loss = dice_total / num_classes
        return self.alpha * focal_loss + (1.0 - self.alpha) * dice_loss

def calculate_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 3) -> float:
    pred_labels = torch.argmax(pred, dim=1)
    ious = []
    for c in range(1, num_classes): # Calculate for Oil Spill (1) and Lookalike (2)
        intersection = ((pred_labels == c) & (target == c)).sum().item()
        union = ((pred_labels == c) | (target == c)).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return float(np.mean(ious)) if ious else 0.0

def train_model(
    data_dir: str = "data/processed_tiles",
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 0.001,
    save_path: str = "weights/best_deeplabv3_sar.pt"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Using compute device: {device}")

    train_ds = SAROilSpillDataset(data_dir=data_dir, split="train", patch_size=256, augment=True)
    val_ds = SAROilSpillDataset(data_dir=data_dir, split="val", patch_size=256, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = DeepLabV3Plus(in_channels=1, num_classes=3).to(device)
    criterion = CombinedFocalDiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_iou = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        scheduler.step()
        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * imgs.size(0)
                val_iou += calculate_iou(outputs, masks) * imgs.size(0)

        val_loss /= len(val_ds)
        val_iou /= len(val_ds)

        print(f"Epoch [{epoch}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Mean IoU: {val_iou:.4f}")

        if val_iou >= best_val_iou or epoch == epochs:
            best_val_iou = val_iou
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_iou": val_iou,
                "classes": ["Ocean", "OilSpill", "Lookalike"]
            }, save_path)
            print(f"--> Saved checkpoint to {save_path}")

    print(f"[Training Complete] Best Validation Mean IoU: {best_val_iou:.4f}")

if __name__ == "__main__":
    train_model(epochs=3, batch_size=4)

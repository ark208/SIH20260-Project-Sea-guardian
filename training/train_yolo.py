"""
Project DRISHTI - YOLOv11 Training on Zenodo SAR Dataset
Reference: https://github.com/CrystalChanB31/yolo11n_reg_zafu_bikes.git
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ultralytics import YOLO

def train_yolo(
    data_yaml: str = "data/yolo_sar/dataset.yaml",
    epochs: int = 3,
    imgsz: int = 512,
    batch: int = 4,
    save_dir: str = "weights"
):
    print(f"[YOLOv11 Training] Initializing YOLO11n on {data_yaml}...")
    model = YOLO("yolo11n.pt") # Loads YOLOv11 base model
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=save_dir,
        name="yolo11_sar_run",
        exist_ok=True,
        verbose=True
    )
    print(f"[YOLOv11 Training Complete] Checkpoint saved.")

if __name__ == "__main__":
    train_yolo(epochs=2)

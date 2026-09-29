"""
Project DRISHTI - Zenodo to YOLOv11 Dataset Converter
Converts Zenodo SAR segmentation masks into standard YOLO format labels (class_id x_center y_center width height).
"""

import os
import glob
import cv2
import numpy as np

def convert_masks_to_yolo(data_dir: str = "data/processed_tiles", yolo_dir: str = "data/yolo_sar"):
    for split in ["train", "val"]:
        img_src = os.path.join(data_dir, split, "images")
        mask_src = os.path.join(data_dir, split, "masks")

        img_dst = os.path.join(yolo_dir, split, "images")
        lbl_dst = os.path.join(yolo_dir, split, "labels")

        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(lbl_dst, exist_ok=True)

        for img_path in glob.glob(os.path.join(img_src, "*.png")):
            base_name = os.path.basename(img_path)
            mask_path = os.path.join(mask_src, base_name)
            
            # Copy image
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            cv2.imwrite(os.path.join(img_dst, base_name), img)

            # Generate YOLO label file
            label_file = os.path.join(lbl_dst, os.path.splitext(base_name)[0] + ".txt")
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                lines = []
                for class_id in [1, 2]: # 1: OilSpill (YOLO class 0), 2: Lookalike (YOLO class 1)
                    bin_mask = (mask == class_id).astype(np.uint8)
                    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        if cv2.contourArea(cnt) < 50:
                            continue
                        x, y, bw, bh = cv2.boundingRect(cnt)
                        # Normalize to [0, 1]
                        xc = (x + bw / 2.0) / w
                        yc = (y + bh / 2.0) / h
                        nw = bw / w
                        nh = bh / h
                        yolo_cls = 0 if class_id == 1 else 1
                        lines.append(f"{yolo_cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
                
                with open(label_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
            else:
                open(label_file, "w").close()

    # Create dataset.yaml
    yaml_content = f"""path: {os.path.abspath(yolo_dir)}
train: train/images
val: val/images
names:
  0: OilSpill
  1: Lookalike
"""
    with open(os.path.join(yolo_dir, "dataset.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"YOLO format SAR dataset generated at: {yolo_dir}")

if __name__ == "__main__":
    convert_masks_to_yolo()

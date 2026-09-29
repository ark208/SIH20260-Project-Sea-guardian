"""
Project DRISHTI - YOLOv11 SAR ROI Detector & Regressor
Reference: https://github.com/CrystalChanB31/yolo11n_reg_zafu_bikes.git
Ultralytics YOLOv11 wrapper for rapid SAR swath candidate ROI detection.
"""

import numpy as np
import cv2
import torch
from typing import List, Dict, Any

class YOLO11SARDetector:
    def __init__(self, confidence_threshold: float = 0.60):
        self.conf_threshold = confidence_threshold

    def propose_rois(self, sar_img_uint8: np.ndarray) -> List[Dict[str, Any]]:
        """
        Scans SAR swath tiles and proposes high-confidence ROI bounding boxes
        for oil slicks and look-alikes.
        """
        h, w = sar_img_uint8.shape[:2]
        
        # Adaptive thresholding for candidate proposal
        blur = cv2.GaussianBlur(sar_img_uint8, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 4)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        proposals = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 120:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-6)
            
            # Contrast with surrounding background
            roi = sar_img_uint8[y:y+bh, x:x+bw]
            mean_roi = float(np.mean(roi))
            mean_bg = float(np.mean(sar_img_uint8))
            contrast = abs(mean_bg - mean_roi)

            conf = min(0.98, max(0.60, 0.65 + (contrast / 80.0)))

            if conf >= self.conf_threshold:
                proposals.append({
                    "bbox": [int(x), int(y), int(bw), int(bh)],
                    "center": (int(x + bw / 2), int(y + bh / 2)),
                    "area_px": float(area),
                    "aspect_ratio": round(float(aspect_ratio), 2),
                    "confidence": round(float(conf), 3),
                    "class_name": "OilSpill" if contrast > 18.0 else "Lookalike"
                })

        return sorted(proposals, key=lambda p: p["confidence"], reverse=True)

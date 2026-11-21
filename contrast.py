"""Mild luminance-only enhancement for faint type on neutral paper.

No global auto-levels, whitening, or color-channel contrast. Saturated covers are
protected; strong shadows remain the responsibility of shadow_removal.
"""
from __future__ import annotations

import cv2
import numpy as np

from image_config import ImageConfig


def enhance_page_contrast(rgb: np.ndarray, cfg: ImageConfig) -> tuple[np.ndarray, np.ndarray | None]:
    """Return RGB image plus diagnostic mask, or unchanged image/None."""
    if not cfg.contrast_enabled or cfg.contrast_strength <= 0:
        return rgb, None
    h, w = rgb.shape[:2]
    if min(h, w) < 70:
        return rgb, None
    scale = min(1.0, 1100.0 / max(h, w))
    small = (cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))),
                        interpolation=cv2.INTER_AREA) if scale < 1 else rgb)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    lab_small = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
    L_small = lab_small[:, :, 0]
    eligible = ((hsv[:, :, 1].astype(np.float32) < 70) & (L_small > 70)).astype(np.float32)
    if eligible.mean() < 0.07:
        return rgb, None
    smooth_mask = cv2.GaussianBlur(eligible, (0, 0), 2.2)
    if scale < 1:
        smooth_mask = cv2.resize(smooth_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float32)
    blurred = cv2.GaussianBlur(L, (0, 0), sigmaX=1.25)
    detail = np.clip(L - blurred, -18, 18)
    difference = detail * (cfg.contrast_strength * 1.55) * smooth_mask
    if np.max(np.abs(difference)) < 0.35:
        return rgb, None
    lab[:, :, 0] = np.clip(np.round(L + difference), 0, 255).astype(np.uint8)
    corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    changed = np.abs(difference) >= 0.35
    output = rgb.copy()
    output[changed] = corrected[changed]
    return output, np.clip(smooth_mask * 255, 0, 255).astype(np.uint8)

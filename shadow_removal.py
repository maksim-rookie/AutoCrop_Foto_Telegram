"""Gentle luminance-only correction of broad soft shadows on neutral surfaces.

Avoids altering saturated/dark book covers. This cannot recover clipped glare or
hard shadows and deliberately does not turn background into artificial white.
"""
from __future__ import annotations

import cv2
import numpy as np

from image_config import ImageConfig


def reduce_soft_shadows(rgb: np.ndarray, config: ImageConfig) -> tuple[np.ndarray, np.ndarray | None]:
    """Return corrected RGB + small diagnostic gain map (or original if not needed)."""
    if not config.shadow_enabled or config.shadow_strength <= 0:
        return rgb, None
    h, w = rgb.shape[:2]
    scale = min(1.0, 900 / max(w, h))
    small = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
    luminance = lab[:, :, 0].astype(np.float32)
    sigma = max(16.0, 0.10 * max(small.shape[:2]))
    illumination = cv2.GaussianBlur(luminance, (0, 0), sigmaX=sigma, sigmaY=sigma)
    spread = float(np.percentile(illumination, 90) - np.percentile(illumination, 10))
    if spread < config.shadow_min_variation:
        return rgb, None
    reference = float(np.percentile(illumination, 76))
    # Divide by estimated illumination with a strict, only-brightening gain cap.
    gain = np.clip((reference + 1.0) / (illumination + 1.0), 1.0,
                   1.0 + 0.40 * config.shadow_strength)
    # Semantic-free bright neutral mask: leave dark covers and saturated printing.
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    eligible = ((hsv[:, :, 1] < 72) & (luminance > 110)).astype(np.uint8) * 255
    eligible = cv2.morphologyEx(eligible, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # Reject false shadow fields caused solely by a dark book on a uniform table.
    neutral = (hsv[:, :, 1] < 72) & (luminance > 135)
    if np.count_nonzero(neutral) > luminance.size * 0.12:
        spread_neutral = float(np.percentile(luminance[neutral], 90) - np.percentile(luminance[neutral], 10))
        if spread_neutral < config.shadow_min_variation:
            return rgb, None
    blend = cv2.GaussianBlur(eligible.astype(np.float32) / 255.0, (0, 0), sigmaX=3)
    # Avoid hard boundaries from the eligibility mask.
    effective_gain = 1 + (gain - 1) * blend
    if scale < 1:
        effective_gain = cv2.resize(effective_gain, (w, h), interpolation=cv2.INTER_LINEAR)
    original_lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = original_lab[:, :, 0].astype(np.float32)
    enhanced = np.clip(L * effective_gain, 0, 255).astype(np.uint8)
    if config.shadow_clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=config.shadow_clahe_clip, tileGridSize=(8, 8))
        mild = clahe.apply(enhanced)
        enhanced = cv2.addWeighted(enhanced, 0.93, mild, 0.07, 0)
    original_lab[:, :, 0] = enhanced
    corrected = cv2.cvtColor(original_lab, cv2.COLOR_LAB2RGB)
    # Blend to control minor RGB roundtrips for already uniform regions.
    changed = effective_gain > 1.004
    output = rgb.copy()
    output[changed] = corrected[changed]
    return output, np.clip((effective_gain - 1) * 800, 0, 255).astype(np.uint8)

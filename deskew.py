"""Bounded affine deskew and perspective warps with a non-clipping margin."""
from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np

from book_detector import BookDetection, side_lengths
from image_config import ImageConfig


def _padding_pixels(width: float, height: float, padding: Sequence[float]) -> tuple[int, int, int, int]:
    left, top, right, bottom = padding
    # Preserve small surrounding slivers even when requested padding is zero.
    return (round(width * max(0.015, left / 100)), round(height * max(0.015, top / 100)),
            round(width * max(0.015, right / 100)), round(height * max(0.015, bottom / 100)))


def perspective_warp(img: np.ndarray, detection: BookDetection,
                     padding: Sequence[float], config: ImageConfig) -> np.ndarray | None:
    """Warp an observed quad only; reject geometry requiring large extrapolation."""
    quad = detection.corners
    if quad is None or detection.confidence < config.min_confidence:
        return None
    lengths = side_lengths(quad)
    ratios = (max(float(lengths[0]), float(lengths[2])) / max(1.0, min(float(lengths[0]), float(lengths[2]))),
              max(float(lengths[1]), float(lengths[3])) / max(1.0, min(float(lengths[1]), float(lengths[3]))))
    if max(ratios) > config.max_perspective_stretch:
        return None
    width = float((lengths[0] + lengths[2]) * 0.5)
    height = float((lengths[1] + lengths[3]) * 0.5)
    if width < 60 or height < 60 or width / height > 4.5 or height / width > 4.5:
        return None
    l, t, r, b = _padding_pixels(width, height, padding)
    w, h = round(width), round(height)
    output_size = (w + l + r, h + t + b)
    if output_size[0] * output_size[1] > 24_000_000:
        return None
    dest = np.array([[l, t], [l + w - 1, t], [l + w - 1, t + h - 1], [l, t + h - 1]], np.float32)
    homography = cv2.getPerspectiveTransform(quad.astype(np.float32), dest)
    # Check degeneracy/over-stretch at the centre, before performing expensive warp.
    if not np.isfinite(homography).all() or abs(float(np.linalg.det(homography))) < 1e-10:
        return None
    # Outside the source photograph, fill by nearest pixel rather than black strips.
    return cv2.warpPerspective(img, homography, output_size, flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def deskew_crop(img: np.ndarray, detection: BookDetection, padding: Sequence[float],
                config: ImageConfig) -> np.ndarray | None:
    """Rotate the entire source on an expanded canvas, then crop around rotated book."""
    quad = detection.corners
    if quad is None:
        return None
    angle = float(detection.angle_degrees)
    if not 0.7 <= abs(angle) <= config.max_deskew_degrees:
        return None
    # Strongly perspectival quadrilaterals have no reliable single deskew angle.
    if detection.perspective_strength > 0.22:
        return None
    height, width = img.shape[:2]
    cx, cy = width / 2, height / 2
    matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cosine, sine = abs(float(matrix[0, 0])), abs(float(matrix[0, 1]))
    new_w = int(math.ceil(height * sine + width * cosine))
    new_h = int(math.ceil(height * cosine + width * sine))
    if new_w * new_h > 30_000_000:
        return None
    matrix[0, 2] += new_w / 2 - cx
    matrix[1, 2] += new_h / 2 - cy
    transformed = cv2.transform(quad[None, :, :], matrix)[0]
    x0, y0 = transformed.min(axis=0)
    x1, y1 = transformed.max(axis=0)
    l, t, r, b = _padding_pixels(float(x1 - x0), float(y1 - y0), padding)
    left, top = max(0, math.floor(x0) - l), max(0, math.floor(y0) - t)
    right, bottom = min(new_w, math.ceil(x1) + r), min(new_h, math.ceil(y1) + b)
    if right <= left or bottom <= top:
        return None
    rotated = cv2.warpAffine(img, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated[top:bottom, left:right].copy()

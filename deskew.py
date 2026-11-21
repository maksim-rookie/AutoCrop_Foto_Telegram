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


def estimate_page_line_angle(bgr: np.ndarray, cfg: ImageConfig) -> float | None:
    """Find conservative mild rotation from distant, parallel open-page edges.

    Requires a long top edge and a matching edge separated by at least 22%
    of image height; never infers page perspective or text orientation.
    """
    if not cfg.line_deskew_enabled or not cfg.deskew_enabled:
        return None
    h0, w0 = bgr.shape[:2]
    if min(h0, w0) < 160 or cfg.max_line_deskew_degrees < 0.7:
        return None
    scale = min(1.0, cfg.analysis_max_side / max(h0, w0))
    if scale < 1:
        bgr = cv2.resize(bgr, (round(w0 * scale), round(h0 * scale)), interpolation=cv2.INTER_AREA)
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 125)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=48,
                            minLineLength=max(60, round(0.25 * w)),
                            maxLineGap=max(12, round(0.025 * w)))
    if lines is None:
        return None
    candidates = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = math.hypot(dx, dy)
        angle = math.degrees(math.atan2(dy, dx))
        while angle > 90:
            angle -= 180
        while angle < -90:
            angle += 180
        if length >= 0.25 * w and abs(angle) <= cfg.max_line_deskew_degrees:
            candidates.append((0.5 * (y1 + y2), angle, length))
    if not candidates:
        return None
    clusters: list[list[tuple[float, float, float]]] = []
    for candidate in sorted(candidates, key=lambda x: x[0]):
        if clusters and candidate[0] - clusters[-1][-1][0] < 0.055 * h:
            clusters[-1].append(candidate)
        else:
            clusters.append([candidate])
    bands = []
    for group in clusters:
        values = np.array(group)
        med_angle = float(np.median(values[:, 1]))
        support = float(min(np.sum(values[:, 2]), 1.2 * w))
        y = float(np.median(values[:, 0]))
        if support >= 0.25 * w:
            bands.append((y, med_angle, support))
    anchors = [b for b in bands if b[0] < 0.22 * h and b[2] >= 0.48 * w]
    if not anchors:
        return None
    top = max(anchors, key=lambda b: b[2])
    partners = [b for b in bands if b[0] - top[0] >= 0.22 * h and
                abs(b[1] - top[1]) <= 1.25]
    if not partners:
        return None
    partner = min(partners, key=lambda b: (abs(b[1] - top[1]), -b[2]))
    angle = float((top[1] * top[2] + partner[1] * partner[2]) /
                  (top[2] + partner[2]))
    if 0.7 <= abs(angle) <= cfg.max_line_deskew_degrees:
        return angle
    return None


def rotate_keep_all(bgr: np.ndarray, angle: float) -> np.ndarray:
    """Affine rotation with an expanded canvas, preserving corners and spine."""
    h, w = bgr.shape[:2]
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    sine, cosine = abs(rot[0, 1]), abs(rot[0, 0])
    new_w = int(math.ceil(w * cosine + h * sine))
    new_h = int(math.ceil(w * sine + h * cosine))
    if new_w * new_h > 30_000_000:
        raise ValueError('Rotation expansion exceeds safe pixel budget')
    rot[0, 2] += (new_w - w) / 2
    rot[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(bgr, rot, (new_w, new_h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)

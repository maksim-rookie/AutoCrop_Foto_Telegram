"""Conservative quadrilateral detection of one book, on a scaled RGB image.

All coordinates returned by detect_book are in the ORIGINAL image reference frame.
A candidate is not equivalent to confirmed detection; see confidence and reasons.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from image_config import ImageConfig


@dataclass(frozen=True)
class BookDetection:
    corners: np.ndarray | None  # clockwise: TL, TR, BR, BL; original pixels
    confidence: float
    method: str
    reasons: tuple[str, ...]
    angle_degrees: float = 0.0
    perspective_strength: float = 0.0


def order_corners(points: np.ndarray) -> np.ndarray:
    """Order a convex quadrilateral TL/TR/BR/BL without relying on sum ties."""
    coords = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = coords.mean(axis=0)
    angles = np.arctan2(coords[:, 1] - center[1], coords[:, 0] - center[0])
    ordered = coords[np.argsort(angles)]
    # For a convex quadrilateral, top-left has the smallest coordinate sum.
    start = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered, -start, axis=0)
    return ordered.astype(np.float32)


def side_lengths(quad: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)


def geometry_metrics(quad: np.ndarray) -> tuple[float, float]:
    """Return small deskew slope and normalized side-length disparity."""
    top = quad[1] - quad[0]
    bottom = quad[2] - quad[3]
    angles = [math.degrees(math.atan2(float(v[1]), float(v[0]))) for v in (top, bottom)]
    # Only a minor in-plane rotation is safe. Perspective skew gives disagreement.
    slope = sum(angles) / 2
    lengths = side_lengths(quad)
    disparity = max(abs(float(lengths[0] - lengths[2])) / max(float(lengths[0]), float(lengths[2]), 1.0),
                    abs(float(lengths[1] - lengths[3])) / max(float(lengths[1]), float(lengths[3]), 1.0))
    v_top = top / max(np.linalg.norm(top), 1)
    v_bottom = bottom / max(np.linalg.norm(bottom), 1)
    disparity = max(disparity, abs(float(v_top[0] * v_bottom[1] - v_top[1] * v_bottom[0])) * 0.5)
    return slope, disparity


def _side_support(edges: np.ndarray, quad: np.ndarray) -> float:
    """Fraction of edge samples supported by actual Canny pixels."""
    h, w = edges.shape
    t = np.linspace(0.04, 0.96, 64, dtype=np.float32)
    support = []
    for index in range(4):
        a, b = quad[index], quad[(index + 1) % 4]
        xs = np.clip(np.rint(a[0] + t * (b[0] - a[0])).astype(int), 5, w - 6)
        ys = np.clip(np.rint(a[1] + t * (b[1] - a[1])).astype(int), 5, h - 6)
        nearby = np.zeros(len(t), dtype=bool)
        for dy in (-5, -3, 0, 3, 5):
            for dx in (-5, -3, 0, 3, 5):
                nearby |= edges[ys + dy, xs + dx] > 0
        support.append(float(nearby.mean()))
    return float(np.mean(support))


def _score(quad: np.ndarray, contour: np.ndarray, edges: np.ndarray, cfg: ImageConfig,
           method: str) -> float:
    height, width = edges.shape
    area = abs(float(cv2.contourArea(quad)))
    frac = area / (height * width)
    if not cfg.min_book_area <= frac <= cfg.max_book_area:
        return 0.0
    lengths = side_lengths(quad)
    long, short = max(float(lengths.max()), 1.0), max(float(lengths.min()), 1.0)
    if short < 35 or long / short > 5.0:
        return 0.0
    # Reject quadrilaterals crossing the frame; extrapolating hidden corners loses text.
    if (quad[:, 0].min() < width * 0.008 or quad[:, 1].min() < height * 0.008 or
            quad[:, 0].max() > width * 0.992 or quad[:, 1].max() > height * 0.992):
        return 0.0
    contour_area = abs(float(cv2.contourArea(contour)))
    fill = min(1.0, contour_area / max(area, 1.0))
    if fill < 0.84:
        return 0.0
    support = _side_support(edges, quad)
    if support < 0.49:
        return 0.0
    # Larger single objects with sustained boundaries are more believable than text boxes.
    area_quality = min(1.0, frac / 0.25)
    confidence = 0.10 + 0.58 * support + 0.12 * fill + 0.20 * area_quality
    if method == 'minAreaRect':
        confidence -= 0.13  # avoid pretending that an inferred rectangle is observed
    return float(np.clip(confidence, 0, 1))


def _hough_orientation(edges: np.ndarray, quad: np.ndarray) -> float | None:
    """Cross-check book-edge angle using HoughLinesP; ignore unrelated short lines."""
    h, w = edges.shape
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=45,
                            minLineLength=max(30, round(min(h, w) * 0.13)),
                            maxLineGap=16)
    if lines is None:
        return None
    expected, _ = geometry_metrics(quad)
    angles, weights = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        length = math.hypot(float(x2 - x1), float(y2 - y1))
        angle = math.degrees(math.atan2(float(y2 - y1), float(x2 - x1)))
        while angle > 90:
            angle -= 180
        while angle <= -90:
            angle += 180
        if abs(angle - expected) < 8:
            angles.append(angle)
            weights.append(length)
    if not angles:
        return None
    return float(np.average(angles, weights=weights))


def detect_book(rgb: np.ndarray, cfg: ImageConfig) -> tuple[BookDetection, np.ndarray]:
    """Detect book quadrilateral with Canny/morphology/contours/Hough.

    Returns a diagnostic edge map for opt-in debug snapshots. No forced rectangle
    is returned when evidence is weak.
    """
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('detect_book ожидает 8-битное RGB изображение.')
    height, width = rgb.shape[:2]
    scale = min(1.0, cfg.analysis_max_side / max(height, width))
    small = cv2.resize(rgb, (max(1, round(width * scale)), max(1, round(height * scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 42, 120)
    # Color transitions can be more informative than luminance for bright covers.
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
    edges |= cv2.Canny(cv2.GaussianBlur(lab[:, :, 1], (5, 5), 0), 38, 105)
    edges |= cv2.Canny(cv2.GaussianBlur(lab[:, :, 2], (5, 5), 0), 38, 105)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, np.ndarray, str]] = []
    # Cost-bounded: examine only the 40 largest contours.
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:40]:
        if cv2.contourArea(contour) < cfg.min_book_area * gray.size * 0.8:
            continue
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        for eps in (0.012, 0.02, 0.035):
            poly = cv2.approxPolyDP(hull, eps * perimeter, True)
            if len(poly) == 4 and cv2.isContourConvex(poly):
                quad = order_corners(poly)
                score = _score(quad, hull, edges, cfg, 'contour')
                if score:
                    candidates.append((score, quad, 'contour'))
        # Geometric fallback only when the contour itself nearly fills a rectangle.
        rect = cv2.minAreaRect(contour)
        quad = order_corners(cv2.boxPoints(rect))
        score = _score(quad, contour, edges, cfg, 'minAreaRect')
        if score:
            candidates.append((score, quad, 'minAreaRect'))
    if not candidates:
        return BookDetection(None, 0.0, 'none', ('четырёхугольник не подтверждён',)), edges
    confidence, quad, method = max(candidates, key=lambda candidate: candidate[0])
    quad = quad / scale
    angle, perspective = geometry_metrics(quad)
    hough = _hough_orientation(edges, quad * scale)
    reasons: list[str] = []
    if hough is not None and abs(hough - angle) > 5:
        confidence *= 0.80
        reasons.append('угол Hough расходится с контуром')
    if confidence < cfg.min_confidence:
        reasons.append('низкая уверенность геометрии')
    return BookDetection(quad, round(confidence, 4), method, tuple(reasons), angle, perspective), edges

"""CPU-only book photo pipeline; safe fallback to unchanged AutoCrop crop_core.

Only opaque uint8 BGR photos opt in to enhancements. Transparent or high-bit-depth
inputs retain the original v1.1 processing path and pixel representation.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
import uuid
from typing import Sequence

import cv2
import numpy as np

from book_detector import BookDetection, detect_book
from crop import legacy_crop
from deskew import deskew_crop, perspective_warp, estimate_page_line_angle, rotate_keep_all
from contrast import enhance_page_contrast
from image_config import ImageConfig
from shadow_removal import reduce_soft_shadows

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    image: np.ndarray | None  # None means re-use the exact original byte stream
    reason: str
    confidence: float
    mode: str
    fallback_reason: str | None


def _snapshot(folder: Path | None, request: str, stage: str, image: np.ndarray) -> None:
    if folder is None:
        return
    try:
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / f'{request}_{stage}.png'), image)
    except (OSError, cv2.error):
        LOGGER.warning('Could not store CV debug snapshot stage=%s', stage)


def process_bgr(image: np.ndarray, padding: Sequence[float], config: ImageConfig) -> PipelineResult:
    """Safe crop with geometry-independent shadow and contrast processing."""
    start = time.perf_counter()
    folder = Path(config.debug_dir) if config.debug_dir else None
    request = uuid.uuid4().hex[:12]
    if not config.enabled or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        cropped, reason = legacy_crop(image, padding)
        return _finish(PipelineResult(cropped, reason, 0.0, 'legacy',
                        'unsupported image type or disabled'), image, folder, request, start)

    cropped = None
    reason = ''
    mode = 'legacy'
    confidence = 0.0
    fallback: list[str] = []
    detection = None
    if config.deskew_enabled or (config.perspective_enabled and not config.rotation_only):
        try:
            detection, edges = detect_book(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), config)
            confidence = detection.confidence
            _snapshot(folder, request, 'canny', edges)
            LOGGER.info('CV detector method=%s confidence=%.3f angle_deg=%.2f reasons=%s',
                        detection.method, confidence, detection.angle_degrees,
                        ', '.join(detection.reasons) or '-')
            if folder is not None:
                try:
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / f'{request}_metrics.json').write_text(json.dumps({
                        'method': detection.method,
                        'confidence': confidence,
                        'angle_degrees': detection.angle_degrees,
                        'perspective_strength': detection.perspective_strength,
                        'fallback_reasons': list(detection.reasons),
                        'corners': detection.corners.tolist() if detection.corners is not None else None
                    }, ensure_ascii=False, indent=2), encoding='utf-8')
                except OSError:
                    LOGGER.warning('Could not store CV debug metrics')
            if detection.corners is not None:
                overlay = image.copy()
                cv2.polylines(overlay, [np.rint(detection.corners).astype(np.int32)],
                              True, (0, 255, 0), 3)
                _snapshot(folder, request, 'quad', overlay)
        except (cv2.error, ValueError, ArithmeticError) as exc:
            LOGGER.warning('CV detector failed safely: %s', type(exc).__name__)
            fallback.append('ошибка детектора; обычная обрезка')

    if detection is not None and detection.corners is not None and confidence >= config.min_confidence:
        if (config.perspective_enabled and not config.rotation_only and
                detection.perspective_strength >= config.perspective_threshold):
            try:
                cropped = perspective_warp(image, detection, padding, config)
            except (cv2.error, ValueError, ArithmeticError) as exc:
                LOGGER.warning('Perspective skipped: %s', type(exc).__name__)
                fallback.append('perspective: ошибка вычисления')
            if cropped is not None:
                reason = 'перспектива скорректирована'
                mode = 'perspective'
            else:
                fallback.append('перспектива отклонена: риск растяжения')
        if cropped is None and config.deskew_enabled:
            try:
                cropped = deskew_crop(image, detection, padding, config)
            except (cv2.error, ValueError, ArithmeticError) as exc:
                LOGGER.warning('Deskew skipped: %s', type(exc).__name__)
                fallback.append('deskew: ошибка вычисления')
            if cropped is not None:
                reason = 'поворот скорректирован'
                mode = 'deskew'
            elif abs(detection.angle_degrees) > config.max_deskew_degrees:
                fallback.append('поворот отклонён: превышен лимит угла')
    elif detection is not None:
        fallback.append(', '.join(detection.reasons) or 'неуверенная геометрия')

    if cropped is None:
        cropped, reason = legacy_crop(image, padding)
        if (config.deskew_enabled and config.line_deskew_enabled and
                (detection is None or detection.corners is None or confidence < config.min_confidence)):
            try:
                candidate = cropped if cropped is not None else image
                angle = estimate_page_line_angle(candidate, config)
                if angle is not None:
                    cropped = rotate_keep_all(candidate, angle)
                    reason += '; небольшой поворот по границам страниц'
                    mode = 'line_deskew'
                    LOGGER.info('Page-line deskew angle_deg=%.2f', angle)
            except (cv2.error, ValueError, ArithmeticError) as exc:
                LOGGER.warning('Page-line rotation skipped: %s', type(exc).__name__)

    working = cropped if cropped is not None else image
    # Importantly, lighting is independent of quadrilateral confidence/fallback.
    if config.shadow_enabled and config.shadow_strength > 0:
        try:
            output_rgb, gain = reduce_soft_shadows(cv2.cvtColor(working, cv2.COLOR_BGR2RGB), config)
            if gain is not None:
                _snapshot(folder, request, 'shadow_gain', gain)
                working = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
                cropped = working
                reason += '; мягкие тени ослаблены'
        except (cv2.error, ValueError, ArithmeticError) as exc:
            LOGGER.warning('Shadow correction skipped: %s', type(exc).__name__)
    if config.contrast_enabled and config.contrast_strength > 0:
        try:
            output_rgb, mask = enhance_page_contrast(cv2.cvtColor(working, cv2.COLOR_BGR2RGB), config)
            if mask is not None:
                _snapshot(folder, request, 'contrast_mask', mask)
                working = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
                cropped = working
                reason += '; контраст текста слегка усилен'
        except (cv2.error, ValueError, ArithmeticError) as exc:
            LOGGER.warning('Contrast correction skipped: %s', type(exc).__name__)
    return _finish(PipelineResult(cropped, reason, confidence, mode,
                                   '; '.join(fallback) or None), image, folder, request, start)

def _finish(result: PipelineResult, original: np.ndarray, folder: Path | None,
            request: str, start: float) -> PipelineResult:
    """Single exit logger and diagnostic snapshot; never logs private pixels."""
    if result.image is not None:
        _snapshot(folder, request, 'final', result.image)
    LOGGER.info('CV pipeline input=%sx%s dtype=%s mode=%s confidence=%.3f fallback=%s duration_ms=%.1f',
                original.shape[1], original.shape[0], original.dtype, result.mode, result.confidence,
                result.fallback_reason or '-', (time.perf_counter() - start) * 1000)
    return result

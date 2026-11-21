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
from deskew import deskew_crop, perspective_warp
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
    """Process decoded, EXIF-corrected BGR image; no I/O unless debug_dir set.

    The historical crop is the only operation performed when detection is weak.
    """
    start = time.perf_counter()
    folder = Path(config.debug_dir) if config.debug_dir else None
    request = uuid.uuid4().hex[:12]
    result: PipelineResult
    if not config.enabled or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        cropped, reason = legacy_crop(image, padding)
        result = PipelineResult(cropped, reason, 0, 'legacy', 'unsupported image type or disabled')
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detection: BookDetection
        try:
            detection, edges = detect_book(rgb, config)
        except (cv2.error, ValueError, ArithmeticError) as exc:
            LOGGER.warning('Geometric detector failed safely: %s', type(exc).__name__)
            cropped, reason = legacy_crop(image, padding)
            return _finish(PipelineResult(cropped, reason, 0.0, 'legacy',
                                          'ошибка детектора; обычная обрезка'), image, folder, request, start)
        LOGGER.info('CV detector method=%s confidence=%.3f angle_deg=%.2f perspective=%.3f reasons=%s',
                    detection.method, detection.confidence, detection.angle_degrees,
                    detection.perspective_strength, ', '.join(detection.reasons) or '-')
        _snapshot(folder, request, 'canny', edges)
        if folder is not None:
            try:
                folder.mkdir(parents=True, exist_ok=True)
                (folder / f'{request}_metrics.json').write_text(json.dumps({
                    'method': detection.method,
                    'confidence': detection.confidence,
                    'angle_degrees': detection.angle_degrees,
                    'perspective_strength': detection.perspective_strength,
                    'fallback_reasons': list(detection.reasons),
                    'corners': detection.corners.tolist() if detection.corners is not None else None,
                }, ensure_ascii=False, indent=2), encoding='utf-8')
            except OSError:
                LOGGER.warning('Could not store CV debug metrics')
        if detection.corners is not None:
            annotated = image.copy()
            cv2.polylines(annotated, [np.rint(detection.corners).astype(np.int32)], True, (0, 255, 0), 3)
            _snapshot(folder, request, 'quad', annotated)
        if detection.corners is None or detection.confidence < config.min_confidence:
            cropped, reason = legacy_crop(image, padding)
            fallback = ', '.join(detection.reasons) or 'неуверенная детекция'
            result = PipelineResult(cropped, reason, detection.confidence, 'legacy', fallback)
        else:
            corrected = None
            mode = 'legacy'
            rejected: list[str] = []
            if config.perspective_enabled and not config.rotation_only and detection.perspective_strength >= config.perspective_threshold:
                try:
                    corrected = perspective_warp(image, detection, padding, config)
                except (cv2.error, ValueError, ArithmeticError) as exc:
                    LOGGER.warning('Perspective warp failed safely: %s', type(exc).__name__)
                    rejected.append('perspective: ошибка вычисления')
                if corrected is not None:
                    mode = 'perspective'
                else:
                    rejected.append('перспектива отклонена: риск растяжения')
            if corrected is None and config.deskew_enabled:
                try:
                    corrected = deskew_crop(image, detection, padding, config)
                except (cv2.error, ValueError, ArithmeticError) as exc:
                    LOGGER.warning('Deskew failed safely: %s', type(exc).__name__)
                    rejected.append('deskew: ошибка вычисления')
                if corrected is not None:
                    mode = 'deskew'
                elif abs(detection.angle_degrees) > config.max_deskew_degrees:
                    rejected.append('поворот отклонён: превышен лимит угла')
            if corrected is None:
                corrected, reason = legacy_crop(image, padding)
                if corrected is None:
                    # No changes: original bytes must remain identical.
                    result = PipelineResult(None, reason, detection.confidence, 'legacy',
                                            '; '.join(rejected) if rejected else None)
                else:
                    result = PipelineResult(corrected, reason, detection.confidence, 'legacy',
                                            '; '.join(rejected) if rejected else None)
            else:
                result = PipelineResult(corrected, 'геометрия скорректирована', detection.confidence,
                                        mode, '; '.join(rejected) if rejected else None)
            # Shadow removal only on a confidently detected book and a real output.
            if result.image is not None and config.shadow_enabled and config.shadow_strength > 0:
                try:
                    output_rgb, gain = reduce_soft_shadows(cv2.cvtColor(result.image, cv2.COLOR_BGR2RGB), config)
                except (cv2.error, ValueError, ArithmeticError) as exc:
                    LOGGER.warning('Shadow correction skipped safely: %s', type(exc).__name__)
                    output_rgb, gain = None, None
                if gain is not None:
                    _snapshot(folder, request, 'shadow_gain', gain)
                    updated = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
                    result = PipelineResult(updated, result.reason + '; мягкие тени ослаблены',
                                            result.confidence, result.mode, result.fallback_reason)
    return _finish(result, image, folder, request, start)


def _finish(result: PipelineResult, original: np.ndarray, folder: Path | None,
            request: str, start: float) -> PipelineResult:
    """Single exit logger and diagnostic snapshot; never logs private pixels."""
    if result.image is not None:
        _snapshot(folder, request, 'final', result.image)
    LOGGER.info('CV pipeline input=%sx%s dtype=%s mode=%s confidence=%.3f fallback=%s duration_ms=%.1f',
                original.shape[1], original.shape[0], original.dtype, result.mode, result.confidence,
                result.fallback_reason or '-', (time.perf_counter() - start) * 1000)
    return result

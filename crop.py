"""Compatibility wrapper for the historical padding/crop implementation."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from crop_core import crop_plan


def legacy_crop(image: np.ndarray, padding: Sequence[float]) -> tuple[np.ndarray | None, str]:
    """Return a cropped view, or None to retain original bytes without re-encoding."""
    bounds, reason = crop_plan(image, padding)
    if bounds is None:
        return None, reason
    x0, y0, x1, y1 = bounds
    return image[y0:y1, x0:x1].copy(), reason

"""Reproducible synthetic wall-clock benchmark, excluding network and disk IO."""
from __future__ import annotations

from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from image_config import ImageConfig
from pipeline import process_bgr


def main() -> None:
    height, width = 3000, 2200
    image = np.full((height, width, 3), (240, 238, 231), dtype=np.uint8)
    polygon = np.array([(350, 350), (1800, 450), (1900, 2600), (280, 2550)], np.int32)
    cv2.fillConvexPoly(image, polygon, (60, 82, 140))
    cv2.polylines(image, [polygon], True, (28, 32, 45), 7)
    durations = []
    for _ in range(3):
        start = time.perf_counter()
        result = process_bgr(image, (10,) * 4, ImageConfig())
        durations.append(time.perf_counter() - start)
    print(f'Synthetic image: {width}x{height}; mode={result.mode}; '
          f'min={min(durations):.3f}s; median={sorted(durations)[1]:.3f}s; max={max(durations):.3f}s')
    print('Synthetic-only timings; measure with your own photos and machine as well.')


if __name__ == '__main__':
    main()

"""Typed, validated settings for the optional computer-vision pipeline."""
from __future__ import annotations

from dataclasses import dataclass, fields
import math
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ImageConfig:
    """Conservative defaults; legacy crop remains available as a fallback."""

    enabled: bool = True
    deskew_enabled: bool = True
    perspective_enabled: bool = False
    shadow_enabled: bool = True
    rotation_only: bool = True
    line_deskew_enabled: bool = True
    max_line_deskew_degrees: float = 6.0
    analysis_max_side: int = 1400
    min_confidence: float = 0.78
    min_book_area: float = 0.07
    max_book_area: float = 0.91
    max_deskew_degrees: float = 18.0
    perspective_threshold: float = 0.075
    max_perspective_stretch: float = 1.35
    shadow_strength: float = 0.45
    contrast_enabled: bool = True
    contrast_strength: float = 0.22
    shadow_min_variation: float = 12.0
    shadow_clahe_clip: float = 0.0
    debug_dir: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> 'ImageConfig':
        """Reject unknown keys, unsafe types and implausible values."""
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError('IMAGE_PROCESSING должен быть объектом JSON.')
        names = {entry.name for entry in fields(cls)}
        unknown = set(data) - names
        if unknown:
            raise ValueError('IMAGE_PROCESSING содержит неизвестные параметры.')
        sample = cls()
        for key, value in data.items():
            original = getattr(sample, key)
            if isinstance(original, bool):
                if not isinstance(value, bool):
                    raise ValueError(f'IMAGE_PROCESSING.{key} должен быть true/false.')
            elif isinstance(original, int):
                if type(value) is not int:
                    raise ValueError(f'IMAGE_PROCESSING.{key} должен быть целым числом.')
            elif key == 'debug_dir':
                if value is not None and (not isinstance(value, str) or not value.strip() or len(value) > 250):
                    raise ValueError('IMAGE_PROCESSING.debug_dir: нужен путь или null.')
            elif not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f'IMAGE_PROCESSING.{key} должен быть конечным числом.')
        config = cls(**data)
        ranges = {
            'analysis_max_side': (512, 2400),
            'min_confidence': (0.55, 0.99),
            'min_book_area': (0.01, 0.30),
            'max_book_area': (0.50, 0.98),
            'max_deskew_degrees': (0.0, 30.0),
            'perspective_threshold': (0.02, 0.30),
            'max_perspective_stretch': (1.05, 1.70),
            'shadow_strength': (0.0, 1.0),
            'contrast_strength': (0.0, 1.0),
            'max_line_deskew_degrees': (0.0, 8.0),
            'shadow_min_variation': (2.0, 80.0),
            'shadow_clahe_clip': (0.0, 2.5),
        }
        for name, (low, high) in ranges.items():
            if not low <= getattr(config, name) <= high:
                raise ValueError(f'IMAGE_PROCESSING.{name}: значение вне диапазона {low}...{high}.')
        if config.min_book_area >= config.max_book_area:
            raise ValueError('IMAGE_PROCESSING: min_book_area должен быть меньше max_book_area.')
        if config.debug_dir is not None and Path(config.debug_dir).is_file():
            raise ValueError('IMAGE_PROCESSING.debug_dir должен быть каталогом.')
        return config

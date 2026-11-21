"""Synthetic image regressions; no network or external photo corpus needed."""
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bot
from book_detector import BookDetection, detect_book
from image_config import ImageConfig
from pipeline import process_bgr
from shadow_removal import reduce_soft_shadows


def synthetic_book(points: list[tuple[int, int]]) -> np.ndarray:
    """Single opaque colorful cover, thick edges, a little print and clean table."""
    image = np.full((620, 830, 3), (233, 225, 218), np.uint8)
    quad = np.array(points, np.int32)
    cv2.fillConvexPoly(image, quad, (48, 76, 121))
    cv2.polylines(image, [quad], True, (15, 28, 32), 4)
    # These lines are outside the book shape in some test variants and intentionally
    # should not force an orientation if they disagree with the actual perimeter.
    return image


class ConfigTests(unittest.TestCase):
    def test_flags_and_manual_rotation_only(self):
        cfg = ImageConfig.from_mapping({'rotation_only': True, 'perspective_enabled': False,
                                        'shadow_strength': 0.2, 'analysis_max_side': 900})
        self.assertTrue(cfg.rotation_only)
        self.assertFalse(cfg.perspective_enabled)
        self.assertEqual(cfg.analysis_max_side, 900)

    def test_bad_values_are_rejected_without_leaking_source_data(self):
        for mapping in ({'deskew_enabled': 1}, {'shadow_strength': float('nan')},
                        {'analysis_max_side': 10}, {'token': 'SECRET'},
                        {'shadow_enabled': 'False'}):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError) as found:
                ImageConfig.from_mapping(mapping)
            self.assertNotIn('SECRET', str(found.exception))


class GeometryTests(unittest.TestCase):
    def test_deskew_rotated_book(self):
        image = synthetic_book([(245, 95), (540, 150), (515, 565), (220, 510)])
        result = process_bgr(image, (10, 10, 10, 10), ImageConfig(shadow_enabled=False))
        self.assertEqual(result.mode, 'deskew')
        self.assertGreaterEqual(result.confidence, 0.78)
        self.assertIsNotNone(result.image)
        self.assertLess(result.image.shape[0], image.shape[0])
        found, _ = detect_book(cv2.cvtColor(result.image, cv2.COLOR_BGR2RGB),
                               ImageConfig(shadow_enabled=False))
        self.assertIsNotNone(found.corners)
        self.assertLess(abs(found.angle_degrees), 2.0)

    def test_perspective_is_conditional(self):
        image = synthetic_book([(200, 120), (610, 95), (555, 510), (230, 535)])
        enabled = ImageConfig(shadow_enabled=False)
        result = process_bgr(image, (5, 5, 5, 5), enabled)
        self.assertEqual(result.mode, 'perspective')
        self.assertLess(result.image.shape[1], image.shape[1])
        self.assertGreaterEqual(result.confidence, enabled.min_confidence)
        # Manual mode must not apply a projective transform, even for same input.
        rotation = process_bgr(image, (5,) * 4, ImageConfig(shadow_enabled=False, rotation_only=True))
        self.assertNotEqual(rotation.mode, 'perspective')

    def test_stretch_guard_never_forces_perspective(self):
        image = synthetic_book([(200, 120), (610, 95), (555, 510), (230, 535)])
        setting = ImageConfig(shadow_enabled=False, max_perspective_stretch=1.05)
        result = process_bgr(image, (5,) * 4, setting)
        self.assertNotEqual(result.mode, 'perspective')
        self.assertIn('перспектива отклонена', result.fallback_reason or '')

    def test_no_confidence_uses_exact_legacy_crop(self):
        image = synthetic_book([(240, 110), (580, 110), (580, 520), (240, 520)])
        fake = BookDetection(None, 0.0, 'none', ('нет контура',))
        with patch('pipeline.detect_book', return_value=(fake, np.zeros(image.shape[:2], np.uint8))):
            actual = process_bgr(image, (10,) * 4, ImageConfig())
        expected = process_bgr(image, (10,) * 4, ImageConfig(enabled=False))
        np.testing.assert_array_equal(actual.image, expected.image)
        self.assertEqual(actual.mode, 'legacy')
        self.assertEqual(actual.fallback_reason, 'нет контура')

    def test_detector_exception_does_not_destroy_image(self):
        image = synthetic_book([(240, 110), (580, 110), (580, 520), (240, 520)])
        with patch('pipeline.detect_book', side_effect=cv2.error('synthetic failure')):
            actual = process_bgr(image, (10,) * 4, ImageConfig())
        expected = process_bgr(image, (10,) * 4, ImageConfig(enabled=False))
        np.testing.assert_array_equal(actual.image, expected.image)
        self.assertEqual(actual.mode, 'legacy')
        self.assertIn('ошибка детектора', actual.fallback_reason or '')

    def test_debug_writes_only_when_explicitly_enabled(self):
        image = synthetic_book([(245, 95), (540, 150), (515, 565), (220, 510)])
        with tempfile.TemporaryDirectory() as folder:
            result = process_bgr(image, (5,) * 4, ImageConfig(shadow_enabled=False, debug_dir=folder))
            self.assertEqual(result.mode, 'deskew')
            stages = {path.stem.rsplit('_', 1)[-1] for path in Path(folder).glob('*.png')}
            self.assertIn('canny', stages)
            self.assertIn('quad', stages)
            self.assertIn('final', stages)

    def test_all_enhancements_off_is_identical_to_original_v11(self):
        image = synthetic_book([(240, 110), (580, 110), (580, 520), (240, 520)])
        ok, encoded = cv2.imencode('.png', image)
        self.assertTrue(ok)
        legacy_result, legacy_name, _ = bot.process_image(encoded.tobytes(), 'book.png', (10,) * 4,
            ImageConfig(enabled=False))
        expected = process_bgr(image, (10,) * 4, ImageConfig(enabled=False)).image
        ok, expected_bytes = cv2.imencode('.png', expected)
        self.assertEqual(legacy_name, 'cropped.png')
        self.assertEqual(legacy_result, expected_bytes.tobytes())


class ShadowTests(unittest.TestCase):
    def test_soft_shadow_is_reduced_without_repainting_cover(self):
        h, w = 500, 640
        source = np.tile(np.array([238, 232, 225], np.float32), (h, w, 1))
        source -= np.linspace(0, 65, w, dtype=np.float32)[None, :, None]
        source = np.clip(source, 0, 255).astype(np.uint8)
        source[100:210, 80:180] = [65, 75, 175]
        cv2.putText(source, 'Printed text', (250, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (34, 34, 34), 2)
        result, field = reduce_soft_shadows(source, ImageConfig(shadow_strength=0.35))
        self.assertIsNotNone(field)
        original_difference = abs(float(source[300, 30, 0]) - float(source[300, 600, 0]))
        result_difference = abs(float(result[300, 30, 0]) - float(result[300, 600, 0]))
        self.assertLess(result_difference, original_difference * 0.75)
        np.testing.assert_array_equal(result[120, 120], source[120, 120])
        self.assertLess(int(result[252, 260, 0]), 85)

    def test_uniform_background_is_not_changed(self):
        image = synthetic_book([(240, 110), (580, 110), (580, 520), (240, 520)])
        result, field = reduce_soft_shadows(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), ImageConfig())
        self.assertIsNone(field)
        np.testing.assert_array_equal(result, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


if __name__ == '__main__':
    unittest.main()

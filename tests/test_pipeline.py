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
from book_detector import BookDetection, detect_book, _hough_orientation
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
    def test_hough_array_layouts_from_different_opencv_versions(self):
        edges = np.zeros((200, 200), np.uint8)
        quad = np.array([[30, 30], [170, 30], [170, 170], [30, 170]], np.float32)
        line = np.array([[30, 30, 170, 30]], np.int32)
        with patch('book_detector.cv2.HoughLinesP', return_value=line):
            flat = _hough_orientation(edges, quad)
        with patch('book_detector.cv2.HoughLinesP', return_value=line.reshape(-1, 1, 4)):
            nested = _hough_orientation(edges, quad)
        self.assertEqual(flat, nested)
        self.assertEqual(flat, 0.0)

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
        enabled = ImageConfig(shadow_enabled=False, contrast_enabled=False, perspective_enabled=True, rotation_only=False)
        result = process_bgr(image, (5, 5, 5, 5), enabled)
        self.assertEqual(result.mode, 'perspective')
        self.assertLess(result.image.shape[1], image.shape[1])
        self.assertGreaterEqual(result.confidence, enabled.min_confidence)
        # Manual mode must not apply a projective transform, even for same input.
        rotation = process_bgr(image, (5,) * 4, ImageConfig(shadow_enabled=False, contrast_enabled=False, rotation_only=True))
        self.assertNotEqual(rotation.mode, 'perspective')

    def test_stretch_guard_never_forces_perspective(self):
        image = synthetic_book([(200, 120), (610, 95), (555, 510), (230, 535)])
        setting = ImageConfig(shadow_enabled=False, contrast_enabled=False, perspective_enabled=True, rotation_only=False, max_perspective_stretch=1.05)
        result = process_bgr(image, (5,) * 4, setting)
        self.assertNotEqual(result.mode, 'perspective')
        self.assertIn('перспектива отклонена', result.fallback_reason or '')

    def test_no_confidence_uses_exact_legacy_crop(self):
        image = synthetic_book([(240, 110), (580, 110), (580, 520), (240, 520)])
        fake = BookDetection(None, 0.0, 'none', ('нет контура',))
        with patch('pipeline.detect_book', return_value=(fake, np.zeros(image.shape[:2], np.uint8))):
            actual = process_bgr(image, (10,) * 4, ImageConfig(shadow_enabled=False, contrast_enabled=False, line_deskew_enabled=False))
        expected = process_bgr(image, (10,) * 4, ImageConfig(enabled=False))
        np.testing.assert_array_equal(actual.image, expected.image)
        self.assertEqual(actual.mode, 'legacy')
        self.assertEqual(actual.fallback_reason, 'нет контура')

    def test_detector_exception_does_not_destroy_image(self):
        image = synthetic_book([(240, 110), (580, 110), (580, 520), (240, 520)])
        with patch('pipeline.detect_book', side_effect=cv2.error('synthetic failure')):
            actual = process_bgr(image, (10,) * 4, ImageConfig(shadow_enabled=False, contrast_enabled=False, line_deskew_enabled=False))
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


class V13RegressionTests(unittest.TestCase):
    def test_safe_default_disables_perspective(self):
        cfg = ImageConfig()
        self.assertFalse(cfg.perspective_enabled)
        self.assertTrue(cfg.rotation_only)
        self.assertTrue(cfg.shadow_enabled)
        self.assertTrue(cfg.contrast_enabled)
        image = synthetic_book([(200, 120), (610, 95), (555, 510), (230, 535)])
        result = process_bgr(image, (5,) * 4, ImageConfig(shadow_enabled=False, contrast_enabled=False))
        self.assertNotEqual(result.mode, 'perspective')

    def test_weak_quad_shadow_is_independent(self):
        h, w = 420, 680
        image = np.tile(np.array([236, 232, 226], np.float32), (h, w, 1))
        image -= np.linspace(0, 70, w, dtype=np.float32)[None, :, None]
        image = image.astype(np.uint8)
        fake = BookDetection(None, 0.0, 'none', ('четырёхугольник не подтверждён',))
        settings = ImageConfig(contrast_enabled=False, line_deskew_enabled=False)
        with patch('pipeline.detect_book', return_value=(fake, np.zeros((h, w), np.uint8))), \
             patch('pipeline.legacy_crop', return_value=(None, 'без обрезки')):
            result = process_bgr(image, (0,) * 4, settings)
        self.assertIsNotNone(result.image)
        self.assertEqual(result.mode, 'legacy')
        self.assertIn('четырёхугольник', result.fallback_reason)
        self.assertIn('тени', result.reason)
        self.assertGreater(float(result.image[220, -50, 0]), float(image[220, -50, 0]))

    def test_detector_error_still_corrects_lighting(self):
        img = np.tile(np.linspace(235, 170, 500, dtype=np.uint8), (400, 1))
        image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        with patch('pipeline.detect_book', side_effect=cv2.error('synthetic')), \
             patch('pipeline.legacy_crop', return_value=(None, 'без обрезки')):
            result = process_bgr(image, (0,) * 4, ImageConfig(contrast_enabled=False))
        self.assertIsNotNone(result.image)
        self.assertIn('ошибка детектора', result.fallback_reason)
        self.assertGreater(result.image[150, 480, 0], image[150, 480, 0])

    def test_line_deskew_requires_two_parallel_page_edges(self):
        from deskew import estimate_page_line_angle
        image = np.full((450, 700, 3), 242, np.uint8)
        cv2.line(image, (55, 65), (640, 79), (10, 10, 10), 3)
        cv2.line(image, (55, 315), (640, 329), (12, 12, 12), 3)
        angle = estimate_page_line_angle(image, ImageConfig())
        self.assertIsNotNone(angle)
        self.assertAlmostEqual(angle, 1.37, delta=0.9)
        cv2.line(image, (55, 315), (640, 329), (242, 242, 242), 6)
        self.assertIsNone(estimate_page_line_angle(image, ImageConfig()))

    def test_line_deskew_rejects_disagreeing_edges(self):
        from deskew import estimate_page_line_angle
        image = np.full((450, 700, 3), 242, np.uint8)
        cv2.line(image, (55, 65), (640, 78), (10, 10, 10), 3)
        cv2.line(image, (55, 350), (640, 305), (10, 10, 10), 3)
        self.assertIsNone(estimate_page_line_angle(image, ImageConfig()))

    def test_contrast_protects_saturated_cover(self):
        from contrast import enhance_page_contrast
        image = np.full((330, 420, 3), (232, 230, 224), np.uint8)
        cv2.putText(image, 'TEXT', (60, 195), cv2.FONT_HERSHEY_SIMPLEX, 1.8,
                    (95, 95, 95), 2)
        image[30:130, 270:390] = [160, 70, 35]
        out, mask = enhance_page_contrast(image, ImageConfig(contrast_strength=0.5))
        self.assertIsNotNone(mask)
        np.testing.assert_array_equal(image[60, 300], out[60, 300])
        self.assertGreater(np.count_nonzero(out != image), 10)

    def test_no_crop_and_no_effect_preserves_bytes(self):
        image = np.full((300, 400, 3), 245, np.uint8)
        out = process_bgr(image, (0,) * 4, ImageConfig(deskew_enabled=False))
        self.assertIsNone(out.image)


if __name__ == '__main__':
    unittest.main()

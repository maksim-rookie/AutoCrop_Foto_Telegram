"""Алгоритм AutoCrop v1.4 из предоставленного архива, без зависимости от Tkinter.
Функции сегментации и кадрирования сохранены без изменений.
"""
import math
import cv2
import numpy as np

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
APP_VERSION = '1.4'
OUTPUT_FOLDER = 'cropped'
DEFAULT_PADDING = (10, 10, 10, 10)
JPEG_QUALITY = 95
MIN_COMPONENT_PIXELS = 9
MIN_COMPONENT_RATIO = 0.0001
RELATIVE_COMPONENT_RATIO = 0.005


def validate_padding(values):
    """Вернуть четыре конечных отступа в диапазоне 0–100%."""
    if len(values) != 4:
        raise ValueError('Укажите четыре отступа: слева, сверху, справа, снизу.')
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or not 0 <= value <= 100 for value in result):
        raise ValueError('Каждый отступ должен быть числом от 0 до 100%.')
    return result


def imread_unicode(path):
    """Чтение с поддержкой Unicode, прозрачности и 16-битного цвета."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if not data.size:
        raise ValueError('Файл пуст.')
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None or img.size == 0:
        raise ValueError('Не удалось декодировать изображение.')
    return img


def _gray8(img):
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3 and img.shape[2] in (3, 4):
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError('Неподдерживаемое число каналов изображения.')
    if gray.dtype == np.uint8:
        return gray
    if gray.dtype == np.uint16:
        return (gray / 257).round().astype(np.uint8)
    raise ValueError(f'Неподдерживаемая глубина цвета: {gray.dtype}.')


def color_foreground(img):
    """Отделить цветной объект от однородного по цвету фона.

    Канал яркости Lab исключён: тень на сером столе не должна становиться
    объектом. Медиана и MAD ограничивают влияние объекта у края кадра.
    При неоднородном фоне или слабом цветовом контрасте вернуть None.
    """
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        return None
    color = img[:, :, :3]
    if color.dtype == np.uint16:
        color = (color / 257).round().astype(np.uint8)
    elif color.dtype != np.uint8:
        return None
    height, width = color.shape[:2]
    if min(height, width) < 12:
        return None
    # Оцениваем маску в уменьшенном масштабе, экономя память на больших фото.
    scale = min(1.0, 1200 / max(height, width))
    small = cv2.resize(color, (max(1, round(width * scale)),
                               max(1, round(height * scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1 else color
    smooth = cv2.GaussianBlur(small, (5, 5), 0)
    chroma = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)[:, :, 1:].astype(np.float32)
    band = max(2, round(min(chroma.shape[:2]) * 0.025))
    border = np.concatenate((chroma[:band].reshape(-1, 2),
                             chroma[-band:].reshape(-1, 2),
                             chroma[band:-band, :band].reshape(-1, 2),
                             chroma[band:-band, -band:].reshape(-1, 2)))
    background = np.median(border, axis=0)
    distances = np.linalg.norm(border - background, axis=1)
    median = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median)))
    if float(np.percentile(distances, 75)) > 12:
        return None
    threshold = max(10.0, median + 4 * 1.4826 * mad)
    mask = (np.linalg.norm(chroma - background, axis=2) > threshold).astype(np.uint8) * 255
    fraction = np.count_nonzero(mask) / mask.size
    if not 0.005 <= fraction <= 0.95 or np.mean(distances > threshold) > 0.30:
        return None
    # Закрываем небольшие разрывы цвета на бумаге, сохраняя внешнюю границу.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # Спроецировать границы клеток маски консервативно на исходное фото.
    if scale < 1:
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask


def segmented_book_mask(img):
    """Уточнить силуэт книги методом GrabCut на уменьшенной фотографии.

    Рамка кадра не считается целиком объектом. Метки фона учитывают цвет
    и освещённость слева/справа в каждой строке; тёмные обложки задают
    передний план вместе с насыщенными участками. Никаких моделей и загрузок.
    """
    if img.ndim != 3 or img.shape[2] not in (3, 4) or max(img.shape[:2]) < 256:
        return None
    color = img[:, :, :3]
    if color.dtype == np.uint16:
        color = (color / 257).round().astype(np.uint8)
    elif color.dtype != np.uint8:
        return None
    height, width = color.shape[:2]
    if min(height, width) < 32:
        return None
    scale = min(1.0, 520 / max(height, width))
    small = cv2.resize(color, (round(width * scale), round(height * scale)),
                       interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    gray = _gray8(small)
    color_mask = color_foreground(small)
    if color_mask is None:
        color_mask = np.zeros_like(gray)
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    labels = np.where((color_mask > 0) | (dark > 0),
                      cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    band = max(2, round(sw * 0.07))
    left = np.median(lab[:, :band], axis=1)
    right = np.median(lab[:, -band:], axis=1)
    ring = np.concatenate((lab[:2, :, 1:].reshape(-1, 2),
                           lab[-2:, :, 1:].reshape(-1, 2),
                           lab[:, :2, 1:].reshape(-1, 2),
                           lab[:, -2:, 1:].reshape(-1, 2)))
    reference = np.median(ring, axis=0)
    left_bad = np.linalg.norm(left[:, 1:] - reference, axis=1) > 9
    right_bad = np.linalg.norm(right[:, 1:] - reference, axis=1) > 9
    old_left, old_right = left.copy(), right.copy()
    left[left_bad & ~right_bad] = old_right[left_bad & ~right_bad]
    right[right_bad & ~left_bad] = old_left[right_bad & ~left_bad]
    x = np.linspace(0, 1, sw, dtype=np.float32)[None, :, None]
    local_background = left[:, None, :] * (1 - x) + right[:, None, :] * x
    chroma_difference = np.linalg.norm(lab[:, :, 1:] - local_background[:, :, 1:], axis=2)
    darkness = local_background[:, :, 0] - lab[:, :, 0]
    large_light_page = (np.count_nonzero(color_mask) > 0.30 * color_mask.size
                        and np.median(gray[color_mask > 0]) > 140)
    background_tolerance = 4 if large_light_page else 7
    labels[(chroma_difference < background_tolerance) & (darkness < 30)] = cv2.GC_BGD
    border = 2
    labels[:border] = labels[-border:] = cv2.GC_BGD
    labels[:, :border] = labels[:, -border:] = cv2.GC_BGD
    strong = (((color_mask > 0) & (chroma_difference > 16))
              | ((gray < 90) & (darkness > 45))).astype(np.uint8)
    strong[:border] = strong[-border:] = 0
    strong[:, :border] = strong[:, -border:] = 0
    count, components, stats, _ = cv2.connectedComponentsWithStats(strong)
    candidates = []
    for index in range(1, count):
        a, b, w, h, area = stats[index]
        edges = sum((a <= border, b <= border,
                     a + w >= sw - border, b + h >= sh - border))
        if area > 20 and edges < 2:
            candidates.append((int(area), index))
    if not candidates:
        return None
    foreground_id = max(candidates)[1]
    sure = cv2.erode((components == foreground_id).astype(np.uint8),
                     np.ones((3, 3), np.uint8)) > 0
    if not np.any(sure):
        sure = components == foreground_id
    labels[sure] = cv2.GC_FGD
    cv2.setRNGSeed(17)
    try:
        cv2.grabCut(small, labels, None, np.zeros((1, 65)), np.zeros((1, 65)),
                    4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    mask = ((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (components == largest).astype(np.uint8)
    if np.count_nonzero(mask) < 0.005 * mask.size:
        return None
    # У раскрытой книги лист может уходить за один край кадра. Возвращаем
    # связанный с книгой цветной участок, а не удаляем его как рамку фона.
    edge_color = (np.linalg.norm(lab[:, :, 1:] - reference, axis=2) > 6).astype(np.uint8)
    occupied_edges = sum(np.mean(edge) > 0.08 for edge in
                         (edge_color[0], edge_color[-1], edge_color[:, 0], edge_color[:, -1]))
    main_box = stats[largest]
    box_fraction = int(main_box[2]) * int(main_box[3]) / mask.size
    if occupied_edges == 1 and box_fraction > 0.30:
        count, components, stats, _ = cv2.connectedComponentsWithStats(edge_color)
        for index in range(1, count):
            a, b, w, h, area = stats[index]
            touches_edge = a == 0 or b == 0 or a + w == sw or b + h == sh
            component = components == index
            if (touches_edge and area > 0.05 * mask.size
                    and np.count_nonzero(component & (mask > 0)) > 0.01 * mask.size):
                mask[component] = 1
    # Запас на округление и сегментацию: 1 пиксель рабочего масштаба.
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8)) * 255
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def detect_object(img):
    """Маска объекта; принадлежность к фону определяется по границе кадра."""
    if img is None or img.size == 0:
        raise ValueError('Изображение пусто.')
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        if np.any(alpha < np.iinfo(alpha.dtype).max):
            # Не удаляем тонкие прозрачные детали морфологическими операциями.
            return np.where(alpha > 0, 255, 0).astype(np.uint8)
    refined = segmented_book_mask(img)
    if refined is not None:
        return refined
    color_mask = color_foreground(img)
    gray = _gray8(img)
    if int(gray.max()) == int(gray.min()):
        return np.zeros(gray.shape, dtype=np.uint8)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Углы учитываются один раз. Для кадра в один пиксель используем всё поле.
    if min(binary.shape) < 2:
        border = binary.ravel()
    else:
        border = np.concatenate((binary[0], binary[-1],
                                 binary[1:-1, 0], binary[1:-1, -1]))
    white_border_ratio = np.count_nonzero(border) / border.size
    if white_border_ratio > 0.5:
        luma_mask = cv2.bitwise_not(binary)
    elif white_border_ratio < 0.5:
        luma_mask = binary
    else:
        luma_mask = (binary if np.count_nonzero(binary) <= binary.size / 2
                     else cv2.bitwise_not(binary))
    if color_mask is None or object_bounds(color_mask) is None:
        return luma_mask
    bounds = object_bounds(luma_mask)
    if bounds is None:
        return color_mask
    x0, y0, x1, y1 = bounds
    height, width = gray.shape
    # Маска, захватившая три стороны кадра, обычно описывает фон.
    edges = sum((x0 <= 1, y0 <= 1, x1 >= width - 1, y1 >= height - 1))
    if edges >= 3:
        return color_mask
    # Цветовой маске нельзя единолично отбрасывать почти чёрную обложку.
    # Объединение может оставить больше фона, но сохраняет весь объект.
    return cv2.bitwise_or(color_mask, luma_mask)


def object_bounds(mask):
    """Объединить значимые компоненты, отбросив мелкий отдельный шум."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return None
    components = stats[1:]
    areas = components[:, cv2.CC_STAT_AREA]
    minimum = max(MIN_COMPONENT_PIXELS, math.ceil(mask.size * MIN_COMPONENT_RATIO),
                  math.ceil(int(areas.max()) * RELATIVE_COMPONENT_RATIO))
    kept = components[areas >= minimum]
    if len(kept) == 0:
        return None
    x0 = int(kept[:, cv2.CC_STAT_LEFT].min())
    y0 = int(kept[:, cv2.CC_STAT_TOP].min())
    x1 = int((kept[:, cv2.CC_STAT_LEFT] + kept[:, cv2.CC_STAT_WIDTH]).max())
    y1 = int((kept[:, cv2.CC_STAT_TOP] + kept[:, cv2.CC_STAT_HEIGHT]).max())
    return x0, y0, x1, y1


def crop_plan(img, padding_percent):
    """Вернуть безопасный прямоугольник и пояснение; None — копия оригинала."""
    left, top, right, bottom = validate_padding(padding_percent)
    bounds = object_bounds(detect_object(img))
    if bounds is None:
        return None, 'границы объекта не определены'
    x0, y0, x1, y1 = bounds
    height, width = img.shape[:2]
    obj_w, obj_h = x1 - x0, y1 - y0
    target = (x0 - round(obj_w * left / 100),
              y0 - round(obj_h * top / 100),
              x1 + round(obj_w * right / 100),
              y1 + round(obj_h * bottom / 100))
    a, b, c, d = target
    clipped = a < 0 or b < 0 or c > width or d > height
    target = (max(0, a), max(0, b), min(width, c), min(height, d))
    if target == (0, 0, width, height):
        return None, 'обрезка не требуется'
    reason = 'отступ у края ограничен имеющимся фоном' if clipped else 'отступы добавлены полностью'
    return target, reason


def auto_crop(img, padding_percent):
    """Срез изображения; None означает, что нужно сохранить исходный файл."""
    bounds, _ = crop_plan(img, padding_percent)
    if bounds is None:
        return None
    x0, y0, x1, y1 = bounds
    return img[y0:y1, x0:x1]


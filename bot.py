"""AutoCrop Telegram: Python 3.10+, long polling, без веб-сервера."""
from __future__ import annotations

# Устанавливаем лимит OpenCV до импорта crop_core/cv2.
import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = '24000000'

import getpass
import logging
import io
import json
from pathlib import Path
import re
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import warnings

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from crop_core import DEFAULT_PADDING, IMAGE_EXTS, JPEG_QUALITY, validate_padding
from image_config import ImageConfig
from pipeline import process_bgr

MAX_INPUT = 20_000_000
MAX_OUTPUT = 49_000_000
MAX_PIXELS = 24_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
HELP = '''Отправьте фото или изображение файлом — я обрежу фон вокруг объекта.
Лучше отправлять как «Файл»: Telegram может сжать обычное фото.
Результат всегда приходит файлом.

/padding 10 — одинаковый отступ 10% со всех сторон
/padding 5 10 5 15 — слева, сверху, справа, снизу, в процентах
/settings — текущие отступы
/reset — вернуть 10%
/id — ваш Telegram ID
/help — эта справка

Отступы считаются от размеров найденного объекта. Лучше всего работает одна книга/объект на однородном фоне.
Форматы: JPG, PNG, BMP, TIFF, WebP; до 20 MB и 24 млн пикселей.
Анимация и многостраничные TIFF не поддерживаются.
Настройки хранятся в памяти до перезапуска.'''


class ConfigError(ValueError):
    """Безопасное для показа сообщение, без значений из конфигурации."""


class ApiError(Exception):
    def __init__(self, code=0, retry_after=0, category='network'):
        explanations = {
            'dns': 'не удалось определить адрес api.telegram.org (DNS)',
            'tls': 'ошибка проверки защищённого соединения TLS/сертификата',
            'timeout': 'истекло время ожидания ответа от Telegram',
            'network': 'не удалось подключиться к api.telegram.org',
            'response': 'получен ответ не в формате Telegram Bot API',
        }
        detail = f'код {code}' if code else explanations.get(category, explanations['network'])
        super().__init__(f'Telegram API: {detail}')
        self.code = code
        self.retry_after = retry_after


class Telegram:
    def __init__(self, token):
        self.base = f'https://api.telegram.org/bot{token}/'
        self.file_base = f'https://api.telegram.org/file/bot{token}/'

    def call(self, method, fields=None, document=None):
        fields = fields or {}
        if document is None:
            body = json.dumps(fields).encode()
            content_type = 'application/json'
        else:
            filename, payload = document
            boundary = uuid.uuid4().hex
            parts = []
            for key, value in fields.items():
                parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
            # Только ASCII-имя, полученное внутри приложения, без пользовательских заголовков.
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="document"; filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode())
            parts.extend([payload, f'\r\n--{boundary}--\r\n'.encode()])
            body = b''.join(parts)
            content_type = f'multipart/form-data; boundary={boundary}'
        request = urllib.request.Request(self.base + method, body, {'Content-Type': content_type})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            try:
                result = json.loads(exc.read())
            except (ValueError, OSError):
                raise ApiError(exc.code) from None
        except OSError as exc:
            # Не печатаем URL и исходный текст исключения: там может быть токен.
            reason = getattr(exc, 'reason', exc)
            category = ('dns' if isinstance(reason, socket.gaierror) else
                        'tls' if isinstance(reason, ssl.SSLError) else
                        'timeout' if isinstance(reason, TimeoutError) else 'network')
            raise ApiError(category=category) from None
        except ValueError:
            raise ApiError(category='response') from None
        if not isinstance(result, dict):
            raise ApiError(category='response')
        if not result.get('ok'):
            raise ApiError(result.get('error_code', 0), result.get('parameters', {}).get('retry_after', 0))
        return result['result']

    def download(self, file_id):
        info = self.call('getFile', {'file_id': file_id})
        if info.get('file_size', 0) > MAX_INPUT:
            raise ValueError('Файл больше 20 MB.')
        path = info.get('file_path', '')
        if not path or path.startswith('/') or '..' in path.split('/') or '://' in path:
            raise ValueError('Telegram не вернул путь к файлу.')
        url = self.file_base + urllib.parse.quote(path, safe='/')
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                payload = response.read(MAX_INPUT + 1)
        except OSError:
            raise ApiError() from None
        if len(payload) > MAX_INPUT:
            raise ValueError('Файл больше 20 MB.')
        return payload


def process_image(payload, name, padding, processing: ImageConfig | None = None):
    """Проверка заголовка до OpenCV; результат и описание без записи на диск."""
    if not payload or len(payload) > MAX_INPUT:
        raise ValueError('Пустой файл или размер больше 20 MB.')
    ext = Path(name).suffix.lower()
    if ext not in IMAGE_EXTS:
        raise ValueError('Нужен JPG, PNG, BMP, TIFF или WebP.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as header:
                if header.width * header.height > MAX_PIXELS:
                    raise ValueError('Изображение больше 24 млн пикселей.')
                if getattr(header, 'n_frames', 1) != 1:
                    raise ValueError('Отправьте один кадр: анимация и многостраничные файлы не поддерживаются.')
                orientation = header.getexif().get(274, 1)
            with Image.open(io.BytesIO(payload)) as validation:
                validation.verify()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ValueError('Изображение слишком большое.') from None
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise ValueError('Не удалось прочитать изображение. Проверьте файл.') from None
    img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None or img.size == 0:
        raise ValueError('Не удалось декодировать изображение.')
    transforms = {
        2: lambda x: np.flip(x, axis=1),
        3: lambda x: np.rot90(x, 2),
        4: lambda x: np.flip(x, axis=0),
        5: lambda x: np.swapaxes(x, 0, 1),
        6: lambda x: np.rot90(x, 3),
        7: lambda x: np.flip(np.swapaxes(x, 0, 1), axis=(0, 1)),
        8: lambda x: np.rot90(x, 1),
    }
    if orientation in transforms:
        img = np.ascontiguousarray(transforms[orientation](img))
    logging.getLogger(__name__).info(
        'Image input: %sx%s bytes=%s dtype=%s channels=%s exif_orientation=%s',
        img.shape[1], img.shape[0], len(payload), img.dtype,
        1 if img.ndim == 2 else img.shape[2], orientation)
    # Existing file validation, orientation, depth and alpha handling stay intact.
    processing = processing if processing is not None else ImageConfig()
    transformed = process_bgr(img, padding, processing)
    if transformed.image is None and orientation not in transforms:
        warning = (f' Предупреждение: {transformed.fallback_reason}.'
                   if transformed.fallback_reason and processing.enabled and img.dtype == np.uint8
                   and img.ndim == 3 and img.shape[2] == 3 else '')
        return payload, 'original' + ext, 'Без обрезки: ' + transformed.reason + '.' + warning
    cropped = transformed.image if transformed.image is not None else img
    output_ext = '.png' if ext == '.webp' else ext
    params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY] if output_ext in {'.jpg', '.jpeg'} else []
    ok, encoded = cv2.imencode(output_ext, cropped, params)
    if not ok:
        raise ValueError('Не удалось сохранить результат.')
    result = encoded.tobytes()
    if len(result) > MAX_OUTPUT:
        raise ValueError('Результат больше 49 MB. Пришлите изображение меньшего размера.')
    caption = (f'Готово: {img.shape[1]}×{img.shape[0]} → {cropped.shape[1]}×{cropped.shape[0]}; '
               f'{transformed.reason}.')
    if transformed.mode in {'deskew', 'line_deskew', 'perspective'}:
        caption += f' Режим: {transformed.mode}, confidence={transformed.confidence:.2f}.'
    if transformed.fallback_reason and processing.enabled and img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8:
        caption += ' Безопасный fallback: ' + transformed.fallback_reason + '.'
    return result, 'cropped' + output_ext, caption


class Bot:
    def __init__(self, api, allowed_ids, image_config: ImageConfig | None = None):
        self.api = api
        self.allowed_ids = allowed_ids
        self.padding = {}
        self.image_config = image_config if image_config is not None else ImageConfig()

    def say(self, chat_id, text):
        self.api.call('sendMessage', {'chat_id': chat_id, 'text': text})

    def handle(self, update):
        message = update.get('message')
        if not message:
            logging.getLogger(__name__).info(
                'Telegram update=%s ignored: no message', update.get('update_id'))
            return
        chat_type = message.get('chat', {}).get('type')
        if chat_type != 'private':
            logging.getLogger(__name__).info(
                'Telegram update=%s ignored: chat_type=%s', update.get('update_id'), chat_type)
            return
        chat_id = message['chat']['id']
        user_id = message.get('from', {}).get('id')
        text = message.get('text', '').strip()
        words = text.split()
        command = words[0].split('@')[0].lower() if words else ''
        args = words[1:]
        safe_command = command if command in {
            '/id', '/start', '/help', '/padding', '/settings', '/reset'
        } else 'other'
        logging.getLogger(__name__).info(
            'Telegram update=%s received: command=%s access=%s',
            update.get('update_id'), safe_command,
            'allowed' if user_id in self.allowed_ids else 'denied')
        if command == '/id':
            self.say(chat_id, f'Ваш Telegram ID: {user_id}')
            return
        if user_id not in self.allowed_ids:
            self.say(chat_id, f'Доступ закрыт. Ваш ID: {user_id}. Владелец должен добавить его в ALLOWED_USER_IDS и перезапустить бота.')
            return
        if command in {'/start', '/help'}:
            self.say(chat_id, HELP)
        elif command == '/padding':
            try:
                values = args * 4 if len(args) == 1 else args
                self.padding[user_id] = validate_padding([x.replace(',', '.') for x in values])
            except (ValueError, TypeError):
                self.say(chat_id, 'Пример: /padding 10 или /padding 5 10 5 15. Значения от 0 до 100%.')
                return
            self.say(chat_id, 'Отступы сохранены. ' + self.settings(user_id))
        elif command == '/reset':
            self.padding.pop(user_id, None)
            self.say(chat_id, self.settings(user_id))
        elif command == '/settings':
            self.say(chat_id, self.settings(user_id))
        elif message.get('document') or message.get('photo'):
            item = message.get('document') or message['photo'][-1]
            name = item.get('file_name', 'photo.jpg')
            try:
                if Path(name).suffix.lower() not in IMAGE_EXTS:
                    raise ValueError('Нужен JPG, PNG, BMP, TIFF или WebP.')
                if item.get('file_size', 0) > MAX_INPUT:
                    raise ValueError('Файл больше 20 MB.')
                self.say(chat_id, 'Обрабатываю изображение…')
                payload = self.api.download(item['file_id'])
                result, filename, caption = process_image(payload, name, self.padding.get(user_id, DEFAULT_PADDING), self.image_config)
                self.api.call('sendDocument', {'chat_id': chat_id, 'caption': caption}, document=(filename, result))
            except ValueError as exc:
                self.say(chat_id, str(exc))
            except cv2.error:
                self.say(chat_id, 'Не удалось обработать изображение. Попробуйте другой JPG или PNG.')
        else:
            self.say(chat_id, 'Отправьте изображение или используйте /help.')

    def settings(self, user_id):
        values = self.padding.get(user_id, DEFAULT_PADDING)
        return 'Отступы слева / сверху / справа / снизу: ' + ' / '.join(f'{v:g}%' for v in values)


def load_config():
    # Только явные ключи из локального JSON, переменные среды имеют приоритет.
    path = Path(__file__).with_name('config.json')
    if not path.exists() and path.with_suffix('.json.txt').exists():
        raise ConfigError('Файл назван config.json.txt. Переименуйте его в config.json.')
    try:
        config = json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}
    except json.JSONDecodeError as exc:
        raise ConfigError(f'Ошибка JSON в config.json: строка {exc.lineno}, столбец {exc.colno}. '
                          'Проверьте двойные кавычки, запятые и фигурные скобки.') from None
    except UnicodeError:
        raise ConfigError('config.json должен быть сохранён в кодировке UTF-8.') from None
    except OSError:
        raise ConfigError('Не удалось прочитать config.json. Проверьте доступ к файлу.') from None
    if not isinstance(config, dict):
        raise ConfigError('В config.json нужен объект в фигурных скобках { ... }.')
    token = os.getenv('TELEGRAM_BOT_TOKEN', config.get('TELEGRAM_BOT_TOKEN', ''))
    if not isinstance(token, str):
        raise ConfigError('TELEGRAM_BOT_TOKEN должен быть строкой в двойных кавычках.')
    token = token.strip()
    if not token or token == 'PASTE_TOKEN_HERE':
        token = getpass.getpass('Токен из BotFather (ввод скрыт): ').strip()
    raw_ids = os.getenv('ALLOWED_USER_IDS', str(config.get('ALLOWED_USER_IDS', ''))).strip()
    if not raw_ids or raw_ids == 'YOUR_TELEGRAM_ID':
        raw_ids = input('Ваш числовой Telegram ID (можно несколько через запятую): ').strip()
    if not re.fullmatch(r'\d+:[A-Za-z0-9_-]+', token):
        raise ConfigError('Некорректный формат TELEGRAM_BOT_TOKEN. '
                          'Вставьте только токен BotFather: без bot, URL и лишних кавычек внутри значения.')
    if not re.fullmatch(r'[0-9]+(?:\s*,\s*[0-9]+)*', raw_ids):
        raise ConfigError('ALLOWED_USER_IDS: нужны только числовые ID через запятую, без @ и номера телефона.')
    ids = {int(value.strip()) for value in raw_ids.split(',')}
    if any(value <= 0 for value in ids):
        raise ConfigError('ALLOWED_USER_IDS: каждый ID должен быть положительным числом.')
    return token, ids



def load_processing_config() -> ImageConfig:
    """Load optional CV settings without changing load_config's historical API."""
    path = Path(__file__).with_name('config.json')
    if not path.exists():
        return ImageConfig()
    try:
        config = json.loads(path.read_text(encoding='utf-8-sig'))
        return ImageConfig.from_mapping(config.get('IMAGE_PROCESSING'))
    except (OSError, UnicodeError, ValueError, AttributeError) as exc:
        # Never quote raw JSON values: the file may contain the bot token.
        raise ConfigError('Некорректные параметры IMAGE_PROCESSING в config.json.') from None


def load_flush_updates_on_start() -> bool:
    """Read a top-level bool; never expose token or user IDs in errors."""
    path = Path(__file__).with_name('config.json')
    if not path.exists():
        return True
    try:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        value = data.get('flush_updates_on_start', True)
        if type(value) is not bool:
            raise ValueError('expected boolean')
        return value
    except (OSError, ValueError, AttributeError, TypeError):
        raise ConfigError('flush_updates_on_start должен быть true/false в config.json.') from None


def initial_update_offset(api: Telegram, flush: bool) -> int:
    """Discard pre-restart pending updates via negative Telegram getUpdates offset."""
    if not flush:
        logging.getLogger(__name__).info('Startup update queue flush disabled')
        return 0
    pending = api.call('getUpdates', {'offset': -1, 'limit': 1, 'timeout': 0,
                                       'allowed_updates': ['message']})
    if not pending:
        logging.getLogger(__name__).info('Startup update queue empty')
        return 0
    last_id = pending[-1].get('update_id')
    if type(last_id) is not int or last_id < 0:
        raise ApiError(category='response')
    logging.getLogger(__name__).info('Startup update queue discarded (pending last update seen)')
    return last_id + 1


def main():
    try:
        token, ids = load_config()
        image_settings = load_processing_config()
        flush_updates = load_flush_updates_on_start()
        api = Telegram(token)
        api.call('getMe')
        if api.call('getWebhookInfo').get('url'):
            print('У бота включён webhook. Используйте отдельного бота для этой версии или отключите webhook самостоятельно.')
            return 1
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
        access_source = 'environment' if os.getenv('ALLOWED_USER_IDS') else 'config.json / prompt'
        logging.getLogger(__name__).info(
            'Access config: permitted_users=%s source=%s', len(ids), access_source)
        bot = Bot(api, ids, image_settings)
        offset = initial_update_offset(api, flush_updates)
        print('AutoCrop запущен. Откройте бота и отправьте /start. Остановка: Ctrl+C.')
        while True:
            try:
                updates = api.call('getUpdates', {'offset': offset, 'timeout': 30, 'limit': 10, 'allowed_updates': ['message']})
                for update in updates:
                    try:
                        bot.handle(update)
                    except ApiError as exc:
                        print(f'Не удалось завершить ответ ({exc}). Отправьте изображение повторно.')
                        if exc.retry_after:
                            time.sleep(min(exc.retry_after, 60))
                    except Exception as exc:
                        # Только тип ошибки, без потенциальных секретов в сообщении.
                        print(f'Ошибка обработки: {type(exc).__name__}. Отправьте файл повторно.')
                    finally:
                        offset = update['update_id'] + 1
            except ApiError as exc:
                if exc.code in {401, 404, 409}:
                    print(f'{exc}. Проверьте токен и убедитесь, что запущена только одна копия бота.')
                    return 1
                print(f'{exc}. Повтор подключения…')
                time.sleep(min(max(exc.retry_after, 5), 60))
    except KeyboardInterrupt:
        print('\nБот остановлен.')
        return 0
    except ConfigError as exc:
        print(f'Ошибка настройки: {exc}')
        return 1
    except ApiError as exc:
        if exc.code in {401, 404}:
            print(f'{exc}. Telegram не принял токен/адрес API. Скопируйте токен из BotFather заново.')
        else:
            print(f'Запуск не выполнен: {exc}. Проверьте соединение с Telegram.')
        return 1
    except EOFError:
        print('Не удалось прочитать ввод. Заполните config.json и запустите start_windows.bat.')
        return 1
    except OSError:
        print('Системная ошибка при запуске. Проверьте доступ к файлам и подключение к интернету.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

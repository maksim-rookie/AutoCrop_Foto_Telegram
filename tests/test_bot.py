import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bot
import cv2
import numpy as np
from PIL import Image


def png(img):
    ok, data = cv2.imencode('.png', img)
    assert ok
    return data.tobytes()


class FakeAPI:
    def __init__(self, payload=b''):
        self.calls = []
        self.payload = payload
        self.downloads = 0

    def call(self, method, fields=None, document=None):
        self.calls.append((method, fields, document))
        return True

    def download(self, file_id):
        self.downloads += 1
        return self.payload


def message(uid=1, **fields):
    return {'message': {'chat': {'id': uid, 'type': 'private'}, 'from': {'id': uid}, **fields}}


class CropTests(unittest.TestCase):
    def test_transparent_object_and_padding(self):
        image = np.zeros((100, 120, 4), np.uint8)
        image[30:70, 40:80] = [30, 50, 200, 255]
        result, name, caption = bot.process_image(png(image), 'book.png', (10, 10, 10, 10))
        actual = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_UNCHANGED)
        self.assertEqual(actual.shape, (48, 48, 4))
        self.assertEqual(name, 'cropped.png')
        np.testing.assert_array_equal(actual[4:44, 4:44], image[30:70, 40:80])

    def test_exif_orientations(self):
        original = np.zeros((40, 60, 4), np.uint8)
        original[10:30, 10:50] = [50, 100, 200, 255]
        original[10:15, 10:15] = [200, 10, 20, 255]
        transforms = {
            2: lambda x: np.flip(x, 1), 3: lambda x: np.rot90(x, 2),
            4: lambda x: np.flip(x, 0), 5: lambda x: np.swapaxes(x, 0, 1),
            6: lambda x: np.rot90(x, 3),
            7: lambda x: np.flip(np.swapaxes(x, 0, 1), (0, 1)),
            8: lambda x: np.rot90(x, 1),
        }
        rgba = cv2.cvtColor(original, cv2.COLOR_BGRA2RGBA)
        for orientation, transform in transforms.items():
            with self.subTest(orientation=orientation):
                exif = Image.Exif()
                exif[274] = orientation
                stream = io.BytesIO()
                Image.fromarray(rgba).save(stream, format='PNG', exif=exif)
                result, _, _ = bot.process_image(stream.getvalue(), 'x.png', (0,) * 4)
                actual = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_UNCHANGED)
                np.testing.assert_array_equal(actual, transform(original[10:30, 10:50]))

    def test_no_crop_is_byte_identical(self):
        original = png(np.full((64, 64, 3), 255, np.uint8))
        result, name, caption = bot.process_image(original, 'blank.png', (10,) * 4)
        self.assertEqual(result, original)
        self.assertEqual(name, 'original.png')

    def test_uint16_alpha_preserved(self):
        image = np.zeros((50, 50, 4), np.uint16)
        image[10:40, 10:40] = [1000, 2000, 3000, 65535]
        result, _, _ = bot.process_image(png(image), 'deep.png', (0,) * 4)
        actual = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_UNCHANGED)
        self.assertEqual(actual.dtype, np.uint16)
        self.assertEqual(actual.shape, (30, 30, 4))

    def test_opaque_book_is_cropped(self):
        image = np.full((300, 240, 3), 245, np.uint8)
        image[70:230, 60:180] = [40, 30, 70]
        result, name, _ = bot.process_image(png(image), 'book.png', (10,) * 4)
        actual = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_UNCHANGED)
        self.assertEqual(name, 'cropped.png')
        self.assertLess(actual.shape[0], 300)
        self.assertLess(actual.shape[1], 240)
        self.assertGreaterEqual(np.count_nonzero((actual[:, :, 0] >= 33) & (actual[:, :, 0] <= 47)), 160 * 120 * 0.9)

    def test_bad_inputs(self):
        for payload, name in [(b'', 'x.png'), (b'wrong', 'x.jpg'), (b'wrong', 'x.exe')]:
            with self.subTest(name=name, payload=payload), self.assertRaises(ValueError):
                bot.process_image(payload, name, (10,) * 4)

    def test_pixel_limit_before_decode(self):
        data = png(np.zeros((20, 20, 3), np.uint8))
        with patch.object(bot, 'MAX_PIXELS', 100), patch.object(bot.cv2, 'imdecode') as decode:
            with self.assertRaises(ValueError):
                bot.process_image(data, 'x.png', (10,) * 4)
            decode.assert_not_called()

    def test_animated_image_rejected(self):
        data = io.BytesIO()
        Image.new('RGB', (20, 20), 'red').save(data, format='WEBP', save_all=True,
            append_images=[Image.new('RGB', (20, 20), 'blue')], duration=100)
        with self.assertRaises(ValueError):
            bot.process_image(data.getvalue(), 'x.webp', (10,) * 4)

    def test_webp_becomes_png(self):
        image = Image.new('RGBA', (80, 80), (0, 0, 0, 0))
        image.paste((200, 10, 10, 255), (20, 20, 60, 60))
        data = io.BytesIO()
        image.save(data, format='WEBP', lossless=True)
        result, name, _ = bot.process_image(data.getvalue(), 'x.webp', (0,) * 4)
        self.assertEqual(name, 'cropped.png')
        self.assertTrue(result.startswith(b'\x89PNG'))


class HandlerTests(unittest.TestCase):
    def setUp(self):
        self.api = FakeAPI(png(np.full((60, 60, 3), 255, np.uint8)))
        self.bot = bot.Bot(self.api, {1, 2})

    def test_unauthorized_never_downloads(self):
        self.bot.handle(message(3, document={'file_id': 'abc', 'file_name': 'x.png'}))
        self.assertEqual(self.api.downloads, 0)

    def test_two_allowed_accounts_receive_start_response(self):
        with self.assertLogs('bot', level='INFO') as captured:
            self.bot.handle(message(1, text='/start'))
            self.bot.handle(message(2, text='/start'))
        self.assertEqual([entry[0] for entry in self.api.calls[-2:]],
                         ['sendMessage', 'sendMessage'])
        self.assertEqual([entry[1]['chat_id'] for entry in self.api.calls[-2:]], [1, 2])
        self.assertTrue(any('access=allowed' in log for log in captured.output))
        self.assertFalse(any('received: command=/start' in log and '123456789' in log
                             for log in captured.output))

    def test_denied_account_still_receives_access_message(self):
        with self.assertLogs('bot', level='INFO') as captured:
            self.bot.handle(message(3, text='/start'))
        self.assertEqual(self.api.calls[-1][0], 'sendMessage')
        self.assertIn('Доступ закрыт', self.api.calls[-1][1]['text'])
        self.assertTrue(any('access=denied' in log for log in captured.output))

    def test_per_user_padding_and_reset(self):
        self.bot.handle(message(text='/padding 1 2 3 4'))
        self.bot.handle(message(2, text='/padding 7,5'))
        self.assertEqual(self.bot.padding[1], (1, 2, 3, 4))
        self.assertEqual(self.bot.padding[2], (7.5,) * 4)
        self.bot.handle(message(text='/padding nan'))
        self.assertEqual(self.bot.padding[1], (1, 2, 3, 4))
        self.bot.handle(message(text='/reset'))
        self.assertNotIn(1, self.bot.padding)

    def test_photo_and_document(self):
        for field in ({'photo': [{'file_id': 'low'}, {'file_id': 'high'}]},
                      {'document': {'file_id': 'doc', 'file_name': '../../a.png'}}):
            self.bot.handle(message(**field))
            method, fields, document = self.api.calls[-1]
            self.assertEqual(method, 'sendDocument')
            self.assertEqual(document[0], 'original.jpg' if 'photo' in field else 'original.png')

    def test_oversized_file_not_downloaded(self):
        self.bot.handle(message(document={'file_id': 'doc', 'file_name': 'a.png', 'file_size': bot.MAX_INPUT + 1}))
        self.assertEqual(self.api.downloads, 0)

    def test_group_ignored(self):
        update = message(text='/start')
        update['message']['chat']['type'] = 'group'
        self.bot.handle(update)
        self.assertEqual(self.api.calls, [])

    def test_id_available_without_access(self):
        self.bot.handle(message(123, text='/id'))
        self.assertIn('123', self.api.calls[-1][1]['text'])


class TransportTests(unittest.TestCase):
    def test_rate_limit_is_sanitized(self):
        error = urllib.error.HTTPError('https://example/SECRET', 429, '', {},
            io.BytesIO(json.dumps({'ok': False, 'error_code': 429,
                'parameters': {'retry_after': 3}}).encode()))
        with patch('urllib.request.urlopen', side_effect=error):
            with self.assertRaises(bot.ApiError) as found:
                bot.Telegram('123:SECRET').call('getMe')
        self.assertEqual(found.exception.retry_after, 3)
        self.assertNotIn('SECRET', str(found.exception))

    def test_download_hard_limit(self):
        response = io.BytesIO(b'x' * 11)
        api = bot.Telegram('123:SECRET')
        with patch.object(api, 'call', return_value={'file_path': 'documents/x.png'}), \
             patch('urllib.request.urlopen', return_value=response), patch.object(bot, 'MAX_INPUT', 10):
            with self.assertRaises(ValueError):
                api.download('id')

    def test_multipart_upload(self):
        reply = io.BytesIO(b'{"ok":true,"result":true}')
        with patch('urllib.request.urlopen', return_value=reply) as opened:
            bot.Telegram('123:SECRET').call('sendDocument', {'chat_id': 1}, ('cropped.png', b'payload'))
        request = opened.call_args.args[0]
        self.assertIn(b'filename="cropped.png"', request.data)
        self.assertIn(b'payload', request.data)

class StartupTests(unittest.TestCase):
    def config_result(self, text):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'config.json').write_text(text, encoding='utf-8')
            with patch.object(bot, '__file__', str(Path(folder, 'bot.py'))), patch.dict(bot.os.environ, {}, clear=True):
                return bot.load_config()

    def test_valid_config(self):
        token, ids = self.config_result('{"TELEGRAM_BOT_TOKEN":"123:SECRET","ALLOWED_USER_IDS":"123, 456"}')
        self.assertEqual(ids, {123, 456})
        self.assertEqual(token, '123:SECRET')

    def test_json_error_is_safe_and_located(self):
        with self.assertRaises(bot.ConfigError) as found:
            self.config_result('{"TELEGRAM_BOT_TOKEN":"SECRET",}')
        self.assertIn('строка', str(found.exception))
        self.assertNotIn('SECRET', str(found.exception))

    def test_config_shape(self):
        for value in ['[]', 'null', '123']:
            with self.subTest(value=value), self.assertRaises(bot.ConfigError):
                self.config_result(value)

    def test_invalid_fields_are_safe(self):
        for token, ids in [(123, '1'), ('https://SECRET', '1'), ('123:SECRET', '@name'),
                           ('123:SECRET', '0'), ('123:SECRET', '1,,2')]:
            with self.subTest(token=token, ids=ids), self.assertRaises(bot.ConfigError) as found:
                self.config_result(json.dumps({'TELEGRAM_BOT_TOKEN': token, 'ALLOWED_USER_IDS': ids}))
            self.assertNotIn('SECRET', str(found.exception))

    def test_wrong_extension_detected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'config.json.txt').write_text('{}')
            with patch.object(bot, '__file__', str(Path(folder, 'bot.py'))), self.assertRaises(bot.ConfigError) as found:
                bot.load_config()
        self.assertIn('config.json.txt', str(found.exception))

    def test_network_errors_are_classified_without_token(self):
        import socket
        import ssl
        for reason, expected in [(socket.gaierror('SECRET'), 'DNS'),
                                 (ssl.SSLError('SECRET'), 'TLS'),
                                 (TimeoutError('SECRET'), 'время ожидания')]:
            with self.subTest(expected=expected), patch('urllib.request.urlopen', side_effect=urllib.error.URLError(reason)):
                with self.assertRaises(bot.ApiError) as found:
                    bot.Telegram('123:SECRET').call('getMe')
                self.assertIn(expected, str(found.exception))
                self.assertNotIn('SECRET', str(found.exception))

    def test_startup_authentication_message(self):
        output = io.StringIO()
        with patch.object(bot, 'load_config', return_value=('123:SECRET', {1})), \
             patch.object(bot.Telegram, 'call', side_effect=bot.ApiError(401)), \
             patch('sys.stdout', output):
            self.assertEqual(bot.main(), 1)
        self.assertIn('401', output.getvalue())
        self.assertIn('токен', output.getvalue())
        self.assertNotIn('SECRET', output.getvalue())


class RestartQueueTests(unittest.TestCase):
    def test_flush_uses_last_pending_offset(self):
        api = FakeAPI()
        with patch.object(api, 'call', return_value=[{'update_id': 527}]) as called:
            offset = bot.initial_update_offset(api, flush=True)
        self.assertEqual(offset, 528)
        called.assert_called_once_with('getUpdates', {
            'offset': -1, 'limit': 1, 'timeout': 0, 'allowed_updates': ['message']})

    def test_empty_queue_starts_normally(self):
        api = FakeAPI()
        with patch.object(api, 'call', return_value=[]):
            self.assertEqual(bot.initial_update_offset(api, flush=True), 0)

    def test_flush_can_be_disabled(self):
        api = FakeAPI()
        with patch.object(api, 'call') as called:
            self.assertEqual(bot.initial_update_offset(api, flush=False), 0)
            called.assert_not_called()

    def test_unexpected_queue_result_fails_without_sending_user_data(self):
        api = FakeAPI()
        with patch.object(api, 'call', return_value=[{'update_id': 'secret'}]):
            with self.assertRaises(bot.ApiError) as found:
                bot.initial_update_offset(api, flush=True)
        self.assertNotIn('secret', str(found.exception))

    def test_flush_config_type_is_checked(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            config = Path(d, 'config.json')
            with patch.object(bot, '__file__', str(Path(d, 'bot.py'))):
                config.write_text('{"flush_updates_on_start": true}', encoding='utf8')
                self.assertTrue(bot.load_flush_updates_on_start())
                config.write_text('{"flush_updates_on_start": false}', encoding='utf8')
                self.assertFalse(bot.load_flush_updates_on_start())
                config.write_text('{"flush_updates_on_start": "false"}', encoding='utf8')
                with self.assertRaises(bot.ConfigError):
                    bot.load_flush_updates_on_start()


if __name__ == '__main__':
    unittest.main()

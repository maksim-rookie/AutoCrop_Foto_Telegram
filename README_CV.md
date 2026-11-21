# AutoCrop Telegram — optional CV enhancements (v1.2)

Runs without ML models; dependencies remain OpenCV, NumPy, Pillow. The original v1.1 `crop_core.py` is **not modified**. Existing Telegram API, commands, `/padding` semantics, file size limits, EXIF behavior, access control and startup remain supported.

## Architecture

- `bot.py`: validated Telegram input, EXIF orientation, output encoding, configuration and command routing.
- `image_config.py`: typed and range-validated CV settings with conservative defaults.
- `book_detector.py`: RGB downscaling for analysis (full-size output), Canny on lightness/chroma, morphological closing, convex contours, 4-corner approximation, geometric `minAreaRect` fallback, HoughLinesP angle cross-check and confidence/reason diagnostics.
- `deskew.py`: bounded affine rotation or quadrilateral rectification, both with a margin around all corners. If the required stretch is excessive, the warp is rejected.
- `shadow_removal.py`: limited illumination-field division, light/low-saturation masks, morphology, optional gentle CLAHE. No hard whitening and no recovery of saturated glare.
- `crop.py`: original AutoCrop wrapper for safe fallback.
- `pipeline.py`: feature flags, fallback handling, logging and opt-in debug snapshots.

**Safety:** Uncertain detections use legacy crop (or the exact original file if no crop is needed). 16-bit or alpha-bearing input retains the legacy path instead of a destructive 8-bit conversion. CV analysis uses RGB; the output is encoded using the original file format rules. EXIF is applied *before* analysis. Setting `enabled: false` invokes only the old algorithm. No new Telegram commands are introduced.

## Configuration

Copy `config.example.json` to `config.json` and set the Telegram token/IDs as in v1.1. In `IMAGE_PROCESSING`:

| Setting | Default | Meaning |
|---|---:|---|
| `enabled` | `true` | Master feature flag: `false` = v1.1 behavior |
| `deskew_enabled` | `true` | Correct small tilt when book contour is certain |
| `perspective_enabled` | `true` | Warp only sufficiently nonparallel book edges |
| `shadow_enabled` | `true` | Neutral-surface soft shadow reduction |
| `rotation_only` | `false` | Manual mode: never use perspective warp |
| `analysis_max_side` | `1400` | Long side for detection; full output resolution retained |
| `min_confidence` | `0.78` | Confidence gate for all enhancements |
| `min_book_area`, `max_book_area` | `0.07`, `0.91` | Reject tiny or frame-filling candidates |
| `max_deskew_degrees` | `18.0` | Prevent accidental 90° or large rotations |
| `perspective_threshold` | `0.075` | Required side disparity before perspective is attempted |
| `max_perspective_stretch` | `1.35` | Maximum unequal opposite-edge length ratio |
| `shadow_strength` | `0.35` | `0` disables effect; recommended `0.2–0.5` |
| `shadow_min_variation` | `12.0` | Ignore effectively uniform illumination |
| `shadow_clahe_clip` | `0.0` | Optional mild CLAHE, recommended off for cover color |
| `debug_dir` | `null` | Folder for debug masks, quadrilateral overlay, angle JSON and output |

`debug_dir` is **opt-in**: diagnostics may contain users' photographs. Use a private writable folder, remove the snapshots after debugging and never commit them to GitHub. Token and Telegram IDs are never stored in diagnostics.

## Run and test

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 -m unittest discover -s tests -v
py -3.12 bot.py
```

No Telegram connection is needed for unit tests. Smoke-test the image processing logic independently:

```powershell
py -3.12 scripts/process_local.py "sample.jpg" "sample_cropped.jpg"
py -3.12 scripts/process_local.py "sample.jpgNo readable content was found in this file for the requested mode.ваются до кадрирования.
При отсутствии обрезки и преобразования ориентации возвращаются исходные байты.
При повторном кодировании метаданные EXIF/ICC не переносятся.
Обычные фото Telegram может сжимать ещё до получения ботом.
Результат ограничен 49 MB; при превышении бот попросит меньший исходник.

РАБОТА И ЗАПУСК НА СЕРВЕРЕ

Бот работает, пока запущен bot.py и компьютер подключён к интернету.
Закрытая консоль, выключенный компьютер или спящий режим остановят обработку.
Для работы 24/7 перенесите папку на постоянно включённый ПК или VPS.
Отдельный домен, публичный IP и входящие порты не нужны: используется long polling.
Нужен исходящий HTTPS к api.telegram.org и доступ к PyPI для установки.

Linux/macOS:
  python3 -m venv .venv
  .venv/bin/python -m pip install -r requirements.txt
  .venv/bin/python bot.py

Можно задать TELEGRAM_BOT_TOKEN и ALLOWED_USER_IDS в переменных среды
вместо config.json. Переменные среды имеют приоритет.
Для постоянного сервера задайте автозапуск через systemd/диспетчер служб.
Сам этот комплект не настраивает сервер и не регистрирует службу.
Файлы обрабатываются в оперативной памяти без сохранения локальных копий.
Загрузка и ответы проходят через Telegram и остаются в чате по его правилам.
Обработка последовательная, рассчитана на личное использование/небольшую команду.
Настройки и offset хранятся в памяти: после аварии возможен повтор последнего ответа.
После ошибки отправки пришлите изображение повторно.

ЕСЛИ НЕ РАБОТАЕТ

Версия комплекта 1.1: подробная диагностика запуска без вывода токена.
Теперь отдельно сообщаются ошибки JSON (строка/столбец), неправильные
поля конфигурации, config.json.txt, отказ API, DNS, TLS и тайм-ауты.
Для обновления с 1.0 достаточно заменить bot.py; свой config.json сохраните.
Если ошибка осталась, передайте текст нового сообщения, без токена.

401/404: проверьте токен. При утечке отзовите его через @BotFather.
409: остановите другие копии этого бота.
Webhook: используйте отдельного бота или самостоятельно отключите прежний webhook.
Комплект проверяет webhook и не отключает его автоматически.
Нет доступа: проверьте ALLOWED_USER_IDS и перезапустите бота.
Нет подключения: проверьте доступ к api.telegram.org на вашем ПК/сервере.
Ошибка установки: проверьте версию Python, интернет и вывод pip в консоли.

ПРОВЕРКИ И СОСТАВ

crop_core.py — алгоритм из исходного auto_crop.py, без Tkinter.
bot.py — Telegram Bot API, контроль доступа и обработка в памяти.
start_windows.bat — создание среды, установка зависимостей и запуск.
config.example.json — образец, не содержит настоящего токена.
tests/test_bot.py — автоматические проверки с подменой Telegram API.

Запуск проверок:
  python -m unittest discover -s tests -v

Проверено в Linux с Python 3.12, NumPy 2.5.4, OpenCV 5.0.0, Pillow 13.0.0.
Проверки не подключаются к Telegram. Живой обмен с ботом и запуск .bat на Windows
в этой среде не проверены — для этого нужен ваш токен и ваш компьютер.

Документация Telegram: https://core.telegram.org/bots/api

НОВОЕ В ПАТЧЕ v1.2 CV

Новые модули обнаруживают четырёхугольник книги, выполняют осторожное
выравнивание наклона и перспективы, ослабляют мягкие тени. При низкой
уверенности применяется старый crop_core.py или возвращается исходник.
Функции включаются флагами IMAGE_PROCESSING в config.json. Чтобы полностью
вернуть поведение v1.1, укажите "enabled": false; для режима только
поворота — "rotation_only": true. Команды бота не менялись.
Смотрите README_CV.md: параметры, диагностика, тесты, ограничения и риски.

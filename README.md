# AutoCrop Telegram — v1.3.0

CPU-only Telegram-бот для аккуратной обрезки фотографий книг. Основной режим: **crop → осторожный поворот → мягкое выравнивание света → небольшой контраст**, без изменения перспективы. При отсутствии уверенно обнаруженного четырёхугольника выполняются обычная обрезка и независимая коррекция освещения.

## Документы
- [Журнал ревизий v1.1 → v1.3.0](CHANGELOG.md)
- [Инженерная рецензия, замечания и риски](docs/ENGINEERING_REVIEW.md)
- [Обработка фотографий и настройки](README_CV.md)
- [Инструкция Windows / Telegram](README_RU.txt)

## Установка

1. Скопируйте `config.example.json` в `config.json` и укажите токен BotFather и числовые `ALLOWED_USER_IDS` (через запятую). Не загружайте `config.json` в GitHub.
2. На Windows запустите `start_windows.bat`. Или:

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 -m unittest discover -s tests -v
py -3.12 bot.py
```

**После перезапуска очередь старых сообщений Telegram очищается по умолчанию.** Чтобы обработать сообщения после простоя, установите `"flush_updates_on_start": false` в `config.json`.

Для проверки без Telegram: `python scripts/process_local.py input.jpg output.jpg`.
Для старого кадрирования: `"IMAGE_PROCESSING": {"enabled": false}`. Для полного отключения поворотов: `"deskew_enabled": false`.

Не используется тяжёлая ML-модель. Пользовательские фото и секреты в репозиторий не входят.

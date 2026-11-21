# AutoCrop Telegram

CPU-only Telegram book photo autocrop: optional deskew, perspective rectification, and gentle soft-shadow correction. Based on the user-supplied AutoCrop Telegram v1.1 implementation.

- [Инструкция по установке и работе с ботом (RU)](README_RU.txt)
- [Настройка CV, архитектура, тесты, риски](README_CV.md)

Quick check: `python -m pip install -r requirements.txt && python -m unittest discover -s tests -v`.

**No ML models; no Telegram token is included.** `config.json` is intentionally untracked (see `config.example.json`).

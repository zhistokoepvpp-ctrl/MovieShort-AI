# MovieShort AI

**Automatic movie clipping for YouTube Shorts** — intelligently extracts the best moments from full-length movies, crops them to 9:16 vertical format, adds proper subtitles, and prepares them for YouTube Shorts publishing.

> Built with PySceneDetect, faster-whisper, FFmpeg, and LLM scoring (Yandex AI Studio DeepSeek V4 / Gemini).

![Status](https://img.shields.io/badge/status-stable-green)
![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-3776AB?logo=python&logoColor=white)
![FFmpeg](https://img.shields.io/badge/ffmpeg-required-007ACC?logo=ffmpeg&logoColor=white)
![Platform](https://img.shields.io/badge/platform-Windows-0078D4?logo=windows&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Release](https://img.shields.io/github/v/release/zhistokoepvpp-ctrl/MovieShort-AI)

---

<p align="center">
  <b>🌐 English</b> &nbsp;|&nbsp; <a href="#русский">Русский</a>
</p>

---

## English

**MovieShort AI** automatically extracts the best moments from full-length movies and prepares them for YouTube Shorts:

- Scene detection → transcription → LLM scoring → vertical crop → subtitles
- AI-powered clip selection with knowledge-aware scoring (1-10)
- Fully automated pipeline — just select a movie and pick the best clips

### Features

- **Smart scene detection** — PySceneDetect finds scene boundaries, merges short scenes, splits by dialogue pauses
- **AI-powered clip selection** — LLM (DeepSeek / Gemini) scores each scene (1-10) with names, uses knowledge about the movie
- **Face tracking** — OpenCV Haar cascade detects faces, centers the crop on the most active area
- **Vertical crop (9:16)** — Professional vertical format for YouTube Shorts with blurred background
- **Word-level subtitles** — faster-whisper transcription with word-level timestamps, customizable font/size/color/position
- **Batch processing** — Process multiple movies in sequence with full progress tracking
- **Smart centering** — For long scenes, finds the dialogue-dense window to start the clip
- **Diversity filter** — Evenly distributes clips across the full movie runtime
- **Bilingual interface** — English and Russian UI, translatable LLM prompts
- **Anti-copyright mode** — Horizontal flip + contrast correction to avoid Content ID flags

### Quick Start

```bash
# 1. Install (one-time)
setup.bat

# 2. Run
run.bat
```

Opens at `http://localhost:7860`.

#### Prerequisites

- **Python 3.9–3.11**
- **FFmpeg** (auto-installed via `setup.bat`)
- **NVIDIA GPU** (optional, speeds up transcription 5-10x)

### Interface

#### Modes

| Mode | Description |
|------|-------------|
| **Contextual (LLM)** | Default. LLM receives all scenes with transcription and selects the best ones by score (1-10) with names. |
| **Standard (no LLM)** | Random scene selection. No API key required. No clip names generated. |
| **Manual** | Enter custom timestamps — clips are cut exactly at those points. |

#### Editing Options

| Option | Default | Description |
|--------|---------|-------------|
| Subtitles | On | Speech recognition + subtitle overlay |
| Smart centering | On | Face detection, centers frame on active area |
| Banner areas | On | Top/bottom padding for banners (300px configurable) |
| Blurred background | On | Fills empty space with blurred video |
| Anti-copyright | On | Mirror + contrast for Content ID avoidance |

### Setup

#### Yandex AI Studio (recommended)

1. Get an API key from [Yandex AI Studio](https://console.cloud.yandex.com/folders)
2. Go to **Settings** in the app → Provider: **Yandex AI Studio**
3. Enter **API Key** and **Folder ID**, select model (e.g., `deepseek-v4-flash`)
4. Click **Verify key**

#### Gemini

1. Get an API key from [Google AI Studio](https://aistudio.google.com/apikey)
2. Go to **Settings** > **Gemini** and paste the key

### Pipeline

1. **Scene detection** — PySceneDetect finds boundaries by frame changes
2. **Short scene merging** — Buffer algorithm merges short scenes (max 120s)
3. **Transcription** — faster-whisper (medium) once for the full movie
4. **Pause splitting** — Scenes split at dialogue pauses >2s
5. **LLM scoring** — Batch by 30 scenes, score 1-10 with names
6. **Clip selection** — Score ≥ 7, max 20 clips, diversity filter
7. **FFmpeg processing** — Cut → 9:16 crop → subtitles → blurred background → 1080×1920

### License

[MIT](LICENSE) — free to use, modify, and distribute.

---

## Русский

**MovieShort AI** — автоматическая нарезка фильмов на YouTube Shorts. Интеллектуально извлекает лучшие моменты из полнометражных фильмов, обрезает их в вертикальный формат 9:16, добавляет субтитры и готовит к публикации.

> Сделано с PySceneDetect, faster-whisper, FFmpeg и скорингом через LLM (Yandex AI Studio DeepSeek V4 / Gemini).

### Возможности

- **Умный поиск сцен** — PySceneDetect находит границы сцен, объединяет короткие, делит по паузам в диалогах
- **AI-выбор клипов** — LLM (DeepSeek / Gemini) оценивает каждую сцену (1-10) и даёт название, используя знания о фильме
- **Face tracking** — OpenCV находит лица, центрирует кадр по самой активной области
- **Вертикальный кроп (9:16)** — Профессиональный формат YouTube Shorts с размытым фоном
- **Субтитры по словам** — faster-whisper с таймстемпами на уровне слов; шрифт, размер, цвет, положение настраиваются
- **Пакетная обработка** — Несколько фильмов подряд с полным отслеживанием прогресса
- **Smart centering** — Для длинных сцен находит окно с максимальной плотностью диалога
- **Diversity-фильтр** — Равномерно распределяет клипы по всему фильму
- **Двуязычный интерфейс** — Русский и английский язык интерфейса, промпты LLM на языке фильма
- **Anti-copyright** — Зеркало + коррекция контраста для обхода Content ID

### Быстрый старт

```bash
# 1. Установка (один раз)
setup.bat

# 2. Запуск
run.bat
```

Откроется `http://localhost:7860`.

#### Зависимости

- **Python 3.9–3.11**
- **FFmpeg** (установится автоматически через `setup.bat`)
- **NVIDIA GPU** (опционально, ускоряет транскрипцию в 5-10×)

### Интерфейс

#### Режимы работы

| Режим | Описание |
|-------|----------|
| **Контекстный (LLM)** | По умолчанию. LLM получает все сцены с транскрипцией и сама выбирает лучшие по оценке (1-10) с названиями. |
| **Стандартный (без LLM)** | Случайный выбор сцен. Не требует API-ключа. Названия не генерируются. |
| **Ручной** | Укажи таймкоды — программа нарежет клипы по ним. |

#### Опции монтажа

| Опция | По умолч. | Описание |
|-------|-----------|----------|
| Субтитры | Вкл | Распознавание речи + наложение субтитров |
| Smart centering | Вкл | Детекция лиц, центровка кадра |
| Баннерные поля | Вкл | Поля сверху/снизу для баннеров (300px) |
| Размытый фон | Вкл | Заполняет пустое пространство размытым видео |
| Anti-copyright | Вкл | Зеркало + контраст для обхода Content ID |

### Настройка

#### Yandex AI Studio (рекомендуется)

1. Получи ключ: [Yandex AI Studio](https://console.cloud.yandex.com/folders) → создать сервисный аккаунт
2. В интерфейсе: **Settings** → провайдер **Yandex AI Studio**
3. Введи **API Key**, **Folder ID**, выбери модель (например, `deepseek-v4-flash`)
4. Нажми **Verify key** — зелёный = работает

#### Gemini

1. Получи ключ: [Google AI Studio](https://aistudio.google.com/apikey)
2. Вставь API-ключ в **Settings** > **Gemini** — приложение само определит OpenRouter или Google AI

### Пайплайн обработки

1. **Детекция сцен** — PySceneDetect находит границы сцен по изменению кадра
2. **Слияние коротких сцен** — Буферный алгоритм объединяет короткие сцены (макс. 120с)
3. **Транскрипция** — faster-whisper (medium) распознаёт речь 1 раз для всего фильма
4. **Разбивка по паузам** — Сцены делятся по паузам в диалоге >2с
5. **LLM скоринг** — Батчами по 30 сцен; оценка 1-10 + название
6. **Выбор клипов** — Оценка ≥ 7, макс. 20 клипов, diversity-фильтр
7. **FFmpeg обработка** — Нарезка → кроп 9:16 → субтитры → размытый фон → 1080×1920

### Структура проекта

```
MovieShort-AI/
├── main.py              # Точка входа (+ monkey-patch gradio_client)
├── config.py            # Конфигурация (Whisper, FFmpeg, API, баннеры)
├── core/
│   ├── pipeline.py      # Пайплайн обработки одного клипа
│   ├── processor.py     # Face tracking + вертикальный кроп
│   ├── batch.py         # Пакетная обработка фильма
│   └── subtitle.py      # Генерация SRT (word-group, JSON persistence)
├── analyzers/
│   ├── detector.py      # Стандартный режим (без LLM)
│   ├── scene_analyzer.py# Детекция сцен, слияние, транскрипция
│   └── text_analyzer.py # LLM анализ (промпты, API, парсинг)
├── gui/
│   └── app.py           # Gradio веб-интерфейс
├── utils/
│   ├── ffmpeg_utils.py  # FFmpeg команды
│   ├── user_config.py   # Сохранение настроек
│   └── log_capture.py   # Захват лога для GUI
├── models/
│   └── haarcascade_frontalface_default.xml
├── output/              # Выходные файлы (создаётся автоматически)
├── requirements.txt     # Зависимости
├── setup.bat           # Установка (venv + CUDA)
└── run.bat             # Запуск
```

### Технические детали

#### Видео на выходе

- **Полный кадр** — исходное 16:9 масштабируется до 1080×1320 (контентная область)
- **Размытый фон** (вкл/выкл) — контент накладывается на размытый full-frame фон (1080×1920)
- **Баннерные поля** (вкл/выкл, настраивается) — поля сверху/снизу для баннеров
- **Anti-copyright** (вкл/выкл) — горизонтальный флип + коррекция контраста/яркости

#### Субтитры

- **Word-group SRT** — группировка слов по 2-4, одна строка без переносов
- **Выше баннера** — субтитры подняты на `position_y` px от низа
- **Точная синхронизация** — по word-level таймстемпам Whisper
- **Транскрипция фильма** делается 1 раз, сохраняется в JSON, переиспользуется для всех клипов

#### Smart Centering

Для длинных сцен (>60с) алгоритм ищет окно с максимальной плотностью слов — клип начинается там, где самый насыщенный диалог. Шаг скользящего окна — 1с.

#### GPU auto-detection

При запуске Whisper: проверка CUDA → проба compute types (`float16` → `int8_float16` → CPU `int8`) → печать модели GPU и расчётного времени.

#### Слияние сцен (без снежного кома)

Буферный алгоритм: короткие сцены (<15с) объединяются, но никогда не расширяют уже готовую сцену. Паузы >2с создают границу. Итог: осмысленные фрагменты 15-120с вместо гигантских 400-секундных сцен.

### Лицензия

[MIT](LICENSE) — можно свободно использовать, модифицировать и распространять.

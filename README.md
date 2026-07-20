# MovieShort AI

**Automatic movie clipping for YouTube Shorts** — intelligently extracts the best moments from full-length movies, crops them to 9:16 vertical format, adds proper subtitles, and prepares them for YouTube Shorts publishing.

> Built with PySceneDetect, faster-whisper, FFmpeg, and LLM scoring (Yandex AI Studio DeepSeek V4 / Gemini).

![Status](https://img.shields.io/badge/status-stable-green)
![Python](https://img.shields.io/badge/python-3.9--3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

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

## Quick Start

```bash
# 1. Install (one-time)
setup.bat

# 2. Run
run.bat
```

Opens at `http://localhost:7860`.

### Prerequisites

- **Python 3.9–3.11**
- **FFmpeg** (auto-installed via `setup.bat` using winget, or manual download)
- **NVIDIA GPU** (optional, speeds up transcription 5-10x)

## Interface

### Modes

| Mode | Description |
|------|-------------|
| **Contextual (LLM)** | Default. LLM receives all scenes with transcription and selects the best ones by score (1-10) with names. No scene matching needed. |
| **Standard (no LLM)** | Random scene selection. No API key required. No clip names generated. |
| **Manual** | Enter custom timestamps — clips are cut exactly at those points. |

### Editing Options

| Option | Default | Description |
|--------|---------|-------------|
| Subtitles | On | Speech recognition + subtitle overlay |
| Smart centering | On | Face detection, centers frame on active area |
| Banner areas | On | Top/bottom padding for banners (300px configurable) |
| Blurred background | On | Fills empty space with blurred video |
| Anti-copyright | On | Mirror + contrast adjustment for Content ID avoidance |

### Subtitle Editor (Settings > Subtitle Editor)

Customize font, size (8-48px), color (6 options), outline (0-5), bold/italic/shadow, bottom margin (50-800px). **Live preview** updates in real time.

## Configuration

### config.py

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BANNER_TOP` | 300 px | Top banner padding |
| `BANNER_BOTTOM` | 300 px | Bottom banner padding |
| `FACE_TRACKING_INTERVAL` | 5 | Analyze every Nth frame |
| `WHISPER_MODEL` | medium | tiny/base/small/medium/large-v3 |
| `WHISPER_BEAM_SIZE` | 5 | Transcription quality |
| `SCENE_THRESHOLD` | 27.0 | Scene detector sensitivity |
| `MIN_SCENE_DURATION` | 15s | Minimum scene duration after merge |
| `MAX_MERGE_DURATION` | 120s | Maximum merge duration |
| `DIALOGUE_PAUSE_THRESHOLD` | 2s | Pause >2s creates scene boundary |
| `DEFAULT_MIN_CLIP_DURATION` | 15s | Minimum clip length |
| `DEFAULT_MAX_CLIP_DURATION` | 60s | Maximum clip length |
| `VERTICAL_WIDTH` | 1080 | Output video width |
| `VERTICAL_HEIGHT` | 1920 | Output video height |

## Setup

### Yandex AI Studio (recommended)

1. Get an API key from [Yandex AI Studio](https://console.cloud.yandex.com/folders)
2. Go to **Settings** in the app
3. Provider: **Yandex AI Studio**
4. Enter **API Key** and **Folder ID**
5. Select model (e.g., `deepseek-v4-flash`)
6. Click **Verify key** — green = ready

### Gemini

1. Get an API key from [Google AI Studio](https://aistudio.google.com/apikey)
2. Go to **Settings** > **Gemini**
3. Paste the API key — the app auto-detects OpenRouter or Google AI

### GPU Acceleration

During `setup.bat`, you will be asked: *"Install PyTorch with CUDA? [Y/N]"*

- **Y** > Installs PyTorch with CUDA 12.4 (requires NVIDIA GPU)
- **N** > Installs CPU-only version (slower transcription)

Verify: run `run.bat` — the log will show `GPU detected: ...` or `CUDA not found`.

## Pipeline

1. **Scene detection** — PySceneDetect finds boundaries by frame changes (with progress bar)
2. **Short scene merging** — Buffer algorithm merges short scenes into meaningful segments (max 120s)
3. **Transcription** — faster-whisper recognizes speech once for the full movie; word-level timestamps; GPU auto-detection
4. **Pause splitting** — Scenes split at dialogue pauses >2s (continuous conversation per scene)
5. **LLM scoring** — Batch by 30 scenes; model scores each (1-10) with a name; knowledge-aware prompts
6. **Clip selection** — Score >= threshold (default 7), max 20 clips; diversity filter distributes across movie
7. **Processing** — FFmpeg: cut > 9:16 crop > subtitles > blurred background > full 1080x1920 frame

## Technical Details

### Video Output

- **Full frame** — Original 16:9 video scaled to 1080x1320 (content area)
- **Blurred background** (toggle) — Content area overlaid on blurred full-frame background (1080x1920)
- **Banner areas** (toggle, configurable) — Top/bottom padding for YouTube Shorts banners
- **Anti-copyright** (toggle) — Horizontal flip + contrast/brightness adjustment

### Subtitle Pipeline

- **Word-group SRT** — Groups 2-4 words per line, no line breaks
- **Above banner** — Subtitles raised by `position_y` px from bottom
- **Precise sync** — Word-level Whisper timestamps
- **Full movie transcription** — Done once, cached as JSON, reused for all clips

### Smart Centering

For long scenes (>60s), the algorithm finds the window with maximum word density — the clip starts at the most dialogue-rich section, not at scene start. Sliding window step: 1s.

### GPU Auto-detection

On Whisper startup:
1. Checks CUDA availability
2. Tries compute types: `float16` > `int8_float16` > CPU `int8`
3. Prints GPU model and estimated transcription time

### Scene Merging (No Snowball)

Buffer algorithm: short scenes (<15s) merge but never expand an already-formed scene. Dialogue pauses >2s create scene boundaries. Result: meaningful 15-120s fragments instead of giant 400s scenes.

### Pinned Versions (requirements.txt)

- **gradio 4.44.1** — Stable, tested compatibility
- **gradio_client 1.3.0** — Compatible with gradio 4.x
- **huggingface_hub 0.22.2** — Last version with HfFolder
- **scenedetect <1.0** — API 0.6.x differs from 1.x

### Monkey-patch (main.py)

`gradio_client==1.3.0` has a bug: if JSON Schema contains `additionalProperties: false`, the parser crashes with `TypeError: argument of type 'bool' is not iterable`. `main.py` contains a monkey-patch that fixes this before importing Gradio.

## Project Structure

```
MovieShort-AI/
├── main.py              # Entry point (+ gradio_client monkey-patch)
├── config.py            # Configuration (Whisper, FFmpeg, API, banners)
├── core/
│   ├── pipeline.py      # Single clip processing pipeline
│   ├── processor.py     # Face tracking + vertical crop
│   ├── batch.py         # Batch movie processing
│   └── subtitle.py      # SRT generation (word-group, JSON persistence)
├── analyzers/
│   ├── detector.py      # Standard mode (no LLM)
│   ├── scene_analyzer.py# Scene detection, merging, transcription
│   └── text_analyzer.py # LLM analysis (prompts, API, parsing)
├── gui/
│   └── app.py           # Gradio web interface
├── utils/
│   ├── ffmpeg_utils.py  # FFmpeg commands
│   ├── user_config.py   # Settings persistence
│   └── log_capture.py   # Log capture for GUI
├── models/
│   └── haarcascade_frontalface_default.xml
├── output/              # Generated clips (created automatically)
├── requirements.txt     # Dependencies
├── setup.bat           # Installation (venv + CUDA)
└── run.bat             # Launch
```

## License

[MIT](LICENSE) — free to use, modify, and distribute.

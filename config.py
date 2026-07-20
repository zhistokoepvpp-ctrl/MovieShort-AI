"""
MovieShort AI — Configuration
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = OUTPUT_DIR / "temp"

# Whisper settings
WHISPER_MODEL = "medium"        # tiny/base/small/medium/large-v3
WHISPER_LANGUAGE = "ru"          # Default language for transcription (ru/en)
WHISPER_DEVICE = "auto"          # "auto", "cpu", or "cuda"
FORCE_CPU = False                # True = force CPU even if GPU available
WHISPER_BEAM_SIZE = 5            # Beam size for transcription accuracy

# Video processing defaults
DEFAULT_MAX_CLIP_DURATION = 60   # seconds
DEFAULT_MIN_CLIP_DURATION = 15   # seconds
VERTICAL_WIDTH = 1080
VERTICAL_HEIGHT = 1920

# Banner padding (top/bottom space for banners in shorts)
BANNER_TOP = 300                 # pixels
BANNER_BOTTOM = 300              # pixels

# Face tracking
FACE_TRACKING_INTERVAL = 5       # Analyze every Nth frame

# Scene detection
SCENE_THRESHOLD = 27.0
SCENE_FRAME_SKIP = 2             # Process every N+1th frame (0=all, 2=every 3rd)
MIN_SCENE_DURATION = 15.0        # seconds — merge raw scenes shorter than this
MAX_MERGE_DURATION = 120         # seconds — don't merge beyond this (prevents giant scenes)
DIALOGUE_PAUSE_THRESHOLD = 2.0   # seconds — gap > this = scene boundary

# LLM
LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
LLM_MODEL = "gemini-2.0-flash"

# Anti-copyright measures (slight transformations to avoid Content ID)
ANTI_COPYRIGHT = True           # master toggle
AC_MIRROR = True                # horizontal flip
AC_CONTRAST = 1.05              # 1.0 = no change
AC_BRIGHTNESS = 0.02            # 0.0 = no change
AC_SATURATION = 1.05            # 1.0 = no change

# Processing options (defaults for GUI)
DEFAULT_BANNER_TOP = 300
DEFAULT_BANNER_BOTTOM = 300
DEFAULT_BLUR_BACKGROUND = True
DEFAULT_ANTI_COPYRIGHT = True
DEFAULT_SUBTITLES = True
DEFAULT_FACE_TRACKING = True
DEFAULT_NUM_CLIPS = 10

# Subtitle editor defaults
SUBTITLE_FONT = "Arial"
SUBTITLE_SIZE = 13
SUBTITLE_COLOR = "&H00FFFFFF"
SUBTITLE_OUTLINE = 1
SUBTITLE_BOLD = True
SUBTITLE_ITALIC = False
SUBTITLE_SHADOW = False
SUBTITLE_POSITION_Y = 400       # px from bottom

# LLM Provider: "gemini" or "yandex"
LLM_PROVIDER = "gemini"

# API key (set in Settings → API Key inside the app, or in user_config.json)
GEMINI_API_KEY = ""

# Yandex AI Studio settings
YANDEX_API_KEY = ""
YANDEX_FOLDER_ID = ""
YANDEX_MODEL = "yandexgpt-lite"          # selected model name
# Available Yandex models (for GUI dropdown)
YANDEX_MODEL_LIST = [
    "yandexgpt-lite",           # cheap, fast, 32K context
    "yandexgpt-5.1",            # YandexGPT Pro 5.1, 32K
    "yandexgpt-5-pro",          # YandexGPT Pro 5, 32K
    "yandexgpt-5-lite",         # YandexGPT Lite 5, 32K
    "aliceai-llm",              # Alice AI LLM, 128K
    "aliceai-llm-flash",        # Alice AI LLM Flash, 64K
    "deepseek-v4-flash",        # DeepSeek V4 Flash, 1M context
    "qwen3-235b-a22b-fp8",      # Qwen3 235B, 256K
    "gpt-oss-120b",             # OSS 120B, 128K
    "gpt-oss-20b",             # OSS 20B, 128K
]
YANDEX_BASE_URL = "https://ai.api.cloud.yandex.net/v1"

# Analysis modes
ANALYSIS_MODES = ["standard", "context"]

# Gradio
GRADIO_PORT = 7860
GRADIO_SHARE = False

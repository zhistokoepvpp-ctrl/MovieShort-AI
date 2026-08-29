"""
MovieShort AI — User config persistence (saves/loads user_config.json).

Guard: protected fields (API keys / folder id) are never overwritten with an
empty value while the on-disk file holds a non-empty one. Legitimate clearing
of a key = manual edit of user_config.json or restore from user_config.json.bak
— there is no GUI path for clearing.
"""
import json
import os
import shutil

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "user_config.json"
)

# Fields that must not be wiped by a GUI save with empty inputs.
PROTECTED = ("api_key", "yandex_api_key", "yandex_folder_id", "openrouter_api_key", "opencode_zen_api_key")

DEFAULT_CONFIG = {
    "api_key": "",
    "movie_title": "",
    "num_clips": 10,
    "min_duration": 15,
    "max_duration": 60,
    "subtitles": True,
    "face_tracking": True,
    "llm_provider": "yandex",
    "yandex_api_key": "",
    "yandex_folder_id": "",
    "yandex_model": "deepseek-v4-flash",
    # OpenRouter provider (https://openrouter.ai)
    "openrouter_api_key": "",
    "openrouter_model": "deepseek/deepseek-chat-v3-0324",
    # OpenCode Zen provider (https://opencode.ai/zen)
    "opencode_zen_api_key": "",
    "opencode_zen_model": "nemotron-3-ultra-free",
    # Processing options
    "banner_top": 300,
    "banner_bottom": 300,
    "blur_background": True,
    "anti_copyright": True,
    "analysis_mode": "context",
    "score_threshold": 7.0,
    "film_language": "ru",
    "ui_language": "ru",
    "auto_cleanup": True,
    # Subtitle editor
    "subtitle_font": "Arial",
    "subtitle_font_name": "Bebas Neue",  # display name from utils/font_manager.POPULAR_FONTS
    "subtitle_size": 13,
    "subtitle_color": "&H00FFFFFF",
    "subtitle_outline": 1,
    "subtitle_bold": True,
    "subtitle_italic": False,
    "subtitle_shadow": False,
    "subtitle_position_y": 400,
    # Cost tracking (rub per minute of film with DeepSeek V4 Flash)
    "cost_per_minute": 0.0,
}


def load():
    """Load user config from JSON file, returns dict."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        merged = dict(DEFAULT_CONFIG)
        merged.update(data)
        return merged
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(DEFAULT_CONFIG)


def _is_blank(value):
    """Whitespace-only strings and None count as empty."""
    return value is None or (isinstance(value, str) and not value.strip())


def save(config_dict):
    """Save user config dict to JSON file (atomic tmp+replace, .bak rotation).

    Guard: if a PROTECTED field arrives empty/whitespace while the current
    on-disk file holds a non-empty value, the old value is kept and a warning
    is printed. Clearing a key is only possible by manual edit of
    user_config.json or restore from user_config.json.bak.
    """
    existing = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (OSError, json.JSONDecodeError):
            existing = {}
        if not isinstance(existing, dict):
            existing = {}

    for field in PROTECTED:
        old_val = existing.get(field)
        if _is_blank(config_dict.get(field)) and not _is_blank(old_val):
            config_dict[field] = old_val
            msg = (
                f"⚠ {field}: сохранено существующее значение "
                f"({str(old_val)[-4:]}), пустая перезапись заблокирована; "
                f"очистка ключа — только ручная правка файла"
            )
            try:
                print(msg)
            except UnicodeEncodeError:
                # piped/redirected stdout on cp1251 (RU Windows) can't encode
                # ⚠/Cyrillic — degrade to ASCII instead of breaking save().
                print(msg.encode("ascii", "replace").decode("ascii"))

    bak_path = CONFIG_PATH + ".bak"
    tmp_path = CONFIG_PATH + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        if os.path.exists(CONFIG_PATH):
            shutil.copyfile(CONFIG_PATH, bak_path)
        os.replace(tmp_path, CONFIG_PATH)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

"""
MovieShort AI — Gradio GUI
"""
import os
import re
import threading
import time as time_module
from typing import List, Tuple

import gradio as gr

import config as app_config
from core.pipeline import process_multiple
from core.batch import process_movie
from utils.log_capture import LogCapture
from utils import user_config
from analyzers.text_analyzer import check_api_key
from utils import fmt_duration


def parse_timestamps(text: str) -> List[Tuple[str, str]]:
    pairs = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(
            r"(\d{1,2}:\d{2}:\d{2})\s*[-–]\s*(\d{1,2}:\d{2}:\d{2})", line
        )
        if m:
            pairs.append((m.group(1), m.group(2)))
    return pairs


# ---------------------------------------------------------------------------
# UI language dictionary
# ---------------------------------------------------------------------------

UI = {
    "ru": {
        "tab_manual": "Ручной режим",
        "tab_auto": "Автоматический режим",
        "settings": "Настройки",
        "process": "Обработать",
        "auto_process": "Анализировать и обработать очередь",
        "file_label": "Выберите видео",
        "movie_title": "Название фильма (необязательно)",
        "timestamps": "Таймкоды (start - end)",
        "ts_placeholder": "00:01:30 - 00:02:30\n00:05:00 - 00:06:00",
        "max_len": "Макс. длина клипа (сек)",
        "processing_opts": "Опции монтажа",
        "subs_label": "Субтитры",
        "subs_info": "Распознаёт речь и накладывает субтитры на видео",
        "face_label": "Smart centering (Face tracking)",
        "face_info": "Анализирует положение лиц и центрирует кадр по ним",
        "banner_label": "Баннерные поля",
        "banner_info": "Добавляет поля сверху/снизу для баннеров в YouTube редакторе",
        "banner_top": "Верхнее поле (px)",
        "banner_bottom": "Нижнее поле (px)",
        "blur_label": "Размытый фон",
        "blur_info": "Заполняет пустое пространство размытой копией видео (формат 9:16)",
        "anti_label": "Anti-copyright",
        "anti_info": "Небольшие искажения (отражение, контраст, яркость) для обхода Content ID",
        "film_language": "Язык фильма",
        "film_language_info": "Язык диалогов в фильме. Определяет язык транскрипции и субтитров.",
        "llm_provider": "LLM Провайдер",
        "analysis_mode": "Режим анализа",
        "analysis_mode_info": "Стандартный: детекция сцен + LLM скоринг | Контекстный: LLM видит текст всех сцен → сам выбирает лучшие",
    "analysis_mode_std": "Стандартный",
    "analysis_mode_ctx": "Контекстный (ИИ)",
        "min_len": "Мин. длина (сек)",
        "num_clips": "Количество клипов",
        "score_threshold": "Порог оценки",
        "score_threshold_info": "Минимальная оценка сцены (1-10) для включения в результат",
        "waiting": "Ожидание запуска...",
        "error_no_file": "Ошибка: загрузите видеофайл.",
        "error_no_ts": "Ошибка: укажите хотя бы один промежуток",
        "console": "Консоль отладки",
        "subtitle_editor": "Редактор субтитров",
        "font": "Шрифт",
        "font_size": "Размер",
        "font_color": "Цвет",
        "outline": "Обводка",
        "bold": "Жирный",
        "italic": "Курсив",
        "shadow": "Тень",
        "position_y": "Отступ от низа (px)",
        "preview": "Предпросмотр",
        "preview_text": "Пример текста субтитров",
        "auto_cleanup": "Авто-очистка temp",
        "auto_cleanup_info": "Удалять временные файлы после обработки каждого фильма",
        "ui_language": "Язык интерфейса",
        "batch_files": "Выберите фильмы (можно несколько)",
        "process_all": "Обработать все",
        "process_queue": "Очередь обработки",
        "wait_start": "Ожидание запуска...",
        "apply_lang": "Применить язык",
        "save": "Сохранить",
        "general": "Общие",
        "not_saved": "не сохранено",
        "saved_ok": "✅ Настройки сохранены",
        "saved_sub_ok": "✅ Настройки субтитров сохранены",
        "reset_defaults": "Сбросить к дефолтным",
        "reset_sub_ok": "✅ Настройки субтитров сброшены к заводским",
        "add_to_queue": "Добавить в очередь",
        "queue_empty": "Очередь пуста",
        "restart_for_lang": "✅ Настройки сохранены (перезапустите для смены языка)",
        "support_author": "Поддержать автора:",
        "file_label_suffix": " — перетащите или нажмите для выбора",
        "movie_placeholder": "Например: Матрица, 1+1, Побег из Шоушенка...",
        "queue_header": "Очередь ({n})",
        "llm_provider_info": "Gemini (Google AI) или Yandex AI Studio",
        "lang_russian": "Русский",
        "lang_english": "English",
        "tab_gemini": "Google AI (Gemini)",
        "tab_yandex": "Yandex AI Studio",
        "api_key_label": "API ключ {provider}",
        "save_key": "Сохранить ключ",
        "check_key": "Проверить ключ",
        "status_not_checked": "⏳ не проверен",
        "folder_id_label": "Folder ID",
        "yandex_model_label": "Модель Yandex",
        "yandex_model_info": "DeepSeek V4 Flash — лучшая для названий, YandexGPT Lite — быстрая и дешёвая",
        "cost_info": "💰 Стоимость DeepSeek V4 Flash:",
        "cost_info_text": "от {low} до {high} руб/мин в зависимости от количества клипов",
        "no_api_key_title": "Без API-ключа",
        "no_api_key_desc": "Приложение будет работать, но сцены будут выбираться случайно — без анализа содержания.",
        "no_api_key_works": "Что работает:",
        "no_api_key_works_list": "детекция сцен, транскрипция (Whisper), нарезка клипов, субтитры, face tracking",
        "no_api_key_lost": "Что теряется:",
        "no_api_key_lost_list": "интеллектуальный отбор лучших сцен (LLM скоринг), контекстный режим, генерация названий моментов",
        "no_api_key_where": "Где взять ключ:",
        "no_api_key_free": "(бесплатно)",
        "no_api_key_yandex_free": "(нужна карта для активации, есть бесплатный грант)",
        "color_white": "Белый",
        "color_yellow": "Жёлтый",
        "color_black": "Чёрный",
        "color_red": "Красный",
        "color_cyan": "Голубой",
        "color_green": "Зелёный",
        "file_label_x": "Файл: {name}",
        "timestamps_count": "Таймкодов: {n}",
        "options_label": "Опции: subs={subs}, face={face}, blur={blur}, anti={anti}",
        "processing_elapsed": "Обработка... прошло {time}",
        "error_generic": "ОШИБКА: {msg}",
        "error_short": "Ошибка!",
        "done_count": "Готово: {ok}/{total} клипов",
        "clip_error": "Клип {n}: ошибка обработки",
        "done_count_files": "Готово: {ok}/{total} клипов (файлов: {files})",
        "no_results": "Ошибка: нет результата",
        "no_scenes": "Не найдено подходящих сцен.",
        "no_scenes_short": "Не найдено сцен",
        "total_time": "⏱  Общее время: {time}",
        "movie_header": "MOVIESHORT AI — ФАЙЛ {i}/{total}",
        "movie_file": "Файл: {name}",
        "movie_title_label": "Фильм: {title}",
        "provider_mode": "Провайдер: {prov}, Режим: {mode}",
        "film_lang_label": "Язык: {lang}",
        "no_api_key_warn": "⚠️  API-ключ не задан — сцены будут выбраны случайно!",
        "mode_context": "Контекстный",
        "mode_standard": "Стандартный",
        "processing_file": "[{i}/{total}] {name} — прошло {time}",
        "key_saved": "API ключ сохранён ({provider})",
        "key_saved_check": "Сохранено. Проверка: {error}",
        "api_ok": "✅ API работает",
        "key_valid_quota": "⚠️ Ключ валиден, но {error}",
        "api_error": "❌ {error}",
        "reset_to_defaults": "Сбросить к дефолтным",
        "status_unknown": "статус неизвестен",
    },
    "en": {
        "tab_manual": "Manual mode",
        "tab_auto": "Automatic mode",
        "settings": "Settings",
        "process": "Process",
        "auto_process": "Analyze & process queue",
        "file_label": "Select video file",
        "movie_title": "Movie title (optional)",
        "timestamps": "Timestamps (start - end)",
        "ts_placeholder": "00:01:30 - 00:02:30\n00:05:00 - 00:06:00",
        "max_len": "Max clip length (s)",
        "processing_opts": "Processing options",
        "subs_label": "Subtitles",
        "subs_info": "Recognizes speech and overlays subtitles on video",
        "face_label": "Smart centering (Face tracking)",
        "face_info": "Analyzes face positions and centers the frame on them",
        "banner_label": "Banner padding",
        "banner_info": "Adds top/bottom padding for banners in YouTube editor",
        "banner_top": "Top padding (px)",
        "banner_bottom": "Bottom padding (px)",
        "blur_label": "Blurred background",
        "blur_info": "Fills empty space with a blurred copy of the video (9:16 format)",
        "anti_label": "Anti-copyright",
        "anti_info": "Subtle transformations (mirror, contrast, brightness) to bypass Content ID",
        "film_language": "Film language",
        "film_language_info": "Language of dialogue in the movie. Determines transcription and subtitle language.",
        "llm_provider": "LLM Provider",
        "analysis_mode": "Analysis mode",
        "analysis_mode_info": "Standard: scene detection + LLM scoring | Context: LLM sees all scene transcripts → picks best",
    "analysis_mode_std": "Standard",
    "analysis_mode_ctx": "Context (AI)",
        "min_len": "Min length (s)",
        "num_clips": "Number of clips",
        "score_threshold": "Score threshold",
        "score_threshold_info": "Minimum scene score (1-10) to include in results",
        "waiting": "Waiting...",
        "error_no_file": "Error: please upload a video file.",
        "error_no_ts": "Error: please enter at least one time range",
        "console": "Debug console",
        "subtitle_editor": "Subtitle Editor",
        "font": "Font",
        "font_size": "Size",
        "font_color": "Color",
        "outline": "Outline",
        "bold": "Bold",
        "italic": "Italic",
        "shadow": "Shadow",
        "position_y": "Bottom offset (px)",
        "preview": "Preview",
        "preview_text": "Subtitle text preview",
        "auto_cleanup": "Auto-cleanup temp",
        "auto_cleanup_info": "Delete temporary files after processing each movie",
        "ui_language": "Interface language",
        "batch_files": "Select movie files (multiple allowed)",
        "process_all": "Process all",
        "process_queue": "Processing queue",
        "wait_start": "Waiting...",
        "apply_lang": "Apply language",
        "save": "Save",
        "general": "General",
        "not_saved": "not saved",
        "saved_ok": "✅ Settings saved",
        "saved_sub_ok": "✅ Subtitle settings saved",
        "reset_defaults": "Reset to defaults",
        "reset_sub_ok": "✅ Subtitle settings reset to defaults",
        "add_to_queue": "Add to queue",
        "queue_empty": "Queue is empty",
        "restart_for_lang": "✅ Settings saved (restart to apply language)",
        "support_author": "Support the author:",
        "file_label_suffix": " — drag & drop or click to select",
        "movie_placeholder": "e.g. The Matrix, The Shawshank Redemption...",
        "queue_header": "Queue ({n})",
        "llm_provider_info": "Gemini (Google AI) or Yandex AI Studio",
        "lang_russian": "Russian",
        "lang_english": "English",
        "tab_gemini": "Google AI (Gemini)",
        "tab_yandex": "Yandex AI Studio",
        "api_key_label": "{provider} API key",
        "save_key": "Save key",
        "check_key": "Check key",
        "status_not_checked": "⏳ not checked",
        "folder_id_label": "Folder ID",
        "yandex_model_label": "Yandex Model",
        "yandex_model_info": "DeepSeek V4 Flash — best for titles, YandexGPT Lite — fast and cheap",
        "cost_info": "💰 DeepSeek V4 Flash cost:",
        "cost_info_text": "from {low} to {high} RUB/min depending on clip count",
        "no_api_key_title": "Without API key",
        "no_api_key_desc": "The app will still work, but scenes will be selected randomly — without content analysis.",
        "no_api_key_works": "What works:",
        "no_api_key_works_list": "scene detection, transcription (Whisper), clip cutting, subtitles, face tracking",
        "no_api_key_lost": "What is lost:",
        "no_api_key_lost_list": "intelligent scene selection (LLM scoring), context mode, scene title generation",
        "no_api_key_where": "Where to get the key:",
        "no_api_key_free": "(free)",
        "no_api_key_yandex_free": "(card required for activation, free grant available)",
        "color_white": "White",
        "color_yellow": "Yellow",
        "color_black": "Black",
        "color_red": "Red",
        "color_cyan": "Cyan",
        "color_green": "Green",
        "file_label_x": "File: {name}",
        "timestamps_count": "Timestamps: {n}",
        "options_label": "Options: subs={subs}, face={face}, blur={blur}, anti={anti}",
        "processing_elapsed": "Processing... elapsed {time}",
        "error_generic": "ERROR: {msg}",
        "error_short": "Error!",
        "done_count": "Done: {ok}/{total} clips",
        "clip_error": "Clip {n}: processing error",
        "done_count_files": "Done: {ok}/{total} clips (files: {files})",
        "no_results": "Error: no result",
        "no_scenes": "No suitable scenes found.",
        "no_scenes_short": "No scenes found",
        "total_time": "⏱  Total time: {time}",
        "movie_header": "MOVIESHORT AI — FILE {i}/{total}",
        "movie_file": "File: {name}",
        "movie_title_label": "Movie: {title}",
        "provider_mode": "Provider: {prov}, Mode: {mode}",
        "film_lang_label": "Language: {lang}",
        "no_api_key_warn": "⚠️  No API key set — scenes will be selected randomly!",
        "mode_context": "Context",
        "mode_standard": "Standard",
        "processing_file": "[{i}/{total}] {name} — elapsed {time}",
        "key_saved": "API key saved ({provider})",
        "key_saved_check": "Saved. Check: {error}",
        "api_ok": "✅ API is working",
        "key_valid_quota": "⚠️ Key is valid, but {error}",
        "api_error": "❌ {error}",
        "reset_to_defaults": "Reset to defaults",
        "status_unknown": "status unknown",
    },
}

LANG_RU = "ru"
LANG_EN = "en"


def _t(key, lang="ru", default=None):
    """Translate a UI string by key and language.

    Args:
        key: translation key
        lang: language code ('ru' or 'en')
        default: fallback if key not found (defaults to key itself)
    """
    fallback = key if default is None else default
    return UI.get(lang, UI["ru"]).get(key, fallback)


def _get_font_style(cfg):
    """Build font_style dict from user config."""
    return {
        "font": cfg.get("subtitle_font", "Arial"),
        "size": cfg.get("subtitle_size", 13),
        "color": cfg.get("subtitle_color", "&H00FFFFFF"),
        "outline": cfg.get("subtitle_outline", 1),
        "bold": cfg.get("subtitle_bold", True),
        "italic": cfg.get("subtitle_italic", False),
        "shadow": cfg.get("subtitle_shadow", False),
        "position_y": cfg.get("subtitle_position_y", 400),
    }


def _make_subtitle_preview_html(font_style: dict, lang="ru") -> str:
    """Build an HTML preview of subtitle text with the given font_style."""
    fs = font_style
    font_family = fs.get("font", "Arial")
    font_size = fs.get("size", 13)
    font_color = fs.get("color", "&H00FFFFFF")
    # Convert ASS color &HAABBGGRR to CSS #RRGGBB
    if font_color.startswith("&H"):
        hex_part = font_color[2:].zfill(8)
        b, g, r = hex_part[2:4], hex_part[4:6], hex_part[6:8]
        css_color = f"#{r}{g}{b}"
    else:
        css_color = "#FFFFFF"
    outline_w = fs.get("outline", 1)
    bold = "bold" if fs.get("bold", True) else "normal"
    italic = "italic" if fs.get("italic", False) else "normal"
    shadow = fs.get("shadow", False)
    position_y = fs.get("position_y", 400)
    shadow_style = "2px 2px 4px rgba(0,0,0,0.8)," if shadow else ""
    preview_text = _t("preview_text", lang)

    return f"""<div style="
    width:100%;height:120px;
    background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
    border-radius:8px;display:flex;align-items:flex-end;
    justify-content:center;padding:0 20px 20px 20px;
    box-sizing:border-box;overflow:hidden;
">
<div style="
    font-family:'{font_family}',Arial,sans-serif;
    font-size:{font_size}px;
    color:{css_color};
    font-weight:{bold};
    font-style:{italic};
    text-shadow:{shadow_style}0 0 2px rgba(0,0,0,0.8);
    background:rgba(0,0,0,0.3);
    padding:6px 14px;
    border-radius:4px;
    text-align:center;
    max-width:90%;word-wrap:break-word;
">{preview_text}</div>
</div>"""


def _get_default_opts(cfg):
    """Build default checkbox values from config."""
    opts = []
    if cfg.get("subtitles", True):
        opts.append("Субтитры")
    if cfg.get("face_tracking", True):
        opts.append("Face tracking")
    return opts


def _make_progress_html(pct: float, label: str = "") -> str:
    """Build an inline HTML progress bar with label."""
    color = "#4CAF50" if pct < 80 else "#2196F3"
    pct_clamped = max(0, min(100, pct))
    escaped_label = label.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    bar_style = (
        f"width:{pct_clamped:.0f}%;height:100%;"
        f"background:{color};border-radius:12px;"
        f"transition:width 0.5s ease;"
    )
    return f"""<div style="margin:12px 0;">
  <div style="display:flex;justify-content:space-between;font-size:13px;color:#444;margin-bottom:2px;">
    <span>{escaped_label}</span>
    <span style="font-weight:bold;">{pct_clamped:.0f}%</span>
  </div>
  <div style="width:100%;height:24px;background:#e8e8e8;border-radius:12px;overflow:hidden;box-shadow:inset 0 2px 4px rgba(0,0,0,0.08);">
    <div style="{bar_style}"></div>
  </div>
</div>"""


def _check_for_update():
    """Check GitHub for newer release. Returns banner HTML or empty string."""
    try:
        import httpx
        import html as html_mod
        resp = httpx.get(
            "https://api.github.com/repos/zhistokoepvpp-ctrl/MovieShort-AI/releases/latest",
            timeout=5,
        )
        if resp.status_code != 200:
            return ""

        data = resp.json()
        tag_name = data.get("tag_name", "")
        if not tag_name.startswith("v"):
            return ""

        remote_ver = tag_name.lstrip("v")
        local_ver = getattr(app_config, "APP_VERSION", "0.0.0")

        remote_parts = [int(x) for x in remote_ver.split(".")]
        local_parts = [int(x) for x in local_ver.split(".")]
        while len(remote_parts) < 3:
            remote_parts.append(0)
        while len(local_parts) < 3:
            local_parts.append(0)

        if remote_parts <= local_parts:
            return ""

        html_url = data.get("html_url", "#")
        body = data.get("body", "")[:300]

        safe_tag = html_mod.escape(tag_name)
        safe_body = html_mod.escape(body[:200].strip())
        safe_url = html_mod.escape(html_url)

        return f"""<div id="update-banner" style="
background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
color:white;padding:14px 20px;border-radius:10px;
margin-bottom:16px;display:flex;align-items:center;
justify-content:space-between;flex-wrap:wrap;gap:10px;
box-shadow:0 4px 15px rgba(102,126,234,0.4);
">
<div style="display:flex;align-items:center;gap:12px;flex:1;min-width:200px;">
    <span style="font-size:24px;">🎬</span>
    <div>
        <div style="font-weight:bold;font-size:15px;">
            Доступна новая версия {safe_tag}
        </div>
        <div style="font-size:13px;opacity:0.9;margin-top:2px;">
            {safe_body}
        </div>
    </div>
</div>
<div style="display:flex;gap:8px;align-items:center;">
    <a href="{safe_url}" target="_blank"
       style="display:inline-block;padding:8px 20px;
              background:white;color:#667eea;border-radius:6px;
              text-decoration:none;font-weight:600;font-size:14px;">
        👀 Смотреть
    </a>
    <button onclick="this.parentElement.parentElement.style.display='none'"
            style="background:transparent;color:white;border:1px solid rgba(255,255,255,0.5);
                   border-radius:6px;padding:8px 12px;cursor:pointer;font-size:16px;">
        ✕
    </button>
</div>
</div>"""
    except Exception:
        return ""


def create_app() -> gr.Blocks:
    # Load persisted settings
    cfg = user_config.load()
    # Actual cost per minute from billing data: 31 руб / 246 мин = 0.13 руб/мин
    cfg["cost_per_minute"] = 0.13
    ui_lang = cfg.get("ui_language", "ru")

    # Sync API key to runtime config
    import config as cfg_module
    if cfg.get("api_key"):
        cfg_module.GEMINI_API_KEY = cfg["api_key"]

    with gr.Blocks(
        title="MovieShort AI",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("# MovieShort AI")

        # Donation buttons — always visible above tabs
        with gr.Row():
            donate_text = _t("support_author", ui_lang)
            gr.HTML(
                value=f"""<div style="text-align:center;margin:-8px 0 6px 0;font-size:14px;">
  <span style="color:#666;">{donate_text}</span>
  <a href="https://donatex.gg/donate/nzeronfourme" target="_blank"
     style="display:inline-block;padding:4px 14px;margin:0 6px;
            background:#FF6B35;color:white;border-radius:6px;
            text-decoration:none;font-weight:500;font-size:13px;">
    💸 DonateX
  </a>
  <a href="https://boosty.to/nzeronfourme/donate" target="_blank"
     style="display:inline-block;padding:4px 14px;margin:0 6px;
            background:#F15A24;color:white;border-radius:6px;
            text-decoration:none;font-weight:500;font-size:13px;">
    🎗 Boosty
  </a>
</div>"""
            )

        # Update notification banner (checked synchronously at startup)
        _banner_html = _check_for_update()
        if _banner_html:
            gr.HTML(value=_banner_html)

        with gr.Tabs():
            # ── Tab 1: Manual ──────────────────────────────────
            with gr.Tab(_t("tab_manual", ui_lang)):
                video_file = gr.File(
                    label=_t("file_label", ui_lang),
                    file_types=[".mp4", ".avi", ".mkv", ".mov"],
                )
                timestamps_box = gr.Textbox(
                    label=_t("timestamps", ui_lang),
                    lines=4,
                    placeholder=_t("ts_placeholder", ui_lang),
                )
                with gr.Group():
                    gr.Markdown(f"### {_t('processing_opts', ui_lang)}")
                    m_subs = gr.Checkbox(value=cfg.get("subtitles", True),
                        label=_t("subs_label", ui_lang),
                        info=_t("subs_info", ui_lang))
                    m_face = gr.Checkbox(value=cfg.get("face_tracking", True),
                        label=_t("face_label", ui_lang),
                        info=_t("face_info", ui_lang))
                    m_banner = gr.Checkbox(value=True,
                        label=_t("banner_label", ui_lang),
                        info=_t("banner_info", ui_lang))
                    with gr.Row():
                        m_banner_top = gr.Slider(0, 500, value=cfg.get("banner_top", 300),
                            label=_t("banner_top", ui_lang))
                        m_banner_bottom = gr.Slider(0, 500, value=cfg.get("banner_bottom", 300),
                            label=_t("banner_bottom", ui_lang))
                    m_blur = gr.Checkbox(value=cfg.get("blur_background", True),
                        label=_t("blur_label", ui_lang),
                        info=_t("blur_info", ui_lang))
                    m_anti = gr.Checkbox(value=cfg.get("anti_copyright", True),
                        label=_t("anti_label", ui_lang),
                        info=_t("anti_info", ui_lang))
                max_clip_slider = gr.Slider(
                    minimum=15, maximum=120, value=60, step=5,
                    label=_t("max_len", ui_lang),
                )
                run_btn = gr.Button(_t("process", ui_lang), variant="primary")
                manual_progress = gr.HTML(
                    value=_make_progress_html(0, _t("wait_start", ui_lang)),
                    elem_id="manual-progress",
                )
                manual_log = gr.Textbox(
                    label=_t("console", ui_lang), lines=10, max_lines=20,
                    interactive=False, elem_id="console-log",
                    value=_t("wait_start", ui_lang) + "\n"
                )

                def on_process(file, ts_text, subs, face, banner, bt, bb, blur, anti, max_dur, _lang=ui_lang):
                    if file is None:
                        yield (_t("error_no_file", _lang),
                               _make_progress_html(0, _t("error_no_file", _lang)))
                        return
                    pairs = parse_timestamps(ts_text or "")
                    if not pairs:
                        yield (_t("error_no_ts", _lang),
                               _make_progress_html(0, _t("error_no_ts", _lang)))
                        return

                    video_path = file.name if hasattr(file, 'name') else str(file)
                    _options = {
                        "subtitles": subs,
                        "face_tracking": face,
                        "max_duration": max_dur,
                        "anti_copyright": anti,
                        "blur_background": blur,
                        "banner_top": bt,
                        "banner_bottom": bb,
                    }

                    capture = LogCapture()
                    capture.start_capture()

                    print(f"Файл: {os.path.basename(video_path)}")
                    print(f"Таймкодов: {len(pairs)}")
                    print(f"Опции: subs={subs}, face={face}, blur={blur}, anti={anti}")

                    results_container = []
                    error_container = []

                    def worker():
                        try:
                            results = process_multiple(video_path, pairs, _options)
                            results_container.append(results)
                        except Exception as e:
                            error_container.append(str(e))
                            import traceback
                            error_container.append(traceback.format_exc())

                    thread = threading.Thread(target=worker, daemon=True)
                    thread.start()

                    all_lines = []
                    start_ts = time_module.time()
                    while thread.is_alive():
                        new_lines = capture.get_new_lines()
                        if new_lines:
                            all_lines.extend(new_lines)
                        elapsed = time_module.time() - start_ts
                        pct = min(95, int(elapsed / 120 * 100))
                        label = f"Обработка... прошло {_fmt_duration(elapsed)}"
                        yield ("\n".join(all_lines[-40:]),
                               _make_progress_html(pct, label))
                        time_module.sleep(0.3)

                    thread.join(timeout=2)
                    new_lines = capture.get_new_lines()
                    if new_lines:
                        all_lines.extend(new_lines)

                    if error_container:
                        print(f"\nОШИБКА: {error_container[0]}")
                        label = "Ошибка!"
                    elif results_container:
                        results = results_container[0]
                        successes = [r for r in results if r is not None]
                        print(f"\nГотово: {len(successes)}/{len(results)} клипов")
                        for r in successes:
                            print(f"  + {os.path.basename(r)}")
                        for i, r in enumerate(results):
                            if r is None:
                                print(f"  - Клип {i+1}: ошибка обработки")
                        label = f"Готово: {len(successes)}/{len(results)} клипов"
                        elapsed_total = time_module.time() - start_ts
                        print(f"\n⏱  Общее время: {_fmt_duration(elapsed_total)}")
                    else:
                        print("\nОшибка: нет результата")
                        label = "Ошибка"

                    capture.stop_capture()
                    yield ("\n".join(capture.get_all()[-40:]),
                           _make_progress_html(100, label))

                run_btn.click(
                    fn=on_process,
                    inputs=[video_file, timestamps_box,
                            m_subs, m_face, m_banner, m_banner_top, m_banner_bottom,
                            m_blur, m_anti, max_clip_slider],
                    outputs=[manual_log, manual_progress],
                )

            # ── Tab 2: Automatic ───────────────────────────────
            with gr.Tab(_t("tab_auto", ui_lang)):
                queue_state = gr.State([])

                auto_file = gr.File(
                    label=_t("file_label", ui_lang) + _t("file_label_suffix", ui_lang),
                    file_count="single",
                )
                with gr.Row():
                    movie_title_box = gr.Textbox(
                        label=_t("movie_title", ui_lang),
                        placeholder=_t("movie_placeholder", ui_lang),
                        scale=3,
                    )
                    add_queue_btn = gr.Button("➕ " + _t("add_to_queue", ui_lang),
                        variant="secondary", scale=1, min_width=140)
                queue_display = gr.HTML(value='<div style="color:gray;padding:8px"><i>' + _t("queue_empty", ui_lang) + '</i></div>')

                def _render_queue(q):
                    if not q:
                        return '<div style="color:gray;padding:8px"><i>' + _t("queue_empty", ui_lang) + '</i></div>'
                    items = ""
                    for i, item in enumerate(q):
                        title_part = f" — {item['title']}" if item.get("title") else ""
                        items += f'<div style="padding:6px 10px;border-bottom:1px solid #ddd;display:flex;align-items:center">'
                        items += f'<span style="margin-right:8px;font-weight:bold;color:#888">{i+1}.</span>'
                        items += f'<span>📁 {item["name"]}{title_part}</span></div>'
                    n = len(q)
                    label = _t("queue_header", ui_lang).format(n=n)
                    return f'<div style="border:1px solid #ccc;border-radius:6px;max-height:200px;overflow-y:auto"><div style="padding:6px 10px;background:#f5f5f5;font-weight:bold;border-bottom:1px solid #ccc">{label}</div>{items}</div>'

                def _add_to_queue(q, file, title):
                    if file is None:
                        return q, _render_queue(q), None, title
                    import os
                    fpath = file.name if hasattr(file, "name") else str(file)
                    new_item = {"path": fpath, "title": title or "", "name": os.path.basename(fpath)}
                    new_q = list(q or []) + [new_item]
                    return new_q, _render_queue(new_q), None, ""

                add_queue_btn.click(
                    fn=_add_to_queue,
                    inputs=[queue_state, auto_file, movie_title_box],
                    outputs=[queue_state, queue_display, auto_file, movie_title_box],
                )
                llm_provider = gr.Radio(
                    choices=["Gemini", "Yandex"],
                    value="Gemini" if cfg.get("llm_provider", "gemini") == "gemini" else "Yandex",
                    label=_t("llm_provider", ui_lang),
                    info=_t("llm_provider_info", ui_lang),
                )
                analysis_mode = gr.Radio(
                    choices=[_t("analysis_mode_std", ui_lang, "Стандартный"),
                             _t("analysis_mode_ctx", ui_lang, "Контекстный (ИИ)")],
                    value=_t("analysis_mode_ctx", ui_lang, "Контекстный (ИИ)") if cfg.get("analysis_mode", "context") == "context" else _t("analysis_mode_std", ui_lang, "Стандартный"),
                    label=_t("analysis_mode", ui_lang),
                    info=_t("analysis_mode_info", ui_lang),
                )
                film_lang = gr.Radio(
                    choices=[_t("lang_russian", ui_lang), _t("lang_english", ui_lang)],
                    value=_t("lang_english", ui_lang) if cfg.get("film_language", "ru") == "en" else _t("lang_russian", ui_lang),
                    label=_t("film_language", ui_lang),
                    info=_t("film_language_info", ui_lang),
                )
                with gr.Row():
                    min_dur = gr.Slider(
                        minimum=15, maximum=60,
                        value=cfg.get("min_duration", 15),
                        label=_t("min_len", ui_lang)
                    )
                    max_dur2 = gr.Slider(
                        minimum=30, maximum=120,
                        value=cfg.get("max_duration", 60),
                        label=_t("max_len", ui_lang)
                    )
                with gr.Row():
                    num_clips = gr.Slider(
                        minimum=5, maximum=20, step=1,
                        value=cfg.get("num_clips", 10),
                        label=_t("num_clips", ui_lang)
                    )
                    score_thresh = gr.Slider(
                        minimum=1, maximum=10, step=0.5,
                        value=cfg.get("score_threshold", 7.0),
                        label=_t("score_threshold", ui_lang),
                        info=_t("score_threshold_info", ui_lang),
                    )
                with gr.Group():
                    gr.Markdown(f"### {_t('processing_opts', ui_lang)}")
                    a_subs = gr.Checkbox(value=cfg.get("subtitles", True),
                        label=_t("subs_label", ui_lang),
                        info=_t("subs_info", ui_lang))
                    a_face = gr.Checkbox(value=cfg.get("face_tracking", True),
                        label=_t("face_label", ui_lang),
                        info=_t("face_info", ui_lang))
                    a_banner = gr.Checkbox(value=True,
                        label=_t("banner_label", ui_lang),
                        info=_t("banner_info", ui_lang))
                    with gr.Row():
                        a_banner_top = gr.Slider(0, 500, value=cfg.get("banner_top", 300),
                            label=_t("banner_top", ui_lang))
                        a_banner_bottom = gr.Slider(0, 500, value=cfg.get("banner_bottom", 300),
                            label=_t("banner_bottom", ui_lang))
                    a_blur = gr.Checkbox(value=cfg.get("blur_background", True),
                        label=_t("blur_label", ui_lang),
                        info=_t("blur_info", ui_lang))
                    a_anti = gr.Checkbox(value=cfg.get("anti_copyright", True),
                        label=_t("anti_label", ui_lang),
                        info=_t("anti_info", ui_lang))
                auto_progress = gr.HTML(
                    value=_make_progress_html(0, _t("wait_start", ui_lang)),
                    elem_id="auto-progress",
                )
                auto_btn = gr.Button(_t("auto_process", ui_lang), variant="primary")
                auto_log = gr.Textbox(
                    label=_t("console", ui_lang), lines=12, max_lines=20,
                    interactive=False, elem_id="console-log",
                    value=_t("wait_start", ui_lang) + "\n"
                )

                def on_auto_process(queue, min_d, max_d,
                                    n_clips, s_thresh,
                                    subs, face, banner, bt, bb, blur, anti,
                                    provider, mode, film_lang_val,
                                    _lang=ui_lang):
                    if not queue:
                        yield (_t("error_no_file", _lang),
                               _make_progress_html(0, _t("error_no_file", _lang)))
                        return

                    # Save current settings as defaults
                    cfg_save = user_config.load()
                    cfg_save["min_duration"] = min_d
                    cfg_save["max_duration"] = max_d
                    cfg_save["subtitles"] = subs
                    cfg_save["face_tracking"] = face
                    cfg_save["banner_top"] = bt
                    cfg_save["banner_bottom"] = bb
                    cfg_save["blur_background"] = blur
                    cfg_save["anti_copyright"] = anti
                    cfg_save["num_clips"] = n_clips
                    cfg_save["score_threshold"] = s_thresh
                    cfg_save["llm_provider"] = "yandex" if provider == "Yandex" else "gemini"
                    is_context = mode and ("Контекстный" in mode or "Context" in mode)
                    cfg_save["analysis_mode"] = "context" if is_context else "standard"
                    cfg_save["film_language"] = "en" if film_lang_val and "English" in film_lang_val else "ru"
                    cleanup = cfg_save.get("auto_cleanup", True)
                    user_config.save(cfg_save)

                    # Get API key from runtime config
                    import config as cfg_runtime
                    llm_provider_val = "yandex" if provider == "Yandex" else "gemini"
                    analysis_mode_val = "context" if is_context else "standard"
                    api_key = cfg_runtime.YANDEX_API_KEY if llm_provider_val == "yandex" else cfg_runtime.GEMINI_API_KEY
                    film_language = "en" if film_lang_val and "English" in film_lang_val else "ru"

                    # Process each file in queue
                    all_results = []
                    total_files = len(queue)
                    for file_idx, item in enumerate(queue):
                        video_path = item["path"]
                        movie_title = item.get("title", "")

                        settings = {
                            "min_duration": min_d,
                            "max_duration": max_d,
                            "subtitles": subs,
                            "face_tracking": face,
                            "anti_copyright": anti,
                            "blur_background": blur,
                            "banner_top": bt,
                            "banner_bottom": bb,
                            "num_clips": n_clips,
                            "score_threshold": s_thresh,
                            "api_key": api_key,
                            "movie_title": movie_title,
                            "llm_provider": llm_provider_val,
                            "analysis_mode": analysis_mode_val,
                            "film_language": film_language,
                            "auto_cleanup": cleanup,
                        }

                        capture = LogCapture()
                        capture.start_capture()

                        file_label = os.path.basename(video_path)
                        print("=" * 60)
                        print(_t("movie_header", _lang).format(i=file_idx+1, total=total_files))
                        print(f"{_t('file_label', _lang)}: {file_label}")
                        print("=" * 60)
                        if movie_title:
                            print(f"{_t('movie_title', _lang)}: {movie_title}")
                        print(f"{_t('llm_provider', _lang)}: {provider}, {_t('analysis_mode', _lang)}: {_t('analysis_mode_ctx' if is_context else 'analysis_mode_std', _lang)}")
                        print(f"{_t('film_language', _lang)}: {film_language}")
                        if not api_key:
                            print(f"⚠️ {_t('no_api_key_warn', _lang)}")
                        print()

                        results_container = []
                        error_container = []

                        def worker():
                            try:
                                results = process_movie(video_path, settings)
                                results_container.append(results)
                            except Exception as e:
                                error_container.append(str(e))
                                import traceback
                                error_container.append(traceback.format_exc())

                        thread = threading.Thread(target=worker, daemon=True)
                        thread.start()

                        all_lines = []
                        start_ts = time_module.time()
                        EST_TOTAL = 2100

                        while thread.is_alive():
                            new_lines = capture.get_new_lines()
                            if new_lines:
                                all_lines.extend(new_lines)
                            elapsed = time_module.time() - start_ts
                            pct = min(97, int(elapsed / EST_TOTAL * 100))
                            label = _t("processing_file", _lang).format(i=file_idx+1, total=total_files, name=file_label, time=_fmt_duration(elapsed))
                            yield ("\n".join(all_lines[-40:]),
                                   _make_progress_html(pct, label))
                            time_module.sleep(0.3)

                        thread.join(timeout=2)
                        new_lines = capture.get_new_lines()
                        if new_lines:
                            all_lines.extend(new_lines)

                        if results_container:
                            all_results.extend(results_container[0])

                    # Final summary
                    if error_container:
                        print(f"\n{_t('error_generic', _lang).format(msg=error_container[0])}")
                        label = _t("error_short", _lang)
                    elif all_results:
                        s = [r for r in all_results if r is not None]
                        print(f"\n{_t('done_count', _lang).format(ok=len(s), total=len(all_results))}")
                        for r in s:
                            print(f"  + {os.path.basename(r)}")
                        label = _t("done_count_files", _lang).format(ok=len(s), total=len(all_results), files=total_files)
                    else:
                        print(f"\n{_t('no_scenes', _lang)}")
                        label = _t("no_scenes_short", _lang)

                    yield ("\n".join(capture.get_all()[-40:]),
                           _make_progress_html(100, label))

                # Clear queue after processing; _render_queue is already defined above
                def _clear_queue():
                    return [], _render_queue([])

                auto_btn.click(
                    fn=on_auto_process,
                    inputs=[queue_state, min_dur, max_dur2,
                            num_clips, score_thresh,
                            a_subs, a_face, a_banner, a_banner_top, a_banner_bottom,
                            a_blur, a_anti,
                            llm_provider, analysis_mode, film_lang],
                    outputs=[auto_log, auto_progress],
                ).then(
                    fn=_clear_queue,
                    inputs=[],
                    outputs=[queue_state, queue_display],
                )

        # ── Settings ───────────────────────────────────────────
        with gr.Accordion(_t("settings", ui_lang), open=False):
            with gr.Tabs():
                with gr.Tab(_t("tab_gemini", ui_lang)):
                    gemini_key_box = gr.Textbox(
                        label=_t("api_key_label", ui_lang).format(provider="Gemini"),
                        type="password",
                        value=cfg.get("api_key", ""),
                    )
                    with gr.Row():
                        save_gemini_btn = gr.Button(_t("save_key", ui_lang))
                        check_gemini_btn = gr.Button(_t("check_key", ui_lang), variant="secondary")
                    gemini_status = gr.HTML(
                        value='<span style="color:gray">' + _t("status_not_checked", ui_lang) + '</span>',
                    )
                    _gemini_title = _t("no_api_key_title", ui_lang)
                    _gemini_desc = _t("no_api_key_desc", ui_lang)
                    _gemini_works = _t("no_api_key_works", ui_lang)
                    _gemini_works_list = _t("no_api_key_works_list", ui_lang)
                    _gemini_lost = _t("no_api_key_lost", ui_lang)
                    _gemini_lost_list = _t("no_api_key_lost_list", ui_lang)
                    _gemini_where = _t("no_api_key_where", ui_lang)
                    _gemini_free = _t("no_api_key_free", ui_lang)
                    gr.HTML(
                        value=f"""<div style="margin-top:12px;padding:16px;background:#FFF3E0;border-left:4px solid #E65100;border-radius:6px;font-size:14px;line-height:1.7;color:#000000 !important;">
  <strong style="color:#000000 !important;font-size:15px;">{_gemini_title}</strong><br>
  <span style="color:#000000 !important;">{_gemini_desc}</span>
  <br><br>
  <strong style="color:#000000 !important;">{_gemini_works}</strong><span style="color:#000000 !important;"> {_gemini_works_list}</span>
  <br>
  <strong style="color:#000000 !important;">{_gemini_lost}</strong><span style="color:#000000 !important;"> {_gemini_lost_list}</span>
  <br><br>
  <strong style="color:#000000 !important;">{_gemini_where}</strong>
  <a href="https://aistudio.google.com/apikey" target="_blank" style="color:#1565C0 !important;font-weight:500;">Google AI Studio</a>
  <span style="color:#000000 !important;"> {_gemini_free}</span>
</div>""",
                    )

                with gr.Tab(_t("tab_yandex", ui_lang)):
                    with gr.Row():
                        yandex_key_box = gr.Textbox(
                            label=_t("api_key_label", ui_lang).format(provider="Yandex"),
                            type="password",
                            value=cfg.get("yandex_api_key", ""),
                        )
                        yandex_folder_box = gr.Textbox(
                            label=_t("folder_id_label", ui_lang),
                            value=cfg.get("yandex_folder_id", ""),
                        )
                        yandex_model_dropdown = gr.Dropdown(
                            choices=app_config.YANDEX_MODEL_LIST,
                            label=_t("yandex_model_label", ui_lang),
                            value=cfg.get("yandex_model", "yandexgpt-lite"),
                            info=_t("yandex_model_info", ui_lang),
                        )
                    with gr.Row():
                        save_yandex_btn = gr.Button(_t("save_key", ui_lang))
                        check_yandex_btn = gr.Button(_t("check_key", ui_lang), variant="secondary")
                    yandex_status = gr.HTML(
                        value='<span style="color:gray">' + _t("status_not_checked", ui_lang) + '</span>',
                    )
                    _y_title = _t("no_api_key_title", ui_lang)
                    _y_desc = _t("no_api_key_desc", ui_lang)
                    _y_works = _t("no_api_key_works", ui_lang)
                    _y_works_list = _t("no_api_key_works_list", ui_lang)
                    _y_lost = _t("no_api_key_lost", ui_lang)
                    _y_lost_list = _t("no_api_key_lost_list", ui_lang)
                    _y_where = _t("no_api_key_where", ui_lang)
                    _y_free = _t("no_api_key_yandex_free", ui_lang)
                    gr.HTML(
                        value=f"""<div style="margin-top:12px;padding:16px;background:#E8F5E9;border-left:4px solid #2E7D32;border-radius:6px;font-size:14px;line-height:1.7;color:#000000 !important;">
  <strong style="color:#000000 !important;font-size:15px;">{_y_title}</strong><br>
  <span style="color:#000000 !important;">{_y_desc}</span>
  <br><br>
  <strong style="color:#000000 !important;">{_y_works}</strong><span style="color:#000000 !important;"> {_y_works_list}</span>
  <br>
  <strong style="color:#000000 !important;">{_y_lost}</strong><span style="color:#000000 !important;"> {_y_lost_list}</span>
  <br><br>
  <strong style="color:#000000 !important;">{_y_where}</strong>
  <a href="https://aistudio.yandex.cloud/platform/" target="_blank" style="color:#1565C0 !important;font-weight:500;">Yandex AI Studio</a>
  <span style="color:#000000 !important;"> {_y_free}</span>
</div>""",
                    )
                    cost_info = _t("cost_info", ui_lang)
                    gr.HTML(value=f"""<div style="margin-top:8px;padding:12px 16px;background:#FFF8E1;border-left:4px solid #F9A825;border-radius:6px;font-size:14px;line-height:1.7;">
  <strong style="color:#000000 !important;">{cost_info}</strong>
  <span style="color:#000000 !important;">
    {_t("cost_info_text", ui_lang).format(low="0.13", high="0.21")}
  </span>
</div>""")

                # ── Subtitle Editor tab ──
                with gr.Tab(_t("subtitle_editor", ui_lang)):
                    initial_fs = _get_font_style(cfg)
                    sub_font = gr.Textbox(label=_t("font", ui_lang), value=initial_fs["font"])
                    with gr.Row():
                        sub_size = gr.Slider(8, 48, value=initial_fs["size"], step=1,
                            label=_t("font_size", ui_lang))
                        sub_outline = gr.Slider(0, 5, value=initial_fs["outline"], step=1,
                            label=_t("outline", ui_lang))
                    _COLOR_MAP_RU = {
                        "Белый": "&H00FFFFFF",
                        "Жёлтый": "&H0000FFFF",
                        "Чёрный": "&H00000000",
                        "Красный": "&H000000FF",
                        "Голубой": "&H00FFFF00",
                        "Зелёный": "&H0000FF00",
                    }
                    _COLOR_MAP_EN = {
                        _t("color_white", "en"): "&H00FFFFFF",
                        _t("color_yellow", "en"): "&H0000FFFF",
                        _t("color_black", "en"): "&H00000000",
                        _t("color_red", "en"): "&H000000FF",
                        _t("color_cyan", "en"): "&H00FFFF00",
                        _t("color_green", "en"): "&H0000FF00",
                    }
                    _COLOR_MAP = _COLOR_MAP_EN if ui_lang == "en" else _COLOR_MAP_RU
                    _COLOR_TO_NAME = {v: k for k, v in _COLOR_MAP.items()}
                    _initial_color_name = _COLOR_TO_NAME.get(initial_fs["color"], list(_COLOR_MAP.keys())[0])
                    sub_color = gr.Dropdown(
                        choices=list(_COLOR_MAP.keys()),
                        value=_initial_color_name, label=_t("font_color", ui_lang),
                    )
                    with gr.Row():
                        sub_bold = gr.Checkbox(value=initial_fs["bold"], label=_t("bold", ui_lang))
                        sub_italic = gr.Checkbox(value=initial_fs["italic"], label=_t("italic", ui_lang))
                        sub_shadow = gr.Checkbox(value=initial_fs["shadow"], label=_t("shadow", ui_lang))
                    sub_position = gr.Slider(50, 800, value=initial_fs["position_y"], step=10,
                        label=_t("position_y", ui_lang))
                    sub_preview = gr.HTML(
                        value=_make_subtitle_preview_html(initial_fs, ui_lang),
                        label=_t("preview", ui_lang),
                    )
                    with gr.Row():
                        save_sub_btn = gr.Button(_t("save", ui_lang), variant="primary")
                        reset_sub_btn = gr.Button(_t("reset_defaults", ui_lang), variant="secondary")
                    sub_status = gr.HTML(value=f'<span style="color:gray">{_t("not_saved", ui_lang)}</span>')

                    # Live preview update
                    _UPDATE_COLOR_MAP = _COLOR_MAP_EN if ui_lang == "en" else _COLOR_MAP_RU
                    def _update_preview(font, size, outline, color_name, bold, italic, shadow, pos, _lang=ui_lang):
                        cval = _UPDATE_COLOR_MAP.get(color_name, "&H00FFFFFF")
                        fs = {"font": font, "size": size, "outline": outline,
                              "color": cval, "bold": bold, "italic": italic,
                              "shadow": shadow, "position_y": pos}
                        return _make_subtitle_preview_html(fs, _lang)
                    sub_font.change(fn=_update_preview,
                        inputs=[sub_font, sub_size, sub_outline, sub_color,
                                sub_bold, sub_italic, sub_shadow, sub_position],
                        outputs=[sub_preview])
                    sub_size.change(fn=_update_preview,
                        inputs=[sub_font, sub_size, sub_outline, sub_color,
                                sub_bold, sub_italic, sub_shadow, sub_position],
                        outputs=[sub_preview])
                    sub_outline.change(fn=_update_preview,
                        inputs=[sub_font, sub_size, sub_outline, sub_color,
                                sub_bold, sub_italic, sub_shadow, sub_position],
                        outputs=[sub_preview])
                    sub_color.change(fn=_update_preview,
                        inputs=[sub_font, sub_size, sub_outline, sub_color,
                                sub_bold, sub_italic, sub_shadow, sub_position],
                        outputs=[sub_preview])
                    sub_bold.change(fn=_update_preview,
                        inputs=[sub_font, sub_size, sub_outline, sub_color,
                                sub_bold, sub_italic, sub_shadow, sub_position],
                        outputs=[sub_preview])
                    sub_italic.change(fn=_update_preview,
                        inputs=[sub_font, sub_size, sub_outline, sub_color,
                                sub_bold, sub_italic, sub_shadow, sub_position],
                        outputs=[sub_preview])
                    sub_shadow.change(fn=_update_preview,
                        inputs=[sub_font, sub_size, sub_outline, sub_color,
                                sub_bold, sub_italic, sub_shadow, sub_position],
                        outputs=[sub_preview])
                    sub_position.change(fn=_update_preview,
                        inputs=[sub_font, sub_size, sub_outline, sub_color,
                                sub_bold, sub_italic, sub_shadow, sub_position],
                        outputs=[sub_preview])

                    _SAVE_COLOR_MAP = _COLOR_MAP_EN if ui_lang == "en" else _COLOR_MAP_RU
                    def _save_sub_settings(font, size, outline, color_name, bold, italic, shadow, pos):
                        cval = _SAVE_COLOR_MAP.get(color_name, "&H00FFFFFF")
                        cfg_k = user_config.load()
                        cfg_k["subtitle_font"] = font
                        cfg_k["subtitle_size"] = size
                        cfg_k["subtitle_outline"] = outline
                        cfg_k["subtitle_color"] = cval
                        cfg_k["subtitle_bold"] = bold
                        cfg_k["subtitle_italic"] = italic
                        cfg_k["subtitle_shadow"] = shadow
                        cfg_k["subtitle_position_y"] = pos
                        user_config.save(cfg_k)
                        return f'<span style="color:green">{_t("saved_sub_ok", cfg_k.get("ui_language", "ru"))}</span>'
                    save_sub_btn.click(fn=_save_sub_settings,
                        inputs=[sub_font, sub_size, sub_outline, sub_color,
                                sub_bold, sub_italic, sub_shadow, sub_position],
                        outputs=[sub_status])

                    def _reset_sub_defaults(_lang=ui_lang):
                        """Reset subtitle editor to factory defaults."""
                        cfg_k = user_config.load()
                        cfg_k["subtitle_font"] = "Arial"
                        cfg_k["subtitle_size"] = 13
                        cfg_k["subtitle_outline"] = 1
                        cfg_k["subtitle_color"] = "&H00FFFFFF"
                        cfg_k["subtitle_bold"] = True
                        cfg_k["subtitle_italic"] = False
                        cfg_k["subtitle_shadow"] = False
                        cfg_k["subtitle_position_y"] = 400
                        user_config.save(cfg_k)
                        fs = {"font": "Arial", "size": 13, "outline": 1,
                              "color": "&H00FFFFFF", "bold": True, "italic": False,
                              "shadow": False, "position_y": 400}
                        _reset_color_map = _COLOR_MAP_EN if _lang == "en" else _COLOR_MAP_RU
                        default_color = list(_reset_color_map.keys())[0]
                        preview = _make_subtitle_preview_html(fs, _lang)
                        msg = f'<span style="color:green">{_t("reset_sub_ok", cfg_k.get("ui_language", "ru"))}</span>'
                        return "Arial", 13, 1, default_color, True, False, False, 400, preview, msg
                    reset_sub_btn.click(fn=_reset_sub_defaults,
                        inputs=[],
                        outputs=[sub_font, sub_size, sub_outline, sub_color,
                                 sub_bold, sub_italic, sub_shadow, sub_position,
                                 sub_preview, sub_status])

                # ── General Settings tab ──
                with gr.Tab(_t("general", ui_lang)):
                    auto_cleanup_cb = gr.Checkbox(
                        value=cfg.get("auto_cleanup", True),
                        label=_t("auto_cleanup", ui_lang),
                        info=_t("auto_cleanup_info", ui_lang),
                    )
                    ui_lang_radio = gr.Radio(
                        choices=[_t("lang_russian", ui_lang), _t("lang_english", ui_lang)],
                        value=_t("lang_english", ui_lang) if cfg.get("ui_language", "ru") == "en" else _t("lang_russian", ui_lang),
                        label=_t("ui_language", ui_lang),
                    )
                    save_general_btn = gr.Button(_t("save", ui_lang), variant="primary")
                    general_status = gr.HTML(value=f'<span style="color:gray">{_t("not_saved", ui_lang)}</span>')

                    def _save_general(cleanup, ui_lang_val):
                        cfg_k = user_config.load()
                        cfg_k["auto_cleanup"] = cleanup
                        cfg_k["ui_language"] = "en" if ui_lang_val and "English" in ui_lang_val else "ru"
                        user_config.save(cfg_k)
                        lang_for_msg = cfg_k["ui_language"]
                        return f'<span style="color:green">{_t("restart_for_lang", lang_for_msg)}</span>'
                    save_general_btn.click(fn=_save_general,
                        inputs=[auto_cleanup_cb, ui_lang_radio],
                        outputs=[general_status])

            def _lang_for_keys():
                cfg_k = user_config.load()
                return cfg_k.get("ui_language", "ru")

            def save_gemini_key(key: str):
                cfg_k = user_config.load()
                cfg_k["api_key"] = key
                cfg_k["llm_provider"] = "gemini"
                user_config.save(cfg_k)
                import config as cfg_runtime
                cfg_runtime.GEMINI_API_KEY = key
                cfg_runtime.LLM_PROVIDER = "gemini"
                lk = _lang_for_keys()
                result = check_api_key(key, "gemini")
                if result.get("ok"):
                    return _t("key_saved", lk).format(provider="Google AI")
                return _t("key_saved_check", lk).format(error=result.get('error', _t("status_unknown", lk)))

            def verify_gemini_key(key: str):
                lk = _lang_for_keys()
                result = check_api_key(key, "gemini")
                if result["ok"]:
                    return f'<span style="color:green">{_t("api_ok", lk)}</span>'
                error = result.get("error", _t("status_unknown", lk))
                if any(x in error.lower() for x in ["лимит", "quota", "429", "запрещён", "limit", "forbidden"]):
                    return f'<span style="color:#FF8C00">{_t("key_valid_quota", lk).format(error=error)}</span>'
                return f'<span style="color:red">❌ {error}</span>'

            def save_yandex_key(key: str, folder_id: str, model: str):
                cfg_k = user_config.load()
                cfg_k["yandex_api_key"] = key
                cfg_k["yandex_folder_id"] = folder_id
                cfg_k["yandex_model"] = model
                cfg_k["llm_provider"] = "yandex"
                user_config.save(cfg_k)
                import config as cfg_runtime
                cfg_runtime.YANDEX_API_KEY = key
                cfg_runtime.YANDEX_FOLDER_ID = folder_id
                cfg_runtime.YANDEX_MODEL = model
                cfg_runtime.LLM_PROVIDER = "yandex"
                lk = _lang_for_keys()
                result = check_api_key(key, "yandex")
                if result.get("ok"):
                    return _t("key_saved", lk).format(provider="Yandex AI")
                return _t("key_saved_check", lk).format(error=result.get('error', _t("status_unknown", lk)))

            def verify_yandex_key(key: str):
                lk = _lang_for_keys()
                result = check_api_key(key, "yandex")
                if result["ok"]:
                    return f'<span style="color:green">{_t("api_ok", lk)}</span>'
                error = result.get("error", _t("status_unknown", lk))
                return f'<span style="color:red">❌ {error}</span>'


            save_gemini_btn.click(fn=save_gemini_key, inputs=[gemini_key_box], outputs=[])
            check_gemini_btn.click(fn=verify_gemini_key, inputs=[gemini_key_box], outputs=[gemini_status])
            save_yandex_btn.click(fn=save_yandex_key, inputs=[yandex_key_box, yandex_folder_box, yandex_model_dropdown], outputs=[])
            check_yandex_btn.click(fn=verify_yandex_key, inputs=[yandex_key_box], outputs=[yandex_status])

            # Load keys into runtime config on startup (without verifying)
            if cfg.get("api_key"):
                import config as cfg_runtime
                cfg_runtime.GEMINI_API_KEY = cfg["api_key"]
            if cfg.get("yandex_api_key"):
                import config as cfg_runtime
                cfg_runtime.YANDEX_API_KEY = cfg["yandex_api_key"]
                cfg_runtime.YANDEX_FOLDER_ID = cfg.get("yandex_folder_id", "")
                cfg_runtime.YANDEX_MODEL = cfg.get("yandex_model", "yandexgpt-lite")

    return app


def _fmt_duration(seconds: float) -> str:
    return fmt_duration(seconds)


if __name__ == "__main__":
    app = create_app()
    app.launch(server_port=7860)

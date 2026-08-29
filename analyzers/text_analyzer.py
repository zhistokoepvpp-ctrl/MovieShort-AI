"""
MovieShort AI — Text Analyzer
Scene scoring via Gemini 2.0 Flash or Yandex AI Studio.
"""

from __future__ import annotations

import json
import os
import re
import time
from urllib.parse import urlparse

import httpx

import config

# Provider keys-page hints for HTTP 403 errors (host -> URL).
_KEYS_HINT = {
    "openrouter.ai": "openrouter.ai/keys",
    "opencode.ai": "opencode.ai/auth",
}

PROMPT_SCENE_SPLIT_OR_KEEP = (
    "Ты анализируешь сцену из фильма «{movie_title}» для YouTube Shorts.\n"
    "Вот диалог сцены:\n\n"
    "{dialogue}\n\n"
    "Длительность сцены: {scene_duration:.0f} секунд.\n\n"
    "Ответь на два вопроса:\n"
    "1) Это ОДИН цельный смысловой момент (один монолог, один диалог-перепалка,\n"
    "   одна непрерывная сцена) или НЕСКОЛЬКО разных моментов, которые\n"
    "   можно показывать отдельно?\n"
    "2) Если НЕСКОЛЬКО — укажи таймкоды (в секундах от начала сцены), где\n"
    "   заканчивается одна часть и начинается следующая.\n\n"
    "Правила:\n"
    "- Если это единый непрерывный разговор/монолог — пиши ОДНА\n"
    "- Если внутри есть явные границы (смена темы, пауза, другой персонаж\n"
    "   начинает новую мысль) — пиши НЕСКОЛЬКО и укажи где\n"
    "- Каждая часть должна быть ≥ 15 секунд\n"
    "- Каждая часть должна быть законченной мыслью\n"
    "- Не более 4 частей\n\n"
    "Формат ответа (строго):\n"
    "РЕШЕНИЕ: ОДНА|НЕСКОЛЬКО\n"
    "ЧАСТИ: (только если НЕСКОЛЬКО)\n"
    "ЧАСТЬ 1: 0 — {end1}\n"
    "ЧАСТЬ 2: {end1} — {end2}\n"
    "...\n\n"
    "Пример 1 (монолог):\n"
    "РЕШЕНИЕ: ОДНА\n"
    "\n"
    "Пример 2 (смена сцен):\n"
    "РЕШЕНИЕ: НЕСКОЛЬКО\n"
    "ЧАСТИ:\n"
    "ЧАСТЬ 1: 0 — 25\n"
    "ЧАСТЬ 2: 25 — 47\n"
    "ЧАСТЬ 3: 47 — 62\n"
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TRANSIENT_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


def _is_transient_error(exc):
    resp = getattr(exc, "response", None)
    if resp is None:
        return True   # network error / timeout / malformed payload -> retry justified
    return getattr(resp, "status_code", 0) in _TRANSIENT_STATUSES


def _call_gemini_api(prompt_text: str, api_key: str, max_tokens: int = 256) -> str:
    """Send a prompt to Google AI Gemini and return raw response text."""
    base = config.LLM_BASE_URL.rstrip("/")
    model = config.LLM_MODEL
    print(f"  Gemini model: {model}")
    url = f"{base}/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    body = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": max_tokens},
    }

    last_exc: Exception | None = None
    quota_exhausted = False
    attempts = 0
    for attempt in range(3):
        try:
            resp = httpx.post(url, json=body, headers=headers, timeout=120)
            if resp.status_code == 429:
                if attempt < 2:
                    wait = 5 * (attempt + 1)
                    time.sleep(wait)
                    continue
                else:
                    resp.raise_for_status()
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (httpx.HTTPError, KeyError, IndexError) as exc:
            attempts = attempt + 1
            last_exc = exc
            if hasattr(exc, 'response') and exc.response is not None:
                if exc.response.status_code in (429, 403):
                    quota_exhausted = True
            if not _is_transient_error(exc):
                break
            if attempt < 2:
                time.sleep(2)

    msg = f"Gemini API failed after {attempts} attempt{'s' if attempts != 1 else ''}: {last_exc}"
    if quota_exhausted:
        msg += " (QUOTA_EXHAUSTED)"
    if isinstance(last_exc, httpx.HTTPStatusError) and last_exc.response.status_code == 403:
        keys_url = _KEYS_HINT.get(urlparse(base).netloc, "проверь ключ у провайдера")
        msg += f"; проверь ключ: {keys_url}"
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Yandex AI Studio API
# ---------------------------------------------------------------------------

def _call_yandex_api(prompt_text: str, api_key: str = "", max_tokens: int = 256, model: str = "") -> str:
    """Call Yandex AI Studio (OpenAI-compatible chat/completions API)."""
    if not api_key:
        api_key = getattr(config, 'YANDEX_API_KEY', '') or os.environ.get('YANDEX_API_KEY', '')
    folder_id = getattr(config, 'YANDEX_FOLDER_ID', '') or os.environ.get('YANDEX_FOLDER_ID', '')
    if not model:
        model = getattr(config, 'YANDEX_MODEL', 'yandexgpt-lite')
    base_url = getattr(config, 'YANDEX_BASE_URL', 'https://ai.api.cloud.yandex.net/v1')

    print(f"  Yandex model: {model} (folder: ...{folder_id[-6:]})")

    if not api_key or not folder_id:
        raise RuntimeError("Yandex API key or Folder ID not set")

    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = {
        "model": f"gpt://{folder_id}/{model}",
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }

    last_exc: Exception | None = None
    quota_exhausted = False
    attempts = 0
    for attempt in range(3):
        try:
            resp = httpx.post(url, json=body, headers=headers, timeout=120)
            if resp.status_code == 429:
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                    continue
                else:
                    resp.raise_for_status()
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            if content is None:
                raise RuntimeError(f"Yandex API returned null content (response: {resp.status_code})")
            return content.strip()
        except Exception as exc:
            attempts = attempt + 1
            last_exc = exc
            if hasattr(exc, 'response') and exc.response is not None:
                if exc.response.status_code in (429, 403):
                    quota_exhausted = True
            if not _is_transient_error(exc):
                break
            if attempt < 2:
                time.sleep(2)

    msg = f"Yandex API failed after {attempts} attempt{'s' if attempts != 1 else ''}: {last_exc}"
    if quota_exhausted:
        msg += " (QUOTA_EXHAUSTED)"
    if isinstance(last_exc, httpx.HTTPStatusError) and last_exc.response.status_code == 403:
        keys_url = _KEYS_HINT.get(urlparse(base_url).netloc, "проверь ключ у провайдера")
        msg += f"; проверь ключ: {keys_url}"
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# OpenRouter API
# ---------------------------------------------------------------------------

def _call_openrouter_api(prompt_text: str, api_key: str = "", max_tokens: int = 256, model: str = "") -> str:
    """Call OpenRouter (OpenAI-compatible chat/completions API)."""
    if not api_key:
        api_key = getattr(config, 'OPENROUTER_API_KEY', '') or os.environ.get('OPENROUTER_API_KEY', '')
    if not model:
        model = getattr(config, 'OPENROUTER_MODEL', 'deepseek/deepseek-chat-v3-0324')
    base_url = getattr(config, 'OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')

    print(f"  OpenRouter model: {model} (key: ...{api_key[-4:] if api_key else '(no key)'})")

    if not api_key:
        raise RuntimeError("OpenRouter API key not set")

    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        # Optional attribution headers per OpenRouter docs
        "HTTP-Referer": "https://github.com/zhistokoepvpp-ctrl/MovieShort-AI",
        "X-Title": "MovieShort AI",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }

    last_exc: Exception | None = None
    quota_exhausted = False
    attempts = 0
    for attempt in range(3):
        try:
            resp = httpx.post(url, json=body, headers=headers, timeout=120)
            if resp.status_code == 429:
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                    continue
                else:
                    resp.raise_for_status()
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            if content is None:
                raise RuntimeError(f"OpenRouter API returned null content (response: {resp.status_code})")
            return content.strip()
        except Exception as exc:
            attempts = attempt + 1
            last_exc = exc
            if hasattr(exc, 'response') and exc.response is not None:
                if exc.response.status_code in (429, 403):
                    quota_exhausted = True
            if not _is_transient_error(exc):
                break
            if attempt < 2:
                time.sleep(2)

    msg = f"OpenRouter API failed after {attempts} attempt{'s' if attempts != 1 else ''}: {last_exc}"
    if quota_exhausted:
        msg += " (QUOTA_EXHAUSTED)"
    if isinstance(last_exc, httpx.HTTPStatusError):
        # Show the real reason from the OpenRouter response body (error.message),
        # degrade silently to the generic message on malformed/non-JSON bodies.
        try:
            err_data = last_exc.response.json()
            server_reason = str(err_data["error"]["message"])[:200]
            msg += f"; причина от сервера: {server_reason}"
        except Exception:
            pass
    if isinstance(last_exc, httpx.HTTPStatusError) and last_exc.response.status_code == 403:
        keys_url = _KEYS_HINT.get(urlparse(base_url).netloc, "проверь ключ у провайдера")
        msg += f"; проверь ключ: {keys_url}"
        # stealth/ox-alpha 403s are usually an account-level model opt-in
        # issue, not a bad key — point at privacy settings too (todo 43).
        try:
            msg += ("; модель stealth/ox-alpha может требовать включения в настройках "
                    "аккаунта OpenRouter (openrouter.ai/settings/privacy) или выберите "
                    "другую модель в выпадающем списке")
        except Exception:
            pass
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# OpenCode Zen API (https://opencode.ai/zen/v1 — OpenAI-compatible)
# ---------------------------------------------------------------------------

def _call_opencode_zen_api(prompt_text: str, api_key: str = "", max_tokens: int = 256, model: str = "") -> str:
    """Call OpenCode Zen (OpenAI-compatible chat/completions API)."""
    if not api_key:
        api_key = getattr(config, 'OPENCODE_ZEN_API_KEY', '') or os.environ.get('OPENCODE_ZEN_API_KEY', '')
    if not model:
        model = getattr(config, 'OPENCODE_ZEN_MODEL', 'nemotron-3-ultra-free')
    base_url = getattr(config, 'OPENCODE_ZEN_BASE_URL', 'https://opencode.ai/zen/v1')

    print(f"  OpenCode Zen model: {model} (key: ...{api_key[-4:] if api_key else '(no key)'})")

    if not api_key:
        raise RuntimeError("OpenCode Zen API key not set")

    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/zhistokoepvpp-ctrl/MovieShort-AI",
        "X-Title": "MovieShort AI",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }

    last_exc: Exception | None = None
    quota_exhausted = False
    attempts = 0
    for attempt in range(3):
        try:
            resp = httpx.post(url, json=body, headers=headers, timeout=120)
            if resp.status_code == 429:
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                    continue
                else:
                    resp.raise_for_status()
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            if content is None:
                raise RuntimeError(f"OpenCode Zen API returned null content (response: {resp.status_code})")
            return content.strip()
        except Exception as exc:
            attempts = attempt + 1
            last_exc = exc
            if hasattr(exc, 'response') and exc.response is not None:
                if exc.response.status_code in (429, 403):
                    quota_exhausted = True
            if not _is_transient_error(exc):
                break
            if attempt < 2:
                time.sleep(2)

    msg = f"OpenCode Zen API failed after {attempts} attempt{'s' if attempts != 1 else ''}: {last_exc}"
    if quota_exhausted:
        msg += " (QUOTA_EXHAUSTED — смените модель на nemotron-3-ultra-free)"
    if isinstance(last_exc, httpx.HTTPStatusError):
        try:
            err_data = last_exc.response.json()
            server_reason = str(err_data["error"]["message"])[:200]
            msg += f"; причина от сервера: {server_reason}"
        except Exception:
            pass
    if isinstance(last_exc, httpx.HTTPStatusError) and last_exc.response.status_code == 403:
        keys_url = _KEYS_HINT.get(urlparse(base_url).netloc, "проверь ключ у провайдера")
        msg += f"; проверь ключ: {keys_url}"
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Unified LLM call
# ---------------------------------------------------------------------------

def call_llm(prompt_text: str, api_key: str = "", provider: str = "", max_tokens: int = 256) -> str:
    """Unified LLM call — routes to Gemini, Yandex, OpenRouter or OpenCode Zen based on provider."""
    if not provider:
        provider = getattr(config, 'LLM_PROVIDER', 'gemini')
    if provider == "yandex":
        return _call_yandex_api(prompt_text, api_key, max_tokens=max_tokens)
    elif provider == "openrouter":
        return _call_openrouter_api(prompt_text, api_key, max_tokens=max_tokens)
    elif provider == "opencode_zen":
        return _call_opencode_zen_api(prompt_text, api_key, max_tokens=max_tokens)
    else:
        return _call_gemini_api(prompt_text, api_key, max_tokens=max_tokens)


def _parse_score(text: str) -> int:
    """Extract an integer 1-10 from model output."""
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        score = int(digits[:2])
        return max(1, min(10, score))
    return 5


def _parse_split_or_keep(raw: str, scene_duration: float) -> dict:
    """
    Parse LLM response for scene split-or-keep decision.
    
    Returns {"decision": "keep"} or {"decision": "split", "parts": [(start, end), ...]}
    """
    m = re.search(r'РЕШЕНИЕ:\s*ОДНА', raw)
    if m:
        return {"decision": "keep"}
    
    m = re.search(r'РЕШЕНИЕ:\s*НЕСКОЛЬКО', raw)
    if m:
        # Parse ЧАСТИ: block
        parts_block = re.search(r'ЧАСТИ:\s*\n(.*?)(?:\n\n|\Z)', raw, re.DOTALL)
        if not parts_block:
            parts_block = re.search(r'ЧАСТЬ\s+\d+', raw)
            if parts_block:
                parts_text = raw[parts_block.start():]
            else:
                return {"decision": "keep"}
        else:
            parts_text = parts_block.group(1)
        
        parts = []
        for line in parts_text.split('\n'):
            line = line.strip()
            m2 = re.match(r'ЧАСТЬ\s+\d+\s*:\s*(\d+(?:\.\d+)?)\s*[—–-]\s*(\d+(?:\.\d+)?)', line)
            if m2:
                start = float(m2.group(1))
                end = float(m2.group(2))
                parts.append((start, end))
        
        # Validate
        if not parts:
            return {"decision": "keep"}
        if len(parts) > 4:
            return {"decision": "keep"}
        if any(end - start < 15 for start, end in parts):
            return {"decision": "keep"}
        sorted_parts = sorted(parts, key=lambda x: x[0])
        for i in range(1, len(sorted_parts)):
            if sorted_parts[i][0] < sorted_parts[i-1][1]:
                return {"decision": "keep"}
        if sorted_parts[-1][1] > scene_duration + 1:
            return {"decision": "keep"}
        
        return {"decision": "split", "parts": sorted_parts}
    
    return {"decision": "keep"}



def _parse_batch_response(raw: str, expected: int) -> list[dict] | None:
    """Parse batch response in format: [N] score — title

    Returns list of {score, title} or None if parsing fails.
    """
    entries = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"\[\s*(\d+)\s*\]\s*(\d+)\s*[—–-]\s*(.+)", line)
        if m:
            idx = int(m.group(1))
            score = max(1, min(10, int(m.group(2))))
            title = m.group(3).strip()
            entries.append((idx, score, title))
    if len(entries) != expected:
        return None
    entries.sort(key=lambda x: x[0])
    return [{"score": s, "title": t} for _, s, t in entries]



def check_api_key(api_key: str, provider: str = "") -> dict:
    """Check if the API key is valid. Supports Gemini, Yandex, OpenRouter and OpenCode Zen.

    Returns:
        {"ok": True} if key works,
        {"ok": False, "error": "message"} if key fails.
    """
    if not provider:
        provider = getattr(config, 'LLM_PROVIDER', 'gemini')
    if provider == "yandex":
        return _check_yandex_key(api_key)
    elif provider == "openrouter":
        return _check_openrouter_key(api_key)
    elif provider == "opencode_zen":
        return _check_opencode_zen_key(api_key)
    else:
        return _check_gemini_key(api_key)


def _check_gemini_key(api_key: str) -> dict:
    """Check Google AI Gemini API key."""
    if not api_key or not api_key.strip():
        return {"ok": False, "error": "Ключ не задан"}

    try:
        resp = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
            json={
                "contents": [{"parts": [{"text": "Say OK"}]}],
                "generationConfig": {"maxOutputTokens": 2},
            },
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            if "candidates" in data:
                return {"ok": True, "models_count": 1}
            return {"ok": False, "error": "Google AI: пустой ответ"}
        elif resp.status_code == 400:
            error_msg = resp.json().get("error", {}).get("message", "Неверный ключ")
            return {"ok": False, "error": f"Google AI: {error_msg[:80]}"}
        elif resp.status_code == 403:
            return {"ok": False, "error": "Google AI: доступ запрещён (key отозван или quota)"}
        elif resp.status_code == 429:
            return {"ok": False, "error": "Google AI: превышен лимит запросов"}
        else:
            return {"ok": False, "error": f"Google AI: HTTP {resp.status_code}"}
    except httpx.TimeoutException:
        return {"ok": False, "error": "Google AI: таймаут (15 сек)"}
    except httpx.ConnectError:
        return {"ok": False, "error": "Google AI: не удалось подключиться"}
    except Exception as e:
        return {"ok": False, "error": f"Google AI: {str(e)[:80]}"}


def _check_yandex_key(api_key: str) -> dict:
    """Check Yandex AI Studio API key."""
    folder_id = getattr(config, 'YANDEX_FOLDER_ID', '') or os.environ.get('YANDEX_FOLDER_ID', '')
    if not api_key or not api_key.strip():
        return {"ok": False, "error": "Yandex API ключ не задан"}
    if not folder_id:
        return {"ok": False, "error": "Yandex Folder ID не задан"}

    try:
        # Force yandexgpt-lite for the check (independent of user's model selection)
        _call_yandex_api("Скажи OK", api_key, model="yandexgpt-lite")
        return {"ok": True}
    except Exception as e:
        err = str(e)
        if "401" in err or "auth" in err.lower():
            return {"ok": False, "error": "Yandex: неверный ключ или Folder ID"}
        if "429" in err:
            return {"ok": False, "error": "Yandex: превышен лимит запросов"}
        return {"ok": False, "error": f"Yandex: {err[:80]}"}


def _check_openrouter_key(api_key: str) -> dict:
    """Check OpenRouter API key."""
    if not api_key or not api_key.strip():
        return {"ok": False, "error": "OpenRouter API ключ не задан"}

    try:
        # Pin a non-reasoning probe model: config default stealth/ox-alpha is
        # reasoning-mandatory — max_tokens=16 burns on reasoning, content comes
        # back null and the null-content guard reports a false key failure.
        _call_openrouter_api("Скажи OK", api_key, max_tokens=16,
                             model="deepseek/deepseek-chat-v3-0324")
        return {"ok": True}
    except Exception as e:
        err = str(e)
        if "401" in err or "auth" in err.lower():
            return {"ok": False, "error": "OpenRouter: неверный ключ"}
        if "403" in err:
            return {"ok": False, "error": ("OpenRouter: доступ запрещён — проверь права ключа на выбранную модель "
                                           "(permissions/guardrail/moderation); модель stealth/ox-alpha может "
                                           "требовать включения в настройках аккаунта OpenRouter "
                                           "(openrouter.ai/settings/privacy) или выберите другую модель "
                                           "в выпадающем списке")}
        if "429" in err:
            return {"ok": False, "error": "OpenRouter: превышен лимит запросов"}
        return {"ok": False, "error": f"OpenRouter: {err[:80]}"}


def _check_opencode_zen_key(api_key: str) -> dict:
    """Check OpenCode Zen API key."""
    if not api_key or not api_key.strip():
        return {"ok": False, "error": "OpenCode Zen API ключ не задан"}

    try:
        # Pin default free model with minimal tokens (null-content guard needs real content)
        _call_opencode_zen_api("Скажи OK", api_key, max_tokens=16,
                               model="nemotron-3-ultra-free")
        return {"ok": True}
    except Exception as e:
        err = str(e)
        if "401" in err or "auth" in err.lower():
            return {"ok": False, "error": "OpenCode Zen: неверный ключ"}
        if "403" in err:
            return {"ok": False, "error": ("OpenCode Zen: доступ запрещён — проверь ключ: opencode.ai/auth")}
        if "429" in err:
            return {"ok": False, "error": "OpenCode Zen: превышен лимит запросов — смените модель на nemotron-3-ultra-free"}
        return {"ok": False, "error": f"OpenCode Zen: {err[:80]}"}


# ---------------------------------------------------------------------------
# English Context Mode prompt
# ---------------------------------------------------------------------------

PROMPT_CONTEXT_SCENES_EN = (
    "You are an expert at picking YouTube Shorts from movies.\n"
    "Movie: «{movie_name}»\n\n"
    "Rate each scene for a YouTube Short on a scale of 1-10.\n"
    "Use your KNOWLEDGE of the movie: if a scene is famous,\n"
    "iconic or quotable — give it a high score, even if the\n"
    "dialogue is just a few words or it's a visual scene (action, fight, chase).\n\n"
    "IMPORTANT: Use the ENTIRE scale 1-10. Don't cluster in the upper range.\n"
    "9-10: RARE. Only the most iconic/viral moments (less than 5% of scenes).\n"
    "7-8:  UNCOMMON. Truly great, memorable scenes.\n"
    "5-6:  Normal. Plot-important but not viral.\n"
    "3-4:  Filler. Transitional scenes, padding.\n"
    "1-2:  Boring, can be cut.\n\n"
    "Score can be decimal (e.g. 8.5, 6.2) for precision.\n\n"
    "IMPORTANT: Each TITLE must be short (2-5 words) and\n"
    "describe ONLY what actually happens in this scene\n"
    "based on its dialogue. Don't generate pompous or clickbait titles.\n"
    "Just describe the action: «Hancock drinks on a bench», not «The calm before the storm».\n\n"
    "{scenes_block}\n\n"
    "For each scene write strictly in format:\n"
    "SCENE {{number}} | SCORE: {{score}} | TITLE: {{title}}\n\n"
    "Example:\n"
    "SCENE 5 | SCORE: 9 | TITLE: Hancock saves a beached whale\n"
    "SCENE 12 | SCORE: 3 | TITLE: Ordinary conversation at the station\n"
)


def _parse_context_response_en(raw: str) -> list[dict]:
    """Parse English context mode response.

    Format: SCENE {N} | SCORE: {score} | TITLE: {title}
    Score may be integer or decimal.

    Returns: [{scene_num, score, title}]
    """
    scenes = []
    seen_nums = set()
    for line in raw.split('\n'):
        m = re.match(
            r'SCENE\s+(\d+)\s*\|\s*SCORE[:\s]*(\d+(?:\.\d+)?)\s*\|\s*TITLE[:\s]*(.+)',
            line,
        )
        if m:
            scene_num = int(m.group(1))
            if scene_num < 1 or scene_num > 10000:
                continue
            if scene_num in seen_nums:
                continue
            seen_nums.add(scene_num)
            score_val = float(m.group(2))
            score_val = max(1.0, min(10.0, score_val))
            scenes.append({
                "scene_num": scene_num,
                "score": score_val,
                "title": m.group(3).strip(),
            })
    return scenes


# ---------------------------------------------------------------------------
# Context Mode — LLM sees real scene transcripts, picks best scenes
# ---------------------------------------------------------------------------

PROMPT_CONTEXT_SCENES = (
    "Ты эксперт по YouTube Shorts из фильмов.\n"
    "Фильм: «{movie_name}»\n\n"
    "Оцени каждую сцену для YouTube Shorts по шкале 1-10.\n"
    "Мысленно вспомни этот фильм: какие сцены популярны,\n"
    "культовые, цитируемые, мемные. Если текущая сцена\n"
    "ОДНА ИЗ НИХ — ставь 10 (десять) без колебаний.\n"
    "Не занижай искусственно — если сцена культовая, она\n"
    "получает 10.\n\n"
    "Если сцена не культовая — оценивай дробно (7.3, 8.5, 6.2)\n"
    "для точности:\n\n"
    "8-9.9: Отличные, запоминающиеся, эмоциональные, визуально эффектные\n"
    "5-7.9:  Хорошие, сюжетно важные, нормальные\n"
    "1-4.9:  Слабые, проходные, можно вырезать\n\n"
    "Каждое НАЗВАНИЕ должно быть коротким (2-5 слов) и\n"
    "описывать ТОЛЬКО то, что реально происходит в этой сцене.\n"
    "Просто опиши действие: «Хэнкок пьёт на скамейке»,\n"
    "а не «Тишина перед бурей».\n\n"
    "{scenes_block}\n\n"
    "Для каждой сцены напиши строго в формате:\n"
    "СЦЕНА {{number}} | ОЦЕНКА: {{score}} | НАЗВАНИЕ: {{title}}\n\n"
    "Пример:\n"
    "СЦЕНА 5 | ОЦЕНКА: 10 | НАЗВАНИЕ: Хэнкок спасает выброшенного кита\n"
    "СЦЕНА 12 | ОЦЕНКА: 3 | НАЗВАНИЕ: Обычный диалог в участке\n"
    "СЦЕНА 30 | ОЦЕНКА: 7.5 | НАЗВАНИЕ: Погоня за грузовиком\n"
)


def _parse_context_response(raw: str) -> list[dict]:
    """Parse context mode response.

    Primary format: СЦЕНА {N} | ОЦЕНКА: {score} | НАЗВАНИЕ: {title}
    Score may be integer or decimal (e.g. 8.5).

    Falls back to lenient parsing for non-standard formatting (DeepSeek, etc.).

    Returns: [{scene_num, score, title}]
    """
    scenes = []
    seen_nums = set()

    # Primary strict parser
    for line in raw.split('\n'):
        line = line.strip()
        m = re.match(
            r'СЦЕНА\s+(\d+)\s*\|\s*ОЦЕНКА[:\s]*(\d+(?:\.\d+)?)\s*\|\s*НАЗВАНИЕ[:\s]*(.+)',
            line,
        )
        if m:
            scene_num = int(m.group(1))
            if scene_num < 1 or scene_num > 10000:
                continue
            if scene_num in seen_nums:
                continue
            seen_nums.add(scene_num)
            score_val = float(m.group(2))
            score_val = max(1.0, min(10.0, score_val))
            scenes.append({
                "scene_num": scene_num,
                "score": score_val,
                "title": m.group(3).strip(),
            })

    if scenes:
        return scenes

    # Lenient fallback: strip markdown, extra text, alternate separators
    for line in raw.split('\n'):
        line = line.strip()
        # Remove markdown bold/italic markers
        cleaned = re.sub(r'[*_#]', '', line)
        # Try: СЦЕНА N — ОЦЕНКА: X — НАЗВАНИЕ: Y  (em-dash separator, any order)
        m = re.match(
            r'СЦЕНА\s+(\d+)\s*[—–\-—|:|]\s*ОЦЕНКА[:\s]*(\d+(?:\.\d+)?)\s*[—–\-—|,:]\s*НАЗВАНИЕ[:\s]*(.+)',
            cleaned, re.IGNORECASE,
        )
        if m:
            scene_num = int(m.group(1))
            if scene_num < 1 or scene_num > 10000 or scene_num in seen_nums:
                continue
            seen_nums.add(scene_num)
            score_val = max(1.0, min(10.0, float(m.group(2))))
            scenes.append({
                "scene_num": scene_num,
                "score": score_val,
                "title": m.group(3).strip(),
            })

    if scenes:
        return scenes

    # Very lenient: extract scene number + score + title from any N/X format
    idx = 0
    for line in raw.split('\n'):
        line = line.strip()
        cleaned = re.sub(r'[*_#]', '', line)
        # Match any line like: "5 | 9.3 | Название" or "5 9.3 Название"
        m = re.match(r'^\s*(\d+)\s*[|\-:,\s]+\s*(\d+(?:\.\d+)?)\s*[|\-:,\s]+\s*(.+)', cleaned)
        if m:
            scene_num = int(m.group(1))
            if scene_num < 1 or scene_num > 10000 or scene_num in seen_nums:
                continue
            seen_nums.add(scene_num)
            score_val = max(1.0, min(10.0, float(m.group(2))))
            title = m.group(3).strip()
            # Remove common cruft from title
            title = re.sub(r'^(НАЗВАНИЕ|TITLE|Название)\s*[:\s]', '', title).strip()
            scenes.append({
                "scene_num": scene_num,
                "score": score_val,
                "title": title,
            })
            idx += 1

    return scenes


# ---------------------------------------------------------------------------
# Block-to-Clips prompt + parser (C2 input)
# ---------------------------------------------------------------------------

PROMPT_BLOCK_TO_CLIPS = (
    "Ты эксперт по нарезке фильмов на YouTube Shorts.\n"
    "Фильм: «{movie_name}»\n\n"
    "Диалог блока:\n\n{dialogue}\n\n"
    "Длительность блока: {block_duration} секунд.\n"
    "Количество смен кадра: {cut_count} (высокое = экшн, много действий).\n"
    "Паузы в диалоге (секунды от начала блока): {pause_points}\n\n"
    "Определи, какие клипы из этого блока подойдут для YouTube Shorts.\n\n"
    "Правила:\n"
    "- Каждый клип: 30-75 секунд. Предпочтительно ~60 секунд.\n"
    "- Клипы < 20 секунд допускаются ТОЛЬКО если reason=\"самодостаточен\"\n"
    "  (законченная шутка, реплика-клипса, яркий момент).\n"
    "- start и end — ВРЕМЯ ОТ НАЧАЛА БЛОКА (не от начала фильма).\n"
    "- Клипы НЕ должны пересекаться.\n"
    "- Сколько клипов вернуть — реши сам. 1-3 обычно достаточно.\n"
    "- Если блок неинтересен — верни пустой массив [].\n\n"
    "Формат ответа (строго JSON-массив, без пояснений):\n"
    '[{{"start": 0.0, "end": 60.0, "title": "Название", "score": 8.5, "reason": "описание"}}]\n\n'
    "Примеры:\n"
    '[{{"start": 0.0, "end": 55.0, "title": "Хэнкок спасает кита", "score": 9.0, "reason": "культовая сцена"}},\n'
    ' {{"start": 55.0, "end": 110.0, "title": "Разговор в участке", "score": 6.5, "reason": "юмор в диалоге"}}]'
)


PROMPT_BLOCK_TO_CLIPS_EN = (
    "You are an expert at cutting movie blocks into YouTube Shorts.\n"
    "Movie: «{movie_name}»\n\n"
    "Block dialogue:\n\n{dialogue}\n\n"
    "Block duration: {block_duration} seconds.\n"
    "Cut count (scene changes): {cut_count} (high = action, many events).\n"
    "Dialogue pauses (seconds from block start): {pause_points}\n\n"
    "Determine which clips from this block are suitable for YouTube Shorts.\n\n"
    "Rules:\n"
    "- Each clip: 30-75 seconds. Prefer ~60 seconds.\n"
    "- Clips < 20 seconds are allowed ONLY with reason=\"self_contained\"\n"
    "  (complete joke, quotable line, self-sufficient moment).\n"
    "- start/end are RELATIVE TO BLOCK START (not movie start).\n"
    "- Clips must NOT overlap.\n"
    "- Decide how many clips to return. 1-3 is usually enough.\n"
    "- If the block is uninteresting — return an empty array [].\n\n"
    "Output format (strict JSON array, no explanations):\n"
    '[{{"start": 0.0, "end": 60.0, "title": "Title", "score": 8.5, "reason": "description"}}]\n\n'
    "Examples:\n"
    '[{{"start": 0.0, "end": 55.0, "title": "Hancock saves a whale", "score": 9.0, "reason": "iconic scene"}},\n'
    ' {{"start": 55.0, "end": 110.0, "title": "Station conversation", "score": 6.5, "reason": "dialogue humor"}}]'
)


# ---------------------------------------------------------------------------
# Batch block-to-clips prompt — multiple blocks in one LLM call
# ---------------------------------------------------------------------------

PROMPT_BATCH_TO_CLIPS = (
    "Ты эксперт по нарезке фильмов на YouTube Shorts.\n"
    "Фильм: «{movie_name}»\n\n"
    "Ниже перечислены блоки сцены. Для КАЖДОГО блока определи, какие клипы "
    "подойдут для Shorts.\n"
    "---\n"
    "{blocks_text}\n"
    "---\n\n"
    "Правила (для каждого блока):\n"
    "- Каждый клип: 30-75 секунд. Предпочтительно ~60 секунд.\n"
    "- Клипы < 20 секунд допускаются ТОЛЬКО если reason=\"самодостаточен\"\n"
    "  (законченная шутка, реплика-клипса, яркий момент).\n"
    "- start и end — АБСОЛЮТНЫЕ секунды от НАЧАЛА ФИЛЬМА (не от блока).\n"
    "- Клипы внутри ОДНОГО блока НЕ должны пересекаться.\n"
    "- Сколько клипов на блок — реши сам. 1-3 обычно достаточно.\n"
    "- Если блок неинтересен — не включай его клипы в ответ.\n\n"
    "Формат ответа (строго JSON-массив, без пояснений):\n"
    '[{{"start": 0.0, "end": 60.0, "title": "Название", "score": 8.5,\n'
    '  "reason": "описание", "block": 0}}]\n\n'
    'Поле "block" — 0-based индекс блока (0, 1, 2, 3...).\n'
    "Пример:\n"
    '[{{"start": 5.0, "end": 60.0, "title": "Тони собирает броню", "score": 9.0,\n'
    '  "reason": "культовая сцена", "block": 0}},\n'
    ' {{"start": 130.0, "end": 185.0, "title": "Разговор в башне", "score": 7.0,\n'
    '  "reason": "диалог", "block": 1}}]'
)

PROMPT_BATCH_TO_CLIPS_EN = (
    "You are an expert at cutting movie blocks into YouTube Shorts.\n"
    "Movie: «{movie_name}»\n\n"
    "Below are scene blocks. For EACH block, determine which clips "
    "are suitable for Shorts.\n"
    "---\n"
    "{blocks_text}\n"
    "---\n\n"
    "Rules (per block):\n"
    "- Each clip: 30-75 seconds. Prefer ~60 seconds.\n"
    "- Clips < 20 seconds are allowed ONLY with reason=\"self_contained\"\n"
    "  (complete joke, quotable line, self-sufficient moment).\n"
    "- start/end are ABSOLUTE seconds from MOVIE START (not block-relative).\n"
    "- Clips within ONE block must NOT overlap.\n"
    "- Decide how many clips per block. 1-3 is usually enough.\n"
    "- If a block is uninteresting — don't include its clips.\n\n"
    "Output format (strict JSON array, no explanations):\n"
    '[{{"start": 0.0, "end": 60.0, "title": "Title", "score": 8.5,\n'
    '  "reason": "description", "block": 0}}]\n\n'
    'Field "block" is 0-based block index (0, 1, 2, 3...).\n'
    "Example:\n"
    '[{{"start": 5.0, "end": 60.0, "title": "Tony suits up", "score": 9.0,\n'
    '  "reason": "iconic scene", "block": 0}},\n'
    ' {{"start": 130.0, "end": 185.0, "title": "Tower talk", "score": 7.0,\n'
    '  "reason": "dialogue", "block": 1}}]'
)


def _parse_batch_response(raw: str, block_start_times: list[float]) -> dict[int, list[dict]]:
    """Parse batch LLM response into per-block clip lists.

    Expects a JSON array with each item having "block" field (int).

    Args:
        raw: raw LLM response text
        block_start_times: list of block start times (seconds from film start)

    Returns:
        dict mapping block_index -> list of clip dicts (with absolute timestamps)
    """
    if not raw:
        return {}

    try:
        clips = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        clips = []

    if not isinstance(clips, list):
        return {}

    result: dict[int, list[dict]] = {}
    for item in clips:
        if not isinstance(item, dict):
            continue
        block_idx = item.get("block")
        if not isinstance(block_idx, int):
            continue
        if block_idx < 0 or block_idx >= len(block_start_times):
            continue
        result.setdefault(block_idx, []).append(item)

    return result


def _parse_block_response(raw: str) -> list[dict]:
    """Parse LLM response for block-to-clips分割.

    Expects a JSON array of clip dicts: [{"start": float, "end": float, ...}].
    Handles markdown code fences and falls back to regex extraction.

    Returns list of clip dicts, or [] on failure.
    """
    if not raw or not isinstance(raw, str):
        return []

    content = raw.strip()
    if not content:
        return []

    # 1. Try extracting from markdown code fence ```json [...] ```
    fence_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1))
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

    # 2. Try direct JSON array extraction
    direct_match = re.search(r'(\[.*?\])', content, re.DOTALL)
    if direct_match:
        try:
            parsed = json.loads(direct_match.group(1))
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

    # 3. Fallback: regex for individual clip objects
    clip_pattern = (
        r'\{\s*"start":\s*([\d.]+),\s*"end":\s*([\d.]+),'
        r'\s*"title":\s*"([^"]*)",\s*"score":\s*([\d.]+)'
    )
    clips = []
    for m in re.finditer(clip_pattern, content):
        clips.append({
            "start": float(m.group(1)),
            "end": float(m.group(2)),
            "title": m.group(3),
            "score": float(m.group(4)),
        })

    return clips


def ask_llm_context_mode(
    scenes: list[dict],
    movie_name: str,
    api_key: str = "",
    provider: str = "",
    batch_size: int = 100,
    language: str = "ru",
) -> list[dict]:
    """Context mode: send scene transcripts to LLM, it picks best scenes.

    Parameters
    ----------
    scenes:
        List of {start, end, text} dicts from detect_and_transcribe().
    movie_name:
        Movie title for the prompt.
    api_key:
        API key.
    provider:
        'yandex' or 'gemini'.
    batch_size:
        Max scenes per API call (100 for 32K models, all for 1M models).
    language:
        'ru' or 'en' — selects Russian or English prompt.

    Returns
    -------
    List of {scene_num, score, title} dicts (scene_num is 1-based index).
    """
    if not scenes or not api_key:
        return []

    # Determine batch size based on model context window
    model = getattr(config, 'YANDEX_MODEL', 'yandexgpt-lite')
    if 'deepseek' in model.lower():
        batch_size = 30  # output capped at 2048 tokens — 30 scenes ≈ 1350 tok
    elif 'aliceai' in model.lower():
        batch_size = min(batch_size, 200)

    is_english = language == "en"
    prompt_template = PROMPT_CONTEXT_SCENES_EN if is_english else PROMPT_CONTEXT_SCENES
    parse_fn = _parse_context_response_en if is_english else _parse_context_response

    all_results: list[dict] = []
    total_batches = (len(scenes) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(scenes))
        batch = scenes[start_idx:end_idx]

        # Build scenes block
        lines = []
        for i, scene in enumerate(batch):
            scene_num = start_idx + i + 1  # 1-based global number
            start_m = int(scene["start"]) // 60
            start_s = int(scene["start"]) % 60
            end_m = int(scene["end"]) // 60
            end_s = int(scene["end"]) % 60
            time_range = f"{start_m:02d}:{start_s:02d}-{end_m:02d}:{end_s:02d}"
            text = (scene.get("text") or "").strip()
            if not text:
                text = "(нет диалога)" if not is_english else "(no dialogue)"
            lines.append(f"Сцена {scene_num} ({time_range}):\n«{text}»")

        scenes_block = "\n\n".join(lines)
        prompt = prompt_template.format(
            movie_name=movie_name, scenes_block=scenes_block
        )

        print(f"  Batch {batch_idx + 1}/{total_batches}: "
              f"scenes {start_idx + 1}-{end_idx} "
              f"({len(batch)} scenes)...", end="")

        try:
            raw = call_llm(prompt, api_key, provider, max_tokens=2048)
            parsed = parse_fn(raw)
            if parsed:
                print(f" OK ({len(parsed)} rated)")
                all_results.extend(parsed)
            else:
                print(f" OK (0 rated — fallback to score 5)")
                for i in range(len(batch)):
                    all_results.append({
                        "scene_num": start_idx + i + 1,
                        "score": 5,
                        "title": "",
                    })
        except Exception as e:
            print(f" FAILED: {e}")
            # Fallback: give all scenes score 5
            for i in range(len(batch)):
                all_results.append({
                    "scene_num": start_idx + i + 1,
                    "score": 5,
                    "title": "",
                })

    return all_results

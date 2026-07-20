"""
MovieShort AI — Text Analyzer
Scene scoring via Gemini 2.0 Flash or Yandex AI Studio.
"""

from __future__ import annotations

import json
import os
import random
import re
import time

import httpx

import config

PROMPT_SCENE_ANALYSIS = (
    "Ты анализируешь сцены из фильмов для YouTube Shorts. "
    "Вот диалог сцены.\n\n{text}\n\n"
    "Оцени интересность от 1 до 10 по критериям: "
    "напряжённость, эмоциональность, наличие крылатых фраз. "
    "Ответь ТОЛЬКО числом."
)

PROMPT_SCENE_ANALYSIS_WITH_TITLE = (
    "Ты анализируешь сцены из фильма «{title}» для YouTube Shorts. "
    "Вот диалог сцены.\n\n{text}\n\n"
    "Оцени интересность от 1 до 10: насколько эта сцена известна среди "
    "зрителей, какие эмоции вызывает, есть ли в ней культовые моменты. "
    "Учитывай репутацию фильма и отзывы зрителей. "
    "Ответь ТОЛЬКО числом."
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
    for attempt in range(3):
        try:
            resp = httpx.post(url, json=body, headers=headers, timeout=30)
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
            last_exc = exc
            if hasattr(exc, 'response') and exc.response is not None:
                if exc.response.status_code in (429, 403):
                    quota_exhausted = True
            if attempt < 2:
                time.sleep(2)

    msg = f"Gemini API failed after 3 attempts: {last_exc}"
    if quota_exhausted:
        msg += " (QUOTA_EXHAUSTED)"
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Yandex AI Studio API
# ---------------------------------------------------------------------------

def _call_yandex_api(prompt_text: str, api_key: str = "", max_tokens: int = 256) -> str:
    """Call Yandex AI Studio (OpenAI-compatible chat/completions API)."""
    if not api_key:
        api_key = getattr(config, 'YANDEX_API_KEY', '') or os.environ.get('YANDEX_API_KEY', '')
    folder_id = getattr(config, 'YANDEX_FOLDER_ID', '') or os.environ.get('YANDEX_FOLDER_ID', '')
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
    for attempt in range(3):
        try:
            resp = httpx.post(url, json=body, headers=headers, timeout=30)
            if resp.status_code == 429:
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                    continue
                else:
                    resp.raise_for_status()
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            last_exc = exc
            if hasattr(exc, 'response') and exc.response is not None:
                if exc.response.status_code in (429, 403):
                    quota_exhausted = True
            if attempt < 2:
                time.sleep(2)

    msg = f"Yandex API failed after 3 attempts: {last_exc}"
    if quota_exhausted:
        msg += " (QUOTA_EXHAUSTED)"
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Unified LLM call
# ---------------------------------------------------------------------------

def call_llm(prompt_text: str, api_key: str = "", provider: str = "", max_tokens: int = 256) -> str:
    """Unified LLM call — routes to Gemini or Yandex based on provider."""
    if not provider:
        provider = getattr(config, 'LLM_PROVIDER', 'gemini')
    if provider == "yandex":
        return _call_yandex_api(prompt_text, api_key, max_tokens=max_tokens)
    else:
        return _call_gemini_api(prompt_text, api_key, max_tokens=max_tokens)


def _parse_score(text: str) -> int:
    """Extract an integer 1-10 from model output."""
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        score = int(digits[:2])
        return max(1, min(10, score))
    return 5


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_scenes(
    transcript_segments: list[dict],
    api_key: str,
    model: str | None = None,
    movie_title: str = "",
) -> list[dict]:
    """Score each transcript segment via Gemini.

    Parameters
    ----------
    transcript_segments:
        List of ``{start, end, text}`` dicts.
    api_key:
        OpenRouter / Google AI key.
    model:
        Unused — kept for backward compat. Reads from ``config.LLM_MODEL``.
    movie_title:
        Movie name for context-aware scoring (optional).

    Returns
    -------
    List of ``{start, end, text, score}`` dicts.
    """

    if not api_key:
        return [
            {**seg, "score": random.randint(5, 8)}
            for seg in transcript_segments
        ]

    total = len(transcript_segments)
    results: list[dict] = []
    quota_exhausted = False
    for i, seg in enumerate(transcript_segments, 1):
        if quota_exhausted:
            score = random.randint(5, 8)
            print(f"  Scene {i}/{total}: score={score} (quota exhausted)")
            results.append({**seg, "score": score})
            continue

        if movie_title:
            prompt = PROMPT_SCENE_ANALYSIS_WITH_TITLE.format(
                title=movie_title, text=seg["text"]
            )
        else:
            prompt = PROMPT_SCENE_ANALYSIS.format(text=seg["text"])
        print(f"  Scene {i}/{total}: scoring...", end="")
        try:
            raw = _call_gemini_api(prompt, api_key)
            score = _parse_score(raw)
            print(f" score={score}")
        except Exception as e:
            score = random.randint(5, 8)
            err_str = str(e)
            if "QUOTA_EXHAUSTED" in err_str:
                quota_exhausted = True
                print(f" score={score} (quota exhausted)")
            else:
                print(f" score={score} (fallback)")
        results.append({**seg, "score": score})

    return results


def batch_analyze(
    segments_batches: list[list[dict]],
    api_key: str,
    movie_title: str = "",
    provider: str = "",
) -> list[dict]:
    """Send batches of up to 10 segments in a single API call for efficiency.

    Also generates a short title for each scene (2-4 words, in Russian).

    Parameters
    ----------
    segments_batches:
        Pre-split list of segment lists (caller splits by 10).
    api_key:
        API key.
    movie_title:
        Movie name for context-aware scoring (optional).

    Returns
    -------
    Flat list of ``{start, end, text, score, title}`` dicts.
    """

    if not api_key:
        results: list[dict] = []
        for batch in segments_batches:
            for seg in batch:
                results.append({**seg, "score": random.randint(5, 8), "title": ""})
        return results

    all_results: list[dict] = []
    total_batches = len(segments_batches)
    total_scenes = sum(len(b) for b in segments_batches)
    scored_so_far = 0
    quota_exhausted = False

    for batch_idx, batch in enumerate(segments_batches):
        if quota_exhausted:
            for seg in batch:
                scored_so_far += 1
                score = random.randint(5, 8)
                print(f"  Scene {scored_so_far}/{total_scenes}: score={score} (quota exhausted)")
                all_results.append({**seg, "score": score, "title": ""})
            continue

        numbered = "\n".join(
            f"[{i + 1}] {seg['text']}" for i, seg in enumerate(batch)
        )
        if movie_title:
            prompt = (
                f"Ты анализируешь сцены из фильма «{movie_title}» для YouTube Shorts.\n"
                "Вот диалоги сцен (нумерованные).\n\n"
                f"{numbered}\n\n"
                "Для каждой сцены укажи:\n"
                "1) Оценку от 1 до 10 (насколько известна, эмоциональна, есть ли культовые моменты)\n"
                "2) Короткое название (2-4 слова, по-русски, отражающее суть момента)\n\n"
                "Формат ответа (строго одна строка на сцену):\n"
                "[номер] оценка — название\n\n"
                "Пример:\n"
                "[1] 7 — Культовая фраза\n"
                "[2] 4 — Обычный диалог\n"
                "[3] 9 — Смерть героя"
            )
        else:
            prompt = (
                "Ты анализируешь сцены из фильмов для YouTube Shorts.\n"
                "Вот диалоги сцен (нумерованные).\n\n"
                f"{numbered}\n\n"
                "Для каждой сцены укажи:\n"
                "1) Оценку от 1 до 10 (напряжённость, эмоциональность, наличие крылатых фраз)\n"
                "2) Короткое название (2-4 слова, по-русски)\n\n"
                "Формат ответа (строго одна строка на сцену):\n"
                "[номер] оценка — название\n\n"
                "Пример:\n"
                "[1] 7 — Напряжённый диалог\n"
                "[2] 4 — Обычная сцена\n"
                "[3] 9 — Эпичная битва"
            )

        print(f"  Batch {batch_idx+1}/{total_batches}: scoring {len(batch)} scenes...", end="")
        try:
            raw = call_llm(prompt, api_key, provider)
            # Parse: [N] score — title
            scene_data = _parse_batch_response(raw, len(batch))
            if scene_data:
                scores_str = ",".join(str(s["score"]) for s in scene_data)
                print(f" done ({scores_str})")
            else:
                # Fallback: try parsing as comma-separated scores (old format)
                parts = [p.strip() for p in raw.replace(";", ",").split(",")]
                scores = [_parse_score(p) for p in parts]
                scene_data = [{"score": scores[i] if i < len(scores) else 5, "title": ""}
                             for i in range(len(batch))]
                print(f" done (scores: {','.join(str(s) for s in scores)})")
        except Exception as e:
            err_str = str(e)
            if "QUOTA_EXHAUSTED" in err_str:
                quota_exhausted = True
                print(" quota exhausted")
            else:
                print(f" fallback: {e!s:.60}")
            scene_data = [{"score": random.randint(5, 8), "title": ""} for _ in batch]

        for i, seg in enumerate(batch):
            scored_so_far += 1
            sd = scene_data[i] if i < len(scene_data) else {"score": 5, "title": ""}
            all_results.append({**seg, "score": sd["score"], "title": sd["title"]})

    return all_results


def _parse_batch_response(raw: str, expected: int) -> list[dict] | None:
    """Parse batch response in format: [N] score — title

    Returns list of {score, title} or None if parsing fails.
    """
    import re
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


# ---------------------------------------------------------------------------
# Hybrid mode — ask LLM for best scenes
# ---------------------------------------------------------------------------

PROMPT_HYBRID_BEST_SCENES = (
    "Ты эксперт по YouTube Shorts из фильмов.\n"
    "Фильм: «{movie_name}»\n\n"
    "Назови 7-10 лучших сцен из этого фильма для YouTube Shorts (15-60 сек).\n\n"
    "Для каждой сцены укажи:\n"
    "1) МИНУТА: точная минута от начала фильма (только число, например 23 или 71).\n"
    "   Это ОЧЕНЬ ВАЖНО — укажи максимально точную минуту. Если сцена около 1 часа — пиши 60, если 1 час 10 минут — пиши 70.\n"
    "2) ОПИСАНИЕ: что происходит, кто участвует, о чём говорят (2-3 предложения)\n"
    "3) ЦИТАТА: ключевая реплика из сцены (1-2 предложения)\n"
    "4) НАЗВАНИЕ: короткое название для шортса (2-4 слова, по-русски)\n"
    "   — название должно быть УНИКАЛЬНЫМ для именно этого фильма\n"
    "   — если знаешь имя персонажа — упомяни\n"
    "   — зритель в ленте YouTube должен понять о чём шортс\n"
    "   Примеры для «{movie_name}»:\n"
    "     ПЛОХО: «спасение из огня», «общественное мнение», «диалог»\n"
    "     ХОРОШО: «Хэнкок спасает людей из здания», «Рэйнор спорит с Хэнкоком»\n"
    "5) ПРИЧИНА: почему эта сцена хороша для шортса\n\n"
    "Формат (строго одна сцена на блок, блоки через пустую строку):\n\n"
    "СЦЕНА 1\n"
    "МИНУТА: 23\n"
    "ОПИСАНИЕ: [что происходит]\n"
    "ЦИТАТА: [реплика]\n"
    "НАЗВАНИЕ: [2-4 слова]\n"
    "ПРИЧИНА: [почему подходит]\n"
)


def ask_llm_for_best_scenes(movie_name: str, api_key: str = "", provider: str = "") -> list[dict]:
    """Ask LLM to recommend best scenes from a movie for YouTube Shorts.

    Returns list of dicts: [{scene_num, description, quote, reason}, ...]
    """
    prompt = PROMPT_HYBRID_BEST_SCENES.format(movie_name=movie_name)
    try:
        raw = call_llm(prompt, api_key, provider, max_tokens=2048)
        return _parse_hybrid_response(raw)
    except Exception as e:
        print(f"  LLM hybrid query failed: {e}")
        return []


def _parse_hybrid_response(raw: str) -> list[dict]:
    """Parse hybrid mode response into structured scene list.
    
    New format:
        СЦЕНА 1
        ГДЕ: первая половина
        ОПИСАНИЕ: текст...
        ЦИТАТА: «текст»
        НАЗВАНИЕ: текст
        ПРИЧИНА: текст
    """
    scenes = []
    blocks = re.split(r'СЦЕНА\s+\d+', raw)
    nums = re.findall(r'СЦЕНА\s+(\d+)', raw)

    for i, block in enumerate(blocks[1:] if len(blocks) > 1 else []):
        scene: dict = {"scene_num": int(nums[i]) if i < len(nums) else i + 1}

        # МИНУТА (precise minute from start — primary field in new prompt)
        minute_match = re.search(r'МИНУТА[:\s]*(\d+)', block)

        # Legacy fallback: ГДЕ (old prompt with "23 минута" or "1 час 10 минут")
        if not minute_match:
            where_match = re.search(r'ГДЕ[:\s]*(.+?)(?:\n|ОПИСАНИЕ)', block, re.DOTALL)
            where_raw = where_match.group(1).strip() if where_match else ""
            scene["where"] = where_raw
            # "23 минута" → 23; "1 час 10 минут" → 70
            hour_match = re.search(r'(\d+)\s*час', where_raw)
            min_match = re.search(r'(\d+)\s*мин', where_raw)
            hours = int(hour_match.group(1)) if hour_match else 0
            mins = int(min_match.group(1)) if min_match else 0
            scene["minute"] = hours * 60 + mins
        else:
            scene["minute"] = int(minute_match.group(1))
            scene["where"] = ""

        # ОПИСАНИЕ (full scene description)
        desc_match = re.search(r'ОПИСАНИЕ[:\s]*(.+?)(?:\n|ЦИТАТА)', block, re.DOTALL)
        scene["description"] = desc_match.group(1).strip() if desc_match else ""

        # ЦИТАТА (key quote)
        quote_match = re.search(
            r'ЦИТАТА[:\s]*[«\u00ab"]?(.+?)[»\u00bb"]?\s*(?:\n|НАЗВАНИЕ)',
            block, re.DOTALL,
        )
        scene["quote"] = quote_match.group(1).strip() if quote_match else ""

        # НАЗВАНИЕ (short title)
        title_match = re.search(r'НАЗВАНИЕ[:\s]*(.+?)(?:\n|ПРИЧИНА)', block, re.DOTALL)
        scene["title"] = title_match.group(1).strip() if title_match else ""

        # ПРИЧИНА (why good for shorts)
        reason_match = re.search(r'ПРИЧИНА[:\s]*(.+?)$', block, re.DOTALL)
        scene["reason"] = reason_match.group(1).strip() if reason_match else ""

        if scene["quote"] or scene["description"]:
            scenes.append(scene)

    return scenes


def rank_scenes(
    scenes_with_scores: list[dict],
    top_n: int = 6,
    score_threshold: float = 7.0,
) -> list[dict]:
    """Return best scenes sorted by score, auto-selecting by quality.

    Takes *all* scenes with ``score >= score_threshold``.  If none qualify,
    falls back to the top 2 scenes.

    Parameters
    ----------
    scenes_with_scores:
        List of ``{start, end, duration, text, score}`` dicts.
    top_n:
        **Ignored when score_threshold is used.**  Kept for backward compat.
    score_threshold:
        Minimum score (1-10) to consider a scene worth keeping.
    """

    sorted_scenes = sorted(scenes_with_scores, key=lambda s: s["score"], reverse=True)

    # Auto-select by quality threshold
    selected: list[dict] = []
    for scene in sorted_scenes:
        if scene["score"] >= score_threshold:
            # Check minimum gap with already-selected scenes
            too_close = any(
                abs(scene["start"] - s["start"]) < 5 for s in selected
            )
            if not too_close:
                selected.append(scene)

    # Fallback: if nothing qualifies, take top 2
    if not selected:
        for scene in sorted_scenes[:2]:
            too_close = any(
                abs(scene["start"] - s["start"]) < 5 for s in selected
            )
            if not too_close:
                selected.append(scene)

    return sorted(selected, key=lambda s: s["start"])


def check_api_key(api_key: str, provider: str = "") -> dict:
    """Check if the API key is valid. Supports both Gemini and Yandex.

    Returns:
        {"ok": True} if key works,
        {"ok": False, "error": "message"} if key fails.
    """
    if not provider:
        provider = getattr(config, 'LLM_PROVIDER', 'gemini')
    if provider == "yandex":
        return _check_yandex_key(api_key)
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
        original_model = config.YANDEX_MODEL
        config.YANDEX_MODEL = "yandexgpt-lite"
        try:
            _call_yandex_api("Скажи OK", api_key)
        finally:
            config.YANDEX_MODEL = original_model
        return {"ok": True}
    except Exception as e:
        err = str(e)
        if "401" in err or "auth" in err.lower():
            return {"ok": False, "error": "Yandex: неверный ключ или Folder ID"}
        if "429" in err:
            return {"ok": False, "error": "Yandex: превышен лимит запросов"}
        return {"ok": False, "error": f"Yandex: {err[:80]}"}


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

    Format: СЦЕНА {N} | ОЦЕНКА: {score} | НАЗВАНИЕ: {title}
    Score may be integer or decimal (e.g. 8.5).

    Returns: [{scene_num, score, title}]
    """
    scenes = []
    seen_nums = set()
    for line in raw.split('\n'):
        m = re.match(
            r'СЦЕНА\s+(\d+)\s*\|\s*ОЦЕНКА[:\s]*(\d+(?:\.\d+)?)\s*\|\s*НАЗВАНИЕ[:\s]*(.+)',
            line,
        )
        if m:
            scene_num = int(m.group(1))
            # Ignore scene numbers outside reasonable range
            if scene_num < 1 or scene_num > 10000:
                continue
            # Ignore duplicate scene numbers
            if scene_num in seen_nums:
                continue
            seen_nums.add(scene_num)
            score_val = float(m.group(2))
            score_val = max(1.0, min(10.0, score_val))
            scenes.append({
                "scene_num": scene_num,  # 1-based
                "score": score_val,
                "title": m.group(3).strip(),
            })
    return scenes


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
            print(f" OK ({len(parsed)} rated)")
            all_results.extend(parsed)
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

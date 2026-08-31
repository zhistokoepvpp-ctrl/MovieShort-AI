"""T7 (v1-7-ui-polish) + T49 OpenCode Zen provider — API layer tests.

Covers request shape, null-content guard, routing through call_llm,
check_api_key no-key path and transient-only retry inheritance.
"""
import httpx
import pytest

import analyzers.text_analyzer as ta

OPENROUTER_OK = {"choices": [{"message": {"content": "  OK  "}}]}
OPENROUTER_NULL = {"choices": [{"message": {"content": None}}]}


class FakeResponse:
    """httpx.Response stand-in; raise_for_status() raises with .response attached."""

    def __init__(self, status_code, body=None):
        self.status_code = status_code
        self._body = body if body is not None else {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code < 400:
            return
        request = httpx.Request("POST", "https://fake.local")
        response = httpx.Response(self.status_code, request=request, json=self._body)
        raise httpx.HTTPStatusError(
            f"HTTP {self.status_code}", request=request, response=response
        )


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(ta.time, "sleep", lambda *_: None)


def _install_post(monkeypatch, responses):
    """Queue of FakeResponse/Exception; the last item repeats forever."""
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None, **kwargs):
        calls.append({"url": url, "json": json, "headers": headers})
        item = responses[min(len(calls) - 1, len(responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(ta.httpx, "post", fake_post)
    return calls


def test_openrouter_request_shape(monkeypatch):
    calls = _install_post(monkeypatch, [FakeResponse(200, OPENROUTER_OK)])
    ta._call_openrouter_api("prompt", api_key="test-key", max_tokens=256)
    assert len(calls) == 1
    assert calls[0]["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer test-key"
    assert calls[0]["headers"]["HTTP-Referer"]
    assert calls[0]["headers"]["X-Title"]
    assert calls[0]["json"]["model"] == ta.config.OPENROUTER_MODEL
    assert calls[0]["json"]["max_tokens"] == 256


def test_openrouter_null_content_raises(monkeypatch):
    _install_post(monkeypatch, [FakeResponse(200, OPENROUTER_NULL)])
    with pytest.raises(RuntimeError):
        ta._call_openrouter_api("prompt", api_key="test-key")


def test_openrouter_success_returns_text(monkeypatch):
    _install_post(monkeypatch, [FakeResponse(200, OPENROUTER_OK)])
    assert ta._call_openrouter_api("prompt", api_key="test-key") == "OK"


def test_call_llm_routes_openrouter(monkeypatch):
    calls = _install_post(monkeypatch, [FakeResponse(200, OPENROUTER_OK)])
    assert ta.call_llm("prompt", "test-key", provider="openrouter") == "OK"
    assert calls[0]["url"] == "https://openrouter.ai/api/v1/chat/completions"


def test_check_api_key_openrouter_no_key_invalid(monkeypatch):
    calls = _install_post(monkeypatch, [FakeResponse(200, OPENROUTER_OK)])
    result = ta.check_api_key("", provider="openrouter")
    assert result["ok"] is False
    assert result.get("error")
    assert calls == []


def test_check_openrouter_key_pins_probe_model(monkeypatch):
    # Key-check probe pins a non-reasoning model explicitly: config default
    # stealth/ox-alpha is reasoning-mandatory, max_tokens=16 burns on reasoning
    # -> null content -> null-content guard raises -> false key failure.
    calls = _install_post(monkeypatch, [FakeResponse(200, OPENROUTER_OK)])
    result = ta.check_api_key("test-key", provider="openrouter")
    assert result["ok"] is True
    assert len(calls) == 1
    assert calls[0]["json"]["model"] == "deepseek/deepseek-chat-v3-0324"
    assert calls[0]["json"]["max_tokens"] == 16
    assert ta.config.OPENROUTER_MODEL == "stealth/ox-alpha"


def test_openrouter_503_retries_three_times(monkeypatch):
    calls = _install_post(monkeypatch, [FakeResponse(503)])
    with pytest.raises(RuntimeError) as excinfo:
        ta._call_openrouter_api("prompt", api_key="test-key")
    assert len(calls) == 3
    assert "after 3 attempts" in str(excinfo.value)


def test_openrouter_403_keys_hint(monkeypatch):
    _install_post(monkeypatch, [FakeResponse(403)])
    with pytest.raises(RuntimeError) as excinfo:
        ta._call_openrouter_api("prompt", api_key="test-key")
    assert "openrouter.ai/keys" in str(excinfo.value)


def test_openrouter_403_server_reason_in_message(monkeypatch):
    _install_post(
        monkeypatch,
        [FakeResponse(403, {"error": {"message": "Key not allowed for this model"}})],
    )
    with pytest.raises(RuntimeError) as excinfo:
        ta._call_openrouter_api("prompt", api_key="test-key")
    msg = str(excinfo.value)
    assert "Key not allowed for this model" in msg
    # todo 43: model-settings hint must accompany the keys hint on 403.
    assert "openrouter.ai/settings/privacy" in msg


def test_openrouter_403_malformed_body_graceful(monkeypatch):
    # JSON without error key, and a non-dict body — must not crash,
    # generic message with keys hint still raised.
    for body in ({"unexpected": True}, "not-a-json-object"):
        _install_post(monkeypatch, [FakeResponse(403, body)])
        with pytest.raises(RuntimeError) as excinfo:
            ta._call_openrouter_api("prompt", api_key="test-key")
        msg = str(excinfo.value)
        assert "OpenRouter API failed" in msg
        assert "openrouter.ai/keys" in msg


# ── OpenCode Zen (T49) ──────────────────────────────────────────────
OPENCODE_ZEN_OK = {"choices": [{"message": {"content": "  OK  "}}]}
OPENCODE_ZEN_NULL = {"choices": [{"message": {"content": None}}]}


def test_opencode_zen_request_shape(monkeypatch):
    calls = _install_post(monkeypatch, [FakeResponse(200, OPENCODE_ZEN_OK)])
    ta._call_opencode_zen_api("prompt", api_key="test-zen-key", max_tokens=256)
    assert len(calls) == 1
    assert calls[0]["url"] == "https://opencode.ai/zen/v1/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer test-zen-key"
    assert calls[0]["headers"]["HTTP-Referer"]
    assert calls[0]["headers"]["X-Title"]
    assert calls[0]["json"]["model"] == ta.config.OPENCODE_ZEN_MODEL
    assert calls[0]["json"]["max_tokens"] == 256
    assert ta.config.OPENCODE_ZEN_MODEL == "nemotron-3-ultra-free"


def test_opencode_zen_null_content_raises(monkeypatch):
    _install_post(monkeypatch, [FakeResponse(200, OPENCODE_ZEN_NULL)])
    with pytest.raises(RuntimeError):
        ta._call_opencode_zen_api("prompt", api_key="test-zen-key")


def test_opencode_zen_success_returns_text(monkeypatch):
    _install_post(monkeypatch, [FakeResponse(200, OPENCODE_ZEN_OK)])
    assert ta._call_opencode_zen_api("prompt", api_key="test-zen-key") == "OK"


def test_call_llm_routes_opencode_zen(monkeypatch):
    calls = _install_post(monkeypatch, [FakeResponse(200, OPENCODE_ZEN_OK)])
    assert ta.call_llm("prompt", "test-zen-key", provider="opencode_zen") == "OK"
    assert calls[0]["url"] == "https://opencode.ai/zen/v1/chat/completions"


def test_check_api_key_opencode_zen_no_key_invalid(monkeypatch):
    calls = _install_post(monkeypatch, [FakeResponse(200, OPENCODE_ZEN_OK)])
    result = ta.check_api_key("", provider="opencode_zen")
    assert result["ok"] is False
    assert result.get("error")
    assert calls == []


def test_check_opencode_zen_pins_probe_model(monkeypatch):
    calls = _install_post(monkeypatch, [FakeResponse(200, OPENCODE_ZEN_OK)])
    result = ta.check_api_key("test-zen-key", provider="opencode_zen")
    assert result["ok"] is True
    assert len(calls) == 1
    assert calls[0]["json"]["model"] == "nemotron-3-ultra-free"
    assert calls[0]["json"]["max_tokens"] == 16
    assert ta.config.OPENCODE_ZEN_MODEL == "nemotron-3-ultra-free"


def test_opencode_zen_503_retries_three_times(monkeypatch):
    calls = _install_post(monkeypatch, [FakeResponse(503)])
    with pytest.raises(RuntimeError) as excinfo:
        ta._call_opencode_zen_api("prompt", api_key="test-zen-key")
    assert len(calls) == 3
    assert "after 3 attempts" in str(excinfo.value)


def test_opencode_zen_403_keys_hint(monkeypatch):
    _install_post(monkeypatch, [FakeResponse(403)])
    with pytest.raises(RuntimeError) as excinfo:
        ta._call_opencode_zen_api("prompt", api_key="test-zen-key")
    assert "opencode.ai/auth" in str(excinfo.value)


def test_opencode_zen_403_server_reason_in_message(monkeypatch):
    _install_post(
        monkeypatch,
        [FakeResponse(403, {"error": {"message": "Key not allowed for this model"}})],
    )
    with pytest.raises(RuntimeError) as excinfo:
        ta._call_opencode_zen_api("prompt", api_key="test-zen-key")
    msg = str(excinfo.value)
    assert "Key not allowed for this model" in msg
    assert "opencode.ai/auth" in msg


def test_opencode_zen_403_malformed_body_graceful(monkeypatch):
    for body in ({"unexpected": True}, "not-a-json-object"):
        _install_post(monkeypatch, [FakeResponse(403, body)])
        with pytest.raises(RuntimeError) as excinfo:
            ta._call_opencode_zen_api("prompt", api_key="test-zen-key")
        msg = str(excinfo.value)
        assert "OpenCode Zen API failed" in msg
        assert "opencode.ai/auth" in msg


def test_keys_hint_contains_opencode():
    assert "opencode.ai" in ta._KEYS_HINT
    assert ta._KEYS_HINT["opencode.ai"] == "opencode.ai/auth"
    assert "openrouter.ai" in ta._KEYS_HINT


def test_opencode_zen_model_list():
    assert len(ta.config.OPENCODE_ZEN_MODEL_LIST) >= 5
    assert ta.config.OPENCODE_ZEN_MODEL_LIST[0] == "nemotron-3-ultra-free"
    assert ta.config.OPENCODE_ZEN_MODEL == "nemotron-3-ultra-free"
    assert "mimo-v2.5-free" in ta.config.OPENCODE_ZEN_MODEL_LIST
    assert "hy3-free" in ta.config.OPENCODE_ZEN_MODEL_LIST


# ── Yandex batch (T7) — fence / truncate / null ─────────────────────

def test_yandex_null_content_raises(monkeypatch):
    monkeypatch.setattr(ta.config, "YANDEX_FOLDER_ID", "test-folder-id")
    _install_post(monkeypatch, [FakeResponse(200, {"choices": [{"message": {"content": None}}]})])
    with pytest.raises(RuntimeError, match="null content"):
        ta._call_yandex_api("prompt", api_key="test-key")


def test_yandex_fence_parsed():
    raw = '```json\n[{"start":2054,"end":2114,"title":"T","score":8,"block":0}]\n```'
    result = ta._parse_batch_response(raw, [0, 100])
    assert len(result) == 1
    assert 0 in result
    assert len(result[0]) == 1
    assert result[0][0]["title"] == "T"


def test_yandex_truncated_json_tolerant():
    # unclosed array — regex fallback must recover without exception
    raw = '[{"start":2054,"end":2114,"title":"T","score":8,"block":0}, {"start":2186,"end":2246,"title":"X","score":7.5,"block":1}'
    result = ta._parse_batch_response(raw, [0, 100, 200])
    assert isinstance(result, dict)
    assert result != {}
    assert 0 in result
    # malformed single object must not raise and should return {}
    raw2 = '[{"start":1, bad json'
    result2 = ta._parse_batch_response(raw2, [0, 100])
    assert result2 == {}


def test_parse_batch_fence_string_block():
    raw = '```json\n[{"start":0,"end":10,"title":"T","score":8,"block":"0"}]\n```'
    result = ta._parse_batch_response(raw, [0, 100])
    assert 0 in result
    assert len(result[0]) == 1
    # also without fence
    raw2 = '[{"start":1,"end":2,"title":"T","score":5,"block":"0"}]'
    result2 = ta._parse_batch_response(raw2, [0, 100])
    assert 0 in result2

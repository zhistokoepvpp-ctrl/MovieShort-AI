import pytest

@pytest.fixture
def mock_segments():
    """Mock whisper segments with word-level timestamps."""
    return [
        {"start": 0.0, "end": 5.0, "text": "Hello world."},
        {"start": 5.5, "end": 10.0, "text": "This is a test sentence."},
        {"start": 10.5, "end": 15.0, "text": "How are you today?"},
        {"start": 16.0, "end": 20.0, "text": "I am fine thank you."},
        {"start": 22.0, "end": 28.0, "text": "This has punctuation!"},
        {"start": 29.0, "end": 35.0, "text": "What about this one?"},
        {"start": 36.0, "end": 42.0, "text": "Finally the last segment."},
    ]

@pytest.fixture
def mock_keep_response():
    return "РЕШЕНИЕ: ОДНА"

@pytest.fixture
def mock_split_response():
    return """РЕШЕНИЕ: НЕСКОЛЬКО
ЧАСТИ:
ЧАСТЬ 1: 0 — 25
ЧАСТЬ 2: 25 — 47"""

@pytest.fixture
def mock_garbage_response():
    return "Это полная белиберда без формата 42"

@pytest.fixture
def mock_empty_segments():
    return []

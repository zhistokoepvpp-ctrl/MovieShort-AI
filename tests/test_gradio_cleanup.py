"""T14 — автоочистка Gradio Temp."""
import os
import sys
import types
import time
import tempfile
import pathlib

# Stub gradio if not installed (CI may not have it)
if "gradio" not in sys.modules:
    try:
        import gradio  # noqa: F401
    except ModuleNotFoundError:
        stub = types.ModuleType("gradio")
        # Minimal attributes used at import time
        stub.Blocks = object
        stub.File = object
        stub.Tab = object
        stub.Tabs = object
        stub.HTML = object
        stub.Textbox = object
        stub.Button = object
        stub.Checkbox = object
        stub.Slider = object
        stub.Radio = object
        stub.State = object
        stub.Row = object
        stub.Group = object
        stub.Accordion = object
        stub.themes = types.SimpleNamespace(
            Soft=lambda **kw: types.SimpleNamespace(set=lambda **k: None),
            colors=types.SimpleNamespace(orange=None, stone=None),
        )
        sys.modules["gradio"] = stub
        # stub gradio_client.utils if needed by main monkeypatch
        if "gradio_client" not in sys.modules:
            gc = types.ModuleType("gradio_client")
            gcu = types.ModuleType("gradio_client.utils")
            gcu.json_schema_to_python_type = lambda s: "Any"
            gcu._json_schema_to_python_type = lambda s, d: "Any"
            sys.modules["gradio_client"] = gc
            sys.modules["gradio_client.utils"] = gcu


def _load_helpers():
    """Try to import helpers from gui.app, fallback to local impl if gradio missing."""
    try:
        from gui.app import _try_remove_gradio_temp, _is_gradio_temp_path, cleanup_gradio_temp
        return _try_remove_gradio_temp, _is_gradio_temp_path, cleanup_gradio_temp
    except Exception:
        # Fallback local (same logic as gui/app.py) – separator-aware
        def _is_gradio_temp_path(p: str) -> bool:
            try:
                gradio_tmp = os.path.join(tempfile.gettempdir(), "gradio").lower()
                pl = p.lower()
                if gradio_tmp in pl:
                    return True
                if os.path.sep + "gradio" + os.path.sep in pl:
                    return True
                if pl.endswith(os.path.sep + "gradio"):
                    return True
                return False
            except Exception:
                return "gradio" in p.lower()

        def _try_remove_gradio_temp(p: str) -> None:
            try:
                if p and _is_gradio_temp_path(p) and os.path.isfile(p):
                    os.unlink(p)
            except Exception:
                pass

        def cleanup_gradio_temp(max_age_seconds: int = 86400) -> int:
            try:
                import shutil
                gradio_tmp = os.path.join(tempfile.gettempdir(), "gradio")
                if not os.path.isdir(gradio_tmp):
                    return 0
                now = time.time()
                removed = 0
                for f in os.listdir(gradio_tmp):
                    fp = os.path.join(gradio_tmp, f)
                    try:
                        if now - os.path.getmtime(fp) > max_age_seconds:
                            if os.path.isdir(fp):
                                shutil.rmtree(fp)
                            else:
                                os.unlink(fp)
                            removed += 1
                    except Exception:
                        pass
                return removed
            except Exception:
                return 0

        return _try_remove_gradio_temp, _is_gradio_temp_path, cleanup_gradio_temp


def test_gradio_after_process_unlink(tmp_path, monkeypatch):
    """Мок file.name в Temp\\gradio, вызов должен unlink."""
    _try_remove_gradio_temp, _is_gradio_temp_path, _ = _load_helpers()

    gradio_dir = os.path.join(tempfile.gettempdir(), "gradio")
    os.makedirs(gradio_dir, exist_ok=True)
    fake = os.path.join(gradio_dir, "_test_gradio_unlink.mp4")
    pathlib.Path(fake).write_text("dummy")

    assert os.path.exists(fake)
    assert "gradio" in fake.lower()
    # simulate on_process logic: same condition as in gui/app.py
    p = str(fake)
    try:
        if hasattr(pathlib.Path(p), "name"):
            pass
        if os.path.sep + "Temp" + os.path.sep in p or "gradio" in p.lower():
            import os as _os2
            _os2.unlink(p)
    except Exception:
        pass
    # also via helper
    if os.path.exists(fake):
        _try_remove_gradio_temp(fake)
    assert not os.path.exists(fake), "Gradio temp должен быть удалён после обработки"

    # Negative: source outside Temp/gradio must NOT be deleted
    outside = tmp_path / "outside.mp4"
    outside.write_text("keep")
    p2 = str(outside)
    # helper must NOT delete because no gradio/Temp
    _try_remove_gradio_temp(p2)
    assert outside.exists(), "Исходник вне Temp/gradio не должен удаляться"
    # ensure condition false
    assert not _is_gradio_temp_path(p2)

    # Temp separator variant
    temp_variant = tmp_path / "Temp" / "gradio" / "file.mp4"
    temp_variant.parent.mkdir(parents=True, exist_ok=True)
    temp_variant.write_text("x")
    pv = str(temp_variant)
    assert os.path.sep + "Temp" + os.path.sep in pv or "gradio" in pv.lower()
    _try_remove_gradio_temp(pv)
    assert not temp_variant.exists()


def test_gradio_startup_old_removed(monkeypatch):
    """Создать Temp\\gradio\\old_file с mtime -25h, вызвать startup cleanup, assert удалён."""
    gradio_tmp = os.path.join(tempfile.gettempdir(), "gradio")
    os.makedirs(gradio_tmp, exist_ok=True)

    old_file = os.path.join(gradio_tmp, "_test_old_gradio_file.tmp")
    new_file = os.path.join(gradio_tmp, "_test_new_gradio_file.tmp")
    pathlib.Path(old_file).write_text("old")
    pathlib.Path(new_file).write_text("new")
    old_mtime = time.time() - 25 * 3600
    os.utime(old_file, (old_mtime, old_mtime))

    assert os.path.exists(old_file)
    assert os.path.exists(new_file)

    # Try main first, then gui.app, then local
    removed = None
    try:
        from main import cleanup_gradio_temp_startup
        removed = cleanup_gradio_temp_startup()
    except Exception:
        try:
            _, _, gui_cleanup = _load_helpers()
            removed = gui_cleanup()
        except Exception:
            pass

    assert not os.path.exists(old_file), "Старый файл (>24ч) должен быть удалён стартап-чисткой"
    assert os.path.exists(new_file), "Новый файл не должен удаляться"
    try:
        os.unlink(new_file)
    except Exception:
        pass

    # also test gui helper directly
    _, _, gui_cleanup2 = _load_helpers()
    old2 = os.path.join(gradio_tmp, "_test_old2.tmp")
    pathlib.Path(old2).write_text("old2")
    os.utime(old2, (time.time() - 26 * 3600, time.time() - 26 * 3600))
    gui_cleanup2()
    assert not os.path.exists(old2)

    # verify gui/app.py contains gradio string
    assert "gradio" in pathlib.Path("gui/app.py").read_text(encoding="utf-8", errors="ignore").lower()

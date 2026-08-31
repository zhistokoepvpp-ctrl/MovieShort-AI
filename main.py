"""
MovieShort AI — Entry point
Clip movies into YouTube Shorts automatically or manually.
"""
# --- Monkey-patch: gradio_client bug (bool schema crashes "const" check) ---
try:
    import gradio_client.utils as _gcu

    _orig_top = _gcu.json_schema_to_python_type
    def _safe_top(schema):
        return "Any" if not isinstance(schema, dict) else _orig_top(schema)
    _gcu.json_schema_to_python_type = _safe_top

    _orig_inner = _gcu._json_schema_to_python_type
    def _safe_inner(schema, defs):
        return "Any" if not isinstance(schema, dict) else _orig_inner(schema, defs)
    _gcu._json_schema_to_python_type = _safe_inner
except Exception:
    pass
# --- End monkey-patch ---

import config
from gui.app import create_app


def cleanup_gradio_temp_startup():
    """Стартап-чистка Temp/gradio старше 24ч (T14)."""
    try:
        import tempfile
        import os
        import time
        import shutil

        gradio_tmp = os.path.join(tempfile.gettempdir(), "gradio")
        if os.path.isdir(gradio_tmp):
            now = time.time()
            removed = 0
            for f in os.listdir(gradio_tmp):
                fp = os.path.join(gradio_tmp, f)
                try:
                    if now - os.path.getmtime(fp) > 86400:
                        if os.path.isdir(fp):
                            shutil.rmtree(fp)
                        else:
                            os.unlink(fp)
                        removed += 1
                except Exception:
                    pass
            if removed:
                print(f"\U0001f9f9 Gradio temp: удалено {removed} старых файлов")
            return removed
    except Exception:
        pass
    return 0


def main():
    """Launch the MovieShort AI Gradio interface."""
    # T14 стартап-чистка
    try:
        cleanup_gradio_temp_startup()
    except Exception:
        pass
    # также пробуем вызвать из gui.app если там есть
    try:
        from gui.app import cleanup_gradio_temp as _gui_cleanup
        _gui_cleanup()
    except Exception:
        pass
    print(f"[MovieShort AI] starting on http://localhost:{config.GRADIO_PORT}")
    print(f"   Output directory: {config.OUTPUT_DIR}")
    print(f"   Language: Russian / English (select in Auto mode -> Film language)")

    app = create_app()
    app.launch(
        server_port=config.GRADIO_PORT,
        share=config.GRADIO_SHARE,
    )


if __name__ == "__main__":
    main()

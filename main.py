"""
MovieShort AI — Entry point
Clip movies into YouTube Shorts automatically or manually.
"""
import sys

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


def main():
    """Launch the MovieShort AI Gradio interface."""
    print(f"[MovieShort AI] starting on http://localhost:{config.GRADIO_PORT}")
    print(f"   Output directory: {config.OUTPUT_DIR}")
    print(f"   Language: Russian / English (select in Auto mode -> Film language)")

    app = create_app()
    app.launch(
        server_port=config.GRADIO_PORT,
        share=config.GRADIO_SHARE,
        show_error=True,
    )


if __name__ == "__main__":
    main()

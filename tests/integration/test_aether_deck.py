"""Regression tests for the AETHER Telemetry Deck deliverable.

Guards against the three classes of defect that froze the deck UI:
  1. a parse-time SyntaxError in the inline <script> block (aborted the whole script),
  2. DOM wiring bugs (getElementById used as a CSS selector; duplicate handlers),
  3. backend deprecations / broken f-strings / missing broadcast wiring.

Run:  python -m pytest tests/integration/test_aether_deck.py -q
"""
from pathlib import Path
import re
import subprocess
import tempfile

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = PROJECT_ROOT / "dashboard" / "aether_deck.html"
BRIDGE = PROJECT_ROOT / "scripts" / "05_launch_aether_deck.py"
NODE = "node"


def _inline_script() -> str:
    html = DASHBOARD.read_text(encoding="utf-8")
    blocks = re.findall(r"<script>([\s\S]*?)</script>", html)
    assert blocks, "no inline <script> block found"
    return blocks[-1]


def test_inline_script_has_no_stray_quote_syntax_error() -> None:
    js = _inline_script()
    # `descr":"` was the smoking-gun pattern that broke the whole script parse.
    assert 'descr":"' not in js, "stray quote before colon in descr key -> SyntaxError"
    assert 'descr:"' in js or "descr:" in js


def test_inline_script_passes_node_parse_check() -> None:
    js = _inline_script()
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as tf:
        tf.write(js)
        path = tf.name
    try:
        proc = subprocess.run([NODE, "--check", path], capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, f"node --check failed:\n{proc.stderr}"
    finally:
        Path(path).unlink(missing_ok=True)


def test_tab_switching_does_not_use_get_element_by_id_for_selectors() -> None:
    js = _inline_script()
    assert '$("[data-tab' not in js, "getElementById used with a CSS selector -> null dereference"
    assert 'document.querySelector(`[data-tab="${tab}"]`)' in js


def test_freeze_and_snapshot_have_single_handler_path() -> None:
    html = DASHBOARD.read_text(encoding="utf-8")
    for btn_id in ("freezeBtn", "snapshotBtn"):
        btn = re.search(rf'id="{btn_id}"[^>]*', html)
        assert btn, f"{btn_id} not found in HTML"
        tag = btn.group(0)
        assert "onclick=" not in tag, f"{btn_id} still has inline onclick (double-fire risk)"


def test_backend_uses_lifespan_not_deprecated_on_event() -> None:
    src = BRIDGE.read_text(encoding="utf-8")
    assert "@APP.on_event" not in src, "deprecated @APP.on_event usage present"
    assert "lifespan" in src, "lifespan context manager missing"


def test_backend_cors_middleware_present() -> None:
    src = BRIDGE.read_text(encoding="utf-8")
    assert "CORSMiddleware" in src, "CORSMiddleware not configured -> file:// origin blocked"
    assert "allow_origins" in src


def test_backend_no_broken_fstring_in_main() -> None:
    src = BRIDGE.read_text(encoding="utf-8")
    bad = re.findall(r'print\(f"[^"]*:port', src)
    assert not bad, f"broken f-string(s) present: {bad}"


def test_backend_broadcasts_gradcam_and_posterior_types() -> None:
    src = BRIDGE.read_text(encoding="utf-8")
    assert '"type": "gradcam"' in src, "backend never broadcasts gradcam -> live saliency dead"
    assert '"grad_cam_bands"' in src, "backend gradcam broadcast missing bands -> saliency never renders"
    assert '"probabilities"' in src, "backend gradcam broadcast missing probabilities -> live probs dead"
    assert '"type": "eval"' in src, "backend never broadcasts eval -> live macro-F1 dead"

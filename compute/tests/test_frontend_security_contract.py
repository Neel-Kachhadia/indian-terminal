from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_frontend_contract_selection_consumes_authoritative_expiry_api() -> None:
    option_chain = (ROOT / "src" / "components" / "OptionsChain" / "index.tsx").read_text("utf-8")
    strategy_lab = (ROOT / "src" / "components" / "StrategyLab" / "index.tsx").read_text("utf-8")
    source = option_chain + strategy_lab
    assert 'invoke<string[]>("get_expiries"' in source
    assert ".getDay(" not in source and "Thursday" not in source


def test_tauri_csp_has_no_remote_script_frame_or_broad_https_capability() -> None:
    config = json.loads((ROOT / "src-tauri" / "tauri.conf.json").read_text("utf-8"))
    csp = config["app"]["security"]["csp"]
    assert "script-src 'self'" in csp
    assert "frame-src 'none'" in csp
    assert "https:" not in csp
    assert "unsafe-eval" not in csp
    index = (ROOT / "index.html").read_text("utf-8")
    assert "fonts.googleapis.com" not in index
    assert "fonts.gstatic.com" not in index


def test_tauri_event_listeners_have_only_required_event_permissions() -> None:
    capability = json.loads(
        (ROOT / "src-tauri" / "capabilities" / "default.json").read_text("utf-8")
    )
    permissions = set(capability["permissions"])
    assert "core:event:allow-listen" in permissions
    assert "core:event:allow-unlisten" in permissions
    assert "core:event:default" not in permissions


def test_equity_pdf_viewer_is_locally_bundled_and_never_frames_remote_content() -> None:
    viewer = (ROOT / "src" / "components" / "EquityPanel" / "ContentViewer.jsx").read_text(
        "utf-8"
    )
    package = json.loads((ROOT / "package.json").read_text("utf-8"))
    assert package["dependencies"]["pdfjs-dist"]
    assert 'new URL("/pdf.min.mjs", window.location.href).href' in viewer
    assert "import(/* @vite-ignore */ moduleUrl)" in viewer
    assert 'workerSrc = "/pdf.worker.min.mjs"' in viewer
    assert (ROOT / "public" / "pdf.min.mjs").stat().st_size > 300_000
    assert (ROOT / "public" / "pdf.worker.min.mjs").stat().st_size > 500_000
    assert "cdnjs.cloudflare.com" not in viewer
    assert "createElement(\"script\")" not in viewer
    assert "<iframe" not in viewer
    assert 'invoke("fetch_equity_pdf"' in viewer
    assert "fetch(item.pdf_url)" not in viewer
    assert "getDocument(url)" not in viewer


def test_optional_plotly_runtime_is_lazy_and_self_hosted() -> None:
    loader = (ROOT / "src" / "lib" / "plotly.ts").read_text("utf-8")
    assert 'script.src = "/plotly.min.js"' in loader
    assert "document.createElement(\"script\")" in loader
    assert "http://" not in loader and "https://" not in loader
    assert (ROOT / "public" / "plotly.min.js").stat().st_size > 4_000_000
    sources = "".join(
        (ROOT / path).read_text("utf-8")
        for path in (
            "src/components/StrategyLab/index.tsx",
            "src/components/IVSurface/index.tsx",
        )
    )
    assert 'import("plotly.js-dist-min")' not in sources
    assert sources.count("loadPlotly()") == 2
    vite = (ROOT / "vite.config.ts").read_text("utf-8")
    assert "noDiscovery: true" in vite
    assert "holdUntilCrawlEnd: false" in vite
    assert "preTransformRequests: false" in vite
    assert '"react/jsx-dev-runtime"' in vite
    assert '"react-window"' in vite
    assert '"zustand"' in vite

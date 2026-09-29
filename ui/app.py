#!/usr/bin/env python
"""
DROCAT UI - Integrated Web Interface for Connectome Analysis

Launch with: python ui/app.py
Access at: http://127.0.0.1:8080

Design language follows the Photo Selector "gallery" reference:
light canvas, white surfaces, cobalt accent, segmented navigation,
focus-panel + contact-sheet workspace.
"""

import logging
import os
import sys
from pathlib import Path
from urllib.parse import unquote

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nicegui import ui, app

# Serve the local documentation so relative links (docs/ui_guides/...)
# open the local instruction files instead of GitHub.
app.add_static_files("/docs", PROJECT_ROOT / "docs")

from ui.config import APP_TITLE, APP_VERSION, APP_PORT, APP_HOST
from ui.components.banner import create_banner_stack
from ui.run_state import ACTIVE_STATUSES, RUN_MANAGER


class _TimerTeardownNoiseFilter(logging.Filter):
    """Hide the harmless NiceGUI 3.15 timer-teardown race from the console.

    When a browser tab is closed or reloaded, NiceGUI deletes the client's
    elements and then wakes tasks still waiting in ``client.connected()``;
    a per-page ``ui.timer`` can enter its loop one last time against an
    already-deleted parent slot and raise
    ``RuntimeError: The parent slot of Timer(id=...) has been deleted.``
    The page is already gone at that point, so the default exception
    handler's traceback is pure log noise - drop exactly those records.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not (
            "The parent slot of Timer" in message
            and "has been deleted" in message
        )


def _silence_timer_teardown_noise() -> None:
    logger = logging.getLogger("nicegui")
    if not any(isinstance(f, _TimerTeardownNoiseFilter) for f in logger.filters):
        logger.addFilter(_TimerTeardownNoiseFilter())


_silence_timer_teardown_noise()


def _migrate_legacy_neuron_indexes() -> None:
    """One-time upgrade: move legacy cache/ neuron indexes to neuron_indexes/.

    The index is a persistent "system file" now (auto-suggestions and the
    available-neurons viewer read it from ``neuron_indexes/``), so existing
    installations keep their pull state without re-downloading metadata.
    Runs once at startup; every failure is non-fatal.
    """
    try:
        from src.neuron_index_builder import migrate_legacy_neuron_indexes
        migrated = migrate_legacy_neuron_indexes(
            PROJECT_ROOT / "cache",
            PROJECT_ROOT / "neuron_indexes",
        )
        if migrated:
            print(
                "Migrated legacy neuron indexes to neuron_indexes/: "
                + ", ".join(migrated),
                flush=True,
            )
    except Exception:
        pass


# One-time startup migration, skipped under pytest so importing the app
# (as the UI test suite does) never mutates the real repo directory.
if "PYTEST_CURRENT_TEST" not in os.environ:
    _migrate_legacy_neuron_indexes()


def _saved_dark_mode() -> bool | None:
    """Return the persisted theme preference from a plain cookie.

    Values: ``True`` = dark, ``False`` = light, ``None`` = follow the
    system preference. The cookie stores 'dark', 'light' or 'auto' (older
    versions stored '1'/'0') so a reload restores the chosen mode without
    a flash. Falls back to system mode when no preference is saved yet or
    no request context exists (script mode, UI tests).
    """
    try:
        request = ui.context.client.request
    except Exception:
        return None
    saved = request.cookies.get("drocat_dark")
    if saved in ("1", "dark"):
        return True
    if saved in ("0", "light"):
        return False
    # 'auto', missing, or unrecognized -> follow the OS preference.
    return None


_DROCAT_TAB_NAMES = {
    "Complete Paths",
    "Shortest Paths",
    "Network",
    "Paths",
    "Type Validation",
    "Skeleton",
    "Net-Viz",
    "Connectivity",
    "Morphology",
    "Find Lines",
    "Find Neurons",
    "Co-Labeling",
    "Downloader",
    "Settings",
}

# Browser-side user state only: a returning user's saved tab name is migrated
# before the membership check. This is NOT a data back-compat shim.
_TAB_NAME_ALIASES = {"Cross-Dataset": "Paths"}


def _saved_active_tab() -> str:
    """Restore the last selected tab from a lightweight browser cookie."""
    try:
        saved = unquote(str(ui.context.client.request.cookies.get(
            "drocat_active_tab", ""
        )))
    except Exception:
        saved = ""
    saved = _TAB_NAME_ALIASES.get(saved, saved)
    return saved if saved in _DROCAT_TAB_NAMES else "Complete Paths"


from ui.tabs import (
    create_find_path_tab,
    create_find_shortest_tab,
    create_network_tab,
    create_inter_dataset_tab,
    create_type_validation_tab,
    create_skeleton_tab,
    create_net_viz_tab,
    create_connectivity_tab,
    create_morphology_tab,
    create_nb_find_lines_tab,
    create_nb_find_neuron_tab,
    create_nb_colabel_tab,
    create_flylight_tab,
    create_settings_tab,
)


DROCAT_CSS = """
:root {
    --drocat-canvas: #f7f8fa;
    --drocat-surface: #ffffff;
    --drocat-soft: #f1f3f6;
    --drocat-navy: #0b1f3a;
    --drocat-muted: #667085;
    --drocat-faint: #98a2b3;
    --drocat-line: #e2e7ee;
    --drocat-line-strong: #cfd7e3;
    --drocat-cobalt: #145cff;
    --drocat-cobalt-soft: #eaf0ff;
    --drocat-ok: #16a34a;
    --drocat-warn: #d97706;
    --drocat-err: #dc2626;
    --drocat-radius-sm: 10px;
    --drocat-radius-md: 16px;
    --drocat-radius-lg: 24px;
    --drocat-shadow: 0 16px 40px rgba(11, 31, 58, .08);
    /* Tab-group tints, shared by the group headers and their tabs */
    --drocat-tint-connection: #eaf0ff;
    --drocat-tint-crossdataset: #ffeef2;
    --drocat-tint-visualization: #e2f6f5;
    --drocat-tint-similarity: #fff3e0;
    --drocat-tint-nb: #f6f1ff;
    --drocat-tint-flylight: #edf9f0;
    --drocat-tint-settings: #eef1f5;
    /* Glass and solid surfaces whose light values are baked into the CSS */
    --drocat-glass: rgba(255, 255, 255, .86);
    --drocat-panel-glass: rgba(255, 255, 255, .72);
    --drocat-table-glass: rgba(255, 255, 255, .78);
    --drocat-toolbar-soft: rgba(247, 249, 252, .92);
    --drocat-toolbar-grad: linear-gradient(135deg, rgba(234, 240, 255, .78), rgba(255, 255, 255, .96));
    --drocat-query-preview-grad: linear-gradient(135deg, rgba(232, 241, 255, .78), rgba(255, 255, 255, .92));
    --drocat-labelmapper-grad: linear-gradient(180deg, rgba(247, 249, 252, .84), #fff);
    --drocat-tab-hover: rgba(255, 255, 255, .55);
    --drocat-cell: #ffffff;
    --drocat-row-even: #f7fbff;
    --drocat-row-hover: #eaf3ff;
    --drocat-selected: #dcecff;
    --drocat-selected-hover: #cfe3ff;
    /* Floating mapped-view columns: fully opaque and tinted differently
       from regular cells so they read as a separate layer. */
    --drocat-map-cell-bg: #e7ebff;
    --drocat-map-cell-head-bg: #b7c8ff;
    --drocat-badge-bg: #fff8d6;
    --drocat-badge-fg: #946200;
    --drocat-badge-soft-bg: #fffbe8;
    --drocat-badge-soft-fg: #a07a2c;
    --drocat-mark-bg: #fff1b8;
    --drocat-err-border: #fecaca;
}

/* ---------- Dark mode ----------
   NiceGUI/Quasar dark mode toggles the body--dark class on <body>. These
   overrides flip every drocat token; the html:has() arm keeps the page
   canvas dark behind the body as well. */
body.body--dark,
html:has(> body.body--dark) {
    --drocat-canvas: #0c1322;
    --drocat-surface: #131c2e;
    --drocat-soft: #1d2b45;
    --drocat-navy: #e7eef8;
    --drocat-muted: #9aa9c0;
    --drocat-faint: #6b7a93;
    --drocat-line: #25334d;
    --drocat-line-strong: #374a6b;
    --drocat-cobalt: #5b8cff;
    --drocat-cobalt-soft: #1a2c52;
    --drocat-ok: #22c55e;
    --drocat-warn: #f59e0b;
    --drocat-err: #f87171;
    --drocat-shadow: 0 16px 40px rgba(0, 0, 0, .5);
    --drocat-tint-connection: #18233c;
    --drocat-tint-crossdataset: #331e26;
    --drocat-tint-visualization: #132a2c;
    --drocat-tint-similarity: #2c2313;
    --drocat-tint-nb: #251a3d;
    --drocat-tint-flylight: #152a1d;
    --drocat-tint-settings: #1c2637;
    --drocat-glass: rgba(19, 28, 46, .86);
    --drocat-panel-glass: rgba(19, 28, 46, .72);
    --drocat-table-glass: rgba(19, 28, 46, .78);
    --drocat-toolbar-soft: rgba(23, 33, 52, .92);
    --drocat-toolbar-grad: linear-gradient(135deg, rgba(26, 44, 82, .72), rgba(19, 28, 46, .96));
    --drocat-query-preview-grad: linear-gradient(135deg, rgba(26, 44, 82, .6), rgba(19, 28, 46, .92));
    --drocat-labelmapper-grad: linear-gradient(180deg, rgba(28, 42, 63, .84), #131c2e);
    --drocat-tab-hover: rgba(255, 255, 255, .07);
    --drocat-cell: #131c2e;
    --drocat-row-even: #17233a;
    --drocat-row-hover: #1e3050;
    --drocat-selected: #1d3a5f;
    --drocat-selected-hover: #25486f;
    /* Floating mapped-view columns: fully opaque and tinted differently
       from regular cells so they read as a separate layer. */
    --drocat-map-cell-bg: #232f52;
    --drocat-map-cell-head-bg: #43569b;
    --drocat-badge-bg: #3d3113;
    --drocat-badge-fg: #e5b74d;
    --drocat-badge-soft-bg: #332b16;
    --drocat-badge-soft-fg: #c9a44d;
    --drocat-mark-bg: #3d3612;
    --drocat-err-border: #5c2c2c;
}

/* A slightly stronger canvas glow keeps the dark theme from going flat. */
body.body--dark {
    background:
        radial-gradient(circle at 16% 0%, rgba(91, 140, 255, .08), transparent 32%),
        var(--drocat-canvas) !important;
}

/* Release selection guidance stays visible and keyboard-friendly in both
   themes without changing the selector's value implicitly. */
.drocat-release-notice {
    margin-top: 6px;
    padding: 8px 10px;
    border: 1px solid color-mix(in srgb, var(--drocat-cobalt) 28%, var(--drocat-line));
    border-radius: var(--drocat-radius-sm);
    background: var(--drocat-cobalt-soft);
    color: var(--drocat-navy);
    font-size: 12px;
    line-height: 1.35;
}
.drocat-release-notice-text { color: var(--drocat-navy); }
.drocat-release-notice-muted { color: var(--drocat-muted); }
.drocat-release-action .q-btn__content,
.drocat-release-dismiss .q-btn__content { color: var(--drocat-cobalt); }

/* Active group-tab accents need brighter hues on the dark tints. */
body.body--dark .drocat-nav .drocat-tint-crossdataset .drocat-group-tab.drocat-active { color: #fb7185 !important; }
body.body--dark .drocat-nav .drocat-tint-visualization .drocat-group-tab.drocat-active { color: #2dd4bf !important; }
body.body--dark .drocat-nav .drocat-tint-similarity .drocat-group-tab.drocat-active { color: #f59e0b !important; }
body.body--dark .drocat-nav .drocat-tint-nb .drocat-group-tab.drocat-active { color: #a78bfa !important; }
body.body--dark .drocat-nav .drocat-tint-flylight .drocat-group-tab.drocat-active { color: #4ade80 !important; }
body.body--dark .drocat-nav .drocat-tint-settings .drocat-group-tab.drocat-active { color: #94a3b8 !important; }

/* The results mark keeps its white icon on a dark navy chip in both themes. */
body.body--dark .drocat-results-mark { background: #1d2a42; }

/* Theme picker in the header (System / Light / Dark). The button shows a
   persistent sun | moon pair in every mode; the menu highlights the mode. */
.drocat-dark-toggle {
    color: var(--drocat-muted) !important;
    font-size: 20px;
}
.drocat-dark-toggle:hover { color: var(--drocat-cobalt) !important; }
.drocat-theme-icon-pair {
    display: flex;
    align-items: center;
    gap: 5px;
    line-height: 1;
}
.drocat-theme-icon-pair .q-icon {
    /* NiceGUI renders q-icon at 1.715em (the button was 20px -> 34.3px);
       pin the icons to exactly 2/3 of that previous size. */
    font-size: 22.9px;
}
.drocat-theme-sep {
    width: 2.5px;
    height: 14px;
    border-radius: 999px;
    background: var(--drocat-line-strong);
}
.drocat-theme-menu { min-width: 152px; }
.drocat-theme-item { color: var(--drocat-muted); border-radius: 8px; }
.drocat-theme-item .drocat-theme-icon { color: var(--drocat-muted); }
.drocat-theme-item .drocat-theme-check { color: transparent; }
.drocat-theme-item.drocat-theme-item-active,
.drocat-theme-item.drocat-theme-item-active .drocat-theme-icon,
.drocat-theme-item.drocat-theme-item-active .drocat-theme-check {
    color: var(--drocat-cobalt) !important;
}

/* Token reminder box (Settings tab) follows the theme. */
.drocat-token-reminder {
    border: 1px solid #e6a23c;
    background: #fdf6ec;
    border-radius: 8px;
    padding: 10px 12px;
}
body.body--dark .drocat-token-reminder {
    border-color: #8a6a1f;
    background: #2f2514;
}

html, body {
    background:
        radial-gradient(circle at 16% 0%, rgba(20, 92, 255, .04), transparent 32%),
        var(--drocat-canvas) !important;
    color: var(--drocat-navy);
    font-family: Inter, "Avenir Next", "Segoe UI", Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    line-height: 1.45;
}

/* ---------- Header ---------- */
.drocat-header {
    background: var(--drocat-glass) !important;
    backdrop-filter: blur(14px);
    border-bottom: 1px solid var(--drocat-line);
    box-shadow: none !important;
    min-height: 76px !important;
    padding: 0 28px !important;
}
.drocat-brand-mark {
    width: 42px; height: 42px; border-radius: 13px;
    display: grid; place-items: center;
    background: var(--drocat-cobalt);
    color: #fff;
    box-shadow: 0 8px 18px rgba(20, 92, 255, .24);
}
.drocat-brand-title { font-size: 19px; font-weight: 700; letter-spacing: -.02em; color: var(--drocat-navy); }
.drocat-brand-sub { font-size: 12.5px; color: var(--drocat-muted); }
.drocat-version-pill {
    border: 1px solid var(--drocat-line-strong); border-radius: 999px;
    padding: 4px 10px; font-size: 11px; font-weight: 650; color: var(--drocat-muted);
    background: var(--drocat-surface);
}
.drocat-header-link {
    color: var(--drocat-muted) !important; font-size: 12.5px; font-weight: 600;
    text-decoration: none !important;
}
.drocat-header-link:hover { color: var(--drocat-cobalt) !important; }
.drocat-activity-btn {
    color: var(--drocat-muted) !important;
    min-height: 36px;
}
.drocat-activity-btn:hover { color: var(--drocat-cobalt) !important; }
.drocat-activity-card { width: min(560px, 92vw); max-width: 92vw; }
.drocat-activity-list { max-height: min(62vh, 620px); overflow-y: auto; }
.drocat-activity-row {
    border: 1px solid var(--drocat-line);
    border-radius: 10px;
    padding: 9px 10px;
    background: var(--drocat-surface);
}
.drocat-activity-row:hover { background: var(--drocat-soft); }
.drocat-activity-title { color: var(--drocat-navy); font-weight: 650; }
.drocat-activity-meta { color: var(--drocat-muted); font-size: 11px; }
.drocat-doc-link {
    color: var(--drocat-cobalt) !important;
    font-size: 11px;
    font-weight: 650;
    text-decoration: none !important;
    white-space: nowrap;
}
.drocat-doc-link:hover { text-decoration: underline !important; }
.drocat-inline-link {
    color: var(--drocat-cobalt) !important;
    font-size: 12px;
    font-weight: 650;
    text-decoration: none !important;
}
.drocat-inline-link:hover { text-decoration: underline !important; }
.drocat-neuron-search-toolbar {
    margin-top: 8px;
    padding: 10px 12px 8px;
    border: 1px solid var(--drocat-line-strong);
    border-radius: 14px;
    background: var(--drocat-toolbar-grad);
    box-shadow: 0 4px 14px rgba(11, 31, 58, .06);
}
.drocat-neuron-search-toolbar .drocat-neuron-search-field {
    background: var(--drocat-table-glass);
    border-radius: 10px;
}
.drocat-neuron-search-toolbar .drocat-neuron-search-submit {
    /* The row uses bottom alignment; stretch the button to the query field's
       full control height without changing its natural responsive width. */
    align-self: stretch;
}
.drocat-neuron-search-toolbar .q-field--outlined .q-field__control {
    border-radius: 10px;
}
.drocat-neuron-search-toolbar .q-field--outlined .q-field__control:before {
    border-color: var(--drocat-line-strong);
}
.drocat-neuron-search-toolbar .q-field--outlined.q-field--focused .q-field__control:after {
    border-color: var(--drocat-cobalt);
}
.drocat-neuron-search-toolbar .q-field__label {
    color: var(--drocat-muted);
}
.drocat-neuron-results-layout {
    display: grid;
    grid-template-columns: minmax(480px, 520px) minmax(0, 1fr);
    gap: 12px;
    align-items: stretch;
    min-height: 0;
    /* Fixed window height so the two panels never reflow with the (changing)
       result-set size, which otherwise flashes/reshapes while typing. The
       tables scroll inside their panels instead of growing them. */
    height: min(58vh, 680px);
}
.drocat-neuron-match-panel,
.drocat-neuron-full-panel {
    min-width: 0;
    min-height: 0;
    border: 1px solid var(--drocat-line);
    border-radius: 12px;
    padding: 12px;
    background: var(--drocat-surface);
    display: flex;
    flex-direction: column;
    overflow: hidden;
}
.drocat-neuron-match-panel {
    background: linear-gradient(180deg, var(--drocat-cobalt-soft), var(--drocat-surface));
    border-color: var(--drocat-line-strong);
    box-shadow: 0 3px 12px rgba(11, 31, 58, .06);
}
.drocat-neuron-query-preview {
    margin: 10px 0 2px;
    padding: 10px 12px;
    border: 1px solid var(--drocat-line-strong);
    border-radius: 10px;
    background: var(--drocat-query-preview-grad);
}
.drocat-neuron-query-preview-list {
    max-height: 118px;
    overflow-y: auto;
    overflow-x: hidden;
    padding: 2px 2px 3px;
    scrollbar-width: thin;
}
.drocat-neuron-query-preview-list.drocat-neuron-query-preview-expanded {
    max-height: none;
    overflow-y: visible;
}
.drocat-query-preview-expand-btn {
    min-height: 24px;
    padding: 0 6px;
    color: var(--drocat-cobalt);
    font-size: 11px;
}
.drocat-query-preview-expand-btn:hover {
    background: rgba(69, 126, 191, .10);
}
.drocat-neuron-query-chip {
    display: inline-flex;
    align-items: center;
    min-height: 26px;
    padding: 3px 9px;
    border: 1px solid rgba(69, 126, 191, .38);
    border-radius: 999px;
    background: var(--drocat-cell);
    color: var(--drocat-navy);
    font-size: 12px;
    font-weight: 650;
    line-height: 1.2;
}
.drocat-neuron-query-chip-wrap {
    display: inline-flex;
    align-items: center;
    gap: 2px;
    min-height: 26px;
    padding: 3px 5px 3px 9px;
    border: 1px solid rgba(69, 126, 191, .38);
    border-radius: 999px;
    background: var(--drocat-cell);
    color: var(--drocat-navy);
}
.drocat-neuron-query-chip-wrap .drocat-neuron-query-chip {
    min-height: auto;
    padding: 0;
    border: 0;
    background: transparent;
}
.drocat-neuron-query-chip-remove {
    min-height: 20px;
    min-width: 20px;
    color: var(--drocat-muted);
}
.drocat-neuron-query-chip-remove:hover {
    color: var(--drocat-navy);
    background: rgba(69, 126, 191, .10);
}
.drocat-chip-input-shell .drocat-chip-input .q-field__control {
    align-content: flex-start;
    scrollbar-width: thin;
}
.drocat-fixed-neuron-input {
    width: 100%;
    min-width: 0;
}
.drocat-fixed-neuron-input .drocat-neuron-input-row {
    flex-wrap: nowrap;
    min-width: 0;
}
.drocat-fixed-neuron-input .drocat-chip-input-shell {
    flex: 1 1 0%;
    width: 0;
    min-width: 0;
}
.drocat-fixed-neuron-input .drocat-chip-input-shell .q-field {
    width: 100%;
    min-width: 0;
}
.drocat-fixed-neuron-input .drocat-neuron-match-filter {
    flex: 0 0 128px;
    width: 128px !important;
    min-width: 128px;
}
.drocat-fixed-neuron-input .drocat-upload-trigger {
    flex: 0 0 34px;
    width: 34px !important;
    min-width: 34px;
}
@media (max-width: 760px) {
    .drocat-fixed-neuron-input .drocat-neuron-input-row {
        flex-wrap: wrap;
    }
    .drocat-fixed-neuron-input .drocat-chip-input-shell {
        flex-basis: 100%;
        width: 100%;
    }
    .drocat-fixed-neuron-input .drocat-neuron-match-filter {
        flex: 1 1 128px;
    }
}
/* The scroll/clip window must live on .q-field__native (the chips' direct
   parent). The container under .q-field__control carries Quasar's
   `height: inherit` and flex-shrinks the native to 118px while the chips
   keep painting past it — an overflow clip on the control alone did not
   stop that paint (observed as chips rendering below the field border in
   the collapsed state). Clipping where the chips actually live fixes it. */
.drocat-chip-input-shell.drocat-chip-list-collapsed
    .drocat-chip-input .q-field__control {
    max-height: 142px;
    overflow-y: auto;
    overflow-x: hidden;
}
.drocat-chip-input-shell.drocat-chip-list-collapsed
    .drocat-chip-input .q-field__native {
    max-height: 118px;
    overflow-y: auto;
    overflow-x: hidden;
}
.drocat-chip-input-shell.drocat-chip-list-expanded
    .drocat-chip-input .q-field__control {
    max-height: none;
    overflow-y: auto;
    overflow-x: hidden;
}
.drocat-chip-input-shell.drocat-chip-list-expanded
    .drocat-chip-input .q-field__native {
    max-height: none;
    overflow-y: auto;
    overflow-x: hidden;
}
.drocat-chip-expand-btn {
    min-height: 24px;
    padding: 0 6px;
    color: var(--drocat-cobalt);
    font-size: 11px;
}
.drocat-chip-expand-btn:hover {
    background: rgba(69, 126, 191, .10);
}
.drocat-neuron-match-jump {
    min-height: 26px;
    padding: 0 4px;
    color: var(--drocat-navy);
    font-weight: 650;
}
.drocat-neuron-match-jump:hover {
    color: var(--drocat-cobalt);
    background: rgba(69, 126, 191, .10);
}
.drocat-neuron-full-panel {
    background: var(--drocat-panel-glass);
}
.drocat-neuron-viewer-card {
    max-height: 94vh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    padding: 12px 16px 14px;
}
.drocat-neuron-dialog-header {
    min-height: 30px;
    margin-bottom: 2px;
}
.drocat-neuron-dialog-title {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.drocat-neuron-header-meta {
    min-width: 0;
    overflow: hidden;
}
.drocat-neuron-header-meta .q-badge {
    min-height: 22px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 700;
}
.drocat-neuron-header-meta .drocat-neuron-source,
.drocat-neuron-header-meta .drocat-neuron-enriched {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.drocat-neuron-intro-row {
    display: grid;
    grid-template-columns: minmax(320px, .9fr) minmax(0, 1.6fr);
    align-items: stretch;
    gap: 10px;
    margin: 2px 0 4px;
}
.drocat-neuron-intro-row .drocat-neuron-query-preview {
    min-width: 0;
    margin: 0;
    padding: 7px 10px;
}
.drocat-neuron-search-help {
    align-self: center;
    min-width: 0;
    margin: 0;
    line-height: 1.35;
}
/* Zero-hit alias panel: every cross-dataset type-name suggestion for the
   query. A broad query can list dozens of entries; cap the panel and scroll
   it internally so it cannot stretch (and crowd out) the viewer layout. */
.drocat-neuron-alias-panel {
    flex: 0 0 auto;
    max-height: 40vh;
    overflow-y: auto;
    padding: 4px 8px 4px 2px;
    border: 1px solid var(--drocat-line);
    border-radius: 8px;
    background: var(--drocat-toolbar-soft);
}
/* Alias panel sections lay every dataset block into one shared grid, so the
   badges, action buttons, and detail text line up in columns across rows
   instead of shifting with each row's content width. */
.drocat-neuron-alias-grid {
    display: grid;
    grid-template-columns:
        max-content max-content max-content max-content minmax(0, 1fr);
    gap: 8px 12px;
    align-items: start;
    margin-top: 4px;
}
/* The auto-mapping section has no action buttons: badge + text only. */
.drocat-neuron-alias-grid-mapped {
    grid-template-columns: max-content minmax(0, 1fr);
}
.drocat-neuron-alias-col-badge,
.drocat-neuron-alias-col-mapped,
.drocat-neuron-alias-col-sankey,
.drocat-neuron-alias-col-network {
    justify-self: start;
}
.drocat-neuron-alias-col-badge { grid-column: 1; }
.drocat-neuron-alias-col-mapped { grid-column: 2; }
.drocat-neuron-alias-col-sankey { grid-column: 3; }
.drocat-neuron-alias-col-network { grid-column: 4; }
/* Button rows center the dataset badge on the Mapped types / Sankey /
   Network buttons' axis: the grid top-aligns items, and the bare badge
   (18px) sat visibly above the taller flat dense buttons (32px at the
   app's font size, hence the em-based min-height). */
.drocat-neuron-alias-badge-centered {
    display: flex;
    align-items: center;
    min-height: 2.2857em;
}
.drocat-neuron-alias-col-details {
    grid-column: 5;
    min-width: 0;
}
.drocat-neuron-viewer-content {
    min-height: 0;
    overflow: auto;
    padding-top: 0;
    padding-bottom: 4px;
}
.drocat-neuron-panel-toolbar {
    flex: 0 0 auto;
    min-height: 34px;
    margin-top: 6px;
    padding: 4px 6px;
    border: 1px solid var(--drocat-line);
    border-radius: 8px;
    background: var(--drocat-toolbar-soft);
}
/* The results window has a fixed height and both panels clip overflow, so the
   inner scroll chains must shrink into each panel's leftover space. Capping
   .q-table__middle at the full window height instead pushed the scroller's
   bottom edge past the panel border, where the clipped last row(s) could never
   be scrolled into view. flex-shrink with min-height: 0 bounds every box to
   the space actually available; short result sets still hug their content. */
.drocat-neuron-match-panel > .drocat-neuron-match-table.q-table__container {
    display: flex;
    flex-direction: column;
    flex: 0 1 auto;
    min-height: 0;
}
.drocat-neuron-match-table .q-table__middle {
    min-height: 0;
    flex: 0 1 auto;
    overflow-x: hidden;
    overflow-y: auto;
    margin-top: 8px;
    border: 1px solid var(--drocat-line);
    border-radius: 8px;
    background: var(--drocat-table-glass);
}
.drocat-neuron-match-table table {
    width: 100%;
    table-layout: fixed;
}
.drocat-neuron-match-table th {
    position: sticky;
    top: 0;
    z-index: 2;
    background: var(--drocat-cobalt-soft) !important;
    color: var(--drocat-navy) !important;
    white-space: nowrap;
}
.drocat-neuron-match-table td,
.drocat-neuron-match-table th {
    height: 40px;
    padding: 7px 9px;
}
.drocat-neuron-match-table td.drocat-neuron-match-by {
    color: var(--drocat-muted);
    font-size: 12px;
    font-weight: 650;
    white-space: nowrap;
}
.drocat-neuron-match-table td.drocat-neuron-match-value {
    color: var(--drocat-navy);
    font-weight: 700;
    overflow: visible;
    white-space: normal;
    overflow-wrap: anywhere;
}
.drocat-neuron-match-table td.drocat-neuron-match-count {
    color: var(--drocat-muted);
    font-variant-numeric: tabular-nums;
    text-align: right;
    white-space: nowrap;
}
.drocat-neuron-match-select-cell {
    width: 36px;
    min-width: 36px;
    padding-left: 7px !important;
    padding-right: 3px !important;
    text-align: center;
}
/* Full-table selection column: the header select-all and the per-row
   checkboxes share one geometry. The th would otherwise inherit the
   browser's centered-text default while the td left-aligns, so the two
   checkbox columns drift apart. */
.drocat-neuron-select-cell {
    padding-left: 9px !important;
    padding-right: 9px !important;
    text-align: center;
}
.drocat-neuron-match-secondary-arrow {
    color: var(--drocat-faint);
    opacity: .9;
}
.drocat-neuron-match-source {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    white-space: nowrap;
}
.drocat-neuron-match-value-line {
    display: flex;
    align-items: center;
    gap: 5px;
    min-width: 0;
}
.drocat-neuron-match-secondary-label {
    display: inline-flex;
    align-items: center;
    min-height: 18px;
    padding: 1px 5px;
    border: 1px solid rgba(234, 179, 8, .68);
    border-radius: 999px;
    background: var(--drocat-badge-bg);
    color: var(--drocat-badge-fg);
    font-size: 10px;
    font-weight: 750;
    letter-spacing: .02em;
    line-height: 1.2;
    text-transform: uppercase;
    white-space: nowrap;
}
.drocat-neuron-match-first {
    margin: 1px 0 0 4px;
    color: var(--drocat-muted);
    font-size: 10px;
    font-weight: 500;
    line-height: 1.2;
    white-space: nowrap;
}
.drocat-neuron-match-table tbody tr.drocat-neuron-match-secondary-row > td {
    height: 28px;
    padding-top: 3px;
    padding-bottom: 3px;
    color: var(--drocat-faint);
    font-size: 11px;
    font-weight: 500;
    line-height: 1.15;
}
.drocat-neuron-match-table tbody tr.drocat-neuron-match-secondary-row
    .drocat-neuron-match-source {
    color: var(--drocat-faint);
    font-size: 11px;
    font-weight: 500;
}
.drocat-neuron-match-table tbody tr.drocat-neuron-match-secondary-row
    .drocat-neuron-match-value-line,
.drocat-neuron-match-table tbody tr.drocat-neuron-match-secondary-row
    .drocat-neuron-match-first {
    padding-left: 12px;
}
.drocat-neuron-match-table tbody tr.drocat-neuron-match-secondary-row
    .drocat-neuron-match-jump {
    min-height: 21px;
    padding: 0 3px;
    color: var(--drocat-muted);
    font-size: 11px;
    font-weight: 550;
}
.drocat-neuron-match-table tbody tr.drocat-neuron-match-secondary-row
    .drocat-neuron-match-first {
    margin-top: 0;
    color: var(--drocat-faint);
    font-size: 9px;
}
.drocat-neuron-match-table tbody tr.drocat-neuron-match-secondary-row
    .drocat-neuron-match-secondary-label {
    min-height: 15px;
    padding: 0 4px;
    margin-right: 2px;
    border-color: rgba(234, 179, 8, .42);
    background: var(--drocat-badge-soft-bg);
    color: var(--drocat-badge-soft-fg);
    font-size: 9px;
    font-weight: 650;
}
.drocat-neuron-match-expand-btn {
    margin-right: 1px;
    color: var(--drocat-muted);
    opacity: .85;
}
.drocat-neuron-match-expand-btn:hover {
    opacity: 1;
    color: var(--drocat-cobalt);
}
.drocat-neuron-match-table tbody tr.drocat-neuron-match-subtype-panel-row > td {
    height: auto;
    padding: 4px 10px 10px 36px;
    background: var(--drocat-cobalt-soft);
    box-shadow: inset 0 2px 4px rgba(11, 31, 58, .06);
}
.drocat-neuron-match-subtype-cell {
    padding: 0 !important;
}
.drocat-neuron-match-subtype-head {
    margin: 2px 0 6px;
    color: var(--drocat-muted);
    font-size: 10px;
    font-weight: 750;
    letter-spacing: .03em;
    text-transform: uppercase;
}
.drocat-neuron-match-subtype-list {
    display: flex;
    flex-direction: column;
    gap: 1px;
    max-height: 240px;
    overflow-y: auto;
    border: 1px solid var(--drocat-line);
    border-radius: 8px;
    background: var(--drocat-surface);
    padding: 4px 6px;
}
.drocat-neuron-match-subtype-item {
    display: flex;
    align-items: center;
    gap: 4px;
    min-height: 26px;
    padding: 0 2px;
    border-radius: 6px;
}
.drocat-neuron-match-subtype-item:hover {
    background: var(--drocat-cobalt-soft);
}
.drocat-neuron-match-subtype-name {
    flex: 1 1 auto;
    min-width: 0;
    color: var(--drocat-navy);
    font-size: 12px;
    font-weight: 650;
    overflow-wrap: anywhere;
}
.drocat-neuron-match-subtype-count {
    color: var(--drocat-muted);
    font-size: 11px;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}
.drocat-neuron-match-subtype-note {
    margin: 4px 2px 2px;
    color: var(--drocat-muted);
    font-size: 10px;
    line-height: 1.35;
}
.drocat-neuron-match-panel .q-table__control {
    color: var(--drocat-cobalt);
}
.drocat-data-viewer-scroll {
    position: relative;
    max-width: 100%;
    overflow: auto;
    border: 1px solid var(--drocat-line);
    border-radius: 10px;
}
.drocat-neuron-full-panel > .drocat-data-viewer-scroll {
    display: flex;
    flex-direction: column;
    flex: 0 1 auto;
    min-height: 0;
}
.drocat-neuron-full-panel .drocat-data-viewer-table.q-table__container {
    display: flex;
    flex-direction: column;
    flex: 0 1 auto;
    min-height: 0;
}
.drocat-data-viewer-table .q-table__middle {
    min-height: 0;
    flex: 0 1 auto;
    overflow: auto;
}
.drocat-data-viewer-table .q-table__middle thead tr:first-child th {
    position: sticky;
    top: 0;
    z-index: 8;
    background: var(--drocat-cobalt-soft) !important;
    color: var(--drocat-navy) !important;
    box-shadow: 0 1px 0 var(--drocat-line-strong), 0 3px 8px rgba(11, 31, 58, .08);
}
.drocat-data-viewer-table th { white-space: nowrap; }
.drocat-data-viewer-table td,
.drocat-data-viewer-table th {
    height: 40px;
    padding: 7px 9px;
}
.drocat-data-viewer-table tbody tr > td {
    /* A QTable selection can otherwise leave its default gray row fill in
       place while the viewer updates its persistent selection sets. The
       viewer owns the two intentional states below: theme surface for
       ordinary cells and blue for selected rows. */
    background: var(--drocat-cell) !important;
    color: var(--drocat-navy) !important;
}
.drocat-data-viewer-table.q-table__container,
.drocat-data-viewer-table .q-table__middle,
.drocat-data-viewer-table .q-table,
.drocat-data-viewer-table tbody,
.drocat-data-viewer-table tbody tr:not(.drocat-neuron-selected-row) {
    background: var(--drocat-cell) !important;
}
.drocat-data-viewer-table tbody tr.q-tr--selected:not(.drocat-neuron-selected-row) > td,
.drocat-data-viewer-table tbody tr[aria-selected="true"]:not(.drocat-neuron-selected-row) > td {
    background: var(--drocat-cell) !important;
}
.drocat-data-viewer-table tbody tr:hover > td {
    background: var(--drocat-cell) !important;
}
.drocat-data-viewer-table tr.drocat-neuron-selected-row > td {
    /* Direct body-ID selections and matched-group selections share the same
       Quasar row-selection state. Keep that state visibly distinct from the
       yellow cell used for the query match. */
    background: var(--drocat-selected) !important;
    color: var(--drocat-navy) !important;
}
.drocat-data-viewer-table tr.drocat-neuron-selected-row:hover > td {
    background: var(--drocat-selected-hover) !important;
}
.drocat-data-viewer-table tr.drocat-neuron-selected-row > td:first-child {
    box-shadow: inset 3px 0 0 var(--drocat-cobalt);
}
.drocat-data-viewer-table td.drocat-neuron-hit-cell {
    /* The matched source cell follows the scroll until it would leave the
       viewport, then floats at the nearest horizontal edge. This keeps the
       actual highlighted cell visible in both scroll directions without
       changing the full table's column layout. */
    position: sticky;
    left: 0;
    right: 0;
    z-index: 4;
    /* Keep the cell's normal row background; only the nested mark below is
       yellow. The outline still identifies every matched cell. */
    background: var(--drocat-cell) !important;
    color: var(--drocat-navy) !important;
    font-weight: 600;
    box-shadow: inset 0 0 0 2px #eab308;
}
.drocat-data-viewer-table td.drocat-neuron-secondary-hit-cell {
    position: static;
    z-index: 1;
    background: var(--drocat-cell) !important;
    color: var(--drocat-navy) !important;
    font-weight: 600;
    box-shadow: inset 0 0 0 2px #eab308;
}
.drocat-data-viewer-table tr.drocat-neuron-selected-row > td.drocat-neuron-hit-cell,
.drocat-data-viewer-table tr.drocat-neuron-selected-row > td.drocat-neuron-secondary-hit-cell {
    background: var(--drocat-selected) !important;
    color: var(--drocat-navy) !important;
}
.drocat-data-viewer-table td.drocat-neuron-map-cell,
.drocat-data-viewer-table th.drocat-neuron-map-cell {
    /* Outline + text layout; the opaque distinct background and the sticky
       geometry are enforced by the end-of-sheet reinforcement rules and the
       per-column inline styles. */
    border-left: 1px solid var(--drocat-line-strong);
    text-align: left;
    vertical-align: top;
}
.drocat-data-viewer-table td.drocat-neuron-map-cell {
    z-index: 5;
}
.drocat-data-viewer-table th.drocat-neuron-map-cell {
    z-index: 9;
    white-space: normal;
}
.drocat-data-viewer-table td.drocat-neuron-map-cell.drocat-neuron-map-edge {
    border-right: 1px solid var(--drocat-line-strong);
}
.drocat-data-viewer-table tr.drocat-neuron-selected-row > td.drocat-neuron-map-cell {
    box-shadow: inset 3px 0 0 var(--drocat-cobalt);
}
.drocat-data-viewer-table .drocat-neuron-map-value {
    /* Fixed-width inner box: keeps the pinned column geometry stable no
       matter how long the map text is (auto table layout would otherwise
       expand the cell and desync the pinned offsets). Quasar tables keep
       white-space: nowrap unless wrap-cells is enabled, so long map text
       would paint past the pinned cell and get covered by the neighbour
       or clipped by the scroll edge; confine it to the cell with a
       truncation ellipsis. Hover shows the full text as a tooltip and a
       click expands the cell in place (wrapping, row grows). */
    width: 100%;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    cursor: zoom-in;
}
.drocat-data-viewer-table .drocat-neuron-map-value.drocat-map-cell-expanded {
    /* Click-to-expand: wrap the full text within the fixed column width;
       the row grows to fit. A second click collapses back to one line. */
    white-space: normal;
    overflow-wrap: anywhere;
    word-break: break-word;
    cursor: zoom-out;
}
.drocat-map-cell-tooltip {
    /* Multiline hover display for the mapped-view provenance cells: the
       full cell texts followed by the bridge derivation lines. */
    white-space: pre-line;
    max-width: 540px;
    font-size: 12.5px;
    line-height: 1.45;
}
.drocat-data-viewer-table mark.drocat-neuron-match-text {
    padding: 0 .08em;
    border-radius: 3px;
    background: var(--drocat-mark-bg);
    color: var(--drocat-navy);
    font-weight: 750;
}
@keyframes drocat-neuron-focus-breathe {
    0%, 100% { opacity: 0; }
    50% { opacity: 1; }
}
.drocat-data-viewer-table tr.drocat-neuron-focus-flash > td {
    position: relative;
}
.drocat-data-viewer-table tr.drocat-neuron-focus-flash > td::after {
    content: "";
    position: absolute;
    inset: 0;
    z-index: 3;
    pointer-events: none;
    background: rgba(148, 163, 184, .20);
    animation: drocat-neuron-focus-breathe 1.5s ease-in-out 1;
}
.drocat-data-viewer-table tr.drocat-neuron-focus-flash > td > * {
    position: relative;
    z-index: 4;
}
@media (max-width: 1100px) {
    .drocat-neuron-intro-row {
        grid-template-columns: 1fr;
    }
    .drocat-neuron-results-layout {
        grid-template-columns: 1fr;
        height: auto;
    }
    /* Stacked panels each keep their own fixed window so the internal
       scroll chains stay height-bounded. */
    .drocat-neuron-match-panel,
    .drocat-neuron-full-panel {
        height: min(58vh, 680px);
    }
    .drocat-neuron-match-panel {
        position: static;
    }
}
@media (max-width: 760px) {
    .drocat-neuron-viewer-card {
        padding: 8px 10px 10px;
    }
    .drocat-neuron-header-meta {
        flex-basis: 100%;
    }
    .drocat-neuron-dialog-header {
        align-items: flex-start;
    }
}

/* ---------- Grouped navigation: layered cards ---------- */
/* Every tab group is its own tinted card: the group header sits on top of
   its tab segments INSIDE the same card, so header and tabs are always
   aligned and each group reads as an independent block. Settings is promoted
   to the header (beside the theme toggle), so it no longer occupies a nav card. */
.drocat-nav {
    display: flex;
    gap: 8px;
    align-items: stretch;
}
.drocat-group-card {
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 4px;
    border: 1px solid var(--drocat-line);
    border-radius: 14px;
    box-shadow: inset 0 1px 3px rgba(11, 31, 58, .04);
}
.drocat-group-card.drocat-tint-connection { background: var(--drocat-tint-connection); }
.drocat-group-card.drocat-tint-crossdataset { background: var(--drocat-tint-crossdataset); }
.drocat-group-card.drocat-tint-visualization { background: var(--drocat-tint-visualization); }
.drocat-group-card.drocat-tint-similarity { background: var(--drocat-tint-similarity); }
.drocat-group-card.drocat-tint-nb { background: var(--drocat-tint-nb); }
.drocat-group-card.drocat-tint-flylight { background: var(--drocat-tint-flylight); }
.drocat-group-card.drocat-tint-settings { background: var(--drocat-tint-settings); }
/* Settings stands alone on the right, apart from the function groups. */
.drocat-settings-card { margin-left: auto; flex: 0 0 auto; }
.drocat-group-head {
    min-width: 0;
    text-align: center;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: var(--drocat-navy);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    padding: 2px 4px 0;
}
/* The single NB badge lives on the NeuronBridge group header. */
.drocat-group-head.drocat-head-nb::after {
    content: "NB";
    margin-left: 6px;
    padding: 1px 6px;
    border-radius: 999px;
    background: #7c3aed;
    color: #fff;
    /* The header is plain text; the badge keeps the app font explicitly. */
    font-family: Inter, "Avenir Next", "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .02em;
}
.drocat-group-tabs { display: flex; gap: 2px; flex: 1; }
.drocat-group-tab {
    flex: 1 1 0;
    min-width: 0;
    min-height: 54px;
    padding: 6px 4px;
    border-radius: 9px;
    position: relative;
    transition: background .16s ease, color .16s ease, box-shadow .16s ease;
}
.drocat-group-tab.q-btn--flat { color: var(--drocat-navy); }
/* Two-row segments: icon on top, name below. */
.drocat-group-tab .q-btn__content { flex-direction: column; gap: 2px; }
.drocat-group-tab .q-icon { font-size: 18px; }
.drocat-group-tab .q-btn__label,
.drocat-group-tab .q-btn__content .block {
    min-width: 0;
    font-size: 11px;
    font-weight: 650;
    letter-spacing: -.015em;
    /* Multi-word names render one word per line with tight leading. */
    white-space: pre-line;
    line-height: 1.15;
}
.drocat-group-tab:hover { background: var(--drocat-tab-hover); }
.drocat-group-tab.drocat-active {
    background: var(--drocat-surface) !important;
    box-shadow: 0 2px 6px rgba(11, 31, 58, .10);
}
.drocat-group-tab.drocat-tab-running::after,
.drocat-group-tab.drocat-tab-failed::after,
.drocat-group-tab.drocat-tab-completed::after {
    content: "";
    position: absolute;
    top: 6px;
    right: 6px;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    box-shadow: 0 0 0 2px var(--drocat-surface);
}
.drocat-group-tab.drocat-tab-running::after { background: var(--drocat-cobalt); }
.drocat-group-tab.drocat-tab-failed::after { background: var(--drocat-err); }
.drocat-group-tab.drocat-tab-completed::after { background: var(--drocat-ok); }
/* The active segment takes its group's accent color. The `.drocat-nav`
   ancestor raises specificity above Quasar's `.text-primary`/`.q-btn--flat`
   button-color rules so the per-group accent actually paints. */
.drocat-nav .drocat-tint-connection .drocat-group-tab.drocat-active { color: var(--drocat-cobalt) !important; }
.drocat-nav .drocat-tint-crossdataset .drocat-group-tab.drocat-active { color: #be123c !important; }
.drocat-nav .drocat-tint-visualization .drocat-group-tab.drocat-active { color: #0e7490 !important; }
.drocat-nav .drocat-tint-similarity .drocat-group-tab.drocat-active { color: #b45309 !important; }
.drocat-nav .drocat-tint-nb .drocat-group-tab.drocat-active { color: #7c3aed !important; }
.drocat-nav .drocat-tint-flylight .drocat-group-tab.drocat-active { color: #15803d !important; }
.drocat-nav .drocat-tint-settings .drocat-group-tab.drocat-active { color: #475467 !important; }

@media (max-width: 1100px) {
    /* Cards no longer fit side by side: let the nav strip scroll instead
       of squeezing the segments until their labels collide. */
    .drocat-nav { overflow-x: auto; scrollbar-width: none; }
    .drocat-group-card { min-width: 160px; }
    .drocat-group-head { font-size: 10px; letter-spacing: .05em; }
    .drocat-group-head.drocat-head-nb::after { font-size: 8px; padding: 0 5px; margin-left: 4px; }
    .drocat-group-tab { min-height: 46px; padding: 4px 3px; }
    .drocat-group-tab .q-icon { font-size: 15px; }
    .drocat-group-tab .q-btn__label { font-size: 9px; letter-spacing: -.03em; }
}

/* Panel tag badge (NeuronBridge etc.) */
.drocat-tag-badge {
    font-size: 10px !important;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
}

/* ---------- Page workspace ---------- */
.drocat-shell { width: 100%; max-width: 1500px; margin: 0 auto; padding: 20px 28px 44px; }
.drocat-page { width: 100%; padding: 20px 0 0; }
.drocat-page-head { padding-bottom: 18px; border-bottom: 1px solid var(--drocat-line); margin-bottom: 20px; }
.drocat-page-mark {
    width: 46px; height: 46px; border-radius: 14px; display: grid; place-items: center;
    background: linear-gradient(135deg, var(--drocat-cobalt), #0d3bb8);
    box-shadow: 0 8px 20px rgba(20, 92, 255, .22);
}
.drocat-page-mark .q-icon { font-size: 24px; }
.drocat-page-title { font-size: 22px; font-weight: 700; letter-spacing: -.02em; color: var(--drocat-navy); }
.drocat-page-sub { font-size: 13px; color: var(--drocat-muted); }
.drocat-page-progress {
    padding: 10px 0 12px !important;
    border-color: var(--drocat-line-strong) !important;
    background: linear-gradient(135deg, var(--drocat-cobalt-soft), var(--drocat-surface)) !important;
}
.drocat-page-progress-compact {
    margin: 0 0 12px;
    border-bottom: 1px solid var(--drocat-line);
    background: transparent !important;
}
.drocat-page-progress-head { min-height: 30px; }
.drocat-page-progress-mark {
    width: 30px; height: 30px; border-radius: 9px; display: grid; place-items: center;
    background: var(--drocat-cobalt);
}
.drocat-page-progress-mark .q-icon { font-size: 17px; }
.drocat-page-progress-summary { min-height: 20px; }
.drocat-progress-percent {
    color: var(--drocat-navy); font-weight: 750; font-variant-numeric: tabular-nums;
}
.drocat-progress-bar {
    height: 12px !important;
    min-height: 12px !important;
    border-radius: 999px;
}

.drocat-workspace {
    display: grid !important;
    grid-template-columns: minmax(0, 1.45fr) minmax(430px, 1fr);
    gap: 24px;
    align-items: start;
}
.drocat-form { min-width: 0; }
.drocat-results { min-width: 0; }
.drocat-results-card { position: sticky; top: 96px; }

@media (max-width: 1280px) {
    .drocat-workspace { grid-template-columns: 1fr; }
    .drocat-results-card { position: static; }
}

@media (max-width: 700px) {
    .drocat-nav { overflow-x: auto; scrollbar-width: none; }
    .drocat-group-card { min-width: 170px; }
    .drocat-group-head { font-size: 9px; letter-spacing: .03em; }
    .drocat-header { min-height: 64px !important; padding: 0 16px !important; }
    .drocat-brand-mark { width: 36px; height: 36px; border-radius: 11px; }
    .drocat-brand-sub, .drocat-header-link { display: none; }
    .drocat-shell { padding: 14px 14px 32px; }
    .drocat-page { padding-top: 16px; }
    .drocat-page-head { align-items: flex-start; padding-bottom: 14px; margin-bottom: 16px; }
    .drocat-page-mark { width: 40px; height: 40px; border-radius: 12px; }
    .drocat-page-title { font-size: 19px; }
    .q-card.drocat-card { padding: 15px 14px; border-radius: 14px !important; }
    .drocat-param-grid { grid-template-columns: minmax(0, 1fr) !important; }
}

/* ---------- Cards ---------- */
.q-card.drocat-card {
    background: var(--drocat-surface) !important;
    border: 1px solid var(--drocat-line) !important;
    border-radius: var(--drocat-radius-md) !important;
    box-shadow: 0 2px 10px rgba(11, 31, 58, .05) !important;
    padding: 18px 20px;
}
.drocat-card-title { font-size: 15px; font-weight: 680; letter-spacing: -.01em; color: var(--drocat-navy); }
.drocat-section-head { margin: 2px 0 10px; }
.drocat-section-icon {
    width: 30px; height: 30px; border-radius: 9px; display: grid; place-items: center;
    background: var(--drocat-cobalt-soft);
}
.drocat-section-icon .q-icon { font-size: 17px; }
.drocat-section-title { font-size: 14px; font-weight: 650; color: var(--drocat-navy); }

/* ---------- Net-Viz edge-list editor ---------- */
.drocat-edge-editor {
    overflow: hidden;
    border: 1px solid var(--drocat-line) !important;
    border-radius: var(--drocat-radius-md) !important;
    background: var(--drocat-surface) !important;
    box-shadow: 0 2px 10px rgba(11, 31, 58, .05) !important;
}
.drocat-edge-editor .q-expansion-item__container,
.drocat-edge-editor .q-expansion-item__content {
    background: var(--drocat-surface) !important;
}
.drocat-edge-editor .q-item {
    min-height: 52px;
    padding: 8px 18px;
    color: var(--drocat-navy);
    font-weight: 650;
}
.drocat-edge-editor .q-expansion-item__content {
    padding: 4px 18px 18px;
}
.drocat-edge-table {
    overflow: hidden;
    border-color: var(--drocat-line-strong) !important;
    border-radius: 10px;
}
/* Auto (default) table layout keeps the table inside its card width: the columns
   flex to the available space (the checkbox/layer stay capped narrow, while
   Neuron/colors take the rest), instead of ``fixed`` layout which distributes
   leftover proportionally and can push the colour columns off-screen on narrower
   cards. The header resizer still overrides a column's width via ``--wc-*``. */
.drocat-edge-table .q-table__middle {
    overflow-x: auto;
}
.drocat-edge-table .q-table thead tr {
    background: var(--drocat-cobalt-soft);
}
.drocat-edge-table .q-table th {
    height: 44px;
    color: var(--drocat-navy);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: .02em;
    text-transform: none;
}
.drocat-edge-table .q-table td {
    height: 48px;
    color: var(--drocat-navy);
    font-size: 13px;
}
.drocat-edge-header-cell,
.drocat-edge-cell {
    border-right: 1px solid var(--drocat-line) !important;
}
.drocat-edge-header-cell.drocat-edge-divider,
.drocat-edge-cell.drocat-edge-divider {
    border-right-color: var(--drocat-line-strong) !important;
}
/* Column widths are driven by ``--wc-*`` CSS variables so the header resizer can
   drag them interactively (see ``layer_style_editor._COL_RESIZE_JS``). The
   checkbox and layer columns stay narrow, leaving more room for Neuron/colors. */
.drocat-edge-select-cell {
    width: var(--wc-select, 42px);
    min-width: 38px;
    border-right: 1px solid var(--drocat-line-strong) !important;
    text-align: center;
    position: relative;
}
.drocat-edge-cell {
    padding: 4px 10px !important;
}
.drocat-edge-table .drocat-layer-column {
    width: var(--wc-layer, 68px);
    min-width: 64px;
}
.drocat-edge-table .drocat-neuron-column {
    width: var(--wc-neuron, 400px);
    min-width: 140px;
}
.drocat-edge-table .drocat-color-column {
    width: var(--wc-color, 280px);
    min-width: 110px;
}
.drocat-edge-table .drocat-synapse-color-column {
    width: var(--wc-synapse_color, 280px);
    min-width: 110px;
}
.drocat-edge-table .drocat-pre-synaptic-color-column {
    width: var(--wc-pre_synaptic_color, 280px);
    min-width: 110px;
}
.drocat-edge-table .drocat-post-synaptic-color-column {
    width: var(--wc-post_synaptic_color, 280px);
    min-width: 110px;
}
/* Net-Viz edge-list editor columns (edge_list_editor._columns_for_mode). */
.drocat-edge-table .drocat-source-column {
    width: var(--wc-source, 180px);
    min-width: 90px;
}
.drocat-edge-table .drocat-target-column {
    width: var(--wc-target, 180px);
    min-width: 90px;
}
.drocat-edge-table .drocat-weight-column {
    width: var(--wc-weight, 90px);
    min-width: 64px;
}
.drocat-edge-table .drocat-source-group-column {
    width: var(--wc-source_group, 140px);
    min-width: 90px;
}
.drocat-edge-table .drocat-target-group-column {
    width: var(--wc-target_group, 140px);
    min-width: 90px;
}
.drocat-edge-table .drocat-edge-info-column {
    width: var(--wc-edge_info, 230px);
    min-width: 120px;
}
.drocat-edge-table .drocat-source-info-column {
    width: var(--wc-source_info, 210px);
    min-width: 120px;
}
.drocat-edge-table .drocat-target-info-column {
    width: var(--wc-target_info, 210px);
    min-width: 120px;
}
/* The selected row must keep its cell text and chips readable (dark themes can
   make navy-on-selected low-contrast). Keep the row light and the text navy. */
.drocat-edge-table tr.selected,
.drocat-edge-table tr.selected td {
    background: var(--drocat-selected, #e9f0f9) !important;
    color: var(--drocat-navy) !important;
}
.drocat-edge-table tr.selected .q-field__native,
.drocat-edge-table tr.selected .q-field__input {
    color: var(--drocat-navy) !important;
}
.drocat-edge-table tr.selected .q-chip {
    opacity: 1 !important;
}
/* The narrow Layer column must still show the layer number, not just its arrow. */
.drocat-layer-column .q-field--borderless .q-field__native,
.drocat-layer-column .q-field--borderless .q-field__input {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
}
.drocat-edge-header-cell {
    position: relative;
}
.drocat-col-resizer {
    position: absolute;
    right: 0;
    top: 0;
    bottom: 0;
    width: 14px;
    cursor: col-resize;
    z-index: 30;
    touch-action: none;
    pointer-events: auto;
}
.drocat-col-resizer::after {
    content: "";
    position: absolute;
    right: 1px;
    top: 12px;
    bottom: 12px;
    width: 2px;
    background: var(--drocat-line);
    border-radius: 1px;
}
.drocat-col-resizer:hover,
.drocat-col-resizer:active {
    background: var(--drocat-cobalt);
    opacity: .5;
}
.drocat-edge-table th.drocat-color-column,
.drocat-edge-table th.drocat-synapse-color-column,
.drocat-edge-table th.drocat-pre-synaptic-color-column,
.drocat-edge-table th.drocat-post-synaptic-color-column {
    white-space: normal;
    line-height: 1.15;
}
.drocat-edge-table .q-field--borderless .q-field__control {
    min-height: 34px;
    padding: 0 4px;
}
.drocat-edge-table .q-field--borderless .q-field__native,
.drocat-edge-table .q-field--borderless .q-field__input {
    color: var(--drocat-navy) !important;
    font-size: 13px;
}
.drocat-color-cell {
    min-width: 0;
    gap: 0 !important;
    align-items: center !important;
}
.drocat-color-cell-picker {
    flex: 0 0 18px;
    width: 18px !important;
    min-width: 18px !important;
    height: 18px !important;
    padding: 0 !important;
    color: var(--drocat-muted) !important;
    align-self: center !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    line-height: 1 !important;
}
.drocat-color-cell-picker .q-icon {
    font-size: 13px !important;
}
.drocat-color-cell-picker-set {
    /* The picker background IS the colour swatch when a value is set; hiding the
       nested content avoids the round-button 25px content box that pushed the
       old preview span 6.5px below centre. */
    border-radius: 4px !important;
    box-sizing: border-box;
    background-position: center !important;
    background-size: cover !important;
    transition: none;
}
.drocat-color-cell-picker-set .q-btn__content,
.drocat-color-cell-picker-set .q-icon {
    display: none !important;
}
.drocat-color-cell .q-field {
    flex: 1 1 auto;
    min-width: 0;
    align-self: center;
}
.drocat-color-cell .q-field__control {
    align-items: center;
}
.drocat-color-cell .q-field__native,
.drocat-color-cell .q-field__input {
    min-width: 0 !important;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
/* The swatch button must stay vertically centred in the row (the small
   round/square button is shorter than the cell and carries no nested preview). */
.drocat-color-cell-picker .q-btn__content {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    line-height: 1 !important;
}

/* In-table neuron chip cell: let the field grow with the chips (no clipping).
   Quasar's multi QSelect keeps its chip row ``nowrap`` by default; force wrapping
   so many chips fold onto extra lines and the row grows instead of overflowing. */
.drocat-neuron-cell {
    min-width: 0;
    width: 100%;
}
.drocat-neuron-cell .q-field {
    width: 100%;
    min-width: 0;
}
.drocat-neuron-cell .q-field__control {
    flex-wrap: wrap;
    min-height: 34px;
    padding: 4px 6px;
}
.drocat-neuron-cell .q-field__control-container {
    min-height: 34px;
}
.drocat-neuron-cell .q-field__native,
.drocat-neuron-cell .q-field__input {
    flex-wrap: wrap;
    white-space: normal;
    min-height: 24px;
    max-height: none;
}
.drocat-neuron-cell .q-chip {
    max-width: 100%;
    margin: 1px 2px;
    white-space: normal;
}

/* In-table neuron auto-suggestion overlay: a plain positioned div below the cell.
   It is not a Quasar menu, so opening it never blurs the cell's q-select. */
.drocat-suggest-overlay {
    /* Fixed (viewport) positioning anchors the overlay to the focused cell
       regardless of any positioned/scrollable ancestor it is nested in. */
    position: fixed;
    z-index: 7000;
    display: none;
    max-height: 240px;
    overflow-y: auto;
    overflow-x: hidden;
    /* Click-through everywhere except the items: the transparent spacer over
       the next row lets clicking that cell focus it (no stray commit); only the
       suggestion items (pointer-events:auto) capture a click to commit. */
    pointer-events: none;
}
.drocat-suggest-item {
    display: flex;
    align-items: center;
    padding: 5px 10px;
    cursor: pointer;
    gap: 10px;
    white-space: nowrap;
    pointer-events: auto;
    background: var(--drocat-surface, #ffffff);
    border: 1px solid var(--drocat-line, #c8d4e4);
    border-top: none;
}
.drocat-suggest-item:first-of-type {
    border-radius: 0 0 6px 6px;
}
.drocat-suggest-item:hover {
    background: var(--drocat-cobalt-soft, #e9f0f9);
}
/* Highlight for arrow-key navigation (matches the standard query box). */
.drocat-suggest-item.drocat-suggest-active {
    background: var(--drocat-cobalt-soft, #e9f0f9);
}
/* A row whose value is already a chip in the cell stays listed and is tinted and
   ticked rather than dropped, so the list never shifts under the pointer while
   several entries are picked in a row (the query box's rule). The label carries
   its own color and the gray hint keeps its muted one, so the tint goes on the
   label and the tick. */
.drocat-suggest-check {
    font-size: 14px;
    line-height: 1;
    flex-shrink: 0;
}
.drocat-suggest-item.drocat-suggest-added .drocat-suggest-label,
.drocat-suggest-item.drocat-suggest-added .drocat-suggest-check {
    color: var(--drocat-cobalt, #145cff);
}
.drocat-suggest-header {
    padding: 4px 10px;
    font-size: 11px;
    color: var(--drocat-muted, #6b7a8f);
    background: var(--drocat-surface, #ffffff);
    border: 1px solid var(--drocat-line, #c8d4e4);
    border-top: none;
}
.drocat-select-group-header {
    min-height: 0;
    padding: 8px 12px 2px;
    font-size: 11px;
    color: var(--drocat-muted, #6b7a8f);
    background: inherit;
    cursor: default;
    user-select: none;
}
.drocat-select-group-header .q-item__label--overline {
    letter-spacing: 0.08em;
}
.drocat-suggest-label {
    font-size: 13px;
    color: var(--drocat-navy, #0b1f3a);
    flex-shrink: 0;
    flex-grow: 1;
    min-width: 0;
}
.drocat-suggest-hint {
    font-size: 11px;
    color: var(--drocat-muted, #6b7a8f);
    margin-left: auto;
    text-align: right;
    flex-shrink: 0;
}
/* History-row removal 'x': prunes the entry from the query history without
   committing the row. One class on both surfaces -- the overlay's span and the
   query box's ui.button -- so the reveal rule below covers them together; the
   metrics stay scoped to the overlay row so the Quasar button keeps its own. */
.drocat-suggest-item .drocat-suggest-remove {
    flex-shrink: 0;
    margin-left: 6px;
    width: 16px;
    height: 16px;
    line-height: 16px;
    font-size: 14px;
    text-align: center;
    color: var(--drocat-muted, #6b7a8f);
    border-radius: 3px;
    cursor: pointer;
}
.drocat-suggest-item .drocat-suggest-remove:hover {
    color: var(--drocat-navy, #0b1f3a);
    background: var(--drocat-cobalt-soft, #e9f0f9);
}
/* A marked row's tick removes the chip and the prune removes the history entry;
   with both always visible they read as two identical removals a few pixels
   apart. So the prune only shows while its own row is the one being pointed at:
   hovered on either surface, keyboard-highlighted in the overlay (its rows are
   not focusable), or Tab-focused in the query box (its button is). Held at
   opacity 0 rather than left out of the DOM, because the row must not change
   width partway through a run of picks. */
.drocat-suggest-remove {
    opacity: 0;
    transition: opacity .12s ease;
}
.drocat-suggest-item:hover .drocat-suggest-remove,
.drocat-suggest-item.drocat-suggest-active .drocat-suggest-remove,
.drocat-suggest-menu .q-item:hover .drocat-suggest-remove,
.drocat-suggest-menu .q-item:focus-within .drocat-suggest-remove {
    opacity: 1;
}

/* Single-color picker palette: exactly ten swatches per row, centred. */
.drocat-swatch-grid {
    display: grid !important;
    grid-template-columns: repeat(10, minmax(0, 1fr)) !important;
    gap: 3px;
    justify-items: center;
    align-items: center;
}
.drocat-swatch-grid .drocat-swatch {
    padding: 2px;
}
.drocat-swatch-grid .drocat-swatch > div {
    width: 20px !important;
    height: 20px !important;
}

.drocat-layer-color-control {
    flex: 0 0 auto;
}
.drocat-layer-add-input-row {
    display: grid !important;
    grid-template-columns: minmax(110px, .8fr) minmax(220px, 1.45fr)
        repeat(2, minmax(140px, 1fr));
    align-items: stretch;
    overflow: visible;
    padding-bottom: 2px;
}
.drocat-layer-add-input-row.drocat-layer-add-input-row-pre-post {
    grid-template-columns: minmax(110px, .8fr) minmax(220px, 1.35fr)
        repeat(3, minmax(125px, 1fr));
}
.drocat-layer-add-field {
    flex: none;
    width: auto;
    min-width: 0;
    min-height: 52px;
    height: 52px;
}
.drocat-layer-add-layer {
    width: auto;
}
.drocat-layer-add-neuron-input {
    /* A chip must stay inside the grid track.  Without an explicit
       min/max-width the QSelect's intrinsic chip width can escape its grid
       item and cover the adjacent Color field. */
    width: 100%;
    min-width: 0;
    max-width: 100%;
    flex: none;
    overflow: hidden;
}
.drocat-layer-add-neuron-input .drocat-neuron-input-row,
.drocat-layer-add-neuron-input .drocat-chip-input-shell,
.drocat-layer-add-neuron-input .drocat-layer-add-field-control,
.drocat-layer-add-field.q-field,
.drocat-layer-add-color-control {
    height: 52px;
}
.drocat-layer-add-field.q-field .q-field__control,
.drocat-layer-add-color-input.q-field .q-field__control,
.drocat-layer-add-neuron-input .q-field__control {
    min-height: 52px !important;
    height: 52px;
}
.drocat-layer-add-neuron-input .drocat-neuron-input-row,
.drocat-layer-add-neuron-input .drocat-chip-input-shell,
.drocat-layer-add-neuron-input .drocat-chip-input,
.drocat-layer-add-neuron-input .q-field,
.drocat-layer-add-neuron-input .q-field__control,
.drocat-layer-add-neuron-input .q-field__control-container {
    width: 100%;
    min-width: 0;
    max-width: 100%;
    box-sizing: border-box;
}
.drocat-layer-add-neuron-input .q-field__control {
    position: relative;
}
.drocat-layer-add-neuron-input .drocat-chip-input-shell,
.drocat-layer-add-neuron-input .q-field__control,
.drocat-layer-add-neuron-input .q-field__control-container {
    overflow: hidden;
}
.drocat-layer-add-field.q-field .q-field__native,
.drocat-layer-add-field.q-field .q-field__input,
.drocat-layer-add-neuron-input .q-field__native,
.drocat-layer-add-neuron-input .q-field__input {
    min-height: 28px;
    max-height: 28px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.drocat-layer-add-neuron-input .q-chip {
    min-width: 0;
    max-width: calc(100% - 58px);
    overflow: hidden;
}
.drocat-layer-add-neuron-input .q-chip__content {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.drocat-layer-add-neuron-input .drocat-chip-expand-btn {
    flex: 0 0 auto;
    max-width: 58px;
    min-height: 28px;
    line-height: 20px;
    margin-top: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.drocat-layer-add-color-control {
    width: auto;
    min-width: 0;
    flex: none;
    align-items: stretch !important;
}
.drocat-layer-add-color-input {
    width: 100%;
    min-width: 0;
}
.drocat-layer-add-color-control > .q-field {
    width: 100%;
    min-width: 0;
}
.drocat-layer-add-color-picker {
    flex: 0 0 20px;
    width: 20px !important;
    min-width: 20px !important;
    height: 20px !important;
    margin: 0 !important;
    padding: 0 !important;
}
.drocat-layer-add-color-picker .q-icon {
    font-size: 14px !important;
}
.drocat-layer-add-color-picker-set {
    border-radius: 4px !important;
}
.drocat-layer-add-form .q-field__label,
.drocat-layer-add-form .q-field--float .q-field__label {
    font-size: 14px !important;
    line-height: 20px !important;
    top: 7px !important;
    transform: none !important;
}
.drocat-layer-add-form .q-field__native,
.drocat-layer-add-form .q-field__input,
.drocat-layer-add-form .q-chip__content {
    font-size: 14px !important;
    line-height: 20px !important;
}
.drocat-layer-add-neuron-input .drocat-chip-expand-btn {
    position: static !important;
    max-width: 58px;
    min-height: 22px;
    margin: 0 !important;
    padding: 0 2px !important;
    font-size: 12px !important;
    line-height: 16px !important;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.drocat-layer-add-form .q-field__append {
    align-self: flex-start;
    min-height: 0;
    padding-top: 7px;
    padding-bottom: 0;
}
.drocat-layer-add-form .q-field__control-container {
    min-width: 0;
}
.drocat-layer-add-neuron-input .q-field__append {
    position: absolute !important;
    top: 4px;
    right: 4px;
    z-index: 2;
    display: flex;
    align-items: flex-start;
    min-height: 0 !important;
    height: 22px;
    padding: 0 !important;
}
.drocat-layer-add-color-input .q-field__append {
    padding-left: 2px;
    padding-right: 2px;
}
.drocat-layer-add-actions {
    min-height: 40px;
}
.drocat-layer-add-form .drocat-layer-add-control,
.drocat-layer-add-form .drocat-layer-add-control > .q-field,
.drocat-layer-add-form .drocat-layer-add-control .q-field__control,
.drocat-layer-add-form .drocat-layer-add-control .drocat-chip-input-shell,
.drocat-layer-add-form .drocat-layer-add-control .drocat-neuron-input-row {
    min-height: 52px !important;
    height: 52px !important;
}
.drocat-layer-add-form .drocat-layer-add-control .q-field__control {
    box-sizing: border-box;
}

@media (max-width: 1100px) {
    .drocat-layer-add-input-row,
    .drocat-layer-add-input-row.drocat-layer-add-input-row-pre-post {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
@media (max-width: 700px) {
    .drocat-layer-add-input-row,
    .drocat-layer-add-input-row.drocat-layer-add-input-row-pre-post {
        grid-template-columns: minmax(0, 1fr);
    }
}
.drocat-layer-validation {
    border: 1px solid rgba(185, 28, 28, .28) !important;
    background: rgba(254, 226, 226, .72) !important;
    box-shadow: none !important;
}
.drocat-layer-validation-label {
    white-space: pre-line;
}
.drocat-edge-row-even > td {
    background: var(--drocat-row-even) !important;
}
.drocat-edge-row-odd > td {
    background: var(--drocat-surface) !important;
}
.drocat-edge-row-even:hover > td,
.drocat-edge-row-odd:hover > td {
    background: var(--drocat-row-hover) !important;
}
.drocat-empty-canvas-btn {
    min-height: 40px;
    white-space: nowrap;
}

/* ---------- Form controls ---------- */
.q-field--outlined .q-field__control {
    background: var(--drocat-surface) !important;
    border-radius: var(--drocat-radius-sm) !important;
}
.q-field--outlined .q-field__control::before { border-color: var(--drocat-line-strong) !important; }
.q-field--outlined .q-field__control:hover::before { border-color: var(--drocat-cobalt) !important; }
.q-field--focused .q-field__control::before { border-color: var(--drocat-cobalt) !important; }
.q-field__label { color: var(--drocat-muted) !important; }
.q-field--focused .q-field__label { color: var(--drocat-cobalt) !important; }
.q-field__native, .q-field__input { color: var(--drocat-navy) !important; }
.q-field--dark .q-field__native { color: #fff !important; }
.q-tooltip {
    font-size: 14px !important;
    line-height: 1.35 !important;
}
.drocat-param-grid { gap: 14px 16px !important; }

/* Compact selects that sit beside a related checkbox.  The wrapper keeps
   the field from claiming the entire flex row while still allowing the row
   to wrap cleanly on narrow screens. */
.drocat-inline-select {
    flex: 0 1 170px;
    width: 170px;
    min-width: 150px;
    max-width: 190px;
}
.drocat-inline-select .q-field { width: 100%; }

/* ---------- LabelMapper / custom-group editors ---------- */
.drocat-settings-mapping-controls {
    align-items: center;
}
.drocat-settings-mapping-controls > .q-field,
.drocat-settings-mapping-controls > .q-btn {
    flex: 1 1 0;
    min-width: 280px;
    width: auto !important;
}
.drocat-settings-mapping-button {
    min-height: 48px !important;
    font-size: 16px !important;
}
.drocat-labelmapper-board {
    min-width: 0;
}
.drocat-labelmapper-group {
    min-width: 0;
    position: relative;
    display: grid !important;
    grid-template-columns: minmax(220px, 360px) minmax(0, 1fr);
    grid-template-rows: auto auto;
    column-gap: 24px;
    row-gap: 12px;
    align-items: start;
    padding: 12px 14px 14px;
    border: 1px solid var(--drocat-line-strong);
    border-radius: 14px;
    background: var(--drocat-labelmapper-grad);
    box-shadow: 0 2px 8px rgba(11, 31, 58, .04);
}
.drocat-labelmapper-group-name {
    grid-column: 1;
    grid-row: 1;
    width: 100%;
    min-width: 0;
    z-index: 1;
}
.drocat-labelmapper-group-members {
    grid-column: 2;
    grid-row: 1;
    min-width: 0;
    width: 100%;
    padding-top: 28px;
}
.drocat-labelmapper-row-actions {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
    align-items: center;
    width: 100%;
    max-width: 100%;
    gap: 16px !important;
    min-height: 64px;
    padding: 0;
    margin-bottom: 0;
}
.drocat-labelmapper-query-actions {
    grid-column: 2;
    justify-self: center;
    max-width: 100%;
}
.drocat-labelmapper-group-controls {
    grid-column: 3;
    justify-self: end;
    position: static;
}
.drocat-labelmapper-query-action {
    min-height: 40px;
    padding: 0 16px;
}
.drocat-labelmapper-datasets {
    grid-column: 1 / -1;
    grid-row: 2;
    min-width: 0;
}
.drocat-labelmapper-dataset-row {
    display: grid !important;
    grid-template-columns: minmax(220px, 360px) minmax(0, 1fr);
    column-gap: 24px;
    align-items: start;
    min-width: 0;
    padding: 8px 0 10px;
    border-bottom: 1px solid var(--drocat-line-strong);
}
.drocat-labelmapper-dataset-row:last-child {
    padding-bottom: 0;
    border-bottom: 0;
}
.drocat-labelmapper-dataset-label {
    min-width: 0;
    padding-top: 20px;
    padding-left: 32px;
    color: var(--drocat-navy);
    font-size: 16px;
    line-height: 1.25;
    font-weight: 700;
    overflow-wrap: anywhere;
}
.drocat-labelmapper-dataset-input {
    min-width: 0;
    width: 100%;
}
.drocat-labelmapper-dataset-input .drocat-chip-input-shell,
.drocat-labelmapper-dataset-input .drocat-chip-input {
    min-width: 0;
}
.drocat-labelmapper-dataset-input .drocat-neuron-viewer-card {
    /* Keep the viewer link compact in each mapping row. */
    margin-top: 1px;
}
@media (max-width: 760px) {
    .drocat-settings-mapping-controls > .q-field,
    .drocat-settings-mapping-controls > .q-btn {
        min-width: 100%;
    }
    .drocat-labelmapper-group {
        display: block !important;
        padding: 10px;
    }
    .drocat-labelmapper-group-name {
        width: 100%;
        min-width: 0;
    }
    .drocat-labelmapper-group-members {
        width: 100%;
        margin-top: 14px;
        padding-top: 0;
    }
    .drocat-labelmapper-row-actions {
        grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
        width: 100%;
        max-width: none;
    }
    .drocat-labelmapper-group-controls {
        position: static;
    }
    .drocat-labelmapper-dataset-row {
        grid-template-columns: minmax(140px, 38%) minmax(0, 1fr);
        column-gap: 12px;
    }
    .drocat-labelmapper-dataset-label {
        width: 100%;
        padding-top: 10px;
        padding-left: 16px;
    }
}

/* ---------- Buttons ---------- */
.q-btn { border-radius: var(--drocat-radius-sm); font-weight: 650; }
.q-btn--flat { color: var(--drocat-muted); }
.q-btn--flat:hover { background: var(--drocat-soft) !important; }
.q-btn--unelevated.bg-primary { background: var(--drocat-cobalt) !important; box-shadow: 0 6px 14px rgba(20, 92, 255, .20); }
.q-btn--unelevated.bg-negative { background: var(--drocat-surface) !important; color: var(--drocat-err) !important; border: 1px solid var(--drocat-err-border); }
/* Segmented toggles: white pill container whose inner segments nest
   concentrically (inner radius = outer radius - padding - border), so the
   selected segment's corners snap to the container's corners with a uniform
   ring and no corner gap. */
.q-btn-group {
    background: var(--drocat-surface);
    border: 1px solid var(--drocat-line);
    border-radius: 999px;
    padding: 4px;
    box-shadow: 0 2px 8px rgba(11, 31, 58, .06);
}
.q-btn-group .q-btn { border-radius: 999px; }
.drocat-run-btn { min-width: 150px; height: 44px !important; }
.drocat-cancel-btn { height: 44px !important; }
.drocat-upload-trigger {
    width: 34px !important;
    height: 34px !important;
    margin-bottom: 4px;
    border: 1px solid var(--drocat-line-strong) !important;
    background: var(--drocat-surface) !important;
    color: var(--drocat-muted) !important;
}
.drocat-upload-trigger:hover { border-color: var(--drocat-cobalt) !important; color: var(--drocat-cobalt) !important; }
.drocat-upload-trigger .q-icon { font-size: 17px; }
.q-menu .q-uploader { border: 1px solid var(--drocat-line) !important; border-radius: 12px !important; }
.drocat-clear-btn { color: var(--drocat-faint); font-size: 12px; }
.drocat-clear-btn:hover { color: var(--drocat-err); }

/* Output-directory actions live inside the input append slot.  Give each
   icon a fixed box so the path text never shifts or sits underneath a
   button when the field is resized. */
.drocat-output-dir .q-field__append {
    flex-wrap: nowrap;
    gap: 3px;
    padding-left: 8px;
}
.drocat-output-dir .drocat-dir-icon-btn,
.drocat-output-dir .drocat-dir-reset-btn {
    flex: 0 0 32px;
    width: 32px !important;
    min-width: 32px !important;
    height: 32px !important;
    min-height: 32px !important;
    padding: 0 !important;
    margin: 0;
    align-self: center;
}
.drocat-output-dir .drocat-dir-icon-btn .q-icon,
.drocat-output-dir .drocat-dir-reset-btn .q-icon { font-size: 20px; }

/* ---------- Palette picker (color previews) ---------- */
.drocat-palette-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 8px;
    overflow-y: auto;
    padding: 2px;
}
.drocat-palette-card {
    border: 1px solid var(--drocat-line);
    border-radius: 10px;
    padding: 6px;
    cursor: pointer;
    background: var(--drocat-surface);
    transition: border-color .15s ease, box-shadow .15s ease;
}
.drocat-palette-card:hover { border-color: var(--drocat-line-strong); }
.drocat-palette-card.selected {
    border-color: var(--drocat-cobalt);
    box-shadow: 0 0 0 2px var(--drocat-cobalt-soft);
}
.drocat-palette-swatches {
    overflow: hidden;
    border-radius: 6px;
    border: 1px solid var(--drocat-line);
}
.drocat-palette-swatches > div {
    position: relative;
}
.drocat-palette-remove {
    position: absolute !important;
    top: 1px;
    right: 1px;
    z-index: 2;
    width: 18px !important;
    min-width: 18px !important;
    height: 18px !important;
    padding: 0 !important;
    border-radius: 50%;
    background: rgba(255, 255, 255, .76) !important;
    color: #dc2626 !important;
}
.drocat-palette-remove .q-icon {
    font-size: 15px !important;
    font-weight: 800;
}
.drocat-custom-color-input-row {
    min-height: 42px;
}
.drocat-palette-name {
    font-size: 10.5px;
    font-weight: 650;
    color: var(--drocat-navy);
    margin-top: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.drocat-palette-preview {
    min-height: 26px;
    padding: 4px 8px;
    background: var(--drocat-soft);
    border-radius: 10px;
}
.drocat-palette-expansion {
    border: 1px solid var(--drocat-line);
    border-radius: 12px;
    overflow: hidden;
}
.drocat-swatch {
    padding: 3px;
    border-radius: 50%;
    border: 2px solid transparent;
    cursor: pointer;
    transition: border-color .15s ease, transform .15s ease;
}
.drocat-swatch:hover { transform: scale(1.08); }
.drocat-swatch.selected { border-color: var(--drocat-cobalt); }
.drocat-palette-strip {
    border-radius: 6px;
    border: 1px solid var(--drocat-line);
    min-height: 18px;
    cursor: pointer;
}
.drocat-palette-pick-swatch {
    cursor: pointer;
    transition: box-shadow .15s ease;
}
.drocat-palette-pick-swatch:hover {
    box-shadow: inset 0 0 0 2px var(--drocat-line-strong);
}
.drocat-palette-pick-swatch.selected {
    box-shadow: inset 0 0 0 2px var(--drocat-cobalt);
}
.drocat-select-palette-strip {
    height: 12px;
    border-radius: 4px;
    border: 1px solid var(--drocat-line);
    /* Inside a dropdown option row: fill the width right of the name. */
    flex: 1 1 55%;
    margin-left: 12px;
    min-width: 24px;
}
.drocat-custom-color-row {
    padding: 4px 8px;
    border: 1px solid var(--drocat-line);
    border-radius: 10px;
    background: var(--drocat-surface);
}
.drocat-custom-color-row:hover { background: var(--drocat-soft); }

/* ---------- Layer tree editor (Skeleton tab) ---------- */
.drocat-layer-board { min-height: 8px; }
.drocat-layer-row {
    width: 100%;
    max-width: 100%;
    box-sizing: border-box;
    border: 1px solid var(--drocat-line);
    border-radius: 12px;
    background: var(--drocat-surface);
    padding: 8px 10px;
    cursor: grab;
}
.drocat-layer-row:hover { border-color: var(--drocat-line-strong); }
.drocat-layer-grip { color: var(--drocat-faint); cursor: grab; }
.drocat-layer-neurons {
    width: 100%;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
    padding: 6px 2px 2px;
    border-top: 1px dashed var(--drocat-line);
    min-height: 34px;
}
.drocat-layer-chip {
    display: inline-flex;
    align-items: center;
    gap: 2px;
    padding: 3px 4px 3px 10px;
    border: 1px solid var(--drocat-line-strong);
    border-radius: 8px;
    background: var(--drocat-soft);
    color: var(--drocat-navy);
    font-size: 12px;
    font-weight: 600;
    cursor: grab;
}
.drocat-layer-chip:hover { border-color: var(--drocat-cobalt); }
.drocat-layer-chip .q-btn { width: 22px; height: 22px; }
.drocat-layer-neuron-input {
    width: 100%;
    min-width: 0;
}
.drocat-layer-neuron-input .drocat-chip-input-shell,
.drocat-layer-neuron-input .drocat-chip-input,
.drocat-layer-neuron-input .q-field {
    width: 100%;
    min-width: 0;
}
.drocat-layer-neuron-input .q-chip { cursor: grab; }

/* ---------- Results panel ---------- */
.drocat-results-head { padding-bottom: 12px; }
.drocat-results-mark {
    width: 30px; height: 30px; border-radius: 9px; display: grid; place-items: center;
    background: var(--drocat-navy);
}
.drocat-results-mark .q-icon { font-size: 16px; }
.drocat-action-bar { padding: 12px 0 10px; border-top: 1px solid var(--drocat-line); }
.drocat-run-recovery {
    padding: 4px 8px;
    border-left: 3px solid var(--drocat-cobalt);
    background: var(--drocat-cobalt-soft);
    border-radius: 0 8px 8px 0;
}
.drocat-run-meta {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
}
.drocat-run-summary {
    padding: 7px 9px;
    border: 1px solid var(--drocat-line);
    border-radius: 8px;
    background: var(--drocat-soft);
    color: var(--drocat-navy);
    font-size: 12px;
    line-height: 1.35;
}
.drocat-progress-row { padding-bottom: 4px; }
.drocat-mini-label { font-size: 11px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--drocat-faint); margin: 8px 0 4px; }
.drocat-empty { color: var(--drocat-faint); font-style: italic; font-size: 12.5px; }
.drocat-muted { color: var(--drocat-muted); }
.drocat-ok { color: var(--drocat-ok); }
.drocat-err { color: var(--drocat-err); }
.drocat-warn { color: var(--drocat-warn); }
.drocat-truncate { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 380px; }

/* ---------- Output file list (file-manager style) ---------- */
.drocat-file-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 4px 0;
}
.drocat-file-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 10px;
    border: 1px solid var(--drocat-line);
    border-radius: 10px;
    background: var(--drocat-surface);
    cursor: pointer;
    min-width: 0;
    transition: border-color .15s ease, background .15s ease;
}
.drocat-file-row:hover {
    border-color: var(--drocat-cobalt);
    background: var(--drocat-soft);
}
.drocat-file-icon { color: var(--drocat-cobalt); font-size: 18px; flex: none; }
.drocat-file-name {
    font-size: 12.5px; font-weight: 600; color: var(--drocat-navy);
    word-break: break-all; line-height: 1.3; min-width: 0;
}
.drocat-file-size { flex: none; font-variant-numeric: tabular-nums; }
.drocat-file-open { color: var(--drocat-muted); flex: none; }
.drocat-file-row:hover .drocat-file-open { color: var(--drocat-cobalt); }
.drocat-expansion { border: 1px solid var(--drocat-line); border-radius: 12px; margin-bottom: 8px; overflow: hidden; }
.drocat-expansion .q-expansion-item__container { background: var(--drocat-surface); }
/* Section-header expansions (collapsible blocks inside drocat-cards, e.g.
   Visualization > Skeleton): the title doubles as the section header, so it
   is rendered slightly larger than Quasar's inherited 14px default. */
.drocat-section-expansion .q-item__label { font-size: 18px; }

/* ---------- Status rows ---------- */
.drocat-status-row { padding: 6px 8px; border-radius: 10px; }
.drocat-status-row:hover { background: var(--drocat-soft); }

/* ---------- Footer ---------- */
.drocat-footer {
    background: var(--drocat-glass) !important;
    backdrop-filter: blur(10px);
    border-top: 1px solid var(--drocat-line);
    min-height: 44px !important;
}
.drocat-footer .q-item__label { color: var(--drocat-faint); font-size: 12px; }

/* ---------- Misc ---------- */
.q-separator { background: var(--drocat-line) !important; }
.q-badge { border-radius: 999px; font-weight: 650; }
.q-expansion-item__toggle-icon { color: var(--drocat-muted); }
.q-chip { border-radius: 8px; background: var(--drocat-soft) !important; color: var(--drocat-navy) !important; }
.q-btn:focus-visible,
.q-field:focus-within,
.q-expansion-item:focus-within {
    outline: 3px solid rgba(20, 92, 255, .28);
    outline-offset: 2px;
}
.drocat-file-row:focus-visible {
    outline: 3px solid rgba(20, 92, 255, .28);
    outline-offset: 2px;
}
.q-spinner { color: var(--drocat-cobalt) !important; }
.nicegui-upload { border: 1px dashed var(--drocat-line-strong) !important; border-radius: 10px !important; }
/* Auto-suggest inputs (pathfinding tabs) replace the native QSelect popup
   with a custom suggestion/history menu; hide the empty native one. */
.drocat-native-popup-hidden { display: none !important; }
/* Arrow-key highlight of the active row in the suggestion/history dropdown. */
.drocat-suggest-menu .q-item.drocat-suggest-active {
    background: var(--drocat-row-hover) !important;
}
/* A suggestion whose value is already in the query stays listed so the rows
   never move under the pointer mid-pick; it is tinted (and ticked) instead of
   removed. The gray column hint keeps its own muted color. */
.drocat-suggest-menu .q-item.drocat-suggest-added {
    color: var(--drocat-cobalt) !important;
}
/* The row tint above cannot reach the tick: ui.icon carries .drocat-muted,
   which sets its own color. Both surfaces now label the tick
   .drocat-suggest-check, so the marked row reads the same everywhere. */
.drocat-suggest-menu .q-item.drocat-suggest-added .drocat-suggest-check {
    color: var(--drocat-cobalt) !important;
}
/* Dataset provenance tags on history rows. */
.drocat-suggest-menu .drocat-history-dataset-badge {
    font-size: 10px;
    letter-spacing: .02em;
}
/* Reinforcement, kept last so nothing re-introduces translucency: the
   floating mapped-view columns must stay fully opaque and visually distinct
   from the metadata columns scrolling underneath them. The doubled class
   raises specificity above every generic cell rule above. */
.drocat-data-viewer-table td.drocat-neuron-map-cell.drocat-neuron-map-cell,
.drocat-data-viewer-table th.drocat-neuron-map-cell.drocat-neuron-map-cell {
    position: sticky;
    background: var(--drocat-map-cell-bg) !important;
    color: var(--drocat-navy) !important;
    text-align: left !important;
}
.drocat-data-viewer-table th.drocat-neuron-map-cell.drocat-neuron-map-cell {
    background: var(--drocat-map-cell-head-bg) !important;
}
/* The generic table-header rule below (… .q-table__middle thead tr:first-child
   th) carries higher specificity than the class-only rule above, so the
   distinct mapped-view head tint needs the same ancestor chain. */
.drocat-data-viewer-table .q-table__middle thead tr:first-child th.drocat-neuron-map-cell.drocat-neuron-map-cell {
    background: var(--drocat-map-cell-head-bg) !important;
}
.drocat-data-viewer-table tr.drocat-neuron-selected-row > td.drocat-neuron-map-cell.drocat-neuron-map-cell {
    background: var(--drocat-selected) !important;
}
.drocat-data-viewer-table .drocat-neuron-map-value {
    text-align: left;
}

/* ---------- Persistent in-page banners (download info etc.) ----------
   The banner stack (ui/components/banner.py) replaces the old Quasar
   "Read" notifications: one unified banner size, orange surfaces per
   theme, a corner × to dismiss, and older banners folded behind a
   count row when several are alive at once. z-index sits above
   Quasar's dialogs/menus (6000) and below tooltips (9000). */
.drocat-banner-stack {
    position: fixed;
    left: 50%;
    transform: translateX(-50%);
    bottom: 18px;
    width: min(690px, calc(100vw - 32px));
    z-index: 6500;
    pointer-events: none;
}
.drocat-banner-stack-body { display: flex; flex-direction: column; gap: 8px; }
.drocat-banner-stack .drocat-banner,
.drocat-banner-stack .drocat-banner-fold { pointer-events: auto; }
.drocat-banner {
    position: relative;
    display: flex;
    align-items: flex-start;
    gap: 8px;
    min-height: 42px;
    padding: 10px 34px 10px 12px;
    background: #ffe3c2;
    color: #000;
    border: 1px solid #f0b269;
    border-radius: var(--drocat-radius-sm);
    box-shadow: var(--drocat-shadow);
    font-size: 13px;
    line-height: 1.45;
}
.drocat-banner .drocat-banner-icon { color: #b45f06; font-size: 18px; margin-top: 1px; }
.drocat-banner .drocat-banner-text { white-space: pre-wrap; overflow-wrap: anywhere; }
.drocat-banner .drocat-banner-close { position: absolute; top: 3px; right: 3px; }
.drocat-banner .drocat-banner-close .q-icon { color: rgba(0, 0, 0, .55); font-size: 16px; }
body.body--dark .drocat-banner {
    background: #7c3e0a;
    color: #ececec;
    border-color: #a2611f;
    box-shadow: 0 12px 32px rgba(0, 0, 0, .45);
}
body.body--dark .drocat-banner .drocat-banner-icon { color: #f0b269; }
body.body--dark .drocat-banner .drocat-banner-close .q-icon { color: rgba(236, 236, 236, .75); }
.drocat-banner-fold {
    display: flex;
    align-items: center;
    padding: 6px 12px;
    border-radius: 8px;
    background: #ffe9d4;
    border: 1px dashed #f0b269;
    color: #7a4310;
    font-size: 12px;
    cursor: pointer;
    user-select: none;
}
.drocat-banner-fold:hover { background: #ffe0bd; }
.drocat-banner-fold .drocat-banner-fold-icon { font-size: 16px; color: #7a4310; }
.drocat-banner-fold .drocat-banner-fold-label { font-size: 12px; }
body.body--dark .drocat-banner-fold {
    background: #5d2f08;
    border-color: #8a4a16;
    color: #f0d9c4;
}
body.body--dark .drocat-banner-fold .drocat-banner-fold-icon { color: #f0d9c4; }
.drocat-banner-folded-list { max-height: 40vh; overflow-y: auto; }
/* Block variant for embedded notices (results-card run summary): same
   palette and typography as the stack banners, full card width. */
.drocat-banner-static {
    display: block;
    padding: 10px 12px;
}
"""


def _neuprint_token_configured() -> bool:
    """Whether a NeuPrint token is configured.

    Mirrors the Settings tab: config.json wins per key, the gitignored
    config_local.json fills empty entries, then the
    NEUPRINT_APPLICATION_CREDENTIALS or NEUPRINT_TOKEN environment
    variables. The CAVE token is optional and does not gate the warning.
    """
    from ui.tabs.settings import _token_sources
    token, _source = _token_sources().get("neuprint", ("", ""))
    return bool(token)


@ui.page("/")
def main_page():
    """Main application page with light/dark Photo-Selector-inspired layout."""

    # Global theme
    ui.add_head_html(f"<style>{DROCAT_CSS}</style>")
    # Quasar's anchor helper logs a console error when a q-tooltip's static
    # target id (#cNN) no longer exists. NiceGUI 3.15 binds .tooltip() via
    # an id string, and the first render of each tab panel recreates some
    # inputs with fresh ids, leaving the original tooltip orphaned (it
    # simply never shows — no functional impact). Silence only that exact
    # error; everything else still reaches the console.
    ui.add_head_html(
        "<script>"
        "(function(){"
        "const _err=console.error;"
        "console.error=function(...a){"
        "if(typeof a[0]==='string'&&a[0].startsWith('Anchor: target '))return;"
        "_err.apply(console,a);"
        "};"
        "})();"
        "</script>"
    )

    # Theme: NiceGUI's dark_mode element drives Quasar's body--dark class
    # (all components, popups and menus re-theme with it). The value None
    # follows the OS preference live (Quasar auto mode listens to
    # prefers-color-scheme on macOS and Windows); True/False force
    # dark/light. The persisted preference is restored here so the first
    # paint already uses it.
    dark = ui.dark_mode(value=_saved_dark_mode())

    # Persistent in-page banner stack (download info and similar notices;
    # replaces the old Quasar "Read" notifications). Fixed overlay, so its
    # DOM position is irrelevant — created early so every tab can use it.
    create_banner_stack()

    THEME_OPTIONS = [
        ("System", "brightness_auto", None),
        ("Light", "light_mode", False),
        ("Dark", "dark_mode", True),
    ]

    # Header
    with ui.header(elevated=False).classes("drocat-header items-center justify-between"):
        with ui.row().classes("items-center gap-3"):
            with ui.element("div").classes("drocat-brand-mark"):
                ui.icon("hub")
            with ui.column().classes("gap-0"):
                ui.label("DROCAT").classes("drocat-brand-title")
                ui.label("Drosophila Connectome Analysis Toolkit").classes("drocat-brand-sub")
        with ui.row().classes("items-center gap-4"):
            ui.label(f"v{APP_VERSION}").classes("drocat-version-pill")
            ui.link("Documentation", "docs/ui_guides/README.html").classes("drocat-header-link")
            activity_btn = ui.button("Activity", icon="history").props(
                'flat no-caps id="drocat-activity-button"'
            ).classes("drocat-activity-btn").tooltip(
                "View recent and active executions"
            )
            theme_button = ui.button().props("flat").classes(
                "drocat-dark-toggle"
            ).tooltip("Theme")
            with theme_button:
                with ui.element("div").classes("drocat-theme-icon-pair"):
                    ui.icon("light_mode").classes("drocat-theme-sun")
                    ui.element("div").classes("drocat-theme-sep")
                    ui.icon("dark_mode").classes("drocat-theme-moon")
                theme_menu = ui.menu().classes("drocat-theme-menu")
            # Settings lives in the header beside the theme toggle rather than as
            # a left-nav tool tab. Its navigation is wired once nav_panels exists
            # below (the header is built before the tab panels).
            settings_btn = ui.button(icon="settings").props("flat").classes(
                "drocat-dark-toggle"
            ).tooltip("Settings")
            theme_items = {}
            with theme_menu:
                for label, icon, mode in THEME_OPTIONS:
                    item = ui.item().props("clickable").classes(
                        "drocat-theme-item"
                    )
                    with item:
                        ui.icon(icon).classes("drocat-theme-icon")
                        ui.label(label)
                        ui.space()
                        ui.icon("check").classes("drocat-theme-check")
                    theme_items[mode] = item
                    item.on_click(lambda _event=None, m=mode: _apply_theme(m, persist=True))

            def _apply_theme(mode: bool | None, *, persist: bool) -> None:
                dark.value = mode
                for item_mode, item in theme_items.items():
                    if item_mode is mode:
                        item.classes(add="drocat-theme-item-active")
                    else:
                        item.classes(remove="drocat-theme-item-active")
                # Persist only on user clicks; the initial restore already
                # came from the cookie (writing during build would need the
                # client loop, which UI tests do not run). The broadcast
                # lets already-open docs/ui_guides pages (guide-theme.js)
                # follow the toggle instead of waiting for their next load.
                if persist:
                    mode_value = (
                        "dark" if mode is True else "light" if mode is False else "auto"
                    )
                    ui.run_javascript(
                        "document.cookie = 'drocat_dark=" + mode_value
                        + "; max-age=31536000; path=/; SameSite=Lax';"
                        "try{new BroadcastChannel('drocat-theme')"
                        ".postMessage('" + mode_value + "')}catch(e){}"
                    )

            # Highlight the active choice without touching the browser.
            _apply_theme(dark.value, persist=False)

    # Main content
    with ui.column().classes("w-full drocat-shell gap-3"):
        # Grouped navigation - layered cards. Every group is its own tinted
        # card holding its header on top of its tab segments (no partition,
        # always aligned): Connection blue, Cross-Dataset rose, Visualization
        # teal, Similarity amber, NeuronBridge purple + NB badge, FlyLight
        # green. Settings is a standalone slate card (no header), separated
        # from the groups.
        NAV_GROUPS = [
            ("Connection", "connection", 3, [
                ("Complete Paths", "route"), ("Shortest Paths", "alt_route"),
                ("Network", "schema"),
            ]),
            ("Cross-Dataset", "crossdataset", 2, [
                ("Paths", "sync_alt"), ("Type Validation", "policy"),
            ]),
            ("Visualization", "visualization", 2, [
                ("Skeleton", "view_in_ar"), ("Net-Viz", "account_tree"),
            ]),
            ("Similarity", "similarity", 2, [
                ("Connectivity", "analytics"), ("Morphology", "science"),
            ]),
            ("NeuronBridge", "nb", 3, [
                ("Find Lines", "biotech"), ("Find Neurons", "search"),
                ("Co-Labeling", "layers"),
            ]),
            ("FlyLight", "flylight", 1, [("Downloader", "download")]),
        ]
        tab_buttons = {}

        def group_tab(label: str, icon: str):
            # Newlines + white-space: pre-line stack multi-word names one
            # word per line (Find Lines -> Find / Lines) with tight leading.
            button = ui.button(label.replace(" ", "\n"), icon=icon).props(
                "flat dense no-caps"
            ).classes("drocat-group-tab")
            button.on_click(lambda _event, name=label: nav_panels.set_value(name))
            tab_buttons[label] = button
            return button

        with ui.element("div").classes("drocat-nav w-full"):
            for group_name, tint, size, tabs in NAV_GROUPS:
                with ui.element("div").classes(
                    f"drocat-group-card drocat-tint-{tint}"
                ).style(f"flex: {size} 1 0"):
                    ui.label(group_name).classes(
                        f"drocat-group-head drocat-head-{tint}"
                    )
                    with ui.element("div").classes("drocat-group-tabs"):
                        for label, icon in tabs:
                            group_tab(label, icon)
            # Settings is promoted to the header (beside the theme toggle), so it
            # is no longer rendered as a left-nav tool-tab card below the groups.

        with ui.tab_panels(value=_saved_active_tab()).classes("w-full bg-transparent") as nav_panels:
            # Connection
            with ui.tab_panel("Complete Paths").classes("p-0"):
                create_find_path_tab()
            with ui.tab_panel("Shortest Paths").classes("p-0"):
                create_find_shortest_tab()
            with ui.tab_panel("Network").classes("p-0"):
                create_network_tab()
            # Cross-Dataset
            with ui.tab_panel("Paths").classes("p-0"):
                create_inter_dataset_tab()
            with ui.tab_panel("Type Validation").classes("p-0"):
                create_type_validation_tab()
            # Visualization
            with ui.tab_panel("Skeleton").classes("p-0"):
                skeleton_recovery = create_skeleton_tab()
            with ui.tab_panel("Net-Viz").classes("p-0"):
                create_net_viz_tab()
            # Similarity
            with ui.tab_panel("Connectivity").classes("p-0"):
                create_connectivity_tab()
            with ui.tab_panel("Morphology").classes("p-0"):
                create_morphology_tab()
            # NeuronBridge
            with ui.tab_panel("Find Lines").classes("p-0"):
                create_nb_find_lines_tab()
            with ui.tab_panel("Find Neurons").classes("p-0"):
                create_nb_find_neuron_tab()
            with ui.tab_panel("Co-Labeling").classes("p-0"):
                create_nb_colabel_tab()
            # FlyLight
            with ui.tab_panel("Downloader").classes("p-0"):
                create_flylight_tab()
            # Settings
            with ui.tab_panel("Settings").classes("p-0"):
                create_settings_tab()

        # The header Settings button opens the Settings panel and is tracked in
        # tab_buttons so the active-tab highlight follows it too.
        tab_buttons["Settings"] = settings_btn
        settings_btn.on_click(lambda: nav_panels.set_value("Settings"))

        def sync_active_tab(value):
            for label, button in tab_buttons.items():
                if label == value:
                    button.classes(add="drocat-active")
                else:
                    button.classes(remove="drocat-active")

        def remember_active_tab(value):
            sync_active_tab(value)
            if value not in _DROCAT_TAB_NAMES:
                return
            try:
                ui.run_javascript(
                    "document.cookie = 'drocat_active_tab=' + "
                    "encodeURIComponent(" + repr(value) + ") + "
                    "'; max-age=31536000; path=/; SameSite=Lax'"
                )
            except Exception:
                pass

        nav_panels.on_value_change(lambda event: remember_active_tab(event.value))
        sync_active_tab(nav_panels.value)

        # Application-level execution overview. Each output panel remains the
        # detailed view, while this compact surface makes work discoverable
        # even when it was started in another tab.
        tab_activity_keys = {
            "Complete Paths": ["find_path"],
            "Shortest Paths": ["find_shortest"],
            "Network": ["network"],
            "Paths": ["inter_dataset"],
            "Type Validation": ["type_mapping_validation"],
            "Skeleton": ["visualization_skeleton"],
            "Net-Viz": ["visualization_net_viz"],
            "Connectivity": ["connectivity_similar", "connectivity_comparison"],
            "Morphology": ["morphology_similar", "morphology_comparison"],
            "Find Lines": ["nb_find_lines"],
            "Find Neurons": ["nb_find_neuron"],
            "Co-Labeling": ["nb_colabel"],
            "Downloader": ["flylight"],
            "Settings": [],
        }

        def _latest_activity(label):
            records = [
                RUN_MANAGER.store.latest(
                    key, session_id=RUN_MANAGER.session_id
                )
                for key in tab_activity_keys.get(label, [])
            ]
            records = [record for record in records if record]
            return max(
                records,
                key=lambda record: str(
                    record.get("updated_at") or record.get("created_at") or ""
                ),
                default=None,
            )

        def sync_run_activity() -> None:
            active_count = 0
            for label, button in tab_buttons.items():
                record = _latest_activity(label)
                status = str(record.get("status", "")) if record else ""
                normalized = status.lower()
                if normalized in {item.lower() for item in ACTIVE_STATUSES}:
                    active_count += 1
                try:
                    button.classes(
                        remove="drocat-tab-running drocat-tab-failed drocat-tab-completed"
                    )
                    if normalized in {item.lower() for item in ACTIVE_STATUSES}:
                        button.classes(add="drocat-tab-running")
                    elif normalized in {"failed", "interrupted", "cancelled"}:
                        button.classes(add="drocat-tab-failed")
                    elif normalized == "completed":
                        button.classes(add="drocat-tab-completed")
                    description = f"{label}: {status or 'no recent execution'}"
                    button.props(f'aria-label="{description}"')
                except Exception:
                    pass
            try:
                activity_btn.props(
                    f'aria-label="Activity ({active_count} active)"'
                )
            except Exception:
                pass

        activity_dialog = ui.dialog()
        with activity_dialog, ui.card().classes("drocat-activity-card gap-3"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("Recent Activity").classes("text-h6")
                with ui.row().classes("items-center gap-1"):
                    ui.button(
                        icon="refresh", on_click=lambda: refresh_activity()
                    ).props("flat dense round").tooltip("Refresh activity")
                    ui.button(icon="close", on_click=activity_dialog.close).props(
                        "flat dense round"
                    ).tooltip("Close")
            activity_list = ui.column().classes("w-full gap-2 drocat-activity-list")

        def _activity_time(record):
            value = record.get("finished_at") or record.get("started_at") or ""
            return str(value).replace("T", " ").replace("+00:00", " UTC")

        def _open_activity_tab(label: str) -> None:
            nav_panels.set_value(label)
            activity_dialog.close()

        def refresh_activity() -> None:
            try:
                activity_list.clear()
                records = RUN_MANAGER.recent_runs(20)
                with activity_list:
                    if not records:
                        ui.label("No executions yet.").classes("drocat-empty")
                    for record in records:
                        tab_label = next(
                            (
                                label
                                for label, keys in tab_activity_keys.items()
                                if record.get("tab_key") in keys
                            ),
                            record.get("title", "Tool"),
                        )
                        status = str(record.get("status", "Unknown"))
                        duration = record.get("duration")
                        duration_text = (
                            f"{float(duration):.1f}s"
                            if isinstance(duration, (int, float))
                            else "duration unavailable"
                        )
                        with ui.row().classes(
                            "w-full items-center justify-between gap-3 drocat-activity-row"
                        ):
                            with ui.column().classes("gap-0 min-w-0"):
                                ui.label(record.get("title", "Execution")).classes(
                                    "drocat-activity-title"
                                )
                                ui.label(
                                    f"{tab_label}  ·  {status}  ·  {duration_text}"
                                ).classes("drocat-activity-meta")
                                ui.label(_activity_time(record)).classes(
                                    "drocat-activity-meta"
                                )
                            ui.button(
                                "Open",
                                on_click=lambda label=tab_label: _open_activity_tab(label),
                            ).props("flat dense color=primary")
            except Exception:
                pass

        def open_activity() -> None:
            refresh_activity()
            activity_dialog.open()

        activity_btn.on_click(open_activity)
        sync_run_activity()
        ui.timer(2.0, sync_run_activity)

        # Offer any auto-saved (uncommitted) layer style draft only when the
        # user switches to the Skeleton tab, so the recovery dialog does not
        # pop up on app load or while the user is on another tab.
        # Offer the auto-saved layer-style draft only on the FIRST Skeleton tab
        # activation after app start (a crash/quit recovery). Later in-page
        # switches to the tab do not prompt: the editor keeps its loaded rows
        # (a default "continue").
        skeleton_recovery_offered = {"value": False}

        def _on_nav(value):
            if value != "Skeleton":
                return
            if skeleton_recovery_offered["value"]:
                return
            skeleton_recovery_offered["value"] = True
            skeleton_recovery()

        nav_panels.on_value_change(lambda event: _on_nav(event.value))

    # --- Token warning dialog -------------------------------------------
    # When no NeuPrint token is configured, pop a warning once the page is
    # live, pointing the user to the Settings tab. It is opened from a
    # one-shot timer (same pattern as the pull-state refresh timers) so
    # building the page never needs the client loop.
    token_warning = ui.dialog().props('id="drocat-token-warning"')
    with token_warning, ui.card().classes("w-96 max-w-[90vw] gap-2"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("key_off").classes("drocat-warn")
            ui.label("NeuPrint token not configured").classes("text-h6")
        ui.label(
            "The NeuPrint token is required for NeuPrint datasets. Get one from "
            "neuprint.janelia.org/account and set it in the Settings tab (the "
            "CAVE token is optional; only needed for FlyWire FAFB online fetching)."
        ).classes("text-sm drocat-muted")
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button(
                "Open Settings",
                icon="settings",
                on_click=lambda: (
                    nav_panels.set_value("Settings"),
                    token_warning.close(),
                ),
            ).props('unelevated color="primary" id="drocat-token-warning-settings"')
            ui.button("Later", on_click=token_warning.close).props(
                'flat id="drocat-token-warning-later"'
            )

    def _maybe_open_token_warning() -> None:
        if not _neuprint_token_configured():
            token_warning.open()

    ui.timer(0.5, _maybe_open_token_warning, once=True)

    # Footer
    with ui.footer().classes("drocat-footer justify-center"):
        ui.label(
            f"DROCAT v{APP_VERSION}  ·  Local execution only  ·  Output files saved to your machine"
        ).classes("text-caption")


def main():
    """Entry point for the DROCAT UI application."""
    host = os.environ.get("DROCAT_UI_HOST", APP_HOST)
    port = int(os.environ.get("DROCAT_UI_PORT", APP_PORT))
    show = os.environ.get("DROCAT_UI_SHOW", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }

    # Run-state manifests/logs are refresh-recovery cache, not permanent
    # application data. Remove terminal entries from older UI sessions before
    # serving a new session; output folders and analysis results are preserved.
    RUN_MANAGER.cleanup_stale_cache()

    # NiceGUI prints its ready line during the server lifespan startup.  This
    # handler runs immediately afterward, so the launcher output includes a
    # usable browser fallback directly below that line on macOS, Windows, and
    # Linux when automatic browser opening is unavailable or disabled.
    app.on_startup(
        lambda: print(
            f"Tip: If the browser did not open automatically, copy and open "
            f"http://127.0.0.1:{port} in your browser.",
            flush=True,
        )
    )
    # Keep backend ownership aligned with the UI server lifetime. This is
    # distinct from page/client deletion: refreshing a browser page does not
    # stop the server or invoke this hook.
    app.on_shutdown(RUN_MANAGER.shutdown)
    ui.run(
        title=APP_TITLE,
        host=host,
        port=port,
        reload=False,
        show=show,
        favicon="🧠",
    )


# A spawned cross-dataset worker imports this file as ``__mp_main__``.  Do
# not start a second NiceGUI server in that worker; it only needs to import
# the picklable scan callback.
if __name__ == "__main__":
    main()

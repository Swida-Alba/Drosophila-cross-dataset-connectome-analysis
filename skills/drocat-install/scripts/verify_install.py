#!/usr/bin/env python3
"""Verify a DROCAT installation without requiring network access.

Checks project layout, the supported Python window, dependency consistency,
core/optional imports, token configuration, and the UI package import.
Token configuration is advisory unless ``--require-token`` is passed.

Usage:
    python verify_install.py --project /path/to/drocat
    python verify_install.py --project . --python /path/to/env/bin/python
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


CORE_IMPORTS = [
    "numpy",
    "pandas",
    "polars",
    "scipy",
    "matplotlib",
    "seaborn",
    "plotly",
    "networkx",
    "openpyxl",
    "xlrd",
    "xlsxwriter",
    "bokeh",
    "requests",
    "jinja2",
    "PIL",
    "pydantic",
    "neuprint",
    "nicegui",
    "tqdm",
    "cv2",
    "reportlab",
    "pptx",
    "img2pdf",
    "neuronbridge_client",
]

OPTIONAL_IMPORTS = [
    "caveclient",
    "cloudvolume",
    "navis",
    "flybrains",
    "selenium",
    "webdriver_manager",
    "fitz",
    "psutil",
    "trimesh",
]

IMPORT_PROBE = """
import importlib, json
mods = json.loads('__MODS_JSON__')
out = {}
for name in mods:
    try:
        importlib.import_module(name)
        out[name] = "ok"
    except Exception as exc:
        out[name] = "%s: %s" % (type(exc).__name__, exc)
print(json.dumps(out))
"""

VERSION_PROBE = r"""
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

try:
    from packaging.requirements import Requirement
except ImportError:  # bare venvs may lack `packaging`; skip rather than crash
    print(json.dumps({"__skipped__": "packaging not installed in target env"}))
    sys.exit(0)

results = {}
for filename in sys.argv[1:]:
    manifest = {}
    for raw_line in Path(filename).read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        try:
            installed = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            manifest[requirement.name] = "missing"
            continue
        if requirement.specifier and installed not in requirement.specifier:
            manifest[requirement.name] = (
                f"installed {installed}, expected {requirement.specifier}"
            )
    results[filename] = manifest
print(json.dumps(results))
"""

# Real disk round-trip through the shared atomic writer.  Imports and
# `pip check` cannot catch platform-specific write failures: the BANC/FAFB
# table converters and the lossless re-encoder all publish through
# write_parquet_atomic, whose fsync used to open the temp file read-only and
# abort with [Errno 9] on Windows — with the install still reporting PASS.
# The re-encode leg is checked too: it returned False from an open-handle
# WinError 32 on every Windows install while this probe still printed
# `atomic-ok` (2026-09-18 retest, F6).
ATOMIC_WRITE_PROBE = r"""
import os, shutil, sys, tempfile
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from utils.parquet_utils import (
    parquet_lossless_marker, reencode_parquet_lossless, write_parquet_atomic)
import polars as pl

tmp = tempfile.mkdtemp(prefix="drocat_verify_atomic_")
try:
    target = os.path.join(tmp, "roundtrip.parquet")
    write_parquet_atomic(
        target,
        lambda temp: pl.DataFrame({"v": [1, 2, 3]}).write_parquet(temp))
    assert pl.read_parquet(target)["v"].to_list() == [1, 2, 3]
    reencoded = reencode_parquet_lossless(
        target, {"compression": "zstd", "use_dictionary": False})
    assert reencoded is True, "lossless re-encode did not run"
    assert parquet_lossless_marker(target), "re-encode left no marker"
    assert pl.read_parquet(target)["v"].to_list() == [1, 2, 3]
    leftovers = [n for n in os.listdir(tmp) if n.endswith(".tmp")]
    assert not leftovers, f"re-encode leaked temp files: {leftovers}"
    print("atomic-ok")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
"""


def isolated_python_env() -> dict[str, str]:
    """Keep probes inside the selected environment, excluding user site packages."""
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    return env


def run_probe(python_exe: str, modules: list) -> dict:
    code = IMPORT_PROBE.replace("__MODS_JSON__", json.dumps(modules))
    proc = subprocess.run(
        [python_exe, "-c", code],
        capture_output=True,
        env=isolated_python_env(),
        text=True, encoding='utf-8', errors='replace',
        timeout=300,
    )
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {m: f"probe failed: {proc.stderr[-200:]}" for m in modules}


def run_version_probe(python_exe: str, manifests: list[Path]) -> dict:
    proc = subprocess.run(
        [python_exe, "-c", VERSION_PROBE, *(str(path) for path in manifests)],
        capture_output=True,
        env=isolated_python_env(),
        text=True, encoding='utf-8', errors='replace',
        timeout=300,
    )
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        detail = (proc.stderr or proc.stdout)[-300:]
        return {str(path): {"probe": f"failed: {detail}"} for path in manifests}


def config_neuprint_token(project: Path) -> tuple[bool, str | None]:
    """Return ``(has_token, source)`` from config_local.json then config.json.

    Reads with ``utf-8-sig`` because Windows PowerShell's
    ``Set-Content -Encoding UTF8`` writes a BOM, which plain ``utf-8``
    decoding rejects — that would hide a configured token and fail
    ``--require-token``.
    """
    has_token = False
    source: str | None = None
    for filename in ("config_local.json", "config.json"):
        config_token = project / filename
        if not config_token.exists():
            continue
        source = str(config_token)
        try:
            cfg = json.loads(config_token.read_text(encoding="utf-8-sig"))
            neuprint_value = (cfg.get("tokens") or {}).get("neuprint") or ""
        except Exception:
            neuprint_value = ""
        if (
            isinstance(neuprint_value, str)
            and neuprint_value.strip()
            and "YOUR_" not in neuprint_value
            and len(neuprint_value.strip().strip("'\"")) > 20
        ):
            has_token = True
            break
    return has_token, source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="Path to the DROCAT repository")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to test (default: the interpreter running this script)",
    )
    parser.add_argument(
        "--require-token",
        action="store_true",
        help="Fail when a real NeuPrint token has not been configured",
    )
    args = parser.parse_args()

    project = Path(args.project).expanduser().resolve()
    # abspath, NOT Path.resolve(): a venv's python is a symlink (uv- and
    # python.org-managed venvs alike), and resolving it points the probes at
    # the base interpreter without the venv's site-packages.
    python_exe = os.path.abspath(os.path.expanduser(args.python))
    checks = []
    version = "unknown"

    def check(name: str, ok: bool, detail: str = ""):
        checks.append((name, ok, detail))

    # Project layout
    check("project directory", project.is_dir(), str(project))
    for rel in [
        "mac_DROCAT.command",
        "windows_DROCAT.bat",
        "archive/install/install.sh",
        "archive/install/install.ps1",
        "archive/install/install.bat",
        "requirements.txt",
        "ui/requirements.txt",
        "ui/app.py",
        "src/coana.py",
        "src/neuronbridge_finder.py",
        "src/neuronbridge_client.py",
        "src/visualize_skeleton.py",
        "vispath-subproject/src/vispath_pkg",
    ]:
        check(f"file {rel}", (project / rel).exists(), str(project / rel))

    # Python version
    version = "unknown"
    try:
        proc = subprocess.run(
            [python_exe, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
            capture_output=True,
            env=isolated_python_env(),
            text=True, encoding='utf-8', errors='replace',
            timeout=60,
        )
        version = proc.stdout.strip()
        major_minor = tuple(int(x) for x in version.split("."))
        check(
            "python version 3.10-3.11",
            (3, 10) <= major_minor < (3, 12),
            version,
        )
    except Exception as exc:
        check("python version", False, str(exc))

    # Imports
    core_results = run_probe(python_exe, CORE_IMPORTS)
    for mod, status in core_results.items():
        check(f"import {mod}", status == "ok", status)
    optional_results = run_probe(python_exe, OPTIONAL_IMPORTS)
    for mod, status in optional_results.items():
        if status != "ok":
            check(f"import {mod} (optional)", True, f"missing: {status}")

    manifests = [
        project / ("requirements-windows.txt" if sys.platform == "win32" else "requirements.txt"),
        project / "ui/requirements.txt",
    ]
    version_results = run_version_probe(python_exe, manifests)
    for manifest, failures in version_results.items():
        if manifest == "__skipped__":
            # Whole-run skip (e.g. no packaging in the target env): report
            # it instead of feeding the sentinel into Path.relative_to.
            check(
                "pinned versions",
                True,
                f"skipped: {failures}",
            )
            continue
        manifest_label = str(Path(manifest).relative_to(project))
        if "__skipped__" in failures:
            check(
                f"pinned versions ({manifest_label})",
                True,
                f"skipped: {failures['__skipped__']}",
            )
            continue
        check(
            f"pinned versions ({manifest_label})",
            not failures,
            "; ".join(f"{name}: {detail}" for name, detail in failures.items()),
        )

    # Dependency metadata must also agree. Successful imports alone do not
    # expose conflicts such as NiceGUI and an old NeuronBridge Pydantic pin.
    try:
        proc = subprocess.run(
            [python_exe, "-m", "pip", "check"],
            capture_output=True,
            env=isolated_python_env(),
            text=True, encoding='utf-8', errors='replace',
            timeout=300,
        )
        detail = (proc.stdout or proc.stderr).strip()
        check("pip dependency consistency", proc.returncode == 0, detail)
    except Exception as exc:
        check("pip dependency consistency", False, str(exc))

    # Atomic-write round trip (see ATOMIC_WRITE_PROBE): a real disk write
    # through the shared atomic writer, the code path that broke BANC/FAFB
    # table preparation on Windows while every import check passed.
    try:
        proc = subprocess.run(
            [python_exe, "-c", ATOMIC_WRITE_PROBE],
            cwd=str(project),
            capture_output=True,
            env=isolated_python_env(),
            text=True, encoding='utf-8', errors='replace',
            timeout=300,
        )
        stdout_text = proc.stdout or ""
        stderr_text = proc.stderr or ""
        check(
            "atomic write round-trip",
            proc.returncode == 0 and "atomic-ok" in stdout_text,
            (stderr_text or stdout_text
             or "probe produced no output (reader failed)")[-300:],
        )
    except Exception as exc:
        check("atomic write round-trip", False, str(exc))

    # Token file: config_local.json (gitignored override) then config.json
    # (committed clean defaults).
    has_neuprint, token_source = config_neuprint_token(project)
    if token_source:
        token_label = (
            "tokens (NeuPrint token configured)"
            if has_neuprint
            else "tokens (NeuPrint token not configured - advisory)"
        )
        check(
            token_label + ("" if args.require_token else " (optional)"),
            has_neuprint or not args.require_token,
            token_source,
        )
    else:
        check(
            "tokens file" + ("" if args.require_token else " (optional)"),
            not args.require_token,
            "missing config_local.json / config.json",
        )

    # UI import (project must be on sys.path)
    ui_probe = (
        "import sys; sys.path.insert(0, '.'); "
        "import ui.app; print('ui-ok')"
    )
    try:
        proc = subprocess.run(
            [python_exe, "-c", ui_probe],
            cwd=str(project),
            capture_output=True,
            env=isolated_python_env(),
            text=True, encoding='utf-8', errors='replace',
            timeout=300,
        )
        check("UI imports (ui.app)", "ui-ok" in proc.stdout, proc.stderr[-200:])
    except Exception as exc:
        check("UI imports (ui.app)", False, str(exc))

    # Report
    required_failures = []
    print(f"DROCAT install verification - {project}")
    print(f"Python: {version}")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}" + (f"  ({detail})" if detail and not ok else ""))
        if not ok and not name.endswith("(optional)"):
            required_failures.append(name)

    print()
    if required_failures:
        print(f"FAILED: {len(required_failures)} required check(s) failed")
        return 1
    print("OK: all required checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

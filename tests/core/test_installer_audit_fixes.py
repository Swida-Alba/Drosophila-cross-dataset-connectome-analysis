"""Regression checks for the 2026-10-25 one-click installation audit."""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

requires_bash = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash not available")


def _text(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _shell_function(path: Path, name: str) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        rf"^(?P<indent>[ \t]*){re.escape(name)}\(\) \{{.*?^(?P=indent)\}}",
        text, re.S | re.M)
    assert match, f"{name} not found in {path}"
    return match.group(0)


def test_install_sh_checks_pip_inside_the_retry_loop_and_forces_reinstall():
    text = _text("archive/install/install.sh")
    assert text.count("python -m pip check") == 1
    assert text.index("VERIFY_OK") < text.index("python -m pip check")
    assert "--force-reinstall" in text


def test_install_ps1_checks_pip_inside_the_retry_loop_and_forces_reinstall():
    text = _text("archive/install/install.ps1")
    assert text.index("$Verified") < text.index('"pip", "check"')
    assert "--force-reinstall" in text


def test_windows_installer_accepts_the_documented_python_window():
    text = _text("archive/install/install.ps1")
    assert "in ((3, 10), (3, 11))" in text
    assert "not Python 3.10/3.11" in text


def test_miniconda_downloads_are_pinned_and_verified():
    manifest = _text("archive/install/miniconda_checksums.txt")
    entries = {
        line.split()[0]: line.split()[1]
        for line in manifest.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert len(entries) >= 6
    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in entries.values())
    for rel in ("archive/install/install.sh", "archive/install/install.ps1"):
        text = _text(rel)
        assert "miniconda_checksums.txt" in text
        for name in re.findall(
                r"https://repo\.anaconda\.com/miniconda/(Miniconda3-[^\"']+)",
                text):
            assert name in entries, f"{rel} URL {name} has no SHA256 pin"
            assert "latest" not in name, (
                f"{rel} URL {name} is a moving alias; pin a versioned file")
    assert "verify_miniconda_download" in _text("archive/install/install.sh")
    assert "Get-FileHash" in _text("archive/install/install.ps1")
    assert "Miniconda3-py312_26.9.4-0-Windows-arm64.exe" in _text(
        "archive/install/install.ps1")


@pytest.mark.parametrize(
    "rel", ["archive/install/install.sh", "mac_DROCAT.command"])
@requires_bash
def test_shell_config_autofill_adds_a_missing_envs_object(tmp_path, rel):
    config = tmp_path / "config_local.json"
    config.write_text(
        '{\n  "tokens": {\n    "neuprint": ""\n  }\n}\n',
        encoding="utf-8")
    func = _shell_function(ROOT / rel, "update_config_env")
    script = (
        "set -euo pipefail\n"
        + func
        + f"\nupdate_config_env 4.5.0 drocat-4.5.0 '{config}'\n"
    )
    subprocess.run(["bash", "-c", script], check=True)
    data = json.loads(config.read_text(encoding="utf-8"))
    assert data["envs"]["4.5.0"] == "drocat-4.5.0"
    assert data["tokens"]["neuprint"] == ""


@pytest.mark.parametrize(
    "rel", ["archive/install/install.sh", "mac_DROCAT.command"])
@requires_bash
@pytest.mark.parametrize(
    "body,expected",
    [
        ('{\n  "envs": {\n    "4.4.0": "old"\n  }\n}\n',
         {"4.4.0": "old", "4.5.0": "drocat-4.5.0"}),
        ('{\n  "envs": {\n  }\n}\n',
         {"4.5.0": "drocat-4.5.0"}),
        ('{\n  "envs": {\n    "4.5.0": ""\n  }\n}\n',
         {"4.5.0": "drocat-4.5.0"}),
    ],
)
def test_shell_config_autofill_handles_an_existing_envs_object(
        tmp_path, rel, body, expected):
    config = tmp_path / "config_local.json"
    config.write_text(body, encoding="utf-8")
    func = _shell_function(ROOT / rel, "update_config_env")
    script = (
        "set -euo pipefail\n"
        + func
        + f"\nupdate_config_env 4.5.0 drocat-4.5.0 '{config}'\n"
    )
    subprocess.run(["bash", "-c", script], check=True)
    data = json.loads(config.read_text(encoding="utf-8"))
    assert data["envs"] == expected


@requires_bash
def test_shell_config_autofill_leaves_a_compact_file_unchanged(tmp_path):
    config = tmp_path / "config_local.json"
    body = '{"envs": {"4.4.0": "old"}}\n'
    config.write_text(body, encoding="utf-8")
    func = _shell_function(ROOT / "archive/install/install.sh",
                           "update_config_env")
    script = (
        "set -euo pipefail\n"
        + func
        + f"\nupdate_config_env 4.5.0 drocat-4.5.0 '{config}'\n"
    )
    subprocess.run(["bash", "-c", script], check=True)
    assert config.read_text(encoding="utf-8") == body


@pytest.mark.parametrize(
    "rel", ["archive/install/install.sh", "mac_DROCAT.command"])
@requires_bash
def test_shell_config_autofill_only_edits_the_envs_object(tmp_path, rel):
    config = tmp_path / "config_local.json"
    config.write_text(
        '{\n  "envs": {\n    "4.4.0": "old"\n  },\n'
        '  "other": {\n    "4.5.0": "sentinel"\n  }\n}\n',
        encoding="utf-8")
    func = _shell_function(ROOT / rel, "update_config_env")
    script = (
        "set -euo pipefail\n"
        + func
        + f"\nupdate_config_env 4.5.0 drocat-4.5.0 '{config}'\n"
    )
    subprocess.run(["bash", "-c", script], check=True)
    data = json.loads(config.read_text(encoding="utf-8"))
    assert data["envs"] == {"4.4.0": "old", "4.5.0": "drocat-4.5.0"}
    assert data["other"]["4.5.0"] == "sentinel"


def test_launcher_health_and_port_guards_are_explicit():
    mac = _text("mac_DROCAT.command")
    win = _text("windows_DROCAT.bat")
    assert "coana, statvis, morphology, route_plan" in mac
    assert "coana,statvis,morphology,route_plan" in win
    assert "DROCAT_UI_PORT must be an integer from 1 to 65535" in mac
    assert "DROCAT_UI_PORT must be an integer from 1 to 65535" in win
    assert r"ui[\\/]app\.py" in mac
    assert 'getattr(m, "__file__", None)' in mac
    assert "getattr(m,'__file__',None)" in win


def test_launchers_report_the_configured_bind_host():
    mac = _text("mac_DROCAT.command")
    win = _text("windows_DROCAT.bat")
    assert 'UI_URL="http://$UI_HOST:$APP_PORT"' in mac
    assert '0.0.0.0) UI_BIND_HOST="127.0.0.1"' in mac
    assert 'open "$UI_URL"' in mac
    assert "http://127.0.0.1:$APP_PORT" not in mac
    assert 'set "UI_HOST=%DROCAT_UI_HOST%"' in win
    assert 'http://!UI_HOST!:!APP_PORT!' in win
    assert "http://127.0.0.1:!APP_PORT!" not in win


def test_ui_host_and_port_guards():
    from ui.config import is_loopback_host, resolve_ui_host, resolve_ui_port

    assert resolve_ui_host("127.0.0.1") == "127.0.0.1"
    assert resolve_ui_host("localhost") == "localhost"
    with pytest.raises(ValueError, match="DROCAT_ALLOW_REMOTE"):
        resolve_ui_host("0.0.0.0")
    assert resolve_ui_host("0.0.0.0", allow_remote=True) == "0.0.0.0"
    assert resolve_ui_port("8080") == 8080
    assert resolve_ui_port(None) == 8080
    for bad in ("abc", "0", "65536", "-1"):
        with pytest.raises(ValueError):
            resolve_ui_port(bad)
    assert is_loopback_host("::1")
    assert not is_loopback_host("0.0.0.0")


def test_ui_browser_url_matches_the_bind():
    from ui.config import ui_browser_url

    assert ui_browser_url("127.0.0.1", 8080) == "http://127.0.0.1:8080"
    assert ui_browser_url("0.0.0.0", 8080) == "http://127.0.0.1:8080"
    assert ui_browser_url("::", 8080) == "http://[::1]:8080"
    assert ui_browser_url("::1", 8080) == "http://[::1]:8080"
    assert ui_browser_url("192.168.1.5", 9000) == "http://192.168.1.5:9000"


def test_verifier_requires_core_runtime_imports():
    import importlib.util

    path = ROOT / "skills/drocat-install/scripts/verify_install.py"
    spec = importlib.util.spec_from_file_location(
        "drocat_verify_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = {
        "pyarrow", "zstandard", "boto3", "xarray",
        "coana", "statvis", "morphology", "route_plan",
    }
    assert required <= set(module.CORE_IMPORTS)


def test_new_core_imports_have_pins_or_are_packaged():
    requirements = _text("requirements.txt") + _text("requirements-windows.txt")
    for module in ("pyarrow", "zstandard", "boto3", "xarray"):
        assert re.search(rf"^{re.escape(module)}==", requirements, re.M), (
            f"{module} is a required import but has no pin")
    pyproject = _text("pyproject.toml")
    for module in ("coana", "statvis", "morphology", "route_plan"):
        assert f'"{module}"' in pyproject, (
            f"{module} is a required import but is not packaged")


def test_installer_skill_does_not_claim_a_token_prompt():
    text = _text("skills/drocat-install/SKILL.md")
    assert "does NOT prompt for tokens" in text
    assert "asks for NeuPrint/CAVE tokens" not in text


def test_bootstrap_has_arm64_checksum_and_resilience_paths():
    sh = _text("archive/install/install.sh")
    ps1 = _text("archive/install/install.ps1")
    assert "miniconda3-drocat" in sh
    assert "miniconda3-drocat" in ps1
    assert "Resuming failed; retrying the download from scratch" in sh
    assert "Windows-arm64.exe" in ps1
    assert "PROCESSOR_ARCHITECTURE" in ps1
    assert "miniconda_checksums.txt" in sh


def test_launcher_repair_health_covers_core_and_checks_again():
    text = _text("mac_DROCAT.command")
    assert text.count("env_healthy") >= 3
    assert "is still unhealthy after the repair installer" in text
    assert "coana, statvis, morphology, route_plan" in text


def test_verifier_rejects_namespace_package_stubs(tmp_path, monkeypatch):
    import importlib.util

    path = ROOT / "skills/drocat-install/scripts/verify_install.py"
    spec = importlib.util.spec_from_file_location(
        "drocat_verify_namespace", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "stubpkg").mkdir()
    monkeypatch.chdir(tmp_path)
    result = module.run_probe(sys.executable, ["stubpkg", "json"])
    assert "namespace package" in result["stubpkg"]
    assert result["json"] == "ok"


@requires_bash
def test_miniconda_install_dir_avoids_a_broken_prefix(tmp_path):
    func = _shell_function(ROOT / "archive/install/install.sh",
                           "miniconda_install_dir")
    home = tmp_path / "home"
    home.mkdir()

    def resolve():
        script = (
            "set -euo pipefail\n"
            f"HOME='{home}'\n"
            + func
            + "\nminiconda_install_dir\n"
        )
        return subprocess.run(
            ["bash", "-c", script], check=True,
            capture_output=True, text=True).stdout.strip()

    assert resolve() == str(home / "miniconda3")
    (home / "miniconda3").mkdir()
    assert resolve() == str(home / "miniconda3-drocat")
    (home / "miniconda3" / "bin").mkdir()
    conda = home / "miniconda3" / "bin" / "conda"
    conda.write_text("#!/bin/sh\n")
    conda.chmod(0o755)
    assert resolve() == str(home / "miniconda3")
    conda.unlink()
    assert resolve() == str(home / "miniconda3-drocat")


@requires_bash
def test_miniconda_checksum_helper_rejects_wrong_bytes(tmp_path):
    helper = _shell_function(ROOT / "archive/install/install.sh",
                             "miniconda_checksum")
    sha = _shell_function(ROOT / "archive/install/install.sh", "sha256_file")
    verify = _shell_function(ROOT / "archive/install/install.sh",
                             "verify_miniconda_download")
    manifest = tmp_path / "checksums.txt"
    installer = tmp_path / "Miniconda3-test.sh"
    installer.write_bytes(b"expected installer bytes\n")
    import hashlib
    digest = hashlib.sha256(installer.read_bytes()).hexdigest()
    manifest.write_text(f"Miniconda3-test.sh {digest}\n", encoding="utf-8")
    script = (
        "set -euo pipefail\n"
        "RED=; NC=; command_exists() { command -v \"$1\" >/dev/null 2>&1; }\n"
        f"MINICONDA_CHECKSUMS='{manifest}'\n"
        + helper + "\n" + sha + "\n" + verify + "\n"
        f"verify_miniconda_download '{installer}' Miniconda3-test.sh\n"
    )
    subprocess.run(["bash", "-c", script], check=True)
    installer.write_bytes(b"tampered\n")
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert proc.returncode != 0
    assert "checksum mismatch" in (proc.stdout + proc.stderr)


def _fake_installer_repo(tmp_path):
    root = tmp_path / "root"
    repo = root / "repo"
    (repo / "archive/install").mkdir(parents=True)
    (repo / "ui").mkdir()
    (repo / "skills/drocat-install/scripts").mkdir(parents=True)
    import shutil
    shutil.copy(ROOT / "archive/install/install.sh",
                repo / "archive/install/install.sh")
    shutil.copy(ROOT / "archive/install/miniconda_checksums.txt",
                repo / "archive/install/miniconda_checksums.txt")
    (repo / "ui/config.py").write_text('APP_VERSION = "4.5.0"\n')
    (repo / "pyproject.toml").write_text('[project]\nversion = "4.5.0"\n')
    (repo / "config.json").write_text(
        '{\n  "tokens": {"neuprint": ""},\n  "envs": {"4.5.0": ""}\n}\n')
    (repo / "requirements.txt").write_text("numpy==1.0\n")
    (repo / "ui/requirements.txt").write_text("nicegui==1.0\n")
    verify = repo / "skills/drocat-install/scripts/verify_install.py"
    verify.write_text("#!/bin/sh\nexit 0\n")
    verify.chmod(0o755)
    (root / "envs/drocat-4.5.0/bin").mkdir(parents=True)
    (root / "bin").mkdir()
    (root / "bin/conda").write_text('''#!/bin/sh
if [ "$1" = "env" ] && [ "$2" = "list" ]; then
  echo "# conda environments:"; echo "base /base"; exit 0
fi
if [ "$1" = "run" ]; then
  shift
  while [ "$#" -gt 0 ]; do case "$1" in -n) shift 2;; --no-capture-output) shift;; *) break;; esac; done
  exec "$@"
fi
if [ "$1" = "create" ]; then echo "CREATE $*" >> "$FAKE_LOG"; exit 0; fi
if [ "$1" = "info" ]; then echo "/base"; exit 0; fi
exit 2
''')
    (root / "bin/python").write_text('''#!/bin/sh
if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "check" ]; then
  echo "PIP_CHECK" >> "$FAKE_LOG"
  if [ -n "$PIP_CHECK_FAIL_FIRST" ] && [ ! -f "$PIP_CHECK_MARK" ]; then
    : > "$PIP_CHECK_MARK"
    exit 1
  fi
  exit "${PIP_CHECK_RC:-0}"
fi
if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then
  echo "PIP $*" >> "$FAKE_LOG"; exit 0
fi
if [ "$1" = "-c" ]; then exec /usr/bin/env python3 "$@"; fi
echo "VERIFY $*" >> "$FAKE_LOG"; exit 0
''')
    (root / "bin/conda").chmod(0o755)
    (root / "bin/python").chmod(0o755)
    return root, repo


@requires_bash
def test_simulated_clean_install_reaches_verifier_and_retries_pip_check(tmp_path):
    root, repo = _fake_installer_repo(tmp_path)
    log = root / "log"
    env = {
        **__import__("os").environ,
        "PATH": f"{root / 'bin'}:{__import__('os').environ['PATH']}",
        "HOME": str(root), "FAKE_LOG": str(log),
    }
    result = subprocess.run(
        ["bash", str(repo / "archive/install/install.sh")],
        cwd=repo, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    lines = log.read_text().splitlines()
    assert "PIP_CHECK" in lines
    assert any(line.startswith("VERIFY ") for line in lines)
    log.write_text("")
    env["PIP_CHECK_RC"] = "1"
    result = subprocess.run(
        ["bash", str(repo / "archive/install/install.sh")],
        cwd=repo, env=env, capture_output=True, text=True)
    assert result.returncode == 1
    lines = log.read_text().splitlines()
    assert "--force-reinstall" in "\n".join(lines)
    assert "Verification failed" in result.stdout


@requires_bash
def test_simulated_clean_install_recovers_after_pip_check_failure(tmp_path):
    root, repo = _fake_installer_repo(tmp_path)
    log = root / "log"
    env = {
        **__import__("os").environ,
        "PATH": f"{root / 'bin'}:{__import__('os').environ['PATH']}",
        "HOME": str(root), "FAKE_LOG": str(log),
        "PIP_CHECK_FAIL_FIRST": "1",
        "PIP_CHECK_MARK": str(root / "pip-check-failed"),
    }
    result = subprocess.run(
        ["bash", str(repo / "archive/install/install.sh")],
        cwd=repo, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    lines = log.read_text().splitlines()
    assert lines.count("PIP_CHECK") == 2
    assert "--force-reinstall" in "\n".join(lines)
    assert any(line.startswith("VERIFY ") for line in lines)

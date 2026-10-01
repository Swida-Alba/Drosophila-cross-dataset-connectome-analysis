"""Guards for FlyWire skeleton-backed workflows."""

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.flywire_readiness import (  # noqa: E402
    FlyWireSkeletonAccessError,
    _configured_token_from_config,
    flywire_skeleton_readiness,
    print_download_instructions,
    require_flywire_skeleton_access,
)
from morphology import MorphologyComparer  # noqa: E402
from visualize_skeleton import VisualizeSkeleton  # noqa: E402


# =============================================================================
# Unified missing-local-files download instructions (FAFB and BANC)
# =============================================================================

class TestPrintDownloadInstructions:
    def _capture(self, dataset, tmp_path, capsys):
        print_download_instructions(dataset, tmp_path / "datasets" / dataset)
        return capsys.readouterr().out

    def test_fafb_instructions_are_canonical(self, tmp_path, capsys):
        text = self._capture("flywire_FAFB_v783", tmp_path, capsys)
        assert "ONE-TIME" in text
        assert "one-time" in text.lower()
        assert "https://codex.flywire.ai/api/download?dataset=fafb" in text
        assert "classification.csv.gz" in text
        assert "connections_princeton_no_threshold.csv.gz" in text
        assert "python src/FAFB_file_converter.py" in text
        # downloads folder of the dataset directory
        assert str(tmp_path / "datasets" / "flywire_FAFB_v783" / "downloads") in text
        # required-file hints stay FAFB-specific: BANC files never leak in
        assert "neurons.csv.gz" not in text
        assert "connections_princeton.csv.gz" not in text
        assert "python src/BANC_file_converter.py" not in text

    def test_banc_instructions_use_public_release_and_converter(self, tmp_path, capsys):
        text = self._capture("banc_v626", tmp_path, capsys)
        assert "public BANC" in text
        assert "banc_public_gcs" in text
        assert "https://codex.flywire.ai/api/download?dataset=banc" not in text
        assert "python src/BANC_file_converter.py" in text
        assert "python src/FAFB_file_converter.py" not in text
        # required-file hints stay BANC-specific: FAFB files never leak in
        assert "classification.csv.gz" not in text
        assert "connections_princeton_no_threshold.csv.gz" not in text
        assert "sk_lod1_783_healed.zip" not in text

    def test_dataset_folder_inside_instructions_matches_dataset(self, tmp_path, capsys):
        """The printed downloads folder carries the exact dataset identifier,
        never a generic path."""
        text = self._capture("banc_v888", tmp_path, capsys)
        assert "cache/banc_v888/skeletons/" in text
        assert "flywire_FAFB_v783" not in text

    def test_instructions_work_without_explicit_dataset_dir(self, capsys):
        print_download_instructions("flywire_FAFB_v783")
        text = capsys.readouterr().out
        assert "https://codex.flywire.ai/api/download?dataset=fafb" in text
        assert "python src/FAFB_file_converter.py" in text


# =============================================================================
# Guards for FlyWire skeleton-backed workflows
# =============================================================================

def test_banc_is_always_ready_via_public_bucket(tmp_path):
    """BANC skeletons fetch from the public bucket: never blocked, and the
    readiness payload never depends on a CAVE token."""
    log = []

    status = require_flywire_skeleton_access(
        "banc_v626", project_root=tmp_path, log=log.append
    )

    text = "\n".join(log)
    assert status["ready"] is True
    assert status["is_banc"] is True
    assert "banc_public_gcs" in text
    assert status["local_source"].startswith("banc_public_gcs")


def test_fafb_without_local_source_or_cave_token_is_blocked_with_instructions(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("CAVE_TOKEN", raising=False)
    monkeypatch.delenv("FLYWIRE_CODEX_TOKEN", raising=False)
    log = []

    with pytest.raises(FlyWireSkeletonAccessError, match="no local FAFB"):
        require_flywire_skeleton_access(
            "flywire_FAFB_v783", project_root=tmp_path, log=log.append
        )

    text = "\n".join(log)
    assert "sk_lod1_783_healed.zip" in text
    assert "python src/FAFB_file_converter.py" in text
    assert "CAVE_TOKEN" in text


def test_fafb_local_source_is_ready_without_cave_token(tmp_path, monkeypatch):
    monkeypatch.delenv("CAVE_TOKEN", raising=False)
    skeleton_zip = (
        tmp_path / "datasets" / "flywire_FAFB_v783" / "sk_lod1_783_healed.zip"
    )
    skeleton_zip.parent.mkdir(parents=True)
    skeleton_zip.write_bytes(b"placeholder")

    status = flywire_skeleton_readiness(
        "flywire_FAFB_v783", project_root=tmp_path
    )
    assert status["local_skeletons"] is True
    assert status["cave_token"] is False
    assert status["ready"] is True

    log = []
    require_flywire_skeleton_access(
        "flywire_FAFB_v783", project_root=tmp_path, log=log.append
    )
    assert "CAVE fallback is disabled" in "\n".join(log)


def test_fafb_cave_token_from_config_json_enables_fallback(tmp_path, monkeypatch):
    """A CAVE token in the project config.json counts as configured."""
    monkeypatch.delenv("CAVE_TOKEN", raising=False)
    (tmp_path / "config.json").write_text(
        '{"tokens": {"cave": "cfg-cave-token"}}\n', encoding="utf-8"
    )

    status = flywire_skeleton_readiness(
        "flywire_FAFB_v783", project_root=tmp_path
    )
    assert status["cave_token"] is True
    assert status["ready"] is True

    log = []
    require_flywire_skeleton_access(
        "flywire_FAFB_v783", project_root=tmp_path, log=log.append
    )
    assert "CAVE_TOKEN is configured" in "\n".join(log)
    assert "cfg-cave-token" not in "\n".join(log)


def test_fafb_cave_token_from_config_json_wins_over_config_local(tmp_path, monkeypatch):
    """config.json wins per key; config_local.json only fills empty entries."""
    monkeypatch.delenv("CAVE_TOKEN", raising=False)
    (tmp_path / "config.json").write_text(
        '{"tokens": {"cave": "cfg-cave-token"}}\n', encoding="utf-8"
    )
    (tmp_path / "config_local.json").write_text(
        '{"tokens": {"cave": "local-cave-token"}}\n', encoding="utf-8"
    )

    assert _configured_token_from_config(tmp_path, "cave") == "cfg-cave-token"
    status = flywire_skeleton_readiness(
        "flywire_FAFB_v783", project_root=tmp_path
    )
    assert status["cave_token"] is True


def test_fafb_cave_token_from_config_local_fills_empty_config_json(tmp_path, monkeypatch):
    """An empty config.json entry falls back to config_local.json."""
    monkeypatch.delenv("CAVE_TOKEN", raising=False)
    (tmp_path / "config.json").write_text(
        '{"tokens": {"cave": ""}}\n', encoding="utf-8"
    )
    (tmp_path / "config_local.json").write_text(
        '{"tokens": {"cave": "local-cave-token"}}\n', encoding="utf-8"
    )

    assert _configured_token_from_config(tmp_path, "cave") == "local-cave-token"
    status = flywire_skeleton_readiness(
        "flywire_FAFB_v783", project_root=tmp_path
    )
    assert status["cave_token"] is True


def test_fafb_local_source_keeps_cave_fallback_enabled_with_token(tmp_path, monkeypatch):
    monkeypatch.setenv("CAVE_TOKEN", "configured-token")
    skeleton_zip = (
        tmp_path / "datasets" / "flywire_FAFB_v783" / "sk_lod1_783_healed.zip"
    )
    skeleton_zip.parent.mkdir(parents=True)
    skeleton_zip.write_bytes(b"placeholder")

    log = []
    require_flywire_skeleton_access(
        "flywire_FAFB_v783", project_root=tmp_path, log=log.append
    )
    text = "\n".join(log)
    assert "CAVE API fallback is configured" in text
    assert "will be attempted" in text


def test_fafb_cave_token_is_an_online_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("CAVE_TOKEN", "secret-token-value")
    status = flywire_skeleton_readiness(
        "flywire_FAFB_v783", project_root=tmp_path
    )
    assert status["local_skeletons"] is False
    assert status["cave_token"] is True
    assert status["ready"] is True

    log = []
    require_flywire_skeleton_access(
        "flywire_FAFB_v783", project_root=tmp_path, log=log.append
    )
    text = "\n".join(log)
    assert "CAVE_TOKEN is configured" in text
    assert "secret-token-value" not in text


def test_morphology_query_guard_runs_at_script_entry(tmp_path, capsys):
    comparer = MorphologyComparer(
        query=1,
        dataset="banc_v626",
        project_root=str(tmp_path),
        verbose=True,
    )

    with pytest.raises(FlyWireSkeletonAccessError, match="deferred"):
        comparer.find_similar()


def test_skeleton_visualizer_banc_preparation_runs(tmp_path, capsys, monkeypatch):
    """BANC construction proceeds to one-time dataset preparation.

    The public-bucket fetch is stubbed offline so preparation fails
    deterministically: construction must exit with the bucket-first
    guidance instead of falling through to the FAFB converter.
    """
    import banc_public_data

    monkeypatch.setattr(
        banc_public_data, "download_meta_feather", lambda *a, **k: None)
    monkeypatch.setattr(
        banc_public_data, "download_connections_product", lambda *a, **k: None)

    with pytest.raises(RuntimeError):
        VisualizeSkeleton(
            dataset="banc_v626",
            neuron_layers=["1"],
            script_path=str(tmp_path),
            verbose=True,
        )

    text = capsys.readouterr().out
    assert "public BANC release bucket" in text
    assert "data preparation failed" in text


def test_fafb_plot_guard_runs_before_skeleton_query(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("CAVE_TOKEN", raising=False)
    monkeypatch.delenv("FLYWIRE_CODEX_TOKEN", raising=False)
    import FAFB_file_converter

    monkeypatch.setattr(
        FAFB_file_converter,
        "ensure_flywire_data",
        lambda dataset, dataset_dir: True,
    )
    visualizer = VisualizeSkeleton(
        dataset="flywire_FAFB_v783",
        neuron_layers=["1"],
        script_path=str(tmp_path),
        verbose=True,
    )

    with pytest.raises(FlyWireSkeletonAccessError, match="no local FAFB"):
        visualizer.plot_skeleton()

    text = capsys.readouterr().out
    assert "sk_lod1_783_healed.zip" in text
    assert "CAVE_TOKEN" in text


def test_fafb_codex_token_enables_lazy_access(tmp_path, monkeypatch):
    """Pristine tree + FlyWire Codex token: the guard accepts FAFB (lazy
    per-neuron fetch) instead of raising before the resolver can run."""
    monkeypatch.delenv("CAVE_TOKEN", raising=False)
    monkeypatch.delenv("FLYWIRE_CODEX_TOKEN", raising=False)
    (tmp_path / "config_local.json").write_text(
        '{"tokens": {"flywire_codex": "codex-token"}}\n', encoding="utf-8")

    status = flywire_skeleton_readiness(
        "flywire_FAFB_v783", project_root=tmp_path)
    assert status["flywire_codex"] is True
    assert status["cave_token"] is False
    assert status["local_skeletons"] is False
    assert status["ready"] is True

    log = []
    require_flywire_skeleton_access(
        "flywire_FAFB_v783", project_root=tmp_path, log=log.append
    )
    text = "\n".join(log)
    assert "lazily per neuron" in text
    assert "CAVE fallback is disabled" in text
    assert "codex-token" not in text


def test_fafb_codex_token_from_env_counts(tmp_path, monkeypatch):
    monkeypatch.delenv("CAVE_TOKEN", raising=False)
    monkeypatch.setenv("FLYWIRE_CODEX_TOKEN", "env-codex-token")
    status = flywire_skeleton_readiness(
        "flywire_FAFB_v783", project_root=tmp_path)
    assert status["flywire_codex"] is True
    assert status["ready"] is True

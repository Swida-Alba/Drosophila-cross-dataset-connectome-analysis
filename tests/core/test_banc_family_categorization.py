"""Standalone-BANC categorization regressions (post-audit fixes).

Each test pins a site that predated the modern ``banc_*`` naming and
mis-bucketed BANC — as NeuPrint, as FAFB, or into the FAFB-only CAVE online
paths (which fail silently for BANC):

- ``DatasetConfig`` must expose BANC as its own local source, not as FAFB;
- the homolog renderer's ``cache_neurons`` default must treat BANC like
  FAFB (the sibling call sites already do);
- the FAFB-vs-NeuPrint hemisphere warning must not fire for FAFB+BANC
  mixes (both local releases use the same hemisphere convention);
- coana's online fallbacks must never route BANC into the FAFB-only CAVE
  annotation/synapse tables: BANC metadata is local-table-only and its
  connections live only in the locally prepared merged table.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import coana as coana_mod  # noqa: E402
from comparison import ComparisonParameters  # noqa: E402
from comparison.dataset_config import DatasetConfig  # noqa: E402
from comparison.profile_comparator import HomologFinder  # noqa: E402


# ---------------------------------------------------------------------------
# DatasetConfig source classification
# ---------------------------------------------------------------------------

def test_dataset_config_classifies_banc_as_standalone_local_source():
    """BANC is local and non-NeuPrint, but is not the FAFB release."""
    cfg = DatasetConfig(dataset="banc_v888")
    assert cfg.is_flywire is False
    assert cfg.is_fafb is False
    assert cfg.is_banc is True
    assert cfg.is_local is True
    assert cfg.is_neuprint is False


def test_dataset_config_legacy_banc_name_and_neuprint_unchanged():
    assert DatasetConfig(dataset="flywire_BANC_v888").is_flywire is False
    assert DatasetConfig(dataset="flywire_BANC_v888").is_banc is True
    assert DatasetConfig(dataset="flywire_FAFB_v783").is_flywire is True
    assert DatasetConfig(dataset="male-cns:v1.0").is_flywire is False
    assert DatasetConfig(dataset="male-cns:v1.0").is_neuprint is True


# ---------------------------------------------------------------------------
# Homolog renderer cache_neurons default
# ---------------------------------------------------------------------------

def test_homolog_renderer_caches_neurons_for_banc(tmp_path):
    """The renderer's cache_neurons default keyed on an inline
    ``startswith('flywire_')`` test, so banc_v888 fell to the pipeline-based
    default (False under 'fast') while FAFB cached. The local-source predicate
    must decide."""
    finder = HomologFinder(output_dir=str(tmp_path), verbose=False)
    options = finder._homolog_visualizer_kwargs(
        {"dataset": "banc_v888", "neuprint_skeleton_pipeline": "fast"})
    assert options["cache_neurons"] is True


def test_homolog_renderer_neuprint_default_unchanged(tmp_path):
    finder = HomologFinder(output_dir=str(tmp_path), verbose=False)
    options = finder._homolog_visualizer_kwargs(
        {"dataset": "male-cns:v1.0", "neuprint_skeleton_pipeline": "fast"})
    assert options["cache_neurons"] is False


# ---------------------------------------------------------------------------
# FAFB hemisphere warning vs BANC
# ---------------------------------------------------------------------------

def test_hemisphere_warning_not_raised_for_fafb_banc_mix(capsys):
    """A FAFB+BANC mix must not print the FAFB-vs-NeuPrint hemisphere
    reversal warning."""
    ComparisonParameters(datasets=["flywire_FAFB_v783", "banc_v888"])
    out = capsys.readouterr().out
    assert "hemisphere labels are reversed" not in out


def test_hemisphere_warning_still_fires_for_fafb_neuprint_mix(capsys):
    ComparisonParameters(datasets=["flywire_FAFB_v783", "male-cns:v1.0"])
    assert "hemisphere labels are reversed" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# coana: BANC online fallbacks must not touch the FAFB-only CAVE paths
# ---------------------------------------------------------------------------

def _banc_connection(tmp_path, monkeypatch, **overrides):
    """A hermetic BANC FindNeuronConnection.

    Construction runs ``__post_init__`` -> ``_prepare_banc_data``; the stub
    keeps the test offline (production only ever constructs BANC with
    ``use_cache=True`` — the init guard rejects the online-only mode).
    ``script_path`` points at an empty tmp root, so every converted-table
    lookup misses and the fallback branches under test are reachable.
    """
    monkeypatch.setattr(coana_mod.FindNeuronConnection,
                        "_prepare_banc_data", lambda self: None)
    return coana_mod.FindNeuronConnection(
        sourceNeurons=["720575941416009108"],
        targetNeurons=["720575941596944935"],
        dataset="banc_v888",
        use_cache=True,
        verbose_mode="full",
        script_path=str(tmp_path),
        _warn_notes=[],
        custom_source_group_names=[],
        custom_target_group_names=[],
        kwargs_fetch={},
        **overrides,
    )


def _forbid_cave(fnc, monkeypatch):
    def no_cave():
        raise AssertionError("CAVE fetcher must not be built for BANC")

    def no_flywire_online(*args, **kwargs):
        raise AssertionError(
            "FAFB CAVE online fetch must not serve BANC metadata")

    monkeypatch.setattr(fnc, "_get_cave_fetcher", no_cave)
    monkeypatch.setattr(fnc, "_fetch_flywire_neurons_online",
                        no_flywire_online)


def test_banc_neuron_fetch_missing_table_skips_online_without_cave(
        tmp_path, monkeypatch, capsys):
    """Missing converted table: BANC must get the prepare-tables message
    and an empty frame, never the FAFB codex URL or a CAVE fetch."""
    fnc = _banc_connection(tmp_path, monkeypatch)
    _forbid_cave(fnc, monkeypatch)
    df = fnc._fetch_from_dataset_or_api(
        ["720575941416009108"], ["bodyId", "type"])
    assert df.empty
    out = capsys.readouterr().out
    assert "ensure_banc_data" in out
    assert "codex.flywire.ai" not in out


def test_banc_type_resolution_missing_table_skips_online_without_cave(
        tmp_path, monkeypatch, capsys):
    fnc = _banc_connection(tmp_path, monkeypatch)
    _forbid_cave(fnc, monkeypatch)
    df = fnc._fetch_neurons_by_types(["T1"], columns=["bodyId", "type"])
    assert df.empty
    out = capsys.readouterr().out
    assert "ensure_banc_data" in out
    assert "codex.flywire.ai" not in out


def test_banc_incoming_connections_raise_instead_of_cave(
        tmp_path, monkeypatch):
    """Target-rooted shortest discovery: with no local merged table, BANC
    must raise the clear prepare-tables error (the caller converts it into
    a visible warning note) instead of a CAVE fetch whose BANC config has
    no synapse view and which silently returned nothing."""
    fnc = _banc_connection(tmp_path, monkeypatch)
    _forbid_cave(fnc, monkeypatch)
    # The local-table attempt resolves from the real repo root; hide every
    # dataset dir so the fallback branch under test is reached.
    monkeypatch.setattr(coana_mod, "resolve_flywire_dataset_dir",
                        lambda script_path, dataset: None)
    with pytest.raises(RuntimeError, match="ensure_banc_data"):
        fnc._fetch_incoming_connections_online(["720575941596944935"])


def test_banc_metadata_enrichment_skips_online_fetch(
        tmp_path, monkeypatch, capsys):
    """Post-fetch index enrichment: when the locally prepared table is
    absent, BANC must skip the online fallback with the prepare-tables
    message instead of querying the FAFB-only CAVE annotation tables."""
    fnc = _banc_connection(tmp_path, monkeypatch)
    _forbid_cave(fnc, monkeypatch)
    # No converted table at the resolved dataset dir -> the enrichment
    # fallback branch is reachable.
    monkeypatch.setattr(
        coana_mod, "resolve_flywire_dataset_dir",
        lambda script_path, dataset: str(tmp_path / "datasets" / "banc_v888"))
    monkeypatch.setattr(fnc, "_load_neuron_index", lambda: pd.DataFrame())
    monkeypatch.setattr(fnc, "_save_neuron_index_state",
                        lambda *a, **k: None)
    fnc._update_neuron_index_after_fetch(
        pd.DataFrame(), upstream_bodyIds=["720575941416009108"],
        downstream_bodyIds=["1"])
    assert "no online metadata API for BANC" in capsys.readouterr().out


def test_banc_prepare_failure_does_not_suggest_cave(
        tmp_path, monkeypatch, capsys):
    """The download-failure advice is case-sensitive legacy: lowercase
    ``banc_v888`` used to be shown the FAFB 'use CAVE API' alternative
    that BANC can never use."""
    import BANC_file_converter as bfc

    fnc = _banc_connection(tmp_path, monkeypatch)
    # Route the prep call back onto the BANC-aware compatibility branch the
    # same way a legacy 'flywire' client_type configuration would.
    fnc.client_type = "flywire"
    monkeypatch.setattr(bfc, "ensure_banc_data",
                        lambda dataset, dataset_dir: False)
    with pytest.raises(RuntimeError):
        fnc._prepare_flywire_data()
    out = capsys.readouterr().out
    assert "use CAVE API" not in out

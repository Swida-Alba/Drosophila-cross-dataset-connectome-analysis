"""Targeted coverage tests for misc utility modules.

Covers the specific uncovered branches of:
- src/utils/token_manager.py
- src/utils/flywire_readiness.py
- src/roi_screening.py
- src/roi_sensitive_segmentation.py
- src/skeleton_simplification.py

Hermetic: no network, no real token/config files (everything is isolated
into tmp_path), synthetic small meshes/skeletons only.
"""

import builtins
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.utils.token_manager import TokenManager  # noqa: E402
from utils import flywire_readiness as fwr  # noqa: E402
import roi_screening as rois  # noqa: E402
import roi_sensitive_segmentation as rss  # noqa: E402
import skeleton_simplification as sks  # noqa: E402


# =============================================================================
# token_manager
# =============================================================================


class TestTokenManagerBranches:
    def test_corrupt_config_json_warns_and_loads_nothing(self, tmp_path,
                                                         monkeypatch, capsys):
        (tmp_path / "config.json").write_text("{not valid json", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        manager = TokenManager(project_root=str(tmp_path))
        assert manager.tokens == {}
        assert "Failed to read config.json" in capsys.readouterr().out

    def test_config_without_dict_tokens_section_is_ignored(self, tmp_path,
                                                           monkeypatch):
        # tokens section is a list, not a dict -> skipped silently
        (tmp_path / "config.json").write_text('{"tokens": ["x"]}\n',
                                              encoding="utf-8")
        # top level is not even a dict
        (tmp_path / "config_local.json").write_text('[1, 2]\n', encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        manager = TokenManager(project_root=str(tmp_path))
        assert manager.tokens == {}

    def test_non_string_config_values_are_skipped(self, tmp_path, monkeypatch):
        (tmp_path / "config.json").write_text(
            '{"tokens": {"neuprint": 42, "cave": "   "}}\n', encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        manager = TokenManager(project_root=str(tmp_path))
        assert manager.tokens == {}

    def test_detect_cave_and_long_non_hex_is_unknown(self):
        manager = TokenManager(project_root="/nonexistent-dir-for-isolation")
        assert manager.detect_token_type("0123456789abcdef0123456789abcdef") == "cave"
        # long but without '.' and not hex -> unknown
        assert manager.detect_token_type("z" * 150) == "unknown"
        # short non-hex -> unknown
        assert manager.detect_token_type("zz") == "unknown"

    def test_auto_token_direct_cave_fills_neuprint_from_files(self, tmp_path,
                                                              monkeypatch):
        (tmp_path / "config.json").write_text(
            '{"tokens": {"neuprint": "file-np"}}\n', encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)
        monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("CAVE_TOKEN", raising=False)
        # Keep the server probe hermetic: every token counts as accepted.
        monkeypatch.setattr(
            TokenManager, "neuprint_token_rejected",
            lambda self, token, server=None: False)
        manager = TokenManager(project_root=str(tmp_path))
        result = manager.get_auto_token(direct_input="0123456789abcdef0123456789abcdef")
        assert result["detected_type"] == "cave"
        assert result["cave"] == "0123456789abcdef0123456789abcdef"
        assert result["neuprint"] == "file-np"

    def test_auto_token_unknown_direct_input_honors_prefer_type(self, tmp_path,
                                                                monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)
        monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS", raising=False)
        # Unknown-format input is kept as-is while the server accepts it;
        # the probe stays stubbed so no real request is made.
        monkeypatch.setattr(
            TokenManager, "neuprint_token_rejected",
            lambda self, token, server=None: False)
        manager = TokenManager(project_root=str(tmp_path))
        unknown = "this-token-matches-no-format"
        res_np = manager.get_auto_token(direct_input=unknown, prefer_type="neuprint")
        assert res_np["neuprint"] == unknown
        assert res_np["cave"] is None
        assert res_np["detected_type"] == "unknown"
        res_cave = manager.get_auto_token(direct_input=unknown, prefer_type="cave")
        assert res_cave["cave"] == unknown
        assert res_cave["neuprint"] is None
        # no preference: neither slot is filled
        res_none = manager.get_auto_token(direct_input=unknown)
        assert res_none["neuprint"] is None
        assert res_none["cave"] is None

    def test_auto_token_unknown_rejected_direct_input_falls_to_chain(
            self, tmp_path, monkeypatch):
        """An unknown-format direct input the server refuses falls back to
        the config/env chain instead of failing later."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("NEUPRINT_TOKEN", "env-np")
        monkeypatch.setattr(
            TokenManager, "neuprint_token_rejected",
            lambda self, token, server=None: True)
        try:
            manager = TokenManager(project_root=str(tmp_path))
            result = manager.get_auto_token(
                direct_input="this-token-matches-no-format",
                prefer_type="neuprint")
            assert result["neuprint"] == "env-np"
        finally:
            monkeypatch.delenv("NEUPRINT_TOKEN")

    def test_require_both_tokens_success(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)
        monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("CAVE_TOKEN", raising=False)
        monkeypatch.setattr(
            TokenManager, "neuprint_token_rejected",
            lambda self, token, server=None: False)
        manager = TokenManager(project_root=str(tmp_path))
        manager._token_sources = {
            "NEUPRINT_TOKEN": [("config.json", "np-tok")],
            "CAVE_TOKEN": [("config.json", "cave-tok")],
        }
        result = manager.require_both_tokens()
        assert result["neuprint"] == "np-tok"
        assert result["cave"] == "cave-tok"


# =============================================================================
# flywire_readiness
# =============================================================================


class TestFlywireManualSkeletonInstruction:
    def test_fafb_instruction_with_default_dir(self):
        text = fwr.flywire_manual_skeleton_instruction("flywire_FAFB_v783")
        assert "sk_lod1_783_healed.zip" in text
        assert "python src/FAFB_file_converter.py" in text
        assert "https://codex.flywire.ai/api/download?dataset=fafb" in text
        assert "datasets/flywire_FAFB_v783/downloads" in text.replace("\\", "/")

    def test_banc_instruction_with_explicit_dir(self, tmp_path):
        text = fwr.flywire_manual_skeleton_instruction(
            "banc_v626", dataset_dir=tmp_path / "ds")
        # BANC skeletons need no manual download: they fetch on demand from
        # the public release bucket.
        assert "not needed" in text
        assert "banc_public_data.fetch_banc_swc" in text
        assert "banc_v626" in text


class TestCaveTokenFromConfigBranches:
    def test_top_level_non_dict_and_non_dict_tokens_section(self, tmp_path,
                                                            monkeypatch):
        monkeypatch.delenv("CAVE_TOKEN", raising=False)
        (tmp_path / "config.json").write_text('[1, 2, 3]\n', encoding="utf-8")
        (tmp_path / "config_local.json").write_text('{"tokens": "not-a-dict"}\n',
                                                    encoding="utf-8")
        assert fwr._configured_token_from_config(tmp_path, "cave") is None

    def test_placeholder_and_non_string_values_are_unconfigured(self, tmp_path,
                                                                monkeypatch):
        monkeypatch.delenv("CAVE_TOKEN", raising=False)
        (tmp_path / "config.json").write_text(
            '{"tokens": {"cave": "YOUR_CAVE_TOKEN"}}\n', encoding="utf-8")
        (tmp_path / "config_local.json").write_text(
            '{"tokens": {"cave": 1234}}\n', encoding="utf-8")
        assert fwr._configured_token_from_config(tmp_path, "cave") is None

    def test_env_placeholder_token_is_unconfigured(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CAVE_TOKEN", "YOUR_CAVE_TOKEN_HERE")
        assert fwr._configured_cave_token(tmp_path) is None


class TestFirstExisting:
    def test_dir_with_pkl_files_counts(self, tmp_path):
        populated = tmp_path / "cache"
        populated.mkdir()
        (populated / "a.pkl").write_bytes(b"x")
        assert fwr._first_existing([populated]) == populated

    def test_dir_with_pkl_zst_files_counts(self, tmp_path):
        populated = tmp_path / "cache"
        populated.mkdir()
        (populated / "a.pkl.zst").write_bytes(b"x")
        assert fwr._first_existing([populated]) == populated

    def test_dir_with_swc_cache_files_counts(self, tmp_path):
        populated = tmp_path / "cache"
        populated.mkdir()
        (populated / "a.swc.zst").write_bytes(b"x")
        assert fwr._first_existing([populated]) == populated

    def test_empty_dir_and_missing_paths_are_skipped(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert fwr._first_existing([tmp_path / "nope", empty]) is None

    def test_unreadable_dir_is_skipped_via_oserror(self, monkeypatch):
        class ExplodingPath:
            def is_file(self):
                return False

            def is_dir(self):
                return True

            def rglob(self, pattern):
                raise PermissionError("denied")

        assert fwr._first_existing([ExplodingPath()]) is None


class TestLocalFafbSkeletonSourceFallbacks:
    def test_parquet_source_is_found(self, tmp_path):
        folder = "flywire_FAFB_v783"
        ddir = tmp_path / "datasets" / folder
        ddir.mkdir(parents=True)
        pq = ddir / f"{folder}_skeletons.parquet"
        pq.write_bytes(b"placeholder")
        assert fwr.local_fafb_skeleton_source(folder, tmp_path) == pq

    def test_cache_skeletons_dir_with_pkls_is_found(self, tmp_path):
        folder = "flywire_FAFB_v783"
        cdir = tmp_path / "cache" / folder / "skeletons"
        cdir.mkdir(parents=True)
        (cdir / "one.pkl").write_bytes(b"x")
        assert fwr.local_fafb_skeleton_source(folder, tmp_path) == cdir

    def test_cache_skeletons_dir_with_cave_swc_is_found(self, tmp_path):
        folder = "flywire_FAFB_v783"
        cdir = tmp_path / "cache" / folder / "skeletons" / "cave_skeletons"
        cdir.mkdir(parents=True)
        (cdir / "one.swc.zst").write_bytes(b"x")
        assert fwr.local_fafb_skeleton_source(folder, tmp_path) == (
            tmp_path / "cache" / folder / "skeletons"
        )

    def test_glob_fallback_finds_noncanonical_skeleton_zip(self, tmp_path):
        folder = "flywire_FAFB_v783"
        ddir = tmp_path / "datasets" / folder
        ddir.mkdir(parents=True)
        odd = ddir / "my_custom_skeleton_bundle.zip"
        odd.write_bytes(b"x")
        assert fwr.local_fafb_skeleton_source(folder, tmp_path) == odd

    def test_glob_fallback_finds_noncanonical_skeleton_parquet(self, tmp_path):
        folder = "flywire_FAFB_v783"
        ddir = tmp_path / "datasets" / folder
        ddir.mkdir(parents=True)
        odd = ddir / "custom_skeletons_table.parquet"
        odd.write_bytes(b"x")
        assert fwr.local_fafb_skeleton_source(folder, tmp_path) == odd

    def test_nothing_available_returns_none(self, tmp_path):
        assert fwr.local_fafb_skeleton_source("flywire_FAFB_v783", tmp_path) is None

    def test_glob_oserror_is_tolerated(self, tmp_path, monkeypatch):
        folder = "flywire_FAFB_v783"
        (tmp_path / "datasets" / folder).mkdir(parents=True)
        monkeypatch.setattr(
            Path, "glob",
            lambda self, pattern: (_ for _ in ()).throw(OSError("denied")))
        assert fwr.local_fafb_skeleton_source(folder, tmp_path) is None


class TestReadinessGuardOtherBranches:
    def test_non_flywire_dataset_is_returned_untouched(self, tmp_path):
        log = []
        status = fwr.require_flywire_skeleton_access(
            "hemibrain:v1.2.1", project_root=tmp_path, log=log.append)
        assert status["is_banc"] is False
        assert status["is_fafb"] is False
        assert log == []

    def test_dataset_folder_normalizes_colons_and_dots(self):
        assert fwr.dataset_folder("hemibrain:v1.2.1") == "hemibrain_v1_2_1"
        assert fwr.dataset_folder(None) == ""

    def test_is_fafb_rejects_banc_names(self):
        assert fwr.is_fafb_dataset("banc_v626") is False
        assert fwr.is_fafb_dataset("flywire_FAFB_v783") is True
        assert fwr.is_banc_dataset(None) is False


# =============================================================================
# roi_screening
# =============================================================================

PRIMARY = ["A(L)", "A(R)", "M"]


def write_dataset(root, dataset, counts, metadata_rois=None, fmt="csv",
                  neuron_rows=None, write_sidecar=True, raw_roi_csv=None):
    """Minimal hermetic ROI dataset writer (mirrors the fixture convention
    of the existing roi_screening tests)."""
    folder = rois._dataset_folder(dataset)
    base = root / "datasets" / folder
    base.mkdir(parents=True, exist_ok=True)
    rows = []
    default_neuron_rows = []
    for bid, blocks in counts.items():
        pre_map = blocks.get("pre", {})
        post_map = blocks.get("post", {})
        for roi in set(pre_map) | set(post_map):
            rows.append({"bodyId": bid, "roi": roi,
                         "pre": pre_map.get(roi, 0),
                         "post": post_map.get(roi, 0)})
        default_neuron_rows.append({
            "bodyId": bid, "type": "", "instance": "",
            "pre": sum(pre_map.values()),
            "post": sum(post_map.values()),
        })
    table = pd.DataFrame(rows, columns=["bodyId", "roi", "pre", "post"])
    if raw_roi_csv is not None:
        (base / f"{folder}_allneurons_roi_count_df.csv").write_text(raw_roi_csv)
    elif fmt == "parquet":
        table.to_parquet(base / f"{folder}_allneurons_roi_count_df.parquet",
                         index=False)
    else:
        table.to_csv(base / f"{folder}_allneurons_roi_count_df.csv", index=False)
    pd.DataFrame(neuron_rows if neuron_rows is not None
                 else default_neuron_rows).to_csv(
        base / f"{folder}_allneurons_neuron_df.csv", index=False)
    if write_sidecar:
        rois_list = PRIMARY if metadata_rois is None else metadata_rois
        meta = {"dataset": dataset,
                "roi_coverage": {"roi_list": rois_list,
                                 "roi_count": len(rois_list)}}
        (base / f"{folder}_metadata.json").write_text(json.dumps(meta))
    return base


def twin_counts():
    return {
        1: {"post": {"A(R)": 10, "M": 5}, "pre": {"A(R)": 8, "M": 4}},
        2: {"post": {"A(L)": 10, "M": 5}, "pre": {"A(L)": 8, "M": 4}},
        3: {"post": {"A(R)": 10, "M": 50}, "pre": {"A(R)": 8, "M": 40}},
        4: {"post": {}, "pre": {}},
    }


class TestRoiScreeningSidecarBranches:
    def test_unreadable_sidecar_returns_none_and_logs(self, tmp_path):
        base = write_dataset(tmp_path, "np:v1", twin_counts())
        (base / "np_v1_metadata.json").write_text("{broken")
        log = []
        assert rois.load_primary_rois("np:v1", str(tmp_path), log=log.append) is None
        assert any("unreadable" in msg for msg in log)

    def test_sidecar_with_only_notprimary_entries_returns_none(self, tmp_path):
        write_dataset(tmp_path, "np:v1", twin_counts(),
                      metadata_rois=["NotPrimary", ""])
        assert rois.load_primary_rois("np:v1", str(tmp_path)) is None

    def test_build_raises_when_no_primary_rois_resolvable(self, tmp_path,
                                                          monkeypatch):
        write_dataset(tmp_path, "np:v1", twin_counts(),
                      metadata_rois=["NotPrimary"])
        monkeypatch.setattr(rois, "backfill_dataset_metadata",
                            lambda *a, **k: None)
        store = rois.RoiProfileStore("np:v1", project_root=str(tmp_path))
        with pytest.raises(rois.RoiScreeningUnavailable, match="No primary ROI"):
            store.build()


class TestRoiScreeningValidationBranches:
    def test_empty_roi_table_fails_validation(self, tmp_path):
        write_dataset(tmp_path, "np:v1", twin_counts(),
                      raw_roi_csv="bodyId,roi,pre,post\n")
        assert not rois.validate_primary_rois(PRIMARY, "np:v1", str(tmp_path))

    def test_missing_neuron_table_fails_validation(self, tmp_path):
        base = write_dataset(tmp_path, "np:v1", twin_counts())
        (base / "np_v1_allneurons_neuron_df.csv").unlink()
        assert not rois.validate_primary_rois(PRIMARY, "np:v1", str(tmp_path))

    def test_unreadable_table_fails_validation_with_log(self, tmp_path):
        # header lacks the required columns -> the polars read raises
        write_dataset(tmp_path, "np:v1", twin_counts(),
                      raw_roi_csv="foo,bar\n1,2\n")
        log = []
        ok = rois.validate_primary_rois(PRIMARY, "np:v1", str(tmp_path),
                                        log=log.append)
        assert ok is False
        assert any("partition validation failed" in m for m in log)

    def test_empty_neuron_table_fails_validation(self, tmp_path):
        base = write_dataset(tmp_path, "np:v1", twin_counts())
        (base / "np_v1_allneurons_neuron_df.csv").write_text(
            "bodyId,type,instance,pre,post\n")
        assert not rois.validate_primary_rois(PRIMARY, "np:v1", str(tmp_path))


class TestRoiScreeningBackfillBranches:
    def test_backfill_returns_none_when_tables_missing(self, tmp_path):
        assert rois.backfill_dataset_metadata("np:v1", str(tmp_path)) is None

    def test_backfill_returns_none_when_client_fails(self, tmp_path,
                                                     monkeypatch):
        write_dataset(tmp_path, "np:v1", twin_counts())
        import neuprint
        monkeypatch.setattr(neuprint, "Client",
                            lambda *a, **k: (_ for _ in ()).throw(
                                RuntimeError("offline")))
        log = []
        meta = rois.backfill_dataset_metadata("np:v1", str(tmp_path),
                                              log=log.append)
        assert meta is None
        assert any("backfill failed" in m for m in log)

    def test_backfill_token_exception_is_swallowed(self, tmp_path, monkeypatch):
        """CSV branch + token_manager failure inside backfill."""
        write_dataset(tmp_path, "np:v1", twin_counts())
        import statvis
        captured = {}

        def fake_build(dataset, neuron_df, roi_count_df, client=None):
            captured["n"] = len(neuron_df)
            return {"dataset": dataset,
                    "roi_coverage": {"roi_list": PRIMARY}}

        monkeypatch.setattr(statvis, "_build_dataset_metadata", fake_build)
        import neuprint
        monkeypatch.setattr(neuprint, "Client", lambda *a, **k: None)
        import utils.token_manager as tm_mod
        monkeypatch.setattr(tm_mod, "token_manager", None)  # .get_token -> AttributeError
        meta = rois.backfill_dataset_metadata("np:v1", str(tmp_path))
        assert meta is not None
        assert captured["n"] == 4

    def test_backfill_reads_parquet_and_logs(self, tmp_path, monkeypatch):
        write_dataset(tmp_path, "np:v1", twin_counts(), fmt="parquet")
        import statvis
        monkeypatch.setattr(
            statvis, "_build_dataset_metadata",
            lambda dataset, neuron_df, roi_count_df, client=None:
                {"dataset": dataset, "roi_coverage": {"roi_list": PRIMARY}})
        import neuprint
        monkeypatch.setattr(neuprint, "Client", lambda *a, **k: None)
        log = []
        meta = rois.backfill_dataset_metadata("np:v1", str(tmp_path),
                                              log=log.append)
        assert meta is not None
        assert any("sidecar saved" in m for m in log)


class TestRoiScreeningResolveAndBuildBranches:
    def test_failed_partition_refetch_recovers(self, tmp_path, monkeypatch):
        """Hierarchical sidecar -> refetch rewrites a valid sidecar."""
        counts = {
            1: {"post": {"A(L)": 10, "A(R)": 2, "A": 12},
                "pre": {"A(R)": 8, "A": 8}},
        }
        base = write_dataset(tmp_path, "np:v1", counts,
                             metadata_rois=["A", "A(L)", "A(R)", "M"],
                             neuron_rows=[{"bodyId": 1, "type": "",
                                           "instance": "", "pre": 8, "post": 12}])

        def fake_backfill(dataset, project_root=None, log=None):
            meta = {"dataset": dataset,
                    "roi_coverage": {"roi_list": ["A(L)", "A(R)"]}}
            (base / "np_v1_metadata.json").write_text(json.dumps(meta))
            return meta

        monkeypatch.setattr(rois, "backfill_dataset_metadata", fake_backfill)
        store = rois.RoiProfileStore("np:v1", project_root=str(tmp_path))
        store.build()
        assert store.rois == ["A(L)", "A(R)"]

    def test_absent_rois_are_reported_and_dropped(self, tmp_path):
        counts = {1: {"post": {"A(L)": 3, "A(R)": 5},
                      "pre": {"A(L)": 1, "A(R)": 2}}}
        write_dataset(tmp_path, "np:v1", counts,
                      metadata_rois=["A(L)", "A(R)", "GHOST"])
        log = []
        store = rois.RoiProfileStore("np:v1", project_root=str(tmp_path),
                                     log=log.append)
        store.build()
        assert store.rois == ["A(L)", "A(R)"]
        assert any("absent from the" in m for m in log)

    def test_no_kept_rois_raises(self, tmp_path):
        counts = {1: {"post": {"A(L)": 3}, "pre": {"A(L)": 1}}}
        write_dataset(tmp_path, "np:v1", counts, metadata_rois=["A(L)"])
        store = rois.RoiProfileStore("np:v1", project_root=str(tmp_path))
        import unittest.mock as mock
        # pretend the resolved primary list has no ROI present in the table
        with mock.patch.object(store, "_resolve_primary_rois",
                               return_value=["ZZZ(L)"]):
            with pytest.raises(rois.RoiScreeningUnavailable,
                               match="None of the primary ROIs"):
                store.build()

    def test_unreadable_cache_file_falls_back_to_rebuild(self, tmp_path):
        write_dataset(tmp_path, "np:v1", twin_counts())
        store = rois.RoiProfileStore("np:v1", project_root=str(tmp_path))
        store.ensure()
        cache = tmp_path / "cache" / "np_v1" / "morphology" / "roi_profiles.npz"
        cache.write_bytes(b"this is not an npz archive")
        reloaded = rois.RoiProfileStore("np:v1", project_root=str(tmp_path))
        assert reloaded.load() is False
        reloaded.ensure()  # rebuilds cleanly
        assert int(reloaded.screen([1]).iloc[0]["bodyId"]) == 2

    def test_fingerprint_tolerates_missing_files(self, tmp_path):
        store = rois.RoiProfileStore("np:v1", project_root=str(tmp_path))
        fp = store._fingerprint(PRIMARY)
        payload = json.loads(fp)
        assert payload["roi_csv"] is None
        assert payload["metadata"] is None

    def test_verbose_logging_prints(self, tmp_path, capsys):
        write_dataset(tmp_path, "np:v1", twin_counts())
        store = rois.RoiProfileStore("np:v1", project_root=str(tmp_path),
                                     verbose=True)
        store._log("hello")
        assert "hello" in capsys.readouterr().out


class TestRoiScreeningScreenBranches:
    def _built_store(self, tmp_path):
        write_dataset(tmp_path, "np:v1", twin_counts())
        store = rois.RoiProfileStore("np:v1", project_root=str(tmp_path))
        store.ensure()
        return store

    def test_screen_before_load_raises(self, tmp_path):
        store = rois.RoiProfileStore("np:v1", project_root=str(tmp_path))
        with pytest.raises(RuntimeError, match="not loaded"):
            store.screen([1])

    def test_unknown_query_bodyids_return_empty(self, tmp_path):
        store = self._built_store(tmp_path)
        res = store.screen([999999])
        assert res.empty
        assert list(res.columns) == ["bodyId", "roi_similarity"]

    def test_query_with_no_arbor_in_one_block(self, tmp_path):
        # query neuron has post synapses but zero pre -> the pre block is
        # skipped inside screen(); extra neurons keep the partition median
        # at 1.0 so validation passes.
        counts = {
            1: {"post": {"A(R)": 7}, "pre": {}},
            2: {"post": {"A(L)": 7}, "pre": {"A(L)": 3}},
            3: {"post": {"M": 2}, "pre": {"M": 2}},
            4: {"post": {"A(R)": 1}, "pre": {"A(R)": 1}},
            5: {"post": {"A(L)": 4}, "pre": {"A(L)": 4}},
        }
        write_dataset(tmp_path, "np:v1", counts)
        store = rois.RoiProfileStore("np:v1", project_root=str(tmp_path))
        store.ensure()
        res = store.screen([1])
        assert int(res.iloc[0]["bodyId"]) == 2
        assert res.iloc[0]["roi_similarity"] > 0.9

    def test_query_zero_vectors_return_empty(self, tmp_path):
        store = self._built_store(tmp_path)
        res = store.screen([4])  # the all-zero neuron
        assert res.empty

    def test_top_k_bounds_the_result(self, tmp_path):
        store = self._built_store(tmp_path)
        full = store.screen([1])
        top1 = store.screen([1], top_k=1)
        assert len(top1) == 1
        assert int(top1.iloc[0]["bodyId"]) == int(full.iloc[0]["bodyId"])
        # top_k larger than the candidate pool keeps everything
        assert len(store.screen([1], top_k=100)) == len(full)
        # top_k=0 falls back to the full ordering
        assert len(store.screen([1], top_k=0)) == len(full)

    def test_mirrored_query_reindexes_columns(self, tmp_path):
        store = self._built_store(tmp_path)
        store._ensure_normalized()
        q = store._post_n[0]
        qm = store._mirrored_query(q)
        # A(R) mass lands on the A(L) column; midline stays put
        assert qm.tolist() != q.tolist()
        assert qm.sum() == pytest.approx(q.sum(), rel=1e-6)

    def test_flip_roi(self):
        assert rois._flip_roi("A(L)") == "A(R)"
        assert rois._flip_roi("A(R)") == "A(L)"
        assert rois._flip_roi("M") == "M"

    def test_query_member_with_all_zero_rows_returns_empty(self, tmp_path):
        # a query that exists in the table but has no arbor in either
        # block -> both blocks skipped -> empty result (n_blocks == 0)
        counts = {
            1: {"post": {"A(R)": 3}, "pre": {"A(R)": 3}},
            2: {"post": {"A(L)": 4}, "pre": {"A(L)": 4}},
            3: {"post": {"M": 2}, "pre": {"M": 2}},
            4: {"post": {"A(R)": 0}, "pre": {"A(R)": 0}},
        }
        write_dataset(tmp_path, "np:v1", counts)
        store = rois.RoiProfileStore("np:v1", project_root=str(tmp_path))
        store.ensure()
        assert store.screen([4]).empty

    def test_all_candidates_unrankable_returns_empty(self, tmp_path):
        # the only non-query neuron has zero rows -> masked out -> no
        # finite score survives
        counts = {
            1: {"post": {"A(R)": 3}, "pre": {"A(R)": 3}},
            2: {"post": {"A(R)": 0}, "pre": {"A(R)": 0}},
        }
        write_dataset(
            tmp_path, "np:v1", counts,
            neuron_rows=[{"bodyId": 1, "type": "", "instance": "",
                          "pre": 3, "post": 3}])
        store = rois.RoiProfileStore("np:v1", project_root=str(tmp_path))
        store.ensure()
        assert store.screen([1]).empty


# =============================================================================
# roi_sensitive_segmentation
# =============================================================================


def cube(center=(0, 0, 0), extents=(2, 2, 2)):
    import trimesh

    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


class UnitCube:
    """Analytic [-1, 1]^3 mesh stub with its own contains()/distance.

    trimesh's native contains()/closest_point() need the optional rtree
    dependency (absent here), so the forced-fallback tests use this stub
    to exercise the non-open3d code paths deterministically.
    """

    bounds = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])

    def contains(self, points):
        points = np.asarray(points, dtype=float)
        return (np.abs(points) <= 1.0).all(axis=1)

    def surface_distance(self, points):
        points = np.asarray(points, dtype=float)
        return np.maximum(np.abs(points) - 1.0, 0.0).max(axis=1)


class TestMeshCoercionAndValidation:
    def test_empty_mesh_mapping_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            rss.classify_points_by_rois(np.zeros((1, 3)), {})

    def test_duplicate_mesh_names_raise(self):
        with pytest.raises(ValueError, match="unique"):
            rss.classify_points_by_rois(
                np.zeros((1, 3)), [("a", cube()), ("a", cube())])

    def test_mesh_without_contains_raises_typeerror(self):
        class NoContains:
            bounds = ((0, 0, 0), (1, 1, 1))

        with pytest.raises(TypeError, match="trimesh-compatible"):
            rss.classify_points_by_rois(np.zeros((1, 3)), {"x": NoContains()})

    def test_mesh_loaded_from_file_path(self, tmp_path):
        mesh = cube()
        path = tmp_path / "cube.stl"
        mesh.export(str(path))
        res = rss.classify_points_by_rois(
            pd.DataFrame({"x": [0.0], "y": [0.0], "z": [0.0]}),
            {"box": path}, overlap="first")
        assert res["roi"].tolist() == ["box"]
        assert res["inside"].tolist() == [True]

    def test_navis_volume_is_coerced_via_trimesh(self):
        import navis

        volume = navis.Volume(cube(), name="v")
        res = rss.classify_points_by_rois(np.array([[0.0, 0.0, 0.0]]),
                                          {"v": volume}, overlap="first")
        assert res["inside"].tolist() == [True]

    def test_object_exposing_trimesh_attribute_is_unwrapped(self):
        class VolumeLike:
            def __init__(self, inner):
                self.trimesh = inner

        res = rss.classify_points_by_rois(np.array([[0.0, 0.0, 0.0]]),
                                          {"v": VolumeLike(cube())},
                                          overlap="first")
        assert res["inside"].tolist() == [True]


class TestPointsArrayValidation:
    def test_missing_coordinate_columns_raise(self):
        with pytest.raises(ValueError, match="Missing coordinate columns"):
            rss.classify_points_by_rois(pd.DataFrame({"x": [0.0]}),
                                        {"box": cube()})

    def test_bad_array_shape_raises(self):
        with pytest.raises(ValueError, match=r"\(n, 3\)"):
            rss.classify_points_by_rois(np.zeros((2, 2)), {"box": cube()})

    def test_non_finite_points_raise(self):
        with pytest.raises(ValueError, match="finite"):
            rss.classify_points_by_rois(np.array([[np.nan, 0.0, 0.0]]),
                                        {"box": cube()})


class TestContainmentBackends:
    def test_backend_name_is_known(self):
        assert rss.containment_backend() in {
            "open3d_raycasting_scene", "trimesh_ray_triangle"}

    def test_open3d_import_failure_falls_back_to_trimesh(self, monkeypatch):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "open3d" or name.startswith("open3d."):
                raise ImportError("no open3d in this test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert rss.containment_backend() == "trimesh_ray_triangle"
        res = rss.classify_points_by_rois(
            np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]]),
            {"box": UnitCube()}, overlap="first")
        assert res["inside"].tolist() == [True, False]

    def test_open3d_scene_build_failure_falls_back(self, monkeypatch):
        # keep open3d importable but sabotage its legacy mesh constructor
        try:
            import open3d
        except ImportError:
            pytest.skip("open3d not installed")
        rss._OPEN3D_SCENE_CACHE.clear()

        def boom(*a, **k):
            raise RuntimeError("cannot build scene")

        monkeypatch.setattr(open3d.geometry, "TriangleMesh", boom)
        res = rss.classify_points_by_rois(
            np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]]),
            {"box": UnitCube()}, overlap="first")
        assert res["inside"].tolist() == [True, False]

    def test_scene_cache_hit(self):
        mesh = cube()
        rss._OPEN3D_SCENE_CACHE.clear()
        o3d_first, scene_first = rss._open3d_scene(mesh)
        o3d_second, scene_second = rss._open3d_scene(mesh)
        assert scene_first is scene_second
        if scene_first is not None:
            assert o3d_second is o3d_first

    def test_contains_empty_points(self):
        assert rss._contains(cube(), np.zeros((0, 3))).shape == (0,)

    def test_contains_bad_bounds_keeps_all_candidates(self):
        rss._OPEN3D_SCENE_CACHE.clear()
        inner = UnitCube()

        # wrapper exposing broken bounds but a working contains()
        class Wrapped:
            @property
            def bounds(self):
                raise RuntimeError("no bounds")

            def contains(self, points):
                return inner.contains(points)

        res = rss._contains(Wrapped(), np.array([[0.0, 0.0, 0.0],
                                                 [50.0, 0.0, 0.0]]))
        assert res.tolist() == [True, False]

    def test_contains_failure_raises_runtime_error(self):
        rss._OPEN3D_SCENE_CACHE.clear()

        class Broken:
            bounds = ((0, 0, 0), (1, 1, 1))

            def contains(self, points):
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="point-in-mesh"):
            rss._contains(Broken(), np.array([[0.5, 0.5, 0.5]]))


class TestSurfaceDistanceBranches:
    def test_nearest_roi_assignments_empty_points(self):
        res = rss._nearest_roi_assignments(np.zeros((0, 3)),
                                           [("a", cube())])
        assert res.empty
        assert list(res.columns) == ["roi", "distance"]

    def test_surface_distance_empty_points(self):
        assert rss._surface_distance(cube(), np.zeros((0, 3))).shape == (0,)

    def test_trimesh_fallback_distance(self, monkeypatch):
        # force the non-open3D branch of _surface_distance; rtree is absent
        # here, so stub trimesh's closest_point with an analytic result
        monkeypatch.setattr(rss, "_open3d_scene", lambda mesh: (None, None))
        import trimesh.proximity

        def fake_closest_point(mesh, points):
            points = np.asarray(points, dtype=float)
            dist = np.maximum(np.abs(points) - 1.0, 0.0).max(axis=1)
            return None, dist, None

        monkeypatch.setattr(trimesh.proximity, "closest_point",
                            fake_closest_point)
        distances = rss._surface_distance(UnitCube(), np.array([[2.0, 0.0, 0.0]]))
        assert distances[0] == pytest.approx(1.0, abs=1e-6)

    def test_trimesh_fallback_failure_raises(self, monkeypatch):
        monkeypatch.setattr(rss, "_open3d_scene", lambda mesh: (None, None))

        class NoProximity:
            pass

        with pytest.raises(RuntimeError, match="distance calculation failed"):
            rss._surface_distance(NoProximity(), np.array([[0.0, 0.0, 0.0]]))

    def test_open3d_distance_failure_raises(self, monkeypatch):
        try:
            import open3d
        except ImportError:
            pytest.skip("open3d not installed")

        class FakeScene:
            def compute_distance(self, tensor):
                raise RuntimeError("boom")

        monkeypatch.setattr(rss, "_open3d_scene",
                            lambda mesh: (open3d, FakeScene()))
        with pytest.raises(RuntimeError, match="distance calculation failed"):
            rss._surface_distance(cube(), np.array([[0.0, 0.0, 0.0]]))


class TestSnapOptionValidation:
    def test_negative_max_snap_distance_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            rss._validate_snap_options(True, -1.0)

    def test_non_finite_max_snap_distance_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            rss._validate_snap_options(True, float("nan"))

    def test_max_snap_distance_requires_snap_outside(self):
        with pytest.raises(ValueError, match="snap_outside"):
            rss._validate_snap_options(False, 1.0)


class TestSegmentSynapsesBranches:
    def test_non_dataframe_synapses_raise(self):
        with pytest.raises(TypeError, match="DataFrame"):
            rss.segment_synapses([(0, 0, 0)], {"box": cube()})

    def test_empty_synapses_without_snap(self):
        empty = pd.DataFrame({"x": [], "y": [], "z": []})
        res = rss.segment_synapses(empty, {"box": cube()})
        assert res.empty
        assert {"point_index", "derived_roi", "inside"}.issubset(res.columns)

    def test_empty_synapses_with_snap(self):
        empty = pd.DataFrame({"x": [], "y": [], "z": []})
        res = rss.segment_synapses(empty, {"box": cube()}, snap_outside=True)
        assert res.empty
        assert "snapped_roi" in res.columns

    def test_all_inside_synapses_skip_snapping_lookup(self):
        syn = pd.DataFrame({"x": [0.0], "y": [0.0], "z": [0.0]})
        res = rss.segment_synapses(syn, {"box": cube()}, snap_outside=True)
        assert res["was_snapped"].tolist() == [False]
        assert res["nearest_roi"].tolist() == [None]

    def test_overlap_invalid_value_raises(self):
        syn = pd.DataFrame({"x": [0.0], "y": [0.0], "z": [0.0]})
        with pytest.raises(ValueError, match="overlap"):
            rss.classify_points_by_rois(syn, {"box": cube()}, overlap="bogus")


class TestSkeletonNodeBranches:
    def test_invalid_skeleton_type_raises(self):
        with pytest.raises(TypeError, match="TreeNeuron"):
            rss.segment_skeleton("not-a-skeleton", {"box": cube()})

    def test_treeneuron_like_object_is_accepted(self):
        import navis

        nodes = pd.DataFrame({
            "node_id": [0, 1], "parent_id": [-1, 0],
            "x": [0.0, 0.5], "y": [0.0, 0.0], "z": [0.0, 0.0],
            "radius": [1.0, 1.0],
        })
        neuron = navis.TreeNeuron(nodes)
        result = rss.segment_skeleton(neuron, {"box": cube()},
                                      overlap="first", segment_samples=2)
        assert len(result.nodes) == 2
        assert result.nodes["derived_roi"].tolist() == ["box", "box"]

    def test_non_finite_skeleton_coordinates_raise(self):
        skeleton = pd.DataFrame({
            "node_id": [0], "parent_id": [-1],
            "x": [np.nan], "y": [0.0], "z": [0.0],
        })
        with pytest.raises(ValueError, match="finite"):
            rss.segment_skeleton(skeleton, {"box": cube()})

    def test_missing_y_column_raises(self):
        skeleton = pd.DataFrame({
            "node_id": [0], "parent_id": [-1], "x": [0.0], "z": [0.0],
        })
        with pytest.raises(ValueError, match="y"):
            rss.segment_skeleton(skeleton, {"box": cube()})

    def test_find_column_error_message(self):
        with pytest.raises(ValueError, match="tried"):
            rss._find_column(pd.DataFrame({"a": [1]}), ("b", "c"), "parent ID")


class TestEdgeTableAndSegments:
    def test_invalid_parent_edges_are_skipped(self):
        skeleton = pd.DataFrame({
            "node_id": [0, 1, 2, 3, 4],
            # 99 is missing, 3 is a self-loop, NaN parent is no edge
            "parent_id": [-1, 0, 99, 3, None],
            "x": [0.0, 0.5, 0.6, 0.7, 0.8],
            "y": [0.0, 0.0, 0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0, 0.0, 0.0],
        })
        nodes = rss._skeleton_nodes(skeleton)
        edges = rss._edge_table(nodes)
        assert edges["node_id"].tolist() == [1]

    def test_single_node_skeleton_has_empty_segments(self):
        skeleton = pd.DataFrame({
            "node_id": [0], "parent_id": [-1],
            "x": [0.0], "y": [0.0], "z": [0.0],
        })
        result = rss.segment_skeleton(skeleton, {"box": cube()},
                                      snap_outside=True, segment_samples=3)
        assert result.segments.empty
        assert result.samples.empty
        assert "snapped_roi" in result.samples.columns
        assert len(result.nodes) == 1

    def test_overlap_error_mode_rejects_crossing_segments(self):
        # Nodes sit in exactly one cube each, but the edge's midpoint
        # samples land inside both overlapping cubes.
        skeleton = pd.DataFrame({
            "node_id": [0, 1], "parent_id": [-1, 0],
            "x": [-0.9, 1.1], "y": [0.0, 0.0], "z": [0.0, 0.0],
        })
        meshes = {"a": cube(), "b": cube(center=(0.2, 0, 0))}
        with pytest.raises(ValueError, match="overlapping ROI meshes"):
            rss.segment_skeleton(skeleton, meshes, overlap="error",
                                 segment_samples=1)

    def test_overlap_all_mode_counts_every_containing_roi(self):
        skeleton = pd.DataFrame({
            "node_id": [0, 1], "parent_id": [-1, 0],
            "x": [-2.0, 2.0], "y": [0.0, 0.0], "z": [0.0, 0.0],
        })
        result = rss.segment_skeleton(skeleton, {"box": cube()},
                                      overlap="all", segment_samples=4,
                                      outside_label=None)
        # edge crosses the box: both inside and (no outside label) rows only
        rois_seen = set(result.segments["derived_roi"])
        assert rois_seen == {"box"}
        # outside samples are omitted entirely when outside_label=None
        assert rss.OUTSIDE_ROI not in set(result.samples["derived_roi"])
        # overlap="all" sample rows use the flatnonzero branch
        assert len(result.samples) > 0

    def test_segment_skeleton_invalid_overlap_raises(self):
        skeleton = pd.DataFrame({
            "node_id": [0], "parent_id": [-1],
            "x": [0.0], "y": [0.0], "z": [0.0],
        })
        with pytest.raises(ValueError, match="overlap"):
            rss.segment_skeleton(skeleton, {"box": cube()}, overlap="bogus")


# =============================================================================
# skeleton_simplification
# =============================================================================


def make_chain(n_nodes=2000):
    import navis

    types = ["root"] + ["slab"] * max(0, n_nodes - 2) + ["end"]
    nodes = pd.DataFrame({
        "node_id": np.arange(n_nodes, dtype=np.int64),
        "parent_id": np.array([-1] + list(range(n_nodes - 1)), dtype=np.int64),
        "x": np.arange(n_nodes, dtype=float) * 1000.0,
        "y": np.zeros(n_nodes),
        "z": np.zeros(n_nodes),
        "radius": np.ones(n_nodes),
        "type": types[:n_nodes],
    })
    nrn = navis.TreeNeuron(nodes)
    nrn.soma = None
    return nrn


class TestSomaPreservation:
    def test_soma_node_ids_are_parsed(self):
        class WithSoma:
            soma = [5, None, "10"]

        assert sks._neuron_node_ids(WithSoma()) == [5, 10]

    def test_unparseable_soma_entries_are_skipped(self):
        class WithNan:
            soma = [np.nan]  # int(np.nan) -> ValueError -> swallowed

        assert sks._neuron_node_ids(WithNan()) == []

    def test_non_iterable_soma_is_ignored(self):
        class Scalar:
            soma = 5  # iterating an int -> TypeError -> swallowed

        assert sks._neuron_node_ids(Scalar()) == []

    def test_none_soma_yields_no_extra_ids(self):
        class NoSoma:
            soma = None

        assert sks._neuron_node_ids(NoSoma()) == []

    def test_soma_ids_flow_into_simplification(self):
        neuron = make_chain(2000)
        # navis' soma setter only accepts scalars; the module merely reads
        # the attribute, so assign the store directly
        neuron._soma = [37]
        simplified, stats = sks.simplify_skeleton_nodes(neuron, 0.90)
        kept = set(simplified.nodes.node_id.values)
        assert 37 in kept
        assert stats["achieved_nodes"] == simplified.n_nodes


class TestFactorSearchExpansion:
    def _run_with_fake_downsample(self, neuron, target, count_fn):
        """Replace navis.downsample_neuron with a deterministic
        count-of-factor model to drive the bracket-expansion branches."""
        import navis

        calls = []

        class Fake:
            def __init__(self, n):
                self.n_nodes = n

        def fake_downsample(nrn, downsampling_factor, inplace, preserve_nodes):
            calls.append(downsampling_factor)
            return Fake(count_fn(downsampling_factor))

        reduction = 1.0 - target / neuron.n_nodes
        original = navis.downsample_neuron
        navis.downsample_neuron = fake_downsample
        try:
            simplified, stats = sks.simplify_skeleton_nodes(neuron, reduction)
        finally:
            navis.downsample_neuron = original
        return simplified, stats, calls

    def test_low_side_expansion(self):
        neuron = make_chain(2000)
        # target 300: the initial estimate undershoots the count, and even
        # the first low probe stays below target -> the search expands the
        # low side toward FACTOR_MIN.
        count_fn = lambda f: max(100, int(400 / max(f, 1e-9)))
        simplified, stats, calls = self._run_with_fake_downsample(
            neuron, 300, count_fn)
        assert stats["target_nodes"] == 300
        # expansion probed all the way down to FACTOR_MIN
        assert any(c <= 1.001 for c in calls)
        # the bracket search recovered a count close to the target
        assert 250 <= stats["achieved_nodes"] <= 300

    def test_high_side_expansion(self):
        neuron = make_chain(2000)
        # target 500: a topology floor of 900 keeps every count above
        # target, so the high side expands repeatedly toward FACTOR_MAX.
        count_fn = lambda f: max(900, 2000 - int(f) * 10)
        simplified, stats, calls = self._run_with_fake_downsample(
            neuron, 500, count_fn)
        assert stats["target_nodes"] == 500
        # expansion pushed the factor all the way to the safety clamp
        assert max(calls) == sks.FACTOR_MAX
        assert stats["achieved_nodes"] == count_fn(2000 / 500 - 1.0)

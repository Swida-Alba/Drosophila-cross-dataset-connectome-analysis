"""Offline tests for the public BANC data layer and converter paths.

All network access is mocked: :mod:`banc_public_data.http_get` is
monkeypatched, so these tests run without connectivity.
"""
import io
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import banc_public_data as bpd


LSWC = (b"# DROCAT test swc\n"
        b"1 0 0 0 0 2 -1\n"
        b"2 0 100 0 0 2 1\n")
LSWC_FULL = (b"# full-res stand-in\n"
             b"1 0 0 0 0 3 -1\n"
             b"2 0 50 0 0 3 1\n"
             b"3 0 100 0 0 3 2\n")
LSWC_FULL_FULL = LSWC_FULL


@pytest.fixture(autouse=True)
def _reset_crosswalk_state():
    """The failure set and memo are process-global: isolate every test."""
    for state in (bpd._CROSSWALK_FAILURES, bpd._CROSSWALK_CACHE):
        state.clear()
    yield
    for state in (bpd._CROSSWALK_FAILURES, bpd._CROSSWALK_CACHE):
        state.clear()


@pytest.fixture
def fake_net(monkeypatch):
    """Route http_get to a canned response map (url -> bytes or None)."""
    calls = []

    def _install(responses):
        def fake_get(url, timeout=120, attempts=3, note="",
                     progress_callback=None):
            calls.append(url)
            for pattern, payload in responses.items():
                if pattern in url:
                    if progress_callback is not None and payload is not None:
                        progress_callback(len(payload), len(payload))
                    return payload
            return None
        monkeypatch.setattr(bpd, "http_get", fake_get)
        return calls

    return _install


class TestFetchBancSwc:
    def test_requires_banc_dataset(self, tmp_path):
        with pytest.raises(ValueError):
            bpd.fetch_banc_swc("male-cns:v1.0", "123", project_root=tmp_path)

    def test_resolution_validation(self, tmp_path):
        with pytest.raises(ValueError):
            bpd.fetch_banc_swc("banc_v888", "123", resolution="coarse",
                               project_root=tmp_path)

    def test_l2_fetch_then_cache_roundtrip(self, tmp_path, fake_net):
        calls = fake_net({SWC_URL: LSWC})
        neuron = bpd.fetch_banc_swc("banc_v888", ID888,
                                    project_root=tmp_path)
        assert neuron is not None
        assert len(neuron.nodes) == 2
        assert "nanometer" in str(neuron.units)
        assert neuron.id == int(ID888)
        # Exactly one HTTP call for the preferred suffix.
        assert len(calls) == 1 and "_l2.swc" in calls[0]

        # Second fetch is served from the .swc.zst cache (no network).
        calls.clear()
        cached = bpd.fetch_banc_swc("banc_v888", ID888, project_root=tmp_path)
        assert cached is not None and len(cached.nodes) == 2
        assert calls == []
        cache_file = (tmp_path / "cache" / "banc_v888" / "skeletons"
                      / "raw_skeletons" / f"{ID888}.swc.zst")
        assert cache_file.exists()
        # Cached entry records level 0 + bucket provenance.
        import zstandard as zstd
        blob = b"".join(zstd.ZstdDecompressor().read_to_iter(
            cache_file.read_bytes()))
        head = b"\n".join(blob.split(b"\n")[:3]).decode()
        assert "# DROCAT simpl: 0" in head
        assert "banc_gcs_l2" in head

    def test_404_falls_back_to_other_resolution(self, tmp_path, fake_net):
        # Preferred full-res 404s; the L2 fallback serves.
        fake_net({
            "_skeleton.swc": None,
            SWC_URL: LSWC,
        })
        neuron = bpd.fetch_banc_swc("banc_v888", ID888, resolution="full",
                                    project_root=tmp_path)
        assert neuron is not None and len(neuron.nodes) == 2

    def test_missing_everywhere_returns_none(self, tmp_path, fake_net):
        fake_net({SWC_URL: None})
        assert bpd.fetch_banc_swc("banc_v888", "123", project_root=tmp_path) is None

    def test_v626_uses_crosswalk_then_pcg_um_fallback(self, tmp_path,
                                                      fake_net, monkeypatch):
        # 888-name SWCs missing entirely; crosswalk maps the v626 id.
        fake_net({
            SWC_URL: None,
            PCG_URL: LSWC,  # micrometre fallback
        })
        monkeypatch.setattr(
            bpd, "get_id_crosswalk",
            lambda dataset, project_root=None, force_refresh=False: {
                ID626: ID888})
        neuron = bpd.fetch_banc_swc("banc_v626", ID626, project_root=tmp_path)
        assert neuron is not None and len(neuron.nodes) == 2
        # The pcg-skel product is micrometre: coordinates scale x1000 to nm.
        coords = neuron.nodes[["x", "y", "z"]].to_numpy()
        assert (coords == [[0.0, 0.0, 0.0], [100000.0, 0.0, 0.0]]).all()
        # R1: the cache must hold the SCALED coordinates, not the raw µm
        # download — a cache hit must not shrink the neuron back.
        cached = bpd.fetch_banc_swc("banc_v626", ID626, project_root=tmp_path)
        assert cached is not None
        cached_coords = cached.nodes[["x", "y", "z"]].to_numpy()
        assert (cached_coords == coords).all()

    def test_unified_chain_serves_full_when_l2_missing(self, tmp_path,
                                                       fake_net):
        # Unified chain (L2 -> full -> pcg): the neuron has no L2 file, so
        # the full-res fallback serves regardless of the (ignored)
        # resolution argument.
        calls = fake_net({
            f"{SWC_URL}/{ID888}_skeleton.swc": LSWC_FULL,
        })
        neuron = bpd.fetch_banc_swc("banc_v888", ID888, resolution="full",
                                    project_root=tmp_path)
        assert neuron._drocat_banc_resolution == "full"
        # A later request (any resolution) is satisfied by the cached entry
        # without touching the network.
        calls.clear()
        neuron2 = bpd.fetch_banc_swc("banc_v888", ID888, resolution="l2",
                                     project_root=tmp_path)
        assert neuron2 is not None
        assert calls == []

    def test_full_cache_hit_after_reload(self, tmp_path, fake_net):
        """R2 regression: the provenance header must survive a cache reload.

        Every fetch re-loads the entry from disk, so the second fetch
        exercises the exact path that used to lose
        ``_drocat_banc_resolution`` and re-download the neuron forever.
        """
        calls = fake_net({
            f"{SWC_URL}/{ID888}_skeleton.swc": LSWC_FULL,
        })
        first = bpd.fetch_banc_swc("banc_v888", ID888, resolution="full",
                                   project_root=tmp_path)
        assert first._drocat_banc_resolution == "full"
        # Cache reload: no network, resolution restored from the header.
        calls.clear()
        second = bpd.fetch_banc_swc("banc_v888", ID888, resolution="full",
                                    project_root=tmp_path)
        assert calls == []
        assert second is not None
        assert second._drocat_banc_resolution == "full"

    def test_l2_cache_reload_keeps_l2_provenance(self, tmp_path, fake_net):
        calls = fake_net({SWC_URL: LSWC})
        first = bpd.fetch_banc_swc("banc_v888", ID888,
                                   project_root=tmp_path)
        assert first._drocat_banc_resolution == "l2"
        calls.clear()
        again = bpd.fetch_banc_swc("banc_v888", ID888,
                                   project_root=tmp_path)
        assert calls == []
        assert again._drocat_banc_resolution == "l2"

    def test_resolution_header_parser(self):
        parse = bpd._resolution_from_source_header
        assert parse("# DROCAT simpl: 0\n# DROCAT source: banc_gcs_full\n"
                     "1 0 0 0 0 2 -1\n") == "full"
        assert parse("# DROCAT simpl: 0\n# DROCAT source: banc_gcs_l2\n"
                     "1 0 0 0 0 2 -1\n") == "l2"
        assert parse("# DROCAT simpl: 0\n"
                     "# DROCAT source: banc_gcs_pcg_um_x1000\n"
                     "1 0 0 0 0 2 -1\n") == "l2"
        # Headerless (legacy) entries read as the coarse default.
        assert parse("1 0 0 0 0 2 -1\n") == "l2"


SWC_URL = f"{bpd.BUCKET_HTTP_BASE}/{bpd.SWC_DIR}"
PCG_URL = f"{bpd.BUCKET_HTTP_BASE}/{bpd.PCG_SKEL_DIR}"
ID888 = "720575941442818239"
ID626 = "720575941595794471"


class TestUnifiedChainIgnoresResolution:
    def test_cached_l2_satisfies_full_request(self, tmp_path, monkeypatch):
        # Source selection is removed: once an L2 entry is cached, even a
        # 'full' request is served from it (the chain is always
        # L2 -> full -> pcg; 'resolution' is accepted and ignored).
        def fake_get(url, timeout=120, attempts=3, note="",
                     progress_callback=None):
            if url.endswith("_l2.swc"):
                return LSWC
            if url.endswith("_skeleton.swc"):
                return LSWC_FULL
            return None
        monkeypatch.setattr(bpd, "http_get", fake_get)
        first = bpd.fetch_banc_swc("banc_v888", ID888, resolution="l2",
                                   project_root=tmp_path, use_cache=True)
        assert first._drocat_banc_resolution == "l2"
        second = bpd.fetch_banc_swc("banc_v888", ID888, resolution="full",
                                    project_root=tmp_path)
        assert second is not None
        assert second._drocat_banc_resolution == "l2"


class TestMetaFeatherMapping:
    # Module-level ``meta_frame`` fixture below is shared with
    # TestCrosswalkCache.

    def test_build_neurons_dataframe_dedupes_duplicate_v626_ids(self, tmp_path):
        """R3: v626 carries duplicate root_626 rows (coarser
        materialization); columns must merge positionally BEFORE the
        bodyId dedupe so no row length mismatch can occur."""
        dup_id = "720575941595794471"
        other_id = "720575941442818239"
        frame = pd.DataFrame({
            "root_626": [dup_id, dup_id, other_id],
            "root_888": ["888_a", "888_b", "888_c"],
            "cell_type": ["l-LNv", "other", None],
            "cell_class": ["local_interneuron", "sensory", "sensory"],
            "super_class": ["central_brain_intrinsic", "sensory", "sensory"],
            "side": ["right", "left", "left"],
            "flow": ["intrinsic", "sensory", "sensory"],
            "nerve": [None, "antennal_nerve", None],
            "hemilineage": ["LALv1", None, None],
            "neurotransmitter_predicted": ["acetylcholine", "gaba", None],
            "neurotransmitter_score": [0.95, 0.9, None],
            "neurotransmitter_verified": [None, None, None],
            "neuropeptide_verified": [None, None, None],
            "body_part_sensory": [None, "antenna", None],
        })
        path = tmp_path / "dup_meta.feather"
        frame.to_feather(path)
        neurons = bpd.build_neurons_dataframe(path, "v626")
        # Exactly one row per bodyId; the FIRST row's metadata wins.
        assert len(neurons) == 2
        row = neurons[neurons["bodyId"] == dup_id].iloc[0]
        assert row["type"] == "l-LNv"
        assert row["nt_type"] == "acetylcholine"
        assert row["Class"] == "local_interneuron"

    def test_cable_length_column_is_labeled_um(self, tmp_path, meta_frame):
        """Real-data issue: the release column is ``l2_cable_length_um``;
        the Codex CSV mislabeled the same values as ``(nm)``.  The bucket
        build must emit the honest unit (values stay micrometres)."""
        # l-LNv L2 cable: 315.2 um (a nanometre value would be absurd).
        path = tmp_path / "meta.feather"
        meta_frame.to_feather(path)
        neurons = bpd.build_neurons_dataframe(path, "v888")
        assert "Cable length (µm)" in neurons.columns
        assert "Cable length (nm)" not in neurons.columns
        row = neurons[neurons["bodyId"] == ID888].iloc[0]
        assert row["Cable length (µm)"] == pytest.approx(315.2)

    def test_build_neurons_dataframe_schema(self, tmp_path, meta_frame):
        import numpy as np

        path = tmp_path / "banc_888_meta.feather"
        meta_frame.to_feather(path)
        neurons = bpd.build_neurons_dataframe(path, "v888")

        # The type mapper hard-fails without this exact header.
        assert "Alternative Cell Type(s)" in neurons.columns
        assert list(neurons.columns)[:4] == ["bodyId", "type", "instance", "post"]
        row = neurons.iloc[0]
        assert row["bodyId"] == ID888
        assert row["type"] == "l-LNv"
        # auto: tokens are dropped; cell_type leads the alias list.
        assert row["Alternative Cell Type(s)"] == "l-LNv"
        assert row["nt_type"] == "acetylcholine"
        assert row["Class"] == "local_interneuron"
        assert row["Soma side"] == "right"
        # Untyped neurons fill 'Unknown' (Codex convention), and their alias
        # list stays empty rather than 'nan'.
        row2 = neurons.iloc[1]
        assert row2["type"] == "Unknown"
        assert row2["Alternative Cell Type(s)"] == ""

    @pytest.mark.parametrize("dataset_name", [
        "banc_v888", "flywire_BANC_v888",
    ])
    def test_prepare_dataset_tables_writes_canonical_tables(
            self, tmp_path, fake_net, meta_frame, monkeypatch, dataset_name):
        import polars as pl

        conn = pl.DataFrame({
            "pre_root_id": ["0", ID888, ID888],
            "post_root_id": [ID888, ID626, ID626],
            "neuropil": ["CB_AL_L", "ME_R", "ME_L"],
            "num_synapses": [5, 3, 9],
        })
        buffer = io.BytesIO()
        conn.write_parquet(buffer)
        conn_bytes = buffer.getvalue()

        fake_net({
            bpd.META_FEATHER_PATH: meta_frame_to_bytes(meta_frame),
            "connectioncountsperneuropil": conn_bytes,
        })

        dataset_dir = tmp_path / "datasets" / "banc_v888"
        assert bpd.prepare_dataset_tables(dataset_name, str(dataset_dir),
                                          project_root=tmp_path) is True
        neurons = pd.read_parquet(dataset_dir / "banc_v888_allneurons_neuron_df.parquet")
        assert len(neurons) == 2
        assert set(neurons["bodyId"]) == {ID888, "720575941442818240"}
        conns = pd.read_parquet(dataset_dir / "banc_v888_merged_connections.parquet")
        # The id==0 placeholder row is filtered; weights aggregate per pair.
        assert len(conns) == 1
        assert conns.iloc[0]["weight"] == 12
        assert conns.iloc[0]["roi"] == "ME_L|ME_R"
        # Raw products are kept for offline re-runs, INSIDE the caller's
        # dataset_dir (a tmp dir here — never the repository's datasets/).
        assert (dataset_dir / "downloads" / "banc_888_meta.feather").exists()
        assert (dataset_dir / "downloads"
                / "connections_v888.parquet").exists()

    def test_prepare_isolated_to_dataset_dir(self, tmp_path, monkeypatch):
        """No project_root passed: downloads derive from dataset_dir."""
        import banc_public_data as mod

        touched = {}
        monkeypatch.setattr(
            mod, "download_meta_feather",
            lambda dataset, project_root=None, force=False,
            progress_callback=None:
            touched.__setitem__("root", str(project_root)) or None)
        monkeypatch.setattr(
            mod, "download_connections_product", lambda *a, **k: None)

        dataset_dir = tmp_path / "datasets" / "banc_v888"
        dataset_dir.mkdir(parents=True)
        # Download stubs return None (bucket unavailable) -> False, but the
        # root they were derived from must be the dataset_dir's parent.
        assert mod.prepare_dataset_tables("banc_v888", str(dataset_dir)) is False
        assert touched["root"] == str(tmp_path)


class TestProductValidation:
    def test_bad_existing_meta_is_quarantined_and_refetched(
            self, tmp_path, fake_net, meta_frame):
        local = bpd.meta_feather_path("banc_v888", tmp_path)
        local.parent.mkdir(parents=True)
        local.write_bytes(b"truncated")
        fake_net({bpd.META_FEATHER_PATH: meta_frame_to_bytes(meta_frame)})

        result = bpd.download_meta_feather("banc_v888", tmp_path)

        assert result == local
        assert pd.read_feather(local).shape[0] == len(meta_frame)
        assert list(local.parent.glob("banc_888_meta.feather.corrupt-*"))

    def test_bad_existing_connections_product_is_quarantined_and_refetched(
            self, tmp_path, fake_net):
        import polars as pl

        conn = pl.DataFrame({
            "pre_root_id": [ID888],
            "post_root_id": [ID626],
            "num_synapses": [3],
            "neuropil": ["ME_L"],
        })
        buffer = io.BytesIO()
        conn.write_parquet(buffer)
        local = bpd.connections_product_path("banc_v888", tmp_path)
        local.parent.mkdir(parents=True)
        local.write_bytes(b"truncated")
        fake_net({"connectioncountsperneuropil": buffer.getvalue()})

        result = bpd.download_connections_product("banc_v888", tmp_path)

        assert result == local
        assert pl.read_parquet(local).height == 1
        assert list(local.parent.glob("connections_v888.parquet.corrupt-*"))

    def test_invalid_download_is_not_promoted(self, tmp_path, fake_net):
        fake_net({bpd.META_FEATHER_PATH: b"not a feather"})
        assert bpd.download_meta_feather("banc_v888", tmp_path) is None
        assert not bpd.meta_feather_path("banc_v888", tmp_path).exists()


def meta_frame_to_bytes(frame):
    buffer = io.BytesIO()
    frame.to_feather(buffer)
    return buffer.getvalue()


@pytest.fixture
def meta_frame():
    """Release-meta stand-in shared by the mapper and crosswalk tests."""
    return pd.DataFrame({
        "banc_888_id": [ID888, "720575941442818240"],
        "root_888": [ID888, "720575941442818240"],
        "root_626": [ID626, "720575941442818240"],
        "cell_type": ["l-LNv", None],
        "fafb_cell_type": ["l-LNv", None],
        "manc_cell_type": ["auto:IN19A114", None],
        "malecns_cell_type": [None, None],
        "hemibrain_cell_type": [None, None],
        "fanc_cell_type": [None, None],
        "fafb_alignment_cell_type": [None, None],
        "cell_class": ["local_interneuron", "sensory"],
        "cell_sub_class": [None, None],
        "super_class": ["central_brain_intrinsic", "sensory"],
        "side": ["right", "left"],
        "flow": ["intrinsic", "sensory"],
        "nerve": [None, "antennal_nerve"],
        "hemilineage": ["LALv1", None],
        "neurotransmitter_predicted": ["acetylcholine", None],
        "neurotransmitter_score": [0.95, None],
        "neurotransmitter_verified": [None, None],
        "neuropeptide_verified": [None, None],
        "body_part_sensory": [None, "antenna"],
        "body_part_effector": [None, None],
        "cell_function": [None, None],
        "cell_function_detailed": [None, None],
        "l2_cable_length_um": [315.2, None],
        "volume_nm3": [2.4e11, None],
        "proofread": ["TRUE", "FALSE"],
    })


class TestCrosswalkCache:
    """R4 negative cache + success memoization of the id crosswalk."""

    def _write_crosswalk(self, tmp_path):
        import pandas as pd

        frame = pd.DataFrame({
            "root_626": [ID626],
            "banc_888_id": [ID888],
        })
        path = bpd._crosswalk_cache_path("banc_v626", tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        return path

    def test_reverse_inverts_forward(self, tmp_path):
        self._write_crosswalk(tmp_path)
        reverse = bpd.get_id_crosswalk_reverse("banc_v626", tmp_path)
        assert reverse == {ID888: ID626}

    def test_success_memoizes_no_rereads(self, tmp_path, monkeypatch):
        self._write_crosswalk(tmp_path)
        reads = []
        real_read = __import__("pandas").read_parquet

        def counting_read(path, *a, **k):
            reads.append(str(path))
            return real_read(path, *a, **k)

        monkeypatch.setattr("pandas.read_parquet", counting_read)
        first = bpd.get_id_crosswalk("banc_v626", tmp_path)
        assert first == {ID626: ID888}
        assert len(reads) == 1
        # Second call is served from the process memo: per-neuron callers
        # never re-read the parquet.
        assert bpd.get_id_crosswalk("banc_v626", tmp_path) == first
        assert bpd.get_id_crosswalk("banc_v626", tmp_path) == first
        assert len(reads) == 1

    def test_negative_cache_recovers_on_forced_refresh(
            self, tmp_path, fake_net, meta_frame):
        # First download fails -> negative cache; direct-id fallback.
        calls = fake_net({bpd.META_FEATHER_PATH: None})
        assert bpd.get_id_crosswalk("banc_v626", tmp_path) == {}
        assert "banc_v626" in bpd._CROSSWALK_FAILURES
        assert bpd.get_id_crosswalk("banc_v626", tmp_path) == {}
        assert len(calls) == 1  # no repeated 58 MB attempts (R4)
        # Forced refresh succeeds -> the negative cache must be cleared so
        # later non-forced calls use the real mapping.
        calls.clear()
        fake_net({bpd.META_FEATHER_PATH: meta_frame_to_bytes(meta_frame)})
        recovered = bpd.get_id_crosswalk("banc_v626", tmp_path,
                                         force_refresh=True)
        assert recovered
        assert "banc_v626" not in bpd._CROSSWALK_FAILURES
        calls.clear()
        again = bpd.get_id_crosswalk("banc_v626", tmp_path)
        assert again == recovered
        assert calls == []  # memo hit, not a fresh download

    def test_readable_cache_supersedes_stale_failure(self, tmp_path):
        # A failure is negative-cached, then the parquet appears (written
        # by another path/process): reading it must clear the failure.
        bpd._CROSSWALK_FAILURES.add("banc_v626")
        self._write_crosswalk(tmp_path)
        crosswalk = bpd.get_id_crosswalk("banc_v626", tmp_path)
        assert crosswalk == {ID626: ID888}
        assert "banc_v626" not in bpd._CROSSWALK_FAILURES


class TestReadinessRelaxation:
    def test_banc_always_ready(self, tmp_path):
        from utils.flywire_readiness import flywire_skeleton_readiness

        status = flywire_skeleton_readiness("banc_v888", project_root=tmp_path)
        assert status["ready"] is True
        assert status["is_banc"] is True
        assert "banc_public_gcs" in status["local_source"]
        assert status["cave_token"] is False

    def test_banc_not_rejected(self, tmp_path):
        from utils.flywire_readiness import require_flywire_skeleton_access

        status = require_flywire_skeleton_access("banc_v626",
                                                 project_root=tmp_path)
        assert status["ready"] is True

    def test_fafb_still_requires_local_or_token(self, tmp_path):
        from utils.flywire_readiness import (
            flywire_skeleton_readiness,
            require_flywire_skeleton_access,
            FlyWireSkeletonAccessError,
        )

        status = flywire_skeleton_readiness("flywire_FAFB_v783",
                                            project_root=tmp_path)
        assert status["ready"] is False
        with pytest.raises(FlyWireSkeletonAccessError):
            require_flywire_skeleton_access("flywire_FAFB_v783",
                                            project_root=tmp_path)

    def test_manual_instruction_mentions_bucket(self, tmp_path):
        from utils.flywire_readiness import flywire_manual_skeleton_instruction

        text = flywire_manual_skeleton_instruction("banc_v888", tmp_path)
        assert "not needed" in text
        assert "banc_public_data" in text


class TestSynapseSliceFetch:
    """Ranged download + per-pair derivation of the per-synapse table."""

    def _fake_remote(self, monkeypatch, content: bytes):
        monkeypatch.setattr(bpd, "_remote_size", lambda url: len(content))

        def fake_range(url, start, end):
            return content[start:end + 1]
        monkeypatch.setattr(bpd, "_fetch_range", fake_range)

    def test_resumable_download(self, tmp_path, monkeypatch):
        content = bytes(range(256)) * 100  # 25_600 bytes
        self._fake_remote(monkeypatch, content)
        dest = tmp_path / "downloads" / "table.parquet"
        got = bpd.fetch_range_to_file("http://example/t.parquet", dest,
                                      chunk_bytes=10_000)
        assert got == dest and dest.stat().st_size == len(content)
        assert dest.read_bytes() == content

    def test_resume_from_partial(self, tmp_path, monkeypatch):
        content = bytes(b % 251 for b in range(5_000))
        self._fake_remote(monkeypatch, content)
        dest = tmp_path / "table.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content[:2_000])  # partial leftover
        seen = []
        got = bpd.fetch_range_to_file(
            "http://example/t.parquet", dest,
            chunk_bytes=1_200,
            progress_callback=lambda done, total: seen.append(done))
        assert got == dest and dest.read_bytes() == content
        assert seen and seen[-1] == len(content)

    def test_404_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bpd, "_remote_size", lambda url: None)
        assert bpd.fetch_range_to_file("http://example/x.parquet",
                                       tmp_path / "x.parquet") is None

    def test_build_synapse_table_aggregates_pairs(self, tmp_path):
        import polars as pl

        rows = pl.DataFrame({
            # id 0 placeholders must be dropped; pair (1,9) has 3 synapses
            # (two at the same site, one elsewhere) -> mean x 20, count 3.
            "pre_root_id": [1, 1, 1, 2, 0, 3],
            "post_root_id": [9, 9, 9, 9, 9, 9],
            "pre_x": [10.0, 20.0, 30.0, 50.0, 1.0, 5.0],
            "pre_y": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "pre_z": [4.0, 4.0, 4.0, 4.0, 1.0, 0.0],
            "neuropil": ["AL", "AL", "AL", "AL", "AL", "AL"],
        })
        src = tmp_path / "per_synapse.parquet"
        rows.write_parquet(src)
        derived = tmp_path / "banc_v888_synapse_table.parquet"
        assert bpd.build_synapse_table(src, derived) is True
        out = pl.read_parquet(derived)
        assert out.height == 3  # (1,9)x3, (2,9)x1, (3,9)x1 — id 0 dropped
        ab = out.filter((pl.col("pre_root_id") == 1)
                        & (pl.col("post_root_id") == 9))
        assert ab["syn_count"][0] == 3
        assert ab["x_pre"][0] == 20.0  # mean of 10/20/30
        # post-site coordinates are not persisted: the release has none,
        # and the reader mirrors the pre site onto the post columns.
        assert "x_post" not in out.columns

    def test_compact_synapse_table_slims_legacy_mirrors(self, tmp_path):
        """Legacy tables that mirrored pre-site coords into post columns
        are rewritten without the duplicates; current tables are no-ops."""
        import polars as pl

        legacy = pl.DataFrame({
            "pre_root_id": [1, 2], "post_root_id": [9, 9],
            "syn_count": [3, 1],
            "x_pre": [20.0, 50.0], "y_pre": [0.0, 0.0], "z_pre": [4.0, 4.0],
            "x_post": [20.0, 50.0], "y_post": [0.0, 0.0], "z_post": [4.0, 4.0],
        })
        path = tmp_path / "banc_v888_synapse_table.parquet"
        legacy.write_parquet(path, compression="zstd")
        before = path.stat().st_size

        assert bpd.compact_synapse_table(path) is True
        assert path.stat().st_size < before
        out = pl.read_parquet(path)
        assert out.columns == ["pre_root_id", "post_root_id", "syn_count",
                               "x_pre", "y_pre", "z_pre"]
        assert out["syn_count"].to_list() == [3, 1]
        assert out["x_pre"].to_list() == [20.0, 50.0]
        # Idempotent: the compacted schema has nothing left to drop.
        assert bpd.compact_synapse_table(path) is False

    def test_compact_synapse_table_encodes_and_stamps_marker(self, tmp_path):
        """A current-schema table without the footer marker gets the
        lossless layout applied once; the marker makes the next call a
        no-op while values and columns stay identical."""
        import polars as pl

        current = pl.DataFrame({
            "pre_root_id": [1], "post_root_id": [9], "syn_count": [1],
            "x_pre": [1.0], "y_pre": [2.0], "z_pre": [3.0],
        })
        path = tmp_path / "table.parquet"
        current.write_parquet(path)
        assert bpd.compact_synapse_table(path) is True
        import pyarrow.parquet as pq
        schema = pq.read_schema(path)
        assert schema.metadata.get(b"DROCAT.lossless") == b"v1"
        assert pl.read_parquet(path).columns == list(current.columns)
        assert bpd.compact_synapse_table(path) is False

    def test_compact_cleans_stale_temp_files(self, tmp_path):
        """Temp siblings left by an interrupted compaction are removed
        before a new rewrite starts."""
        import polars as pl

        legacy = pl.DataFrame({
            "pre_root_id": [1], "post_root_id": [9], "syn_count": [1],
            "x_pre": [1.0], "y_pre": [2.0], "z_pre": [3.0],
            "x_post": [1.0], "y_post": [2.0], "z_post": [3.0],
        })
        path = tmp_path / "banc_v888_synapse_table.parquet"
        legacy.write_parquet(path)
        stale = tmp_path / ".banc_v888_synapse_table.parquet.compact.999999.tmp"
        stale.write_bytes(b"junk from a dead process")

        assert bpd.compact_synapse_table(path) is True
        assert not stale.exists()

    def test_build_writes_atomically_and_cleans_stale_temps(self, tmp_path):
        import os as _os
        import polars as pl
        from utils.parquet_utils import temp_sibling

        rows = pl.DataFrame({
            "pre_root_id": [1], "post_root_id": [9],
            "pre_x": [5.0], "pre_y": [6.0], "pre_z": [7.0],
        })
        src = tmp_path / "per_synapse.parquet"
        rows.write_parquet(src)
        derived = tmp_path / "banc_v888_synapse_table.parquet"
        # The production temp naming must be the hidden dotted form the
        # stale-cleanup helper matches; a non-dotted name would leak.
        assert _os.path.basename(
            temp_sibling(str(derived), "build")).startswith(".")
        stale = tmp_path / ".banc_v888_synapse_table.parquet.build.999999.tmp"
        stale.write_bytes(b"junk from a dead process")

        assert bpd.build_synapse_table(src, derived) is True
        # The atomic rename leaves the final file and no temp siblings,
        # and the post-pass stamps the lossless marker.
        assert derived.exists()
        assert not stale.exists()
        assert not list(tmp_path.glob(".*.build.*.tmp"))
        assert not list(tmp_path.glob(".*.compact.*.tmp"))
        import pyarrow.parquet as pq
        assert (pq.read_schema(derived).metadata or {}).get(
            b"DROCAT.lossless") == b"v1"

    def test_ensure_rebuilds_table_truncated_by_interrupted_run(self, tmp_path,
                                                                monkeypatch):
        """A derived table without a valid parquet footer (the previous
        run died mid-write) is discarded and rebuilt — re-fetching the
        raw download if it was already reclaimed — instead of failing at
        read time forever."""
        import polars as pl

        ds_dir = tmp_path / "datasets" / "banc_v888"
        ds_dir.mkdir(parents=True)
        derived = ds_dir / "banc_v888_synapse_table.parquet"
        derived.write_bytes(b"PAR1truncated-no-valid-footer")

        def fake_ensure_synapse_table(dataset, project_root=None,
                                      force=False, progress_callback=None):
            raw = tmp_path / "downloads" / "per_synapse.parquet"
            raw.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame({
                "pre_root_id": [1], "post_root_id": [9],
                "pre_x": [5.0], "pre_y": [6.0], "pre_z": [7.0],
            }).write_parquet(raw)
            return raw

        monkeypatch.setattr(bpd, "ensure_synapse_table",
                            fake_ensure_synapse_table)

        assert bpd.ensure_synapse_derived_table(
            "banc_v888", str(ds_dir), project_root=str(tmp_path)) is True
        out = pl.read_parquet(derived)
        assert out.columns == ["pre_root_id", "post_root_id", "syn_count",
                               "x_pre", "y_pre", "z_pre"]
        assert out.height == 1
        assert not list(ds_dir.glob(".*.build.*.tmp"))

    def test_build_failure_preserves_existing_table(self, tmp_path, monkeypatch):
        """A failed build must not destroy a previously good derived table,
        and must not leave a temp sibling behind."""
        import polars as pl

        good = pl.DataFrame({
            "pre_root_id": [1], "post_root_id": [9], "syn_count": [1],
            "x_pre": [1.0], "y_pre": [2.0], "z_pre": [3.0],
        })
        derived = tmp_path / "banc_v888_synapse_table.parquet"
        good.write_parquet(derived)
        before = derived.read_bytes()
        src = tmp_path / "per_synapse.parquet"
        good.select(["pre_root_id", "post_root_id", "x_pre", "y_pre", "z_pre"]
                    ).write_parquet(src)

        original_write = pl.DataFrame.write_parquet

        def boom(self, path, *args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(pl.DataFrame, "write_parquet", boom)
        assert bpd.build_synapse_table(src, derived) is False
        monkeypatch.setattr(pl.DataFrame, "write_parquet", original_write)

        assert derived.read_bytes() == before
        assert not list(tmp_path.glob(".*.build.*.tmp"))


class TestReaderIntegration:
    def test_reader_reads_derived_table(self, tmp_path, monkeypatch):
        import polars as pl
        import visualize_skeleton as vsmod

        rows = pl.DataFrame({
            "pre_root_id": [100],
            "post_root_id": [200],
            "pre_x": [5.0], "pre_y": [6.0], "pre_z": [300_000.0],
            "neuropil": ["AL"],
        })
        src = tmp_path / "per_synapse.parquet"
        rows.write_parquet(src)
        ds_dir = tmp_path / "datasets" / "banc_v888"
        derived = ds_dir / "banc_v888_synapse_table.parquet"
        assert bpd.build_synapse_table(src, derived) is True

        vs = object.__new__(vsmod.VisualizeSkeleton)
        vs.dataset = "banc_v888"
        vs.script_path = str(tmp_path)
        vs.min_synapse_num = 0
        monkeypatch.setattr(type(vs), "_get_synapse_table_path",
                            lambda self: str(derived), raising=False)

        conn = vs._read_flywire_connection_frame(
            source_ids={"100"}, target_ids={"200"})
        assert conn is not None and len(conn) == 1
        assert conn.iloc[0]["x_pre"] == 5.0
        # the reader mirrors the pre site onto the post columns in memory
        assert conn.iloc[0]["x_post"] == 5.0
        assert int(conn.iloc[0]["syn_count"]) == 1
        # nanometre coordinates must NOT be scaled (z = 300000 stays)
        assert conn.iloc[0]["z_pre"] == 300_000.0

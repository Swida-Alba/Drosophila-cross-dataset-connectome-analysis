"""Tests for `VisualizeSkeleton._resolve_fafb_sources`.

Covers the per-body render priority (identical for tube and line):

    repaired caches (cave_skeletons / extrusion_fixes, status-driven)
    -> raw SWC cache -> healed ZIP -> FlyWire Codex lazy fetch
    -> CAVE skeletonization (tree)
    (`api_only` / force_API_fetching routes straight to CAVE)

and the strict `use_cache=False` policy (cache sources skipped, extrusion
parquet check cache untouched). CAVE replacements are trees persisted in the
dedicated ``cave_skeletons`` store; locally pruned fixes live in
``extrusion_fixes`` and are labelled ``local_repaired``. The
``TestCodexLazyFlow`` class exercises the real lazy + extrusion + local
repair chain with ONLY a FlyWire Codex token (no CAVE token) and the
codex service monkeypatched, so no test touches the network.
"""

import shutil
import sys
import tempfile
import weakref
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import navis  # noqa: E402
from visualize_skeleton import VisualizeSkeleton  # noqa: E402


def make_tree(body_id):
    neuron = navis.TreeNeuron(pd.DataFrame({
        "node_id": np.array([0], dtype=np.int64),
        "parent_id": np.array([-1], dtype=np.int64),
        "x": np.array([0.0]),
        "y": np.array([0.0]),
        "z": np.array([0.0]),
        "radius": np.array([1.0]),
        "type": ["root"],
    }))
    neuron.soma = None
    neuron.id = body_id
    return neuron


def make_spiky_tree(body_id):
    neuron = navis.TreeNeuron(pd.DataFrame({
        "node_id": np.array([0, 1, 2, 3, 4], dtype=np.int64),
        "parent_id": np.array([-1, 0, 1, 1, 3], dtype=np.int64),
        "x": np.array([0.0, 1.0, 2.0, 100.0, 101.0]),
        "y": np.zeros(5),
        "z": np.zeros(5),
        "radius": np.ones(5),
    }))
    neuron.id = body_id
    return neuron


class RecordingResolver:
    """Stubbed resolver recording every source-layer call."""

    def __init__(self, zip_hits=None, raw_hits=None, cave_hits=None,
                 codex_hits=None):
        self.visualizer = object.__new__(VisualizeSkeleton)
        self.visualizer.dataset = "flywire_FAFB_v783"
        self.visualizer.cache_neurons = True
        self.visualizer.auto_fix_extrusions = False
        self.visualizer._vprint = lambda *args, **kwargs: None
        # Hermetic scratch project: repair caches and the extrusion parquet
        # are read/written under this root, never the real repo cache.
        self._tmp = tempfile.mkdtemp(prefix="drocat-resolver-")
        weakref.finalize(self, shutil.rmtree, self._tmp, True)
        self.visualizer.script_path = self._tmp

        self.zip_hits = dict(zip_hits or {})
        self.raw_hits = dict(raw_hits or {})
        self.cave_hits = dict(cave_hits or {})
        self.codex_hits = dict(codex_hits or {})

        self.calls = {"zip": [], "raw": [], "codex": [], "cave": [],
                      "extrusion": []}

        self.visualizer._preload_fafb_skeletons = self._fake_zip
        self.visualizer._load_api_cached_skeletons = self._fake_raw
        self.visualizer._fetch_fafb_skeletons_via_codex = self._fake_codex
        self.visualizer._fetch_fafb_skeletons_via_cave = self._fake_cave
        self.visualizer._detect_extrusions_in_skeletons = self._fake_extrusion

    # --- mock source layers --------------------------------------------
    def _fake_zip(self, body_ids_filter=None):
        ids = list(body_ids_filter or [])
        self.calls["zip"].append(ids)
        return {bid: self.zip_hits[bid] for bid in ids
                if bid in self.zip_hits}

    def _fake_raw(self, body_ids):
        self.calls["raw"].append(list(body_ids))
        found = {bid: self.raw_hits[bid] for bid in body_ids
                 if bid in self.raw_hits}
        missing = [bid for bid in body_ids if bid not in self.raw_hits]
        return found, missing

    def _fake_codex(self, body_ids):
        self.calls["codex"].append(list(body_ids))
        return {bid: self.codex_hits[bid] for bid in body_ids
                if bid in self.codex_hits}

    def _fake_cave(self, body_ids):
        self.calls["cave"].append(list(body_ids))
        return {bid: self.cave_hits[bid] for bid in body_ids
                if bid in self.cave_hits}

    def _fake_extrusion(self, skeletons, **kwargs):
        self.calls["extrusion"].append((list(skeletons), kwargs))
        return []

    def resolve(self, body_ids, **kwargs):
        return self.visualizer._resolve_fafb_sources(body_ids, **kwargs)


class TestSourcePriority:
    def test_raw_then_zip_then_cave(self):
        resolver = RecordingResolver(
            zip_hits={"1": make_tree("1")},
            raw_hits={"2": make_tree("2")},
            cave_hits={"4": make_tree("4")},
        )
        sources, skeleton_cache = resolver.resolve([1, 2, 4])

        assert sources == {"1": "zip", "2": "raw_cache", "4": "cave"}
        assert set(skeleton_cache) == {"1", "2", "4"}
        # Priority: raw SWC cache, then the healed bundle; CAVE
        # skeletonization serves the rest.
        assert resolver.calls["raw"] == [["1", "2", "4"]]
        assert resolver.calls["zip"] == [["1", "4"]]
        assert resolver.calls["cave"] == [["4"]]

    def test_raw_cache_before_bundle(self):
        """The raw SWC cache precedes the healed bundle (pipeline
        priority), and both stay SWC sources."""
        resolver = RecordingResolver(
            zip_hits={"7": make_tree("7")},
            raw_hits={"7": make_tree("7")},
        )
        sources, skeleton_cache = resolver.resolve([7])

        assert sources == {"7": "raw_cache"}
        assert set(skeleton_cache) == {"7"}
        assert resolver.calls["raw"] == [["7"]]
        assert resolver.calls["zip"] == []
        assert resolver.calls["cave"] == []

    def test_repaired_cache_served_before_zip(self):
        """A body with a persisted local fix is served from the repair
        caches; the zip is never consulted for it."""
        resolver = RecordingResolver()
        scratch_root = resolver.visualizer.script_path
        from cave_data_fetcher import CAVEDataFetcher
        from fafb_utils import set_extrusion_repair_status

        fetcher = CAVEDataFetcher(
            dataset="flywire_FAFB_v783", cave_token="",
            project_root=scratch_root, cache_enabled=True,
            verbose=False)
        assert fetcher.save_extrusion_fix_skeleton("7", make_tree("7"))
        set_extrusion_repair_status(
            scratch_root, "flywire_FAFB_v783",
            {"7": "local_fallback"})

        sources, skeleton_cache = resolver.resolve([7])

        assert sources == {"7": "local_repaired"}
        assert set(skeleton_cache) == {"7"}
        assert resolver.calls["zip"] == []
        assert resolver.calls["raw"] == []
        assert resolver.calls["cave"] == []

    def test_every_local_miss_falls_through_to_cave(self):
        cave_tree = make_tree("9")
        resolver = RecordingResolver(cave_hits={"9": cave_tree})
        sources, skeleton_cache = resolver.resolve([9])

        assert sources == {"9": "cave"}
        assert skeleton_cache == {"9": cave_tree}

    def test_api_only_routes_straight_to_cave(self):
        resolver = RecordingResolver(
            zip_hits={"1": make_tree("1")},
            raw_hits={"1": make_tree("1")},
            cave_hits={"1": make_tree("1")},
        )
        sources, skeleton_cache = resolver.resolve([1], api_only=True)

        assert sources == {"1": "cave"}
        assert set(skeleton_cache) == {"1"}
        assert resolver.calls["zip"] == []
        assert resolver.calls["raw"] == []
        assert resolver.calls["cave"] == [["1"]]


class TestStrictUseCache:
    def test_cache_sources_skipped_when_caching_disabled(self):
        resolver = RecordingResolver(
            raw_hits={"2": make_tree("2")},
            cave_hits={"2": make_tree("2")},
        )
        resolver.visualizer.cache_neurons = False

        sources, skeleton_cache = resolver.resolve([2])

        assert sources == {"2": "cave"}
        assert set(skeleton_cache) == {"2"}
        assert resolver.calls["raw"] == []

    def test_healed_zip_stays_eligible_without_cache(self):
        """The healed ZIP is the canonical raw source and stays eligible
        under the strict use_cache=False policy."""
        resolver = RecordingResolver(zip_hits={"5": make_tree("5")})
        resolver.visualizer.cache_neurons = False

        sources, skeleton_cache = resolver.resolve([5])

        assert sources == {"5": "zip"}
        assert set(skeleton_cache) == {"5"}
        assert resolver.calls["raw"] == []
        assert resolver.calls["cave"] == []

    def test_extrusion_check_honors_use_cache_policy(self):
        resolver = RecordingResolver(zip_hits={"6": make_tree("6")})
        resolver.visualizer.auto_fix_extrusions = True

        # caching enabled -> extrusion check may use its parquet cache
        resolver.resolve([6])
        assert resolver.calls["extrusion"][0][1]["use_cache"] is True

        # caching disabled -> strict in-memory-only extrusion check
        resolver.calls["extrusion"].clear()
        resolver.visualizer.cache_neurons = False
        resolver.resolve([6])
        assert resolver.calls["extrusion"][0][1]["use_cache"] is False


class TestExtrusionRepair:
    def test_extrusion_affected_tree_source_moves_to_cave(self):
        cave_tree = make_tree("6")
        resolver = RecordingResolver(
            zip_hits={"6": make_tree("6")},
            cave_hits={"6": cave_tree},
        )
        resolver.visualizer.auto_fix_extrusions = True
        resolver.visualizer._detect_extrusions_in_skeletons = (
            lambda skeletons, **kwargs: ["6"])

        sources, skeleton_cache = resolver.resolve([6])

        assert sources == {"6": "cave"}
        assert skeleton_cache == {"6": cave_tree}
        assert resolver.calls["cave"] == [["6"]]

    def test_failed_cave_repair_prunes_local_extrusion_branch(self):
        resolver = RecordingResolver(
            zip_hits={"7": make_spiky_tree("7")},
        )
        resolver.visualizer.auto_fix_extrusions = True
        resolver.visualizer._detect_extrusions_in_skeletons = (
            lambda skeletons, **kwargs: ["7"])

        sources, skeleton_cache = resolver.resolve([7])

        assert sources == {"7": "local_repaired"}
        assert set(skeleton_cache["7"].nodes["node_id"]) == {0, 1, 2}
        assert resolver.calls["cave"] == [["7"]]


FAFB_SWC = (
    "# SWC format file\n"
    "1 1 0.0 0.0 0.0 1.0 -1\n"
    "2 3 100.0 0.0 0.0 1.0 1\n"
    "3 3 200.0 0.0 0.0 1.0 2\n"
)


class CodexOnlyHarness(RecordingResolver):
    """Real lazy-fetch + extrusion + repair chain, Codex token only.

    ``_fetch_fafb_skeletons_via_codex`` and ``_fetch_fafb_skeletons_via_cave``
    stay the REAL methods; the access flag reports no CAVE token so the
    real no-token gate runs, and the codex service is monkeypatched per
    test — no network is ever touched.
    """

    def __init__(self):
        super().__init__()
        self.visualizer._fetch_fafb_skeletons_via_codex = (
            VisualizeSkeleton._fetch_fafb_skeletons_via_codex.__get__(
                self.visualizer))
        self.visualizer._fetch_fafb_skeletons_via_cave = (
            VisualizeSkeleton._fetch_fafb_skeletons_via_cave.__get__(
                self.visualizer))
        self.visualizer._flywire_skeleton_access = {
            "is_fafb": True, "cave_token": False}


class TestCodexLazyFlow:
    def test_local_miss_fetches_from_codex_without_cave_token(self, monkeypatch):
        import codex_downloader
        monkeypatch.setattr(codex_downloader, "get_codex_token",
                            lambda *a, **k: "codex-token")
        monkeypatch.setattr(
            codex_downloader, "fetch_skeleton_swcs",
            lambda ids, **k: ({str(b): FAFB_SWC for b in ids}, []))

        resolver = CodexOnlyHarness()
        resolver.visualizer.auto_fix_extrusions = True
        sources, skeleton_cache = resolver.resolve(["720575940596125868"])

        assert sources == {"720575940596125868": "codex_lazy"}
        assert set(skeleton_cache["720575940596125868"].nodes["node_id"]) \
            == {1, 2, 3}
        # the fetched tree is persisted into the shared raw cache under the
        # scratch root (later runs resolve from raw_cache instead)
        from skeleton_provenance import raw_skeleton_cache_path
        assert raw_skeleton_cache_path(
            resolver.visualizer.script_path, "flywire_FAFB_v783",
            "720575940596125868").exists()

    def test_codex_tree_extrusion_flagged_repairs_locally_without_cave(
            self, monkeypatch):
        import codex_downloader
        import fafb_utils
        monkeypatch.setattr(codex_downloader, "get_codex_token",
                            lambda *a, **k: "codex-token")
        monkeypatch.setattr(
            codex_downloader, "fetch_skeleton_swcs",
            lambda ids, **k: ({str(b): FAFB_SWC for b in ids}, []))

        resolver = CodexOnlyHarness()
        resolver.visualizer.auto_fix_extrusions = True
        resolver.visualizer._detect_extrusions_in_skeletons = (
            lambda skeletons, **kwargs: list(skeletons))
        repaired_calls = []

        def fake_repair(neuron):
            repaired_calls.append(neuron)
            return make_tree("720575940596125868"), {
                "repaired": True, "removed_nodes": 2}

        monkeypatch.setattr(fafb_utils, "repair_extruded_skeleton",
                            fake_repair)

        sources, _skeleton_cache = resolver.resolve(["720575940596125868"])

        # no CAVE token: the fetch gate returns nothing and the diagnosed
        # branch is pruned locally instead
        assert sources == {"720575940596125868": "local_repaired"}
        assert len(repaired_calls) == 1
        fix_dir = (Path(resolver.visualizer.script_path) / "cache"
                   / "flywire_FAFB_v783" / "skeletons" / "extrusion_fixes")
        saved = [p.name for p in fix_dir.iterdir()
                 if p.name.startswith("720575940596125868")]
        assert saved, "pruned fix was not persisted to extrusion_fixes"
        status = fafb_utils.load_extrusion_repair_status(
            resolver.visualizer.script_path, "flywire_FAFB_v783")
        assert status.get("720575940596125868") == "local_fallback"

    def test_no_codex_token_skips_lazy_and_uses_cave(self, monkeypatch):
        import codex_downloader
        monkeypatch.setattr(codex_downloader, "get_codex_token",
                            lambda *a, **k: None)
        fetch_called = []
        monkeypatch.setattr(
            codex_downloader, "fetch_skeleton_swcs",
            lambda ids, **k: fetch_called.append(list(ids)) or ({}, []))

        resolver = CodexOnlyHarness()
        resolver.visualizer._fetch_fafb_skeletons_via_cave = resolver._fake_cave
        resolver.cave_hits = {"720575940596125868":
                              make_tree("720575940596125868")}
        sources, _skeleton_cache = resolver.resolve(["720575940596125868"])

        assert sources == {"720575940596125868": "cave"}
        assert fetch_called == []  # lazy never reached the service
        assert resolver.calls["cave"] == [["720575940596125868"]]

    def test_codex_fetch_failure_falls_through_to_cave(self, monkeypatch):
        import codex_downloader
        monkeypatch.setattr(codex_downloader, "get_codex_token",
                            lambda *a, **k: "codex-token")

        def boom(ids, **k):
            raise RuntimeError("portal down")

        monkeypatch.setattr(codex_downloader, "fetch_skeleton_swcs", boom)

        resolver = CodexOnlyHarness()
        resolver.visualizer._fetch_fafb_skeletons_via_cave = resolver._fake_cave
        resolver.cave_hits = {"720575940596125868":
                              make_tree("720575940596125868")}
        sources, _skeleton_cache = resolver.resolve(["720575940596125868"])
        assert sources == {"720575940596125868": "cave"}

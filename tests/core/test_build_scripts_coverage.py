"""Hermetic coverage tests for the three ``src/build_*.py`` CLI scripts.

Every path-dependent helper is redirected to ``tmp_path`` (either by
monkeypatching the module's ``__file__`` / ``_PROJECT_ROOT`` constants or by
patching the heavy ``FindNeuronConnection`` / profiler collaborators), so no
real project ``cache/``, ``datasets/`` or ``neuron_indexes/`` data is touched.
"""

import json
import sys
import types
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import build_connection_cache as bcc  # noqa: E402
import build_connectivity_profile_cache as bpcp  # noqa: E402
import build_seed_indexes as bsi  # noqa: E402


def _redirect_module_paths(monkeypatch, module, tmp_path):
    """Point ``Path(__file__).parent``-based lookups into tmp_path."""
    monkeypatch.setattr(
        module, "__file__", str(tmp_path / "src" / (module.__name__ + ".py"))
    )


# ===========================================================================
# build_connection_cache
# ===========================================================================


class FakeConnectionFNC:
    last = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.build_args = None
        FakeConnectionFNC.last = self

    def build_connection_cache(self, **kwargs):
        self.build_args = kwargs
        return {
            "total_neurons": 2,
            "total_connections": 1,
            "cached_neurons": ["a", "b"],
            "failed_neurons": [],
            "elapsed_time": 0.25,
        }


@pytest.fixture
def fake_connection_fnc(monkeypatch):
    import coana

    FakeConnectionFNC.last = None
    monkeypatch.setattr(coana, "FindNeuronConnection", FakeConnectionFNC)
    return FakeConnectionFNC


def _stub_token_manager(monkeypatch, token="tok-123"):
    fake_module = types.ModuleType("utils.token_manager")
    fake_module.token_manager = types.SimpleNamespace(
        get_token=lambda name, direct=None: token
    )
    monkeypatch.setitem(sys.modules, "utils.token_manager", fake_module)


def test_build_cache_resolves_token_and_reports(monkeypatch, fake_connection_fnc, capsys):
    _stub_token_manager(monkeypatch, token="tok-123")
    result = bcc.build_cache(
        "hemibrain:v1.2.1", token=None, neuron_types=["Mi1", "T4a"]
    )
    assert result["total_neurons"] == 2
    fake = FakeConnectionFNC.last
    assert fake.kwargs["token"] == "tok-123"
    assert fake.kwargs["dataset"] == "hemibrain:v1.2.1"
    assert fake.kwargs["use_cache"] is True
    assert fake.build_args["neuron_types"] == ["Mi1", "T4a"]
    output = capsys.readouterr().out
    assert "Cache Build Summary" in output
    assert "Neuron types: ['Mi1', 'T4a']" in output
    assert "cache/hemibrain_v1_2_1/" in output


def test_build_cache_without_types_and_import_error(monkeypatch, fake_connection_fnc, capsys):
    # Force the ``except ImportError: pass`` fallback around the token manager.
    broken = types.ModuleType("utils.token_manager")  # no token_manager attr
    monkeypatch.setitem(sys.modules, "utils.token_manager", broken)
    result = bcc.build_cache("flywire_FAFB_v783", token="direct-token")
    assert result["total_connections"] == 1
    fake = FakeConnectionFNC.last
    assert fake.kwargs["token"] == "direct-token"
    assert "Neuron types: ALL" in capsys.readouterr().out


def test_complete_neuron_count_requires_the_recorded_row_count():
    index_df = pd.DataFrame({
        "bodyId": ["1", "2", "3", "4"],
        "downstream_complete": [True, True, True, False],
        "connection_count": [10, 1, 0, 5],
    })
    conn_df = pd.DataFrame({"bodyId_pre": ["1", "2", "4"]})
    # 1: 1/10 rows -> incomplete; 2: 1/1 -> complete; 3: zero flagged ->
    # complete; 4: flag false and 1/5 -> incomplete.
    assert bcc.complete_neuron_count(index_df, conn_df) == 2


def test_show_stats_full(tmp_path, monkeypatch, capsys):
    _redirect_module_paths(monkeypatch, bcc, tmp_path)
    safe = "hemibrain_v1_2_1"
    cache_dir = tmp_path / "cache" / safe
    cache_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "bodyId_pre": ["1", "1", "2"],
            "bodyId_post": ["2", "3", "3"],
            "weight": [1, 2, 3],
        }
    ).to_parquet(cache_dir / "connections.parquet", index=False)

    index_dir = tmp_path / "neuron_indexes" / safe
    index_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "bodyId": ["1", "2"],
            "downstream_complete": [True, False],
        }
    ).to_parquet(index_dir / "neuron_index.parquet", index=False)

    bcc.show_stats("hemibrain:v1.2.1")
    output = capsys.readouterr().out
    assert "Connection Cache Statistics" in output
    assert "Total connections: 3" in output
    assert "Unique upstream neurons: 2" in output
    assert "Unique downstream neurons: 2" in output
    assert "Total synapse count: 6" in output
    assert "Total neurons indexed: 2" in output
    assert "Fully cached neurons: 2 (100.0%)" in output


def test_show_stats_missing_cache(tmp_path, monkeypatch, capsys):
    _redirect_module_paths(monkeypatch, bcc, tmp_path)
    bcc.show_stats("hemibrain:v1.2.1")
    output = capsys.readouterr().out
    assert "[ERROR] Connection cache not found" in output


def test_show_stats_corrupt_files(tmp_path, monkeypatch, capsys):
    _redirect_module_paths(monkeypatch, bcc, tmp_path)
    safe = "hemibrain_v1_2_1"
    cache_dir = tmp_path / "cache" / safe
    cache_dir.mkdir(parents=True)
    (cache_dir / "connections.parquet").write_bytes(b"not parquet")
    index_dir = tmp_path / "neuron_indexes" / safe
    index_dir.mkdir(parents=True)
    (index_dir / "neuron_index.parquet").write_bytes(b"also not parquet")

    bcc.show_stats("hemibrain:v1.2.1")
    output = capsys.readouterr().out
    assert "[ERROR] Could not read connection cache" in output
    assert "[WARNING] Could not read neuron index" in output


def test_main_stats_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(bcc, "show_stats", lambda dataset: calls.append(dataset))
    monkeypatch.setattr(
        sys, "argv", ["build_connection_cache.py", "--stats", "manc:v1.0"]
    )
    bcc.main()
    assert calls == ["manc:v1.0"]


def test_main_build_path(monkeypatch):
    captured = {}

    def record(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(bcc, "build_cache", record)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_connection_cache.py",
            "hemibrain:v1.2.1",
            "--token",
            "tok",
            "--batch-size",
            "7",
            "--types",
            "Mi1",
            "T4a",
        ],
    )
    bcc.main()
    assert captured["dataset"] == "hemibrain:v1.2.1"
    assert captured["token"] == "tok"
    assert captured["batch_size"] == 7
    assert captured["neuron_types"] == ["Mi1", "T4a"]


# ===========================================================================
# build_connectivity_profile_cache
# ===========================================================================


class _FakeProfile:
    def __init__(self, upstream=None, downstream=None, two_hop=False):
        self.upstream_partners = upstream or {}
        self.downstream_partners = downstream or {}
        self.unique_types_upstream = len(self.upstream_partners)
        self.unique_types_downstream = len(self.downstream_partners)
        self.untyped_upstream_2hop = two_hop
        self.untyped_downstream_2hop = False


class FakeProfilerFNC:
    last = None
    progress_calls = []
    profiles = {}

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeProfilerFNC.last = self

    def build_connectivity_profile_cache(self, **kwargs):
        callback = kwargs["progress_callback"]
        FakeProfilerFNC.progress_calls = []
        for args in ((1, 2, "Mi1"), (2, 2, None), (0, 0, "")):
            callback(*args)
            FakeProfilerFNC.progress_calls.append(args)
        return {
            "profiles": dict(FakeProfilerFNC.profiles),
            "failed_types": [],
            "total_profiles": len(FakeProfilerFNC.profiles),
        }


@pytest.fixture
def fake_profiler_fnc(monkeypatch):
    import coana

    FakeProfilerFNC.last = None
    FakeProfilerFNC.profiles = {}
    monkeypatch.setattr(coana, "FindNeuronConnection", FakeProfilerFNC)
    return FakeProfilerFNC


def test_profile_build_cache_with_env_token(monkeypatch, fake_profiler_fnc, capsys):
    # Isolate the chain: config tokens (if any) win over the env, so clear
    # them and keep only the legacy NEUPRINT_TOKEN env alias set.
    from utils.token_manager import token_manager
    monkeypatch.setattr(token_manager, "tokens", {})
    monkeypatch.setattr(token_manager, "_token_sources", {})
    monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.setenv("NEUPRINT_TOKEN", "env-token")
    FakeProfilerFNC.profiles = {
        "Mi1": _FakeProfile({"A": 1}, {"B": 2}, two_hop=True)
    }
    result = bpcp.build_cache("hemibrain:v1.2.1", token=None, max_neurons=10)
    assert result["total_profiles"] == 1
    fake = FakeProfilerFNC.last
    assert fake.kwargs["token"] == "env-token"
    output = capsys.readouterr().out
    assert "[INFO] Using NeuPrint token from the config/env chain" in output
    assert "Sample Profile: Mi1" in output
    assert "Upstream partners: 1" in output
    assert "Downstream partners: 1" in output
    assert FakeProfilerFNC.progress_calls[0] == (1, 2, "Mi1")


def test_profile_build_cache_no_env_token_no_profiles(
    monkeypatch, fake_profiler_fnc, capsys
):
    monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)
    result = bpcp.build_cache("flywire_FAFB_v783", token="direct", force=True)
    assert result["profiles"] == {}
    assert FakeProfilerFNC.last.kwargs["token"] == "direct"
    output = capsys.readouterr().out
    assert "[INFO] Using token" not in output
    assert "Sample Profile" not in output


class _FakeConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeProfiler:
    def __init__(self, config):
        self.config = config

    def read_connectivity_profile_cache(self, dataset):
        return {
            "Mi1": _FakeProfile({"A": 1, "B": 2}, {"C": 3}, two_hop=True),
            "T4a": _FakeProfile({"A": 1}, {}, two_hop=False),
        }


def test_profile_show_stats_full(tmp_path, monkeypatch, capsys):
    _redirect_module_paths(monkeypatch, bpcp, tmp_path)
    safe = "hemibrain_v1_2_1"
    cache_dir = tmp_path / "cache" / safe
    cache_dir.mkdir(parents=True)
    (cache_dir / "connectivity_profiles.parquet").write_bytes(b"dummy")

    import comparison.connectivity_profiler as profiler_module

    # NOTE: the real ``ProfilerConfig`` rejects ``verbose``, so show_stats()
    # would always land in its except branch (see bug report); patch both
    # collaborators to exercise the happy path.
    monkeypatch.setattr(profiler_module, "ProfilerConfig", _FakeConfig)
    monkeypatch.setattr(profiler_module, "ConnectivityProfiler", _FakeProfiler)

    bpcp.show_stats("hemibrain:v1.2.1")
    output = capsys.readouterr().out
    assert "Total profiles: 2" in output
    assert "Upstream partners: avg=1.5, max=2" in output
    assert "Profiles with 2-hop data: 1 (50.0%)" in output
    assert "Mi1" in output and "T4a" in output


def test_profile_show_stats_missing_file(tmp_path, monkeypatch, capsys):
    _redirect_module_paths(monkeypatch, bpcp, tmp_path)
    bpcp.show_stats("hemibrain:v1.2.1")
    assert "[ERROR] Cache not found" in capsys.readouterr().out


def test_profile_show_stats_read_failure(tmp_path, monkeypatch, capsys):
    _redirect_module_paths(monkeypatch, bpcp, tmp_path)
    safe = "hemibrain_v1_2_1"
    cache_dir = tmp_path / "cache" / safe
    cache_dir.mkdir(parents=True)
    (cache_dir / "connectivity_profiles.parquet").write_bytes(b"dummy")

    class _BrokenProfiler:
        def __init__(self, config):
            pass

        def read_connectivity_profile_cache(self, dataset):
            raise RuntimeError("cannot parse")

    import comparison.connectivity_profiler as profiler_module

    monkeypatch.setattr(profiler_module, "ProfilerConfig", _FakeConfig)
    monkeypatch.setattr(profiler_module, "ConnectivityProfiler", _BrokenProfiler)
    bpcp.show_stats("hemibrain:v1.2.1")
    assert "[WARNING] Could not read profiles: cannot parse" in capsys.readouterr().out


def test_profile_show_stats_unpatched_config_falls_to_warning(
    tmp_path, monkeypatch, capsys
):
    """Documents the real-world behavior: ``ProfilerConfig(verbose=False)``
    raises, so show_stats() degrades to the WARNING branch."""
    _redirect_module_paths(monkeypatch, bpcp, tmp_path)
    safe = "hemibrain_v1_2_1"
    cache_dir = tmp_path / "cache" / safe
    cache_dir.mkdir(parents=True)
    (cache_dir / "connectivity_profiles.parquet").write_bytes(b"dummy")

    bpcp.show_stats("hemibrain:v1.2.1")
    assert "[WARNING] Could not read profiles" in capsys.readouterr().out


def test_profile_show_stats_empty_profiles(tmp_path, monkeypatch, capsys):
    _redirect_module_paths(monkeypatch, bpcp, tmp_path)
    safe = "hemibrain_v1_2_1"
    cache_dir = tmp_path / "cache" / safe
    cache_dir.mkdir(parents=True)
    (cache_dir / "connectivity_profiles.parquet").write_bytes(b"dummy")

    class _EmptyProfiler:
        def __init__(self, config):
            pass

        def read_connectivity_profile_cache(self, dataset):
            return {}

    import comparison.connectivity_profiler as profiler_module

    monkeypatch.setattr(profiler_module, "ProfilerConfig", _FakeConfig)
    monkeypatch.setattr(profiler_module, "ConnectivityProfiler", _EmptyProfiler)
    bpcp.show_stats("hemibrain:v1.2.1")
    assert "Total profiles" not in capsys.readouterr().out


def test_profile_main_stats_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(bpcp, "show_stats", lambda dataset: calls.append(dataset))
    monkeypatch.setattr(
        sys, "argv", ["build_connectivity_profile_cache.py", "--stats"]
    )
    bpcp.main()
    assert calls == [bpcp.DEFAULT_DATASET]


def test_profile_main_build_path(monkeypatch):
    captured = {}

    def record(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(bpcp, "build_cache", record)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_connectivity_profile_cache.py",
            "manc:v1.0",
            "--top-k",
            "15",
            "--top-m",
            "8",
            "--no-expand-2hop",
            "--max-neurons",
            "100",
            "--types",
            "Mi1",
            "--force",
            "--token",
            "tok",
        ],
    )
    bpcp.main()
    assert captured["dataset"] == "manc:v1.0"
    assert captured["top_k"] == 15
    assert captured["top_m"] == 8
    assert captured["expand_2hop"] is False
    assert captured["max_neurons"] == 100
    assert captured["neuron_types"] == ["Mi1"]
    assert captured["force"] is True
    assert captured["token"] == "tok"


# ===========================================================================
# build_seed_indexes
# ===========================================================================


SEED_HEADER = "bodyId,type,instance,post"


def _write_seed_csv(tmp_path, dataset, rows, header=SEED_HEADER):
    folder = dataset.replace(":", "_").replace(".", "_")
    source_dir = tmp_path / "datasets" / folder
    source_dir.mkdir(parents=True, exist_ok=True)
    path = source_dir / f"{folder}_allneurons_neuron_df.csv"
    lines = [header] + [",".join(str(value) for value in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_atomic_write_parquet_creates_parents(tmp_path):
    frame = pl.DataFrame({"bodyId": ["1"], "type": ["Mi1"]})
    target = tmp_path / "deep" / "nested" / "index.parquet"
    bsi._atomic_write_parquet(frame, target)
    assert target.exists()
    assert pl.read_parquet(target).to_dicts() == frame.to_dicts()
    # The temporary file must not survive the atomic replace.
    assert list(target.parent.glob("*.tmp-*")) == []


def test_seed_frame_zeroes_cache_state_and_adds_defaults(tmp_path):
    source = _write_seed_csv(
        tmp_path,
        "fake:v9",
        [("1001", "Mi1", "Mi1_R", 12), ("1002", "T4a", "T4a_L", 7)],
    )
    frame = bsi._seed_frame(source)
    assert frame.columns[:4] == ["bodyId", "type", "instance", "post"]
    assert frame["downstream_complete"].to_list() == [False, False]
    assert frame["last_fetched"].to_list() == ["", ""]
    assert frame["connection_count"].to_list() == [0, 0]
    # Source ``post`` values are preserved by the seed projection.
    assert frame["post"].to_list() == [12, 7]


def test_seed_frame_adds_missing_post(tmp_path):
    source = _write_seed_csv(
        tmp_path,
        "fake:v9",
        [("1001", "Mi1"), ("1002", "T4a")],
        header="bodyId,type",
    )
    frame = bsi._seed_frame(source)
    assert frame["post"].to_list() == [0, 0]


def test_build_seed_index_success(tmp_path, monkeypatch):
    monkeypatch.setattr(bsi, "_PROJECT_ROOT", tmp_path)
    _write_seed_csv(
        tmp_path,
        "fake:v9",
        [("1001", "Mi1", "Mi1_R", 12), ("1002", "T4a", "T4a_L", 7)],
    )
    index_dir = tmp_path / "neuron_indexes"

    entry = bsi.build_seed_index("fake:v9", index_dir)

    assert entry["dataset"] == "fake:v9"
    assert entry["folder"] == "fake_v9"
    assert entry["rows"] == 2
    assert entry["index_bytes"] > 0
    assert entry["search_bytes"] > 0

    index_path = index_dir / "fake_v9" / "neuron_index.parquet"
    search_path = index_dir / "fake_v9" / "neuron_index_search.parquet"
    assert index_path.exists()
    assert search_path.exists()

    frame = pl.read_parquet(index_path)
    assert frame["downstream_complete"].to_list() == [False, False]
    assert frame["bodyId"].to_list() == ["1001", "1002"]


def test_build_seed_index_no_metadata(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bsi, "_PROJECT_ROOT", tmp_path)
    (tmp_path / "datasets").mkdir()
    assert bsi.build_seed_index("ghost:v1", tmp_path / "neuron_indexes") is None
    assert "No local metadata table" in capsys.readouterr().out


def test_build_seed_index_duplicate_bodyids(tmp_path, monkeypatch):
    monkeypatch.setattr(bsi, "_PROJECT_ROOT", tmp_path)
    _write_seed_csv(
        tmp_path,
        "fake:v9",
        [("1001", "Mi1", "Mi1_R", 12), ("1001", "T4a", "T4a_L", 7)],
    )
    with pytest.raises(ValueError, match="duplicate bodyIds"):
        bsi.build_seed_index("fake:v9", tmp_path / "neuron_indexes")


def test_build_seed_index_empty_projection(tmp_path, monkeypatch):
    monkeypatch.setattr(bsi, "_PROJECT_ROOT", tmp_path)
    _write_seed_csv(tmp_path, "fake:v9", [])
    with pytest.raises(ValueError, match="no usable bodyIds"):
        bsi.build_seed_index("fake:v9", tmp_path / "neuron_indexes")


def test_read_manifest_variants(tmp_path):
    manifest_path = tmp_path / "manifest.json"

    assert bsi._read_manifest(tmp_path) == {}

    manifest_path.write_text("{not json", encoding="utf-8")
    assert bsi._read_manifest(tmp_path) == {}

    manifest_path.write_text('{"datasets": {"a": 1}}', encoding="utf-8")
    assert bsi._read_manifest(tmp_path) == {"datasets": {"a": 1}}


def test_seed_main_success(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bsi, "_PROJECT_ROOT", tmp_path)
    _write_seed_csv(
        tmp_path,
        "fake:v9",
        [("1001", "Mi1", "Mi1_R", 12), ("1002", "T4a", "T4a_L", 7)],
    )
    monkeypatch.setattr(sys, "argv", ["build_seed_indexes.py", "--datasets", "fake:v9"])

    assert bsi.main() == 0

    manifest = json.loads(
        (tmp_path / "neuron_indexes" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["index_dir"] == "neuron_indexes"
    assert "generated_utc" in manifest
    assert manifest["datasets"]["fake:v9"]["rows"] == 2
    output = capsys.readouterr().out
    assert "- fake:v9" in output
    assert "Manifest written" in output


def test_seed_main_discovers_all_local_metadata_datasets(tmp_path, monkeypatch):
    monkeypatch.setattr(bsi, "_PROJECT_ROOT", tmp_path)
    _write_seed_csv(tmp_path, "manc:v1.0", [("1", "A")], header="bodyId,type")
    _write_seed_csv(
        tmp_path, "optic-lobe:v1.1", [("2", "B")], header="bodyId,type"
    )
    monkeypatch.setattr(sys, "argv", ["build_seed_indexes.py"])

    assert bsi.main() == 0
    manifest = json.loads(
        (tmp_path / "neuron_indexes" / "manifest.json").read_text(encoding="utf-8")
    )
    assert set(manifest["datasets"]) == {"manc:v1.0", "optic-lobe:v1.1"}


def test_seed_main_failure_returns_one(tmp_path, monkeypatch):
    monkeypatch.setattr(bsi, "_PROJECT_ROOT", tmp_path)
    _write_seed_csv(
        tmp_path,
        "fake:v9",
        [("1001", "Mi1", "Mi1_R", 12), ("1001", "T4a", "T4a_L", 7)],
    )
    monkeypatch.setattr(sys, "argv", ["build_seed_indexes.py", "--datasets", "fake:v9"])
    assert bsi.main() == 1


def test_seed_main_nothing_rebuilt(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bsi, "_PROJECT_ROOT", tmp_path)
    (tmp_path / "datasets").mkdir()
    monkeypatch.setattr(sys, "argv", ["build_seed_indexes.py", "--datasets", "ghost:v0"])
    assert bsi.main() == 1
    assert "Nothing was rebuilt." in capsys.readouterr().out

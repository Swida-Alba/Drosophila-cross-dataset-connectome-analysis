"""scripts/maintenance/record_cache_baseline.py — the CLI surface of the
product's offline cache certification (2026-09-18 retest, finding F5).

The gate itself is tested in tests/core/test_cache_coverage.py; what is
tested here is the script around it:

* a manifest-less, layer-1-clean cache certifies and the written manifest
  records ``source: 'user_certified'``;
* a truncated cache is refused, the product's own message is printed
  verbatim (the script never re-words the remedy), and nothing is written;
* an existing manifest is refused without ``--force`` and left untouched;
* the construction never reaches ``FindNeuronConnection.__post_init__`` —
  the only route to a NeuPrint client — so it works with no token and no
  network;
* ``--list`` reports which local caches can be certified.
"""

import importlib.util
import json
import sys
from pathlib import Path

import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import coana  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "record_cache_baseline",
    PROJECT_ROOT / "scripts" / "maintenance" / "record_cache_baseline.py")
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)


@pytest.fixture(autouse=True)
def _clear_coverage_memo():
    coana._CACHE_COVERAGE_CACHE.clear()
    yield
    coana._CACHE_COVERAGE_CACHE.clear()


@pytest.fixture(autouse=True)
def _no_connected_construction(monkeypatch):
    """Forbid the one route that could touch the network.

    ``FindNeuronConnection.__post_init__`` is where a NeuPrint client is
    built (and where a manifest-less cache-only run refuses). The script
    assembles its instance directly, so these tests fail loudly if it ever
    starts constructing one the connecting way.
    """
    def _forbidden(self, *args, **kwargs):
        raise AssertionError(
            "record_cache_baseline must not run "
            "FindNeuronConnection.__post_init__ (it can connect)")

    monkeypatch.setattr(coana.FindNeuronConnection, "__post_init__",
                        _forbidden)
    # No credential configured anywhere: certification is an offline action.
    monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS", raising=False)


def _write_cache(root: Path, dataset: str, connection_rows, index_rows):
    """A synthetic cache + neuron index under *root* (no manifest)."""
    dataset_safe = coana.dataset_folder(dataset)
    cache_dir = root / "cache" / dataset_safe
    cache_dir.mkdir(parents=True)
    pl.DataFrame({
        "bodyId_pre": [r[0] for r in connection_rows],
        "bodyId_post": [r[1] for r in connection_rows],
        "weight": [r[2] for r in connection_rows],
    }).write_parquet(cache_dir / "connections.parquet")
    if index_rows is not None:
        index_dir = root / "neuron_indexes" / dataset_safe
        index_dir.mkdir(parents=True)
        pl.DataFrame({
            "bodyId": [r[0] for r in index_rows],
            "downstream_complete": [r[1] for r in index_rows],
            "connection_count": [r[2] for r in index_rows],
        }).write_parquet(index_dir / "neuron_index.parquet")
    return dataset, dataset_safe


def _manifest_path(root: Path, dataset_safe: str) -> Path:
    return root / "cache" / dataset_safe / "cache_manifest.json"


CLEAN = dict(
    connection_rows=[(100, 200 + i, 5) for i in range(189)],
    index_rows=[(100, True, 189), (101, True, 0)],
)
TRUNCATED = dict(
    connection_rows=[(100, 200 + i, 10) for i in range(12)],
    index_rows=[(100, True, 189)],
)


# ---------------------------------------------------------------------------
# Certifying a manifest-less cache
# ---------------------------------------------------------------------------

def test_certifies_layer_one_clean_cache(tmp_path, capsys):
    dataset, dataset_safe = _write_cache(tmp_path, "alpha:v1.0",
                                         **CLEAN)
    code = script.main(["--dataset", dataset,
                        "--project-root", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 0, captured.out + captured.err

    path = _manifest_path(tmp_path, dataset_safe)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["source"] == "user_certified"
    assert manifest["dataset"] == dataset
    assert manifest["distinct_connections"] == 189

    # The current state is reported before anything is written.
    assert "current state: none" in captured.out
    # ... and the result names the count and the manifest path.
    assert "distinct_connections: 189" in captured.out
    assert str(path) in captured.out


def test_certification_unlocks_a_cache_only_run(tmp_path):
    """The point of the whole exercise (F5): the refusal is gone."""
    dataset, _safe = _write_cache(tmp_path, "alpha:v1.0", **CLEAN)
    connection = script.build_offline_connection(dataset, tmp_path)
    with pytest.raises(RuntimeError, match="no integrity manifest"):
        connection._enforce_cache_coverage("user_requested")
    script.main(["--dataset", dataset, "--project-root", str(tmp_path)])
    coana._CACHE_COVERAGE_CACHE.clear()
    connection._enforce_cache_coverage("user_requested")  # must not raise


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_truncated_cache_refuses_with_the_products_words(tmp_path, capsys):
    dataset, dataset_safe = _write_cache(tmp_path, "alpha:v1.0", **TRUNCATED)
    # The exact text the product raises (never re-worded by the script).
    probe = script.build_offline_connection(dataset, tmp_path)
    with pytest.raises(RuntimeError) as expected:
        probe.record_cache_baseline()

    code = script.main(["--dataset", dataset,
                        "--project-root", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 1
    assert str(expected.value) in captured.err
    assert "Refusing to certify" in captured.err
    assert not _manifest_path(tmp_path, dataset_safe).exists()


def test_existing_manifest_is_refused_without_force(tmp_path, capsys):
    dataset, dataset_safe = _write_cache(tmp_path, "alpha:v1.0", **CLEAN)
    assert script.main(["--dataset", dataset,
                        "--project-root", str(tmp_path)]) == 0
    path = _manifest_path(tmp_path, dataset_safe)
    recorded = json.loads(path.read_text(encoding="utf-8"))
    capsys.readouterr()

    code = script.main(["--dataset", dataset,
                        "--project-root", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 1
    assert "already has a cache integrity manifest" in captured.err
    # The refusal is the product's; the --force hint is the script's own line.
    assert "--force" in captured.err
    # Nothing was overwritten.
    assert json.loads(path.read_text(encoding="utf-8")) == recorded
    assert "current state: source=user_certified" in captured.out


def test_force_rerecords_an_existing_manifest(tmp_path, capsys):
    dataset, dataset_safe = _write_cache(tmp_path, "alpha:v1.0", **CLEAN)
    script.main(["--dataset", dataset, "--project-root", str(tmp_path)])
    capsys.readouterr()

    code = script.main(["--dataset", dataset, "--force",
                        "--project-root", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 0, captured.out + captured.err
    manifest = json.loads(
        _manifest_path(tmp_path, dataset_safe).read_text(encoding="utf-8"))
    assert manifest["source"] == "user_certified"
    assert "current state: source=user_certified" in captured.out


def test_a_cache_without_an_index_has_nothing_to_certify(tmp_path, capsys):
    """No neuron index: completeness cannot be checked, so nothing is
    certified (and the refusal is the product's)."""
    dataset, dataset_safe = _write_cache(
        tmp_path, "alpha:v1.0",
        connection_rows=[(100, 200, 5)], index_rows=None)
    code = script.main(["--dataset", dataset,
                        "--project-root", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 1
    assert "Nothing to certify" in captured.err
    assert not _manifest_path(tmp_path, dataset_safe).exists()


def test_dataset_is_required_unless_listing():
    with pytest.raises(SystemExit) as exit_info:
        script.main([])
    assert exit_info.value.code == 2


# ---------------------------------------------------------------------------
# Construction and --list
# ---------------------------------------------------------------------------

def test_offline_connection_carries_no_client(tmp_path):
    """The instance is assembled for the cache check only: no NeuPrint
    client, no connecting init (see the autouse guard above)."""
    dataset, _safe = _write_cache(tmp_path, "alpha:v1.0", **CLEAN)
    connection = script.build_offline_connection(dataset, tmp_path)
    assert isinstance(connection, coana.FindNeuronConnection)
    assert connection.dataset == dataset
    assert connection.cache_only is True
    assert connection.client_hemibrain is None
    assert connection._cache_manifest_path().endswith(
        str(Path("cache") / "alpha_v1_0" / "cache_manifest.json"))


def test_script_uses_its_own_offline_factory(tmp_path, monkeypatch):
    """Every instance the script touches comes from the patchable factory,
    never from a connecting ``FindNeuronConnection(...)``."""
    dataset, dataset_safe = _write_cache(tmp_path, "alpha:v1.0", **CLEAN)
    real = script.build_offline_connection
    built = []

    def factory(name, script_path=None):
        built.append(name)
        return real(name, script_path)

    monkeypatch.setattr(script, "build_offline_connection", factory)
    assert script.main(["--dataset", dataset,
                        "--project-root", str(tmp_path)]) == 0
    # One instance for the state report, one for the certification itself.
    assert built == [dataset, dataset]
    manifest = json.loads(
        _manifest_path(tmp_path, dataset_safe).read_text(encoding="utf-8"))
    assert manifest["source"] == "user_certified"


def test_refusal_wording_is_the_products(tmp_path, monkeypatch, capsys):
    """A monkeypatched product refusal reaches stdout/stderr untouched."""
    class _Refusing:
        dataset = "alpha:v1.0"

        def _load_cache_manifest(self):
            return None

        def _cache_manifest_path(self):
            return str(tmp_path / "cache" / "alpha_v1_0"
                       / "cache_manifest.json")

        def record_cache_baseline(self, force=False):
            raise RuntimeError("PRODUCT-OWNED REFUSAL TEXT")

    monkeypatch.setattr(script, "build_offline_connection",
                        lambda *args, **kwargs: _Refusing())
    code = script.main(["--dataset", "alpha:v1.0",
                        "--project-root", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 1
    assert "PRODUCT-OWNED REFUSAL TEXT" in captured.err
    assert "PRODUCT-OWNED REFUSAL TEXT\nHint" not in captured.err


# ---------------------------------------------------------------------------
# --list
# ---------------------------------------------------------------------------

def test_list_reports_what_can_be_certified(tmp_path, capsys):
    _write_cache(tmp_path, "alpha:v1.0", **CLEAN)            # certifiable
    already, already_safe = _write_cache(tmp_path, "beta:v1.0", **CLEAN)
    _write_cache(tmp_path, "gamma:v1.0",                     # no index
                 connection_rows=[(100, 200, 5)], index_rows=None)
    _manifest_path(tmp_path, already_safe).write_text(json.dumps({
        "schema": coana.CACHE_MANIFEST_SCHEMA, "dataset": already,
        "distinct_connections": 189, "source": "baseline"}),
        encoding="utf-8")
    capsys.readouterr()

    assert script.main(["--list", "--project-root", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert captured.out.count("CERTIFIABLE") == 1
    assert "re-record it only with --force" in captured.out
    assert "nothing to certify" in captured.out
    assert "1 connection cache(s) can be certified" in captured.out
    # Folders are shown as the identifier a run would certify under.
    assert "alpha:v1.0" in captured.out
    assert "beta:v1.0" in captured.out
    assert "gamma:v1.0" in captured.out

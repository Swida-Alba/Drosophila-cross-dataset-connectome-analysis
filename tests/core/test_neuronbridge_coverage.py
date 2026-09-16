"""Tests for the NeuronBridge dataset-coverage snapshot service.

The coverage contract (plan
`_plan/plan-nb-find-lines-dataset-aware-queries.md` §5): verdicts are
advisory, ``unknown`` never disables, an unreachable NeuronBridge never
invalidates a persisted snapshot, and a single-body 404 is noise — only a
sample-wide answer settles ``unavailable``.
"""

import json
import sys
import threading
import types
from pathlib import Path

import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import src.neuronbridge_coverage as nbc  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _Record:
    """Minimal stand-in for a by_body EMImage record."""

    def __init__(self, library, published):
        self.libraryName = library
        self.publishedName = published


def _http_style_error(status):
    """A client-style error chain carrying an HTTP status code."""
    cause = Exception(f"{status} Client Error")
    cause.response = types.SimpleNamespace(status_code=status)
    exc = RuntimeError(f"Could not retrieve url: {status}")
    exc.__cause__ = cause
    return exc


class StubClient:
    """``_get_json`` behavior keyed by bodyId in the URL."""

    data_url = "https://stub.example/v3_10_0"
    version = "v3_10_0"

    def __init__(self, behavior=None):
        self.calls = []
        self._behavior = behavior or (lambda url: {"results": []})
        self._lock = threading.Lock()

    def _get_json(self, url):
        with self._lock:
            self.calls.append(url)
        return self._behavior(url)


def _by_body(url):
    return url


def _records_response(*records):
    return {"results": [
        {"libraryName": rec.libraryName, "publishedName": rec.publishedName}
        if isinstance(rec, _Record) else rec
        for rec in records
    ]}


@pytest.fixture
def fresh_root(tmp_path, monkeypatch):
    """Isolate the snapshot store and per-test freshness."""
    root = tmp_path / "nb-cache"
    monkeypatch.setattr(nbc, "default_cache_root", lambda: root)
    return root


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------

class TestClassifyRecords:
    def test_exact_version_match_wins_over_aligned_candidates(self):
        verdict = nbc.classify_records("male-cns:v0.9", [
            _Record("FlyEM_Male_CNS_Brain_v0.9", "male-cns:v0.9:10001"),
        ])
        assert verdict.status == nbc.STATUS_EXACT
        assert verdict.hosted_version == "male-cns:v0.9"
        assert "FlyEM_Male_CNS_Brain_v0.9" in verdict.evidence

    def test_v10_bodies_align_to_hosted_v09_library(self):
        verdict = nbc.classify_records("male-cns:v1.0", [
            _Record("FlyEM_VNC_v0.5", "vnc:v0.5:10001"),
            _Record("FlyEM_MANC_v1.2.1", "manc:v1.2.1:10001"),
            _Record("FlyEM_Male_CNS_Brain_v0.9", "male-cns:v0.9:10001"),
        ])
        # The unrelated bases (VNC, MANC) are ignored; the male-cns base at
        # a different version is the aligned verdict.
        assert verdict.status == nbc.STATUS_ALIGNED
        assert verdict.hosted_version == "male-cns:v0.9"
        assert "FlyEM_MANC_v1.2.1" not in verdict.evidence

    def test_unrelated_base_only_is_unavailable_not_aligned(self):
        # The optic-lobe collision class: small ids that belong to
        # unrelated MANC/VNC neurons must never read as optic-lobe cover.
        verdict = nbc.classify_records("optic-lobe:v1.1", [
            _Record("FlyEM_MANC_v1.2.1", "manc:v1.2.1:27769"),
        ])
        assert verdict.status == nbc.STATUS_UNAVAILABLE

    def test_unresolvable_base_identity_is_unknown(self):
        # 'unknown' (and empty) identities cannot be classified at all —
        # the finder's dataset-identity parser maps them to no base.
        verdict = nbc.classify_records("unknown", [
            _Record("FlyEM_MANC_v1.2.1", "manc:v1.2.1:27769"),
        ])
        assert verdict.status == nbc.STATUS_UNKNOWN


# ---------------------------------------------------------------------------
# probing
# ---------------------------------------------------------------------------

class TestProbeCoverage:
    def test_no_local_sample_is_unknown_never_unavailable(self):
        client = StubClient()
        verdict = nbc.probe_coverage("banc_v626", [], client)
        assert verdict.status == nbc.STATUS_UNKNOWN
        assert client.calls == []

    def test_single_body_misses_are_noise_a_hosted_body_settles_exact(self):
        behavior = {
            "111": _http_style_error(404),
            "222": _records_response(
                _Record("FlyEM_Hemibrain_v1.2.1", "hemibrain:v1.2.1:222")),
        }
        client = StubClient(lambda url: behavior[url.rsplit("/", 1)[1].split(".")[0]])
        verdict = nbc.probe_coverage(
            "hemibrain:v1.2.1", ["111", "222"], client)
        assert verdict.status == nbc.STATUS_EXACT

    def test_all_sample_misses_settle_unavailable(self):
        client = StubClient(
            lambda url: (_ for _ in ()).throw(_http_style_error(404)))
        verdict = nbc.probe_coverage("banc_v626", ["1", "2"], client)
        assert verdict.status == nbc.STATUS_UNAVAILABLE

    def test_network_failure_is_unknown(self):
        def down(url):
            raise RuntimeError("connection refused (no status)")

        client = StubClient(down)
        verdict = nbc.probe_coverage("male-cns:v0.9", ["1", "2"], client)
        assert verdict.status == nbc.STATUS_UNKNOWN

    def test_mixed_errors_and_misses_still_classify(self):
        behavior = {
            "1": _http_style_error(404),
            "2": _http_style_error(404),
        }
        client = StubClient(lambda url: behavior[url.rsplit("/", 1)[1].split(".")[0]])
        verdict = nbc.probe_coverage("male-cns:v1.0", ["1", "2"], client)
        assert verdict.status == nbc.STATUS_UNAVAILABLE


# ---------------------------------------------------------------------------
# snapshot plumbing
# ---------------------------------------------------------------------------

def _write_snapshot(root, payload):
    path = root / nbc.SNAPSHOT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _down_client():
    class _Down:
        data_url = "https://stub/v3"

        @property
        def version(self):
            raise RuntimeError("neuronbridge unreachable")

        def _get_json(self, url):
            raise RuntimeError("unreachable")

    return _Down()


class TestRefresh:
    def test_offline_never_invalidates_persisted_snapshot(self, fresh_root):
        _write_snapshot(fresh_root, {
            "nb_version": "v3_10_0",
            "created_at": "2026-09-15T00:00:00+00:00",
            "datasets": {
                "hemibrain:v1.2.1": {"status": "exact",
                                     "checked_at": "2026-09-15T00:00:00+00:00"},
            },
        })
        snapshot = nbc.refresh(
            ["hemibrain:v1.2.1", "banc_v626"],
            client=_down_client(), cache_root=fresh_root,
        )
        # The unreachable bridge returns the persisted snapshot untouched;
        # the missing dataset is NOT rewritten to unavailable.
        assert snapshot.nb_version == "v3_10_0"
        assert snapshot.coverage_of("hemibrain:v1.2.1").status == "exact"
        assert "banc_v626" not in snapshot.datasets

    def test_fresh_snapshot_reuses_verdicts_without_probing(self, fresh_root):
        _write_snapshot(fresh_root, {
            "nb_version": "v3_10_0",
            "created_at": "2026-09-15T00:00:00+00:00",
            "datasets": {
                "male-cns:v1.0": {
                    "status": "aligned",
                    "hosted_version": "male-cns:v0.9",
                    "checked_at": "2026-09-15T00:00:00+00:00",
                },
            },
        })
        client = StubClient()  # would answer with empty results if probed
        snapshot = nbc.refresh(
            ["male-cns:v1.0"], client=client,
            cache_root=fresh_root, nb_version="v3_10_0",
        )
        assert snapshot.coverage_of("male-cns:v1.0").status == "aligned"
        assert client.calls == []

    def test_version_change_invalidates_and_reprobes(self, fresh_root):
        _write_snapshot(fresh_root, {
            "nb_version": "v3_9_0",
            "created_at": "2026-09-15T00:00:00+00:00",
            "datasets": {
                "manc:v1.2.3": {
                    "status": "unavailable",
                    "checked_at": "2026-09-15T00:00:00+00:00",
                },
            },
        })
        # The new version hosts the same bodies under manc:v1.2.1.
        client = StubClient(lambda url: _records_response(
            _Record("FlyEM_MANC_v1.2.1", "manc:v1.2.1:10120")))
        snapshot = nbc.refresh(
            ["manc:v1.2.3"], client=client,
            cache_root=fresh_root, nb_version="v3_10_0",
        )
        assert snapshot.nb_version == "v3_10_0"
        assert snapshot.coverage_of("manc:v1.2.3").status == "aligned"
        assert snapshot.coverage_of("manc:v1.2.3").hosted_version == "manc:v1.2.1"
        persisted = nbc.load_snapshot(fresh_root)
        assert persisted.coverage_of("manc:v1.2.3").status == "aligned"

    def test_ttl_expiry_reprobes_stale_verdicts(self, fresh_root):
        _write_snapshot(fresh_root, {
            "nb_version": "v3_10_0",
            "created_at": "2026-01-01T00:00:00+00:00",
            "datasets": {
                "manc:v1.2.3": {
                    "status": "unavailable",
                    "checked_at": "2026-01-01T00:00:00+00:00",
                },
            },
        })
        client = StubClient(lambda url: _records_response(
            _Record("FlyEM_MANC_v1.2.1", "manc:v1.2.1:10120")))
        snapshot = nbc.refresh(
            ["manc:v1.2.3"], client=client,
            cache_root=fresh_root, nb_version="v3_10_0",
        )
        assert snapshot.coverage_of("manc:v1.2.3").status == "aligned"


# ---------------------------------------------------------------------------
# snapshot consumers
# ---------------------------------------------------------------------------

def _hand_snapshot():
    return nbc.CoverageSnapshot.from_dict({
        "nb_version": "v3_10_0",
        "created_at": "2026-09-15T00:00:00+00:00",
        "datasets": {
            "hemibrain:v1.2.1": {
                "status": "exact",
                "hosted_version": "hemibrain:v1.2.1",
                "checked_at": "2026-09-15T00:00:00+00:00",
            },
            "male-cns:v1.0": {
                "status": "aligned",
                "hosted_version": "male-cns:v0.9",
                "checked_at": "2026-09-15T00:00:00+00:00",
            },
            "banc_v626": {
                "status": "unavailable",
                "checked_at": "2026-09-15T00:00:00+00:00",
            },
            "banc_v888": {
                "status": "unavailable",
                "checked_at": "2026-09-15T00:00:00+00:00",
            },
        },
    })


class TestSnapshotConsumers:
    def test_covered_datasets_map_selected_to_hosted_releases(self):
        covered = _hand_snapshot().covered_datasets(
            ["hemibrain:v1.2.1", "male-cns:v1.0", "banc_v626", "banc_v888"])
        assert covered == {
            "hemibrain:v1.2.1": "hemibrain:v1.2.1",
            "male-cns:v1.0": "male-cns:v0.9",
        }

    def test_aligned_hosted_name_canonicalizes_to_drocat_spelling(self):
        snapshot = nbc.CoverageSnapshot.from_dict({
            "nb_version": "v3_10_0",
            "created_at": "2026-09-15T00:00:00+00:00",
            "datasets": {
                "flywire_FAFB_v782": {
                    "status": "aligned",
                    "hosted_version": "flywire_fafb:v783",
                    "checked_at": "2026-09-15T00:00:00+00:00",
                },
            },
        })
        covered = snapshot.covered_datasets(["flywire_FAFB_v782"])
        assert covered == {"flywire_FAFB_v782": "flywire_FAFB_v783"}

    def test_unknown_datasets_are_not_disabled_and_never_warn(self):
        snapshot = _hand_snapshot()
        assert snapshot.unavailable_datasets(
            ["banc_v626", "banc_v888"]) == ["banc_v626", "banc_v888"]
        assert snapshot.unavailable_datasets(["male-cns:v0.9"]) == []
        assert nbc.warnings_for(["banc_v626"], _hand_snapshot()) != []
        assert nbc.warnings_for(["male-cns:v0.9"], _hand_snapshot()) == []
        assert nbc.warnings_for(["anything"], None) == []


# ---------------------------------------------------------------------------
# local sampling
# ---------------------------------------------------------------------------

class TestSampleBodyIds:
    def _project_frame(self, rows):
        return pl.DataFrame(
            rows, schema={"bodyId": pl.Utf8, "type": pl.Utf8},
            orient="row",
        )

    def test_samples_one_body_per_distinct_typed_row(self, monkeypatch):
        frame = self._project_frame([
            ("100", "Unknown"),      # untyped: skipped
            ("101", "Tm4"),
            ("102", "Tm4"),          # duplicate type: skipped
            ("103", "L2"),
            ("104", None),           # untyped: skipped
            ("105", "T5a"),
            ("106", "Tm1"),
            ("107", "Tm2"),
        ])
        monkeypatch.setattr(nbc, "_load_local_projection", lambda ds: frame)
        picked = nbc.sample_body_ids("x:y", limit=4)
        assert picked == ["101", "103", "105", "106"]

    def test_missing_table_samples_nothing(self, monkeypatch):
        monkeypatch.setattr(nbc, "_load_local_projection", lambda ds: None)
        assert nbc.sample_body_ids("missing:x") == []

    def test_real_local_table_sampling_is_deterministic(self):
        # Uses whatever is locally pulled; must never raise and must be
        # stable, with untyped rows (e.g. BANC 'Unknown') filtered out.
        first = nbc.sample_body_ids("male-cns:v1.0")
        second = nbc.sample_body_ids("male-cns:v1.0")
        assert first == second
        assert len(first) <= nbc.SAMPLE_FIRST_ROUND + nbc.SAMPLE_SECOND_ROUND

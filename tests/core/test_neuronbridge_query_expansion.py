"""Tests for the NeuronBridge Find Lines cross-dataset expansion.

The orchestration contract (plan
`_plan/plan-nb-find-lines-dataset-aware-queries.md` §7): each name chip
expands through the type mapper's shared policy API into every covered
release and runs one find_lines_batch call per chip; bodyId chips bypass
expansion; mapper-unavailable / coverage-unknown degrade to the ordinary
query path; unavailable datasets warn but never block.
"""

import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import src.neuronbridge_query_expansion as nqe  # noqa: E402
import src.neuronbridge_coverage as nbc  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

class FakeFinder:
    """Records find_lines_batch calls; optionally simulates the finder's
    on-disk outputs (inner timestamped folder + match/summary files) so the
    prune pass has something to act on."""

    def __init__(self, simulate_files: bool = False):
        self.calls = []
        self.simulate_files = simulate_files
        self._progress = lambda *a, **k: None
        self._vprint = lambda *a, **k: None

    def find_lines_batch(self, queries, dataset=None, output_dir=None, **params):
        self.calls.append({
            "queries": list(queries) if isinstance(queries, (list, tuple)) else queries,
            "dataset": dataset,
            "output_dir": output_dir,
            "params": params,
        })
        if self.simulate_files and output_dir:
            inner = Path(output_dir) / "NB-find-lines_ALL_q_20260915_000000"
            inner.mkdir(parents=True, exist_ok=True)
            for q in (queries if isinstance(queries, list) else [queries]):
                (inner / f"{q}_lines.csv").write_text("line,score\n")
            (inner / "line_summary.csv").write_text("line,weighted_score\n")
            (inner / "parameters.json").write_text("{}")
        rows = [
            {"line": "VT000000", "score": 1.0, "source_query": str(q)}
            for q in (queries if isinstance(queries, list) else [queries])
        ]
        return pd.DataFrame(rows)


class FakeMapperSnapshot:
    loaded = True

    def __init__(self, mapper=None, requested=True):
        self.mapper = mapper


def _resolution(status, targets=(), source_type="x", kind=""):
    return types.SimpleNamespace(
        status=status,
        kind=kind,
        source_type=source_type,
        target_types=tuple(targets),
        source_dataset="male-cns:v1.0",
        target_dataset="male-cns:v0.9",
    )


def _install_mapper(monkeypatch, resolve_behavior=None):
    """Patch the module's mapper plumbing with network-free fakes."""
    registry = {"calls": []}

    def fake_resolve(mapper, chip, source_ds, target_ds, **kwargs):
        registry["calls"].append((chip, target_ds))
        return resolve_behavior(chip, target_ds)

    monkeypatch.setattr(nqe, "HAS_TYPE_MAPPER", True)
    monkeypatch.setattr(nqe, "get_type_mapper", lambda: object())
    monkeypatch.setattr(nqe, "MapperSnapshot", FakeMapperSnapshot)
    monkeypatch.setattr(nqe, "resolve_valid_targets", fake_resolve)
    return registry


def _fake_snapshot(monkeypatch, datasets_payload):
    """Inject a hand-built snapshot into the orchestration's plumbing
    (no network, no disk)."""
    snapshot = nbc.CoverageSnapshot.from_dict({
        "nb_version": "v3_10_0",
        "created_at": "2026-09-15T00:00:00+00:00",
        "datasets": datasets_payload,
    })
    monkeypatch.setattr(
        nqe, "refresh_coverage",
        lambda datasets, cache_root=None, **kwargs: snapshot)
    monkeypatch.setattr(nqe, "load_snapshot", lambda root=None: snapshot)
    return snapshot


@pytest.fixture
def fake_finder(monkeypatch):
    """Replace the inner NeuronBridgeFinder construction (network-free)."""
    fake = FakeFinder()
    monkeypatch.setattr(nqe, "_build_finder", lambda **kwargs: fake)
    return fake


@pytest.fixture
def writing_finder(fake_finder):
    """A fake finder that also writes the on-disk outputs a real run would."""
    fake_finder.simulate_files = True
    return fake_finder


def _make_finder():
    return nqe.ExpandedLineFinder(verbose=False)


# ---------------------------------------------------------------------------
# expand_chip (the mapper-policy boundary)
# ---------------------------------------------------------------------------

class TestExpandChip:
    def test_mapped_chip_expands_to_target_local_names(self, monkeypatch):
        _install_mapper(monkeypatch, lambda chip, ds: _resolution(
            "mapped", targets=("DNp01",)))
        expanded = nqe.expand_chip(
            "P9-2", {"male-cns:v1.0": "male-cns:v0.9"}, mapper=object(),
            mapper_snapshot=FakeMapperSnapshot())
        assert [(row["expanded_name"], row["nb_dataset"])
                for row in expanded] == [("DNp01", "male-cns:v0.9")]
        assert expanded[0]["mapping_status"] == "mapped"

    def test_split_chip_expands_to_all_licensed_branches(self, monkeypatch):
        def behavior(chip, ds):
            if ds == "flywire_FAFB_v783":
                return _resolution("valid_split_evidence",
                                   targets=("Tm4a", "Tm4b"))
            return _resolution("mapped", targets=("Tm4",))

        _install_mapper(monkeypatch, behavior)
        expanded = nqe.expand_chip(
            "Tm4", {
                "male-cns:v1.0": "male-cns:v0.9",
                "flywire_FAFB_v782": "flywire_FAFB_v783",
            }, mapper=object(), mapper_snapshot=FakeMapperSnapshot())
        names = {(row["expanded_name"], row["nb_dataset"])
                 for row in expanded}
        assert names == {
            ("Tm4", "male-cns:v0.9"),
            ("Tm4a", "flywire_FAFB_v783"),
            ("Tm4b", "flywire_FAFB_v783"),
        }

    def test_conflict_chip_licenses_no_target_at_that_release(self, monkeypatch):
        _install_mapper(monkeypatch, lambda chip, ds: _resolution(
            "conflict", targets=()))
        expanded = nqe.expand_chip(
            "CB1011", {"banc_v626": "banc_v626"}, mapper=object(),
            mapper_snapshot=FakeMapperSnapshot())
        assert expanded == [{
            "expanded_name": "", "nb_dataset": "banc_v626",
            "mapping_status": "conflict",
            "mapping_kind": "no licensed target",
        }]

    def test_unmapped_chip_keeps_the_raw_name(self, monkeypatch):
        # The real expansion_targets policy returns (source_type,) for
        # unmapped chips — the long-tail fallback keeps the raw name.
        _install_mapper(monkeypatch, lambda chip, ds: _resolution(
            "unmapped", source_type=chip))
        expanded = nqe.expand_chip(
            "WeirdName", {"male-cns:v1.0": "male-cns:v0.9"}, mapper=object(),
            mapper_snapshot=FakeMapperSnapshot())
        assert expanded[0]["expanded_name"] == "WeirdName"
        assert expanded[0]["mapping_status"] == "unmapped"

    def test_mapper_unavailable_passes_the_chip_through(self):
        expanded = nqe.expand_chip(
            "P9-2", {"male-cns:v1.0": "male-cns:v0.9"}, mapper=None)
        assert [row["mapping_status"] for row in expanded] == \
            ["mapper unavailable"]
        assert expanded[0]["expanded_name"] == "P9-2"


# ---------------------------------------------------------------------------
# orchestration (ExpandedLineFinder.run)
# ---------------------------------------------------------------------------

class TestRun:
    def test_chip_expands_and_runs_one_batch_per_chip(
        self, monkeypatch, writing_finder, tmp_path
    ):
        registry = _install_mapper(
            monkeypatch, lambda chip, ds: _resolution("mapped", targets=("DNp01",)))
        _fake_snapshot(monkeypatch, {
            "male-cns:v1.0": {
                "status": "aligned",
                "hosted_version": "male-cns:v0.9",
                "checked_at": "2026-09-15T00:00:00+00:00",
            },
            "banc_v626": {
                "status": "unavailable",
                "checked_at": "2026-09-15T00:00:00+00:00",
            },
        })
        finder = _make_finder()
        result = finder.run(
            queries=["DNp01", "12345"],
            dataset=["male-cns:v1.0", "banc_v626"],
            expand_names=True,
            coverage_datasets=["male-cns:v1.0", "banc_v626"],
            match_type="cds",
            output_dir=str(tmp_path),
        )

        # One expanded call for the name chip at the hosted release, one
        # plain call for the bodyId chip — each in its own chip folder.
        assert len(writing_finder.calls) == 2
        name_call, id_call = writing_finder.calls
        assert name_call["queries"] == ["DNp01"]
        assert name_call["dataset"] == ["male-cns:v0.9"]
        assert name_call["params"]["match_type"] == "cds"
        assert id_call["queries"] == ["12345"]
        assert id_call["dataset"] == ["male-cns:v1.0", "banc_v626"]
        run_root = Path(name_call["output_dir"]).parent
        assert run_root.name.startswith("NB-find-lines-expanded_")
        assert run_root.parent == Path(tmp_path)
        assert Path(name_call["output_dir"]).name == "chip_DNp01"
        assert Path(id_call["output_dir"]) == run_root / "chip_12345"

        # Chip provenance: the name chip's rows carry the original chip
        # (not the expanded name); the bodyId chip carries its own id.
        assert set(result["source_query"]) == {"DNp01", "12345"}

        # The expansion map records chip → name → hosted release, deduped
        # per chip (the same name at two releases is ONE query).
        frame = pd.read_csv(run_root / nqe.EXPANSION_MAP_FILENAME)
        assert set(frame["source_query"]) == {"DNp01"}
        assert frame.iloc[0]["nb_dataset"] == "male-cns:v0.9"

        # The run summary preserves the ORIGINAL selection and routing.
        summary = json.loads(
            (run_root / nqe.EXPANSION_SUMMARY_FILENAME).read_text())
        assert summary["selected_datasets"] == ["male-cns:v1.0", "banc_v626"]
        assert summary["covered_datasets"] == {
            "male-cns:v1.0": "male-cns:v0.9"}
        assert summary["unavailable_datasets"] == ["banc_v626"]

        # Full (default): the simulated match tables survive; the coverage
        # warning lands in the unified user_warning_notes.txt.
        assert (run_root / "chip_DNp01" / "DNp01_lines.csv").exists()
        notes = (run_root / nqe.WARNINGS_FILENAME).read_text()
        assert "coverage: banc_v626" in notes
        assert registry["calls"] == [("DNp01", "male-cns:v0.9")]

    def test_compact_prunes_match_tables_and_audits(
        self, monkeypatch, writing_finder, tmp_path
    ):
        _install_mapper(
            monkeypatch, lambda chip, ds: _resolution("mapped", targets=("DNp01",)))
        _fake_snapshot(monkeypatch, {
            "male-cns:v1.0": {
                "status": "aligned",
                "hosted_version": "male-cns:v0.9",
                "checked_at": "2026-09-15T00:00:00+00:00",
            },
        })
        finder = _make_finder()
        finder.run(
            queries=["DNp01"],
            dataset=["male-cns:v1.0"],
            expand_names=True,
            coverage_datasets=["male-cns:v1.0"],
            output_dir=str(tmp_path),
            keep_per_match_csv=False,
            cleanup_source_images=True,
            compact_keep_last_n=0,  # immediate deletion (no window)
        )
        # recover the run root from the finder call (single-chip -> flat)
        run_root = Path(writing_finder.calls[0]["output_dir"]).parent

        # Compact: the chip folder was hoisted (single-chip run is flat)
        # and the bodyId-level match table is gone; summaries + audit stay.
        assert run_root.name.startswith("NB-find-lines-expanded_")
        assert (run_root / "line_summary.csv").exists()
        assert not (run_root / "DNp01_lines.csv").exists()
        audit = json.loads((run_root / "cleanup_audit.json").read_text())
        assert len(audit["removed"]) == 1
        summary = json.loads(
            (run_root / nqe.EXPANSION_SUMMARY_FILENAME).read_text())
        assert summary["output_detail"] == {
            "keep_per_match_csv": False, "cleanup_source_images": True,
            "compact_keep_last_n": 0}
        assert summary["cleanup"]["total_removed"] == 1

    def test_retention_window_keeps_current_run_match_tables(
        self, monkeypatch, writing_finder, tmp_path
    ):
        _install_mapper(
            monkeypatch, lambda chip, ds: _resolution("mapped", targets=("DNp01",)))
        _fake_snapshot(monkeypatch, {
            "male-cns:v1.0": {
                "status": "exact",
                "hosted_version": "male-cns:v1.0",
                "checked_at": "2026-09-15T00:00:00+00:00",
            },
        })
        # An older Compact sibling run: opted in, so the window may sweep it.
        old_run = tmp_path / "NB-find-lines-expanded_ALL_OLD_20260914_000000"
        old_run.mkdir(parents=True)
        (old_run / "OLD_lines.csv").write_text("line\n")
        # The safety condition: only runs whose summary exists may have
        # their match tables swept.
        (old_run / "line_summary.csv").write_text("line\n")
        (old_run / "parameters.json").write_text(json.dumps({
            "function_params": {"keep_per_match_csv": False}}))

        finder = _make_finder()
        finder.run(
            queries=["DNp01"], dataset=["male-cns:v1.0"],
            expand_names=True, coverage_datasets=["male-cns:v1.0"],
            output_dir=str(tmp_path),
            keep_per_match_csv=False, cleanup_source_images=True,
            compact_keep_last_n=1,
        )
        run_root = Path(writing_finder.calls[0]["output_dir"]).parent

        # The current run is inside the window: its match table survives.
        assert (run_root / "DNp01_lines.csv").exists()
        # The older Compact run was swept.
        assert not (old_run / "OLD_lines.csv").exists()
        audit = json.loads((old_run / "cleanup_audit.json").read_text())
        assert len(audit["removed"]) == 1
        summary = json.loads(
            (run_root / nqe.EXPANSION_SUMMARY_FILENAME).read_text())
        assert summary["retention"]["pruned_runs"] == [old_run.name]
        assert summary["output_detail"]["compact_keep_last_n"] == 1

    def test_full_single_chip_run_is_flat(
        self, monkeypatch, writing_finder, tmp_path
    ):
        _install_mapper(
            monkeypatch, lambda chip, ds: _resolution("mapped", targets=("DNp01",)))
        _fake_snapshot(monkeypatch, {
            "male-cns:v1.0": {
                "status": "exact",
                "hosted_version": "male-cns:v1.0",
                "checked_at": "2026-09-15T00:00:00+00:00",
            },
        })
        finder = _make_finder()
        finder.run(
            queries=["DNp01"], dataset=["male-cns:v1.0"],
            expand_names=True, coverage_datasets=["male-cns:v1.0"],
            output_dir=str(tmp_path),
        )
        run_root = Path(writing_finder.calls[0]["output_dir"]).parent
        # Flat: the chip's files (hoisted from the inner finder folder) sit
        # in the run root next to the expansion reports.
        assert (run_root / "DNp01_lines.csv").exists()
        assert (run_root / "line_summary.csv").exists()
        assert not list(run_root.glob("chip_*"))
        assert not list(run_root.glob("NB-find-lines_ALL_*"))

    def test_expand_off_delegates_untouched(
        self, monkeypatch, fake_finder, tmp_path
    ):
        finder = _make_finder()
        finder.run(
            queries=["DNp01"], dataset=["male-cns:v1.0"],
            expand_names=False, output_dir=str(tmp_path),
            keep_per_match_csv=False, cleanup_source_images=True,
        )
        assert len(fake_finder.calls) == 1
        assert fake_finder.calls[0]["queries"] == ["DNp01"]
        assert fake_finder.calls[0]["dataset"] == ["male-cns:v1.0"]
        # The output-detail flags ride along on the delegated call.
        assert fake_finder.calls[0]["params"]["keep_per_match_csv"] is False
        assert fake_finder.calls[0]["params"]["cleanup_source_images"] is True

    def test_missing_coverage_degrades_to_the_ordinary_path(
        self, monkeypatch, fake_finder, tmp_path
    ):
        _install_mapper(monkeypatch)
        monkeypatch.setattr(
            nqe, "refresh_coverage",
            lambda datasets, cache_root=None, **kwargs:
                (_ for _ in ()).throw(RuntimeError("offline")))
        monkeypatch.setattr(nqe, "load_snapshot", lambda root=None: None)
        finder = _make_finder()
        finder.run(
            queries=["DNp01"], dataset=["male-cns:v1.0"],
            expand_names=True, output_dir=str(tmp_path),
        )
        assert len(fake_finder.calls) == 1
        assert fake_finder.calls[0]["queries"] == ["DNp01"]
        assert fake_finder.calls[0]["dataset"] == ["male-cns:v1.0"]

    def test_mapper_failure_degrades_to_the_ordinary_path(
        self, monkeypatch, fake_finder, tmp_path
    ):
        monkeypatch.setattr(nqe, "HAS_TYPE_MAPPER", True)

        def _boom():
            raise RuntimeError("mapper load failed")

        monkeypatch.setattr(nqe, "get_type_mapper", lambda: _boom())
        _fake_snapshot(monkeypatch, {
            "male-cns:v1.0": {
                "status": "exact",
                "hosted_version": "male-cns:v1.0",
                "checked_at": "2026-09-15T00:00:00+00:00",
            },
        })
        finder = _make_finder()
        finder.run(
            queries=["DNp01"], dataset=["male-cns:v1.0"],
            expand_names=True, output_dir=str(tmp_path),
        )
        assert len(fake_finder.calls) == 1
        assert fake_finder.calls[0]["dataset"] == ["male-cns:v1.0"]

    def test_all_datasets_unknown_scope_still_expands_hosted_only(
        self, monkeypatch, fake_finder, tmp_path
    ):
        """'(all)': expansion targets only the hosted releases; the unknown
        (not-yet-probed) datasets stay on the ordinary query path instead of
        being silently dropped."""
        registry = _install_mapper(
            monkeypatch, lambda chip, ds: _resolution("mapped", targets=("L2",)))
        _fake_snapshot(monkeypatch, {
            "male-cns:v1.0": {
                "status": "exact",
                "hosted_version": "male-cns:v1.0",
                "checked_at": "2026-09-15T00:00:00+00:00",
            },
        })
        finder = _make_finder()
        finder.run(
            queries=["L2"], dataset=None,
            expand_names=True,
            coverage_datasets=["male-cns:v1.0", "banc_v888"],
            output_dir=str(tmp_path),
        )
        assert len(fake_finder.calls) == 1
        assert fake_finder.calls[0]["dataset"] == ["male-cns:v1.0"]

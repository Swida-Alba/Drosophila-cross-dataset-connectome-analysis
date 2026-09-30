"""Round-6 Windows feedback, F-P2: an absent local-release target must
auto-prepare or refuse — never resolve zero pairs against a universe that
does not exist and exit 0 — and an empty type lookup must announce itself
instead of quietly labelling typed neurons "(untyped)".
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from comparison import mapping_validation as mv  # noqa: E402
from comparison.mapping_validation import MappingValidator  # noqa: E402


# ---------------------------------------------------------------------------
# the local-release gate
# ---------------------------------------------------------------------------

def test_gate_passes_through_non_banc_targets(monkeypatch):
    calls = []

    def spy(dataset, dataset_dir):
        calls.append((dataset, dataset_dir))
        return True

    import BANC_file_converter
    monkeypatch.setattr(BANC_file_converter, 'ensure_banc_data', spy)
    assert mv.ensure_local_release_data('hemibrain:v1.2.1') is True
    assert mv.ensure_local_release_data('flywire_FAFB_v783') is True
    assert calls == []  # NeuPrint/flywire targets answer to the cache gates


def test_gate_prepares_banc_target(monkeypatch, tmp_path):
    calls = []

    def spy(dataset, dataset_dir):
        calls.append((dataset, Path(dataset_dir)))
        return True

    import BANC_file_converter
    monkeypatch.setattr(BANC_file_converter, 'ensure_banc_data', spy)
    assert mv.ensure_local_release_data('banc_v888',
                                        project_root=str(tmp_path)) is True
    # R7-1b: dataset_dir is the DATASET FOLDER - the converter writes
    # {dataset_dir}/{name}_*, and the root scattered them loose where the
    # catalog could not see them.
    assert calls == [('banc_v888', tmp_path / 'datasets' / 'banc_v888')]


def test_gate_routes_banc_aliases_to_the_canonical_folder(
        monkeypatch, tmp_path):
    """Alias spellings must prepare into the CANONICAL folder.

    ``ensure_banc_data`` renames the converted files with the canonical
    dataset prefix (``dataset_folder``), while every reader probes
    ``datasets/<canonical>/<canonical>_*`` — so a raw alias ('banc',
    'flywire_BANC') as the FOLDER scattered v626-prefixed tables under a
    directory no lookup ever visits (the run then resolves zero pairs).
    """
    calls = []

    def spy(dataset, dataset_dir):
        calls.append((dataset, Path(dataset_dir)))
        return True

    import BANC_file_converter
    monkeypatch.setattr(BANC_file_converter, 'ensure_banc_data', spy)
    from comparison.connectivity_profiler import canonical_dataset_name

    aliases = ('banc', 'flywire_BANC', 'flywire_BANC_v888', 'banc:v888',
               'banc_v626')
    for alias in aliases:
        assert mv.ensure_local_release_data(alias,
                                            project_root=str(tmp_path)) is True
    # The prepared folder must be exactly the folder the reader side
    # probes (canonical_dataset_name + the same punctuation scrub).
    expected = [tmp_path / 'datasets' /
                canonical_dataset_name(a).replace(':', '_').replace('.', '_')
                for a in aliases]
    assert [folder for _, folder in calls] == expected
    # Pin the two releases the aliases fold into — bare aliases pin to the
    # historical v626 default, versioned ones keep their release.
    assert [folder.name for _, folder in calls] == [
        'banc_v626', 'banc_v626', 'banc_v888', 'banc_v888', 'banc_v626']


def test_release_guard_refuses_unknown_version():
    # R7-1a: an explicit-but-unknown BANC release must be refused BEFORE
    # any download (round 7 saw banc_v999 silently serve v888 data).
    import banc_public_data
    with pytest.raises(ValueError, match='not published'):
        banc_public_data.reject_unknown_banc_release('banc_v999')


def test_release_guard_accepts_known_and_default():
    import banc_public_data
    for name in ('banc', 'flywire_BANC', 'banc_v888', 'banc_v626'):
        banc_public_data.reject_unknown_banc_release(name)  # must not raise


def test_gate_refuses_when_preparation_fails(monkeypatch, capsys):
    def boom(dataset, dataset_dir):
        raise RuntimeError('bucket unreachable')

    import BANC_file_converter
    monkeypatch.setattr(BANC_file_converter, 'ensure_banc_data', boom)
    assert mv.ensure_local_release_data('banc_v888') is False
    out = capsys.readouterr().out
    assert 'refusing' not in out  # that wording lives in the CLI wrapper
    assert 'target preparation for banc_v888 failed' in out


def test_gate_refuses_on_release_valueerror(monkeypatch, capsys):
    import BANC_file_converter

    def unknown_release(dataset, dataset_dir):
        raise ValueError('BANC release v999 is not published')

    monkeypatch.setattr(BANC_file_converter, 'ensure_banc_data',
                        unknown_release)
    assert mv.ensure_local_release_data('banc_v999') is False
    out = capsys.readouterr().out
    assert 'refused' in out and 'not published' in out


def test_gate_refuses_when_preparation_declines(monkeypatch):
    import BANC_file_converter
    monkeypatch.setattr(BANC_file_converter, 'ensure_banc_data',
                        lambda d, dd: False)
    assert mv.ensure_local_release_data('flywire_BANC_v888') is False


# ---------------------------------------------------------------------------
# the empty-lookup anomaly line (resolve_type_pairs)
# ---------------------------------------------------------------------------

class _StubResolver:
    def type_bodyid_pool(self, name, dataset):
        return [720575940606868828, 720575940609174392]


class _EmptyLookupProfiler:
    def get_types_for_bodyids(self, bodyids, dataset):
        return {}


class _TypedLookupProfiler:
    def get_types_for_bodyids(self, bodyids, dataset):
        return {b: 's-LNv' for b in bodyids}


def _validator_for(profiler, logs):
    v = MappingValidator.__new__(MappingValidator)
    v.cfg = mv.MappingValidationConfig(
        source_dataset='flywire_FAFB_v783', target_dataset='banc_v888',
        query_types=['s-LNv'])
    v.resolver = _StubResolver()
    v.profiler = profiler()
    v.log = logs.append
    return v


class _TableProbeProfiler(_EmptyLookupProfiler):
    has_table = True

    def _has_local_table(self, dataset):
        return self.has_table


def test_empty_type_lookup_announces_the_lookup_gap():
    logs = []
    v = _validator_for(_TableProbeProfiler, logs)
    v._pairs_for_type = lambda src_type, pool, query: []
    v._current_query = ''
    v._same_name_excluded = []
    v.pairs = v.resolve_type_pairs()
    joined = '\n'.join(logs)
    assert '! [resolution] type lookup matched 0 of 2 bodyIds' in joined
    assert 'local neuron table: True' in joined
    assert 'names the LOOKUP GAP, not the data' in joined


def test_typed_pool_never_triggers_the_anomaly():
    logs = []
    v = _validator_for(_TypedLookupProfiler, logs)
    v._pairs_for_type = lambda src_type, pool, query: []
    v._current_query = ''
    v._same_name_excluded = []
    v.pairs = v.resolve_type_pairs()
    joined = '\n'.join(logs)
    assert 'type lookup matched 0 of' not in joined


def test_no_pairs_diagnosis_names_mapper_state():
    # R7-1: the diagnosis must carry the mapper's load error (the K4b
    # root cause: the crosswalk needs a dataset the host never
    # initialized) instead of blaming local data that is present.
    logs = []
    v = _validator_for(_TypedLookupProfiler, logs)
    v._pairs_for_type = lambda src_type, pool, query: []
    v._current_query = ''
    v._same_name_excluded = []
    v.mapper = SimpleNamespace(
        last_load_error='neuron_df not found: male-cns table',
        _mapper_snapshot_path=lambda: None)
    diag = v._no_pairs_diagnosis()
    joined = '\n'.join(diag)
    assert 'neuron_df not found: male-cns table' in joined
    assert 'initialize the datasets the mapper needs' in joined
    assert "source dataset 'flywire_FAFB_v783'" in joined
    assert "target dataset 'banc_v888'" in joined


def test_no_pairs_diagnosis_says_unknown_when_it_cannot_probe():
    # R7-1c's own rule: a table state the diagnostic cannot answer must be
    # reported as 'unknown', never asserted as ABSENT (a profiler without
    # _has_local_table used to print ABSENT without ever probing).
    logs = []
    v = _validator_for(_TypedLookupProfiler, logs)
    v._current_query = ''
    v.mapper = SimpleNamespace(last_load_error=None,
                               _mapper_snapshot_path=lambda: None)
    joined = '\n'.join(v._no_pairs_diagnosis())
    assert 'local neuron table unknown' in joined
    assert 'ABSENT' not in joined


def test_no_pairs_line_is_loud_and_readme_greppable():
    source = Path(mv.__file__).read_text()
    assert ("'! [resolution] no valid type pairs resolved; '" in source), (
        'the no-pairs line must carry the README-greppable ! prefix and '
        'point at the target-data cause (round-6 F-P2)')


# ---------------------------------------------------------------------------
# F-D1: the UTF-8 stdio guard lives in parquet_utils itself
# ---------------------------------------------------------------------------

def test_parquet_utils_installs_utf8_guard_on_import(monkeypatch, tmp_path):
    """The module that emits ✓/⚠️ must survive a GBK-bound stdout even when
    nothing imported coana first (round-6 F-D1: a committed re-encode
    aborted on cp936 through FAFB_file_converter's __main__)."""
    import io
    import importlib
    import pyarrow as pa
    import pyarrow.parquet as pq
    import utils.parquet_utils as pu
    import utils.console_encoding as ce

    buf = io.BytesIO()
    gbk_out = io.TextIOWrapper(buf, encoding='gbk')
    monkeypatch.setattr(sys, 'stdout', gbk_out)
    importlib.reload(pu)  # guard runs at import against the GBK stream
    table = tmp_path / 'x.parquet'
    pq.write_table(pa.table({'a': [1, 2, 3]}), table)
    try:
        ok = pu.reencode_parquet_lossless(str(table), 'test')
    finally:
        gbk_out.flush()
        sys.stdout = sys.__stdout__
    assert ok in (True, False)  # no UnicodeEncodeError escaped


def test_banc_dash_spelled_alias_folds_to_its_release():
    """The dash spelling banc-v888 names the same release as banc_v888 —
    it must fold to the canonical form (2026-09-30 code audit F-DP-010)
    while the hidden NeuPrint colon spelling banc:v888 passes through by
    design."""
    from comparison.connectivity_profiler import canonical_dataset_name
    assert canonical_dataset_name('banc-v888') == 'banc_v888'
    assert canonical_dataset_name('banc_v888') == 'banc_v888'
    assert canonical_dataset_name('banc:v888') == 'banc:v888'
    assert canonical_dataset_name('flywire-banc-v888') == 'banc_v888'
    assert canonical_dataset_name('banc') == 'banc_v626'

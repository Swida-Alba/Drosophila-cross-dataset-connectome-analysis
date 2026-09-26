"""Round-6 Windows feedback, F-P2: an absent local-release target must
auto-prepare or refuse — never resolve zero pairs against a universe that
does not exist and exit 0 — and an empty type lookup must announce itself
instead of quietly labelling typed neurons "(untyped)".
"""

import sys
from pathlib import Path

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
    assert calls == [('banc_v888', tmp_path / 'datasets')]


def test_gate_refuses_when_preparation_fails(monkeypatch, capsys):
    def boom(dataset, dataset_dir):
        raise RuntimeError('bucket unreachable')

    import BANC_file_converter
    monkeypatch.setattr(BANC_file_converter, 'ensure_banc_data', boom)
    assert mv.ensure_local_release_data('banc_v888') is False
    out = capsys.readouterr().out
    assert 'refusing' not in out  # that wording lives in the CLI wrapper
    assert 'target preparation for banc_v888 failed' in out


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


def test_empty_type_lookup_announces_the_lookup_gap():
    logs = []
    v = _validator_for(_EmptyLookupProfiler, logs)
    v._pairs_for_type = lambda src_type, pool, query: []
    v._current_query = ''
    v._same_name_excluded = []
    v.pairs = v.resolve_type_pairs()
    joined = '\n'.join(logs)
    assert '! [resolution] type lookup returned nothing for 2 bodyIds' in joined
    assert 'names the LOOKUP GAP, not the data' in joined


def test_typed_pool_never_triggers_the_anomaly():
    logs = []
    v = _validator_for(_TypedLookupProfiler, logs)
    v._pairs_for_type = lambda src_type, pool, query: []
    v._current_query = ''
    v._same_name_excluded = []
    v.pairs = v.resolve_type_pairs()
    joined = '\n'.join(logs)
    assert 'type lookup returned nothing' not in joined


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

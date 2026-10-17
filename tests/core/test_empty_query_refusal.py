"""An unresolvable query is a refusal the caller can see (plan 2026-10-16 §13).

Every connection tool used to `_vprint("Error: … DataFrame is empty")` at level
'always' and RETURN: no run folder, no report, no exception. On the real cache
that made a typo'd or absent type name indistinguishable from a run that
searched and found nothing — a headless caller saw only a clean exit (the
matrix's first zero-connection job hit exactly this). All four tools now raise,
while the cross-dataset tool keeps refusing per UNIT only: a type present in one
dataset release and absent from another is the ordinary cross-dataset case.

Wrong hypotheses pinned here: a raise that also leaves a half-written folder
behind; a batch guard so broad it swallows a real defect; a note-less refusal
that names no dataset.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
for _p in (PROJECT_ROOT, PROJECT_ROOT / 'src',
           PROJECT_ROOT / 'vispath-subproject' / 'src'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import coana  # noqa: E402
from tests.core.test_pathfinding import _make_pipeline_fc  # noqa: E402


def _empty_source(monkeypatch, tmp_path, finder):
    coana._FINDALLPATH_GRAPH_CACHE.clear()
    fc, _c, _l = _make_pipeline_fc(
        monkeypatch, tmp_path, [("S", "A", 5), ("A", "T", 6)],
        max_interlayer=2, source_ids=("S",), target_ids=("T",))
    fc.source_df = pd.DataFrame(columns=['bodyId', 'type'])
    with pytest.raises(ValueError) as exc:
        getattr(fc, finder)()
    return fc, str(exc.value)


@pytest.mark.parametrize("finder", ['FindAllPath', 'FindShortestPath'])
def test_an_empty_source_query_refuses_without_a_folder(monkeypatch, tmp_path,
                                                       finder):
    fc, message = _empty_source(monkeypatch, tmp_path, finder)
    assert 'Source query resolved to no neurons' in message, message
    assert repr(fc.dataset) in message, (
        f'the refusal does not name the dataset it refused: {message}')
    assert 'this is a refusal, not an empty result' in message, message
    # nothing was searched, so nothing was written: no folder, no report
    # (an unset attribute IS the pass — the raise precedes any folder)
    _folder = getattr(fc, 'allpath_folder', '') or ''
    assert not (_folder and Path(_folder).exists()), (
        'the refused run left a folder behind — a reader could mistake it for '
        'a run that searched and found nothing')


def test_an_empty_target_query_names_the_target_side(monkeypatch, tmp_path):
    coana._FINDALLPATH_GRAPH_CACHE.clear()
    fc, _c, _l = _make_pipeline_fc(
        monkeypatch, tmp_path, [("S", "A", 5), ("A", "T", 6)],
        max_interlayer=2, source_ids=("S",), target_ids=("T",))
    fc.target_df = pd.DataFrame(columns=['bodyId', 'type'])
    with pytest.raises(ValueError) as exc:
        fc.FindAllPath()
    assert 'Target query resolved to no neurons' in str(exc.value)


def test_an_empty_network_query_refuses(monkeypatch, tmp_path):
    from tests.core.test_find_network import _make_network_fc

    fc, *_rest = _make_network_fc(
        monkeypatch, tmp_path, [("S", "A", 10)], ["S", "A"])
    fc.source_df = pd.DataFrame(columns=['bodyId', 'type'])
    with pytest.raises(ValueError) as exc:
        fc.FindNetwork()
    message = str(exc.value)
    assert 'resolved to no neurons' in message, message
    _nf = getattr(fc, 'network_folder', '') or ''
    assert not (_nf and Path(_nf).exists()), (
        'the refused Network run created a folder anyway')


def test_a_refused_run_with_an_explicit_saveas_leaves_no_folder_behind(
        tmp_path):
    """The refusal's promise is about the FILESYSTEM, measured on the real
    release: an explicit `saveas` used to be mkdir'd eagerly by
    `InitializeNeuronInfo`, so the first version of this refusal still left a
    named EMPTY folder in the output directory — found on real data
    (2026-10-17), invisible to every offline harness because none of them calls
    `InitializeNeuronInfo`."""
    fafb = PROJECT_ROOT / 'datasets' / 'flywire_FAFB_v783'
    if not (fafb / 'flywire_FAFB_v783_allneurons_neuron_df.csv').exists():
        pytest.skip('needs the local flywire_FAFB_v783 release')
    out = tmp_path / 'runs'
    out.mkdir()
    fc = coana.FindNeuronConnection(
        dataset='flywire_FAFB_v783', sourceNeurons=['NP2245'],
        targetNeurons=['aMe12'], output_dir=str(out),
        saveas='R_refused_should_not_exist', max_interlayer=1,
        use_cache=True, cache_only=True, showfig=False)
    fc.InitializeNeuronInfo()
    with pytest.raises(ValueError, match='resolved to no neurons'):
        fc.FindAllPath()
    assert not (out / 'R_refused_should_not_exist').exists(), (
        'the refused run left an empty named folder — exactly what a refusal '
        'is supposed to avoid')
    assert not any(p.is_dir() and p.name.startswith('find-paths')
                   for p in out.iterdir()), (
        'the refusal also created an auto-named run folder')


def test_the_comparison_tool_refuses_per_unit_and_keeps_going(monkeypatch,
                                                              tmp_path):
    """One dataset of N is not the whole request: the delegate's refusal is
    recorded for THAT unit, and any other ValueError still propagates."""
    from src.comparison.comparison_analyzer import ComparisonAnalyzer

    analyzer = ComparisonAnalyzer.__new__(ComparisonAnalyzer)

    class _Params:
        find_reciprocal = False

    analyzer.parameters = _Params()
    logged = []
    analyzer._log = lambda msg, level='info': logged.append((level, msg))

    class _Refuses:
        def FindAllPath(self, find_reciprocal=False):
            raise ValueError("Source query resolved to no neurons in dataset "
                             "'male-cns:v1.0': ['NoSuchType'].")

    class _OtherDefect:
        def FindAllPath(self, find_reciprocal=False):
            raise ValueError('some other value error')

    analyzer._delegate_path_run(_Refuses(), 'male-cns:v1.0', 5,
                                'FindAllPath')
    assert analyzer._empty_query_refusals == [{
        'dataset': 'male-cns:v1.0', 'threshold': 5, 'method': 'FindAllPath',
        'reason': ("Source query resolved to no neurons in dataset "
                   "'male-cns:v1.0': ['NoSuchType'].")}], (
        analyzer._empty_query_refusals)
    assert logged and logged[0][0] == 'warn', logged
    assert 'contributes no rows' in logged[0][1], logged

    with pytest.raises(ValueError, match='some other value error'):
        analyzer._delegate_path_run(_OtherDefect(), 'flywire_FAFB_v783', 3,
                                    'FindAllPath')
    assert len(analyzer._empty_query_refusals) == 1, (
        'the guard swallowed an unrelated ValueError')


def test_the_report_publishes_the_refusals_it_recorded():
    """A refusal the console alone knows about is a missing column the reader
    has to guess at, so the banner names the unit and the query in the report —
    and stays out of a clean one."""
    from src.comparison.comparison_analyzer import ComparisonAnalyzer

    analyzer = ComparisonAnalyzer.__new__(ComparisonAnalyzer)

    class _Params:
        comparison_mode = 'path'

    analyzer.parameters = _Params()
    assert 'contributed nothing' not in analyzer._generate_mode_specific_note(), (
        'a comparison with no refusal still printed a refusal banner')

    analyzer._empty_query_refusals = [
        {'dataset': 'male-cns:v1.0', 'threshold': 5, 'method': 'FindAllPath',
         'reason': ("Source query resolved to no neurons in dataset "
                    "'male-cns:v1.0': ['NoSuchType'].")}]
    note = analyzer._generate_mode_specific_note()
    assert 'contributed nothing' in note, note[:400]
    assert 'male-cns:v1.0' in note and 'NoSuchType' in note, note[:800]
    assert 'threshold 5' in note, note[:800]
    # the mode caveat the reader also needs is still there
    assert 'Path-Based Filtering Caveat' in note, note[-400:]


def test_the_coverage_warning_names_a_refusal_as_the_cause():
    """The report's red coverage callout fires for a refused dataset too — and
    its advice used to be 'check the delegate run log', a file a refused unit
    never wrote (measured: `dataset_data/<ds>/` holds only
    APPLIED_THRESHOLDS.md + threshold_meta.json). The callout must say WHY the
    dataset is empty, and stop pointing at a log that does not exist."""
    from comparison import html_report_generator as hrg
    from src.comparison.comparison_analyzer import ComparisonAnalyzer

    analyzer = ComparisonAnalyzer.__new__(ComparisonAnalyzer)
    analyzer.dataset_coverage = lambda: {
        'male-cns:v1.0': {'status': 'no_data'},
        'flywire_FAFB_v783': {'status': 'no_data'},
    }
    analyzer._empty_query_refusals = [
        {'dataset': 'male-cns:v1.0', 'threshold': 5, 'method': 'FindAllPath',
         'reason': 'Source query resolved to no neurons.'}]
    html = hrg._generate_dataset_coverage_callout(
        analyzer, ['male-cns:v1.0', 'flywire_FAFB_v783'],
        {'male-cns:v1.0': 'MCNS', 'flywire_FAFB_v783': 'FAFB'})
    assert 'resolved to no neuron' in html, html[:600]
    assert 'no run folder and no run log' in html, html[:600]
    # the OTHER dataset failed some other way, so its log pointer stays
    assert 'flywire_FAFB_v783' in html and 'run_log.txt' in html, html[:900]

    analyzer._empty_query_refusals = analyzer._empty_query_refusals + [
        {'dataset': 'flywire_FAFB_v783', 'threshold': 5,
         'method': 'FindAllPath',
         'reason': 'Target query resolved to no neurons.'}]
    both = hrg._generate_dataset_coverage_callout(
        analyzer, ['male-cns:v1.0', 'flywire_FAFB_v783'],
        {'male-cns:v1.0': 'MCNS', 'flywire_FAFB_v783': 'FAFB'})
    assert 'run_log.txt' not in both, (
        'both datasets were refused; the callout still sends the reader to a '
        'log neither wrote')


def test_a_refusal_reason_cannot_inject_markup():
    """Dataset names and type names are user input and are quoted into the
    report."""
    from src.comparison.comparison_analyzer import ComparisonAnalyzer

    analyzer = ComparisonAnalyzer.__new__(ComparisonAnalyzer)
    analyzer._empty_query_refusals = [{
        'dataset': '<img src=x onerror=alert(1)>', 'threshold': 1,
        'method': 'FindAllPath', 'reason': "it's <b>gone</b>"}]
    banner = analyzer._empty_query_banner()
    assert '<img src=x' not in banner, banner
    assert '&lt;img src=x' in banner, banner
    assert "it&#x27;s &lt;b&gt;gone&lt;/b&gt;" in banner, banner


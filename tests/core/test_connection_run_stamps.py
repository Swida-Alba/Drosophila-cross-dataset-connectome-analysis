"""L4 + L6 of the Connection-tab alignment (plan 2026-10-16 Q8, §"laws").

L4 — every run folder the group writes answers, in its own `parameters.txt`,
which threshold basis gated the graph, which lane the user asked for, and which
path mode produced the folder. A reader who returns to a folder months later
has only these lines; a Network folder that stays silent about its basis printed
`min synapse number: 3` while the graph ran at 1.

L6 — the report's kind recognizes EVERY prefix the group writes, so a run that
produced nothing gets the empty state of ITS OWN tool (the 'Pathfinding'
fallback used to hand a Network run depth- and lane-advice it cannot act on).

Both are measured on real offline runs of the three tools, not on greps.
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
from paths_pair_report import is_pathfinding_run, run_kind  # noqa: E402
from tests.core.test_applied_threshold_provenance import _BUDGET_EDGES  # noqa: E402
from tests.core.test_find_network import _make_network_fc  # noqa: E402
from tests.core.test_pathfinding import _make_pipeline_fc  # noqa: E402

STAMPS = ('path_mode', 'run mode', 'weight basis')


def _stamp(text, key):
    """Read one `parameters.txt` line of the padded grammar the report and the
    refill parser already read — and return its VALUE, so a key printed with
    nothing after the colon fails the law instead of passing a substring test."""
    for line in text.splitlines():
        head, _, tail = line.strip().partition(':')
        if head == key:
            return tail.strip()
    return None


def _path_run(monkeypatch, tmp_path, finder, **overrides):
    coana._FINDALLPATH_GRAPH_CACHE.clear()
    fc, _calls, _logs = _make_pipeline_fc(
        monkeypatch, tmp_path, _BUDGET_EDGES, max_interlayer=2)
    fc.pathfinding = 'StrongestFirst'
    for key, value in overrides.items():
        setattr(fc, key, value)
    getattr(fc, finder)()
    return Path(fc.allpath_folder)


def _text(run_dir):
    return (run_dir / 'parameters.txt').read_text(encoding='utf-8')


@pytest.mark.parametrize("finder,expect_kind", [
    ('FindAllPath', 'Complete Paths'),
    ('FindShortestPath', 'Shortest Paths'),
])
def test_a_path_run_stamps_its_basis_lane_and_mode(monkeypatch, tmp_path,
                                                   finder, expect_kind):
    run_dir = _path_run(monkeypatch, tmp_path, finder)
    text = _text(run_dir)
    values = {key: _stamp(text, key) for key in STAMPS}
    assert all(values.values()), (expect_kind, values, text)
    assert values['weight basis'] == 'synapse', values
    assert run_kind(run_dir.name) == expect_kind, run_dir.name


def test_a_ratio_run_names_its_basis(monkeypatch, tmp_path):
    """The same line must carry the OTHER regime, so the stamp is a fact about
    the graph, not a leftover of the default."""
    run_dir = _path_run(monkeypatch, tmp_path, 'FindAllPath',
                        weight_basis='connection_ratio', min_ratio=0.05)
    assert _stamp(_text(run_dir), 'weight basis') == 'connection_ratio (bodyId)'


def test_a_network_run_stamps_the_same_three_lines(monkeypatch, tmp_path):
    fc, *_rest = _make_network_fc(
        monkeypatch, tmp_path, [("S", "A", 10), ("A", "S", 8), ("S", "B", 6)],
        ["S", "A", "B"])
    fc.FindNetwork()
    run_dir = Path(fc.network_folder)
    text = _text(run_dir)
    values = {key: _stamp(text, key) for key in STAMPS}
    assert all(values.values()), values
    # a direct-connection run has no lane to run: it must SAY so, not stay mute
    assert values['path_mode'] == 'network', values
    assert 'n/a' in values['run mode'], values
    assert values['weight basis'] == 'synapse', values


def test_a_network_run_stamps_the_regime_it_ran_in(monkeypatch, tmp_path):
    """The threshold regime has FOUR lines a reader needs, not three: the
    folder must name the basis, the count that actually gated the graph AFTER
    the ratio floor, the ratio, and how they combined. An inert AND is stamped
    as inactive rather than as a co-threshold that never fired."""
    cases = {
        # basis, min_synapse, min_ratio, combo -> expected stamps
        ('synapse', 3, 0.0, 'or'): ('synapse', '3', '0.0', 'or'),
        ('connection_ratio', 3, 0.05, 'or'):
            ('connection_ratio (bodyId)', '1', '0.05', 'or'),
        ('synapse', 4, 0.02, 'and'):
            ('synapse', '4', '0.02', 'and (synapse count + connection ratio)'),
        ('synapse', 4, 0.0, 'and'):
            ('synapse', '4', '0.0', 'and (inactive: min connection ratio is 0)'),
    }
    for (basis, syn, ratio, combo), (want_basis, want_syn, want_ratio,
                                     want_combo) in cases.items():
        name = f'{basis[:5]}_{syn}_{ratio}_{combo}'
        fc, *_rest = _make_network_fc(
            monkeypatch, tmp_path / name,
            [("S", "A", 10), ("A", "S", 8)], ["S", "A"])
        fc.weight_basis, fc.min_synapse_num = basis, syn
        fc.min_ratio, fc.threshold_combination = ratio, combo
        fc.FindNetwork()
        text = _text(Path(fc.network_folder))
        got = tuple(_stamp(text, key) for key in (
            'weight basis', 'min synapse number', 'min connection ratio',
            'threshold combination'))
        assert got == (want_basis, want_syn, want_ratio, want_combo), (
            name, got)


def test_the_stamp_is_stated_once_per_folder(monkeypatch, tmp_path):
    """A key the file answers TWICE is a key the reader cannot trust — and
    `read_run_provenance` raises on contradictory values for the keys it
    collects, so a duplicate outside that list is the one that slips through.
    The Network folder used to carry `weight basis:` twice: the parameter_dict
    block wrote 'connection_ratio (bodyId)' and the provenance append then
    wrote 'connection_ratio'."""
    counts = []
    for name, finder in (('all', 'FindAllPath'),
                         ('shortest', 'FindShortestPath')):
        out = (tmp_path / name)
        out.mkdir()
        counts.append(_path_run(monkeypatch, out, finder))
    fc, *_rest = _make_network_fc(
        monkeypatch, tmp_path, [("S", "A", 10)], ["S", "A"])
    fc.FindNetwork()
    counts.append(Path(fc.network_folder))
    for run_dir in counts:
        text = _text(run_dir)
        for key in STAMPS:
            lines = [ln for ln in text.splitlines()
                     if ln.strip().split(':')[0] == key]
            assert len(lines) == 1, (run_dir.name, key, lines)


def test_the_network_folder_token_carries_the_live_thresholds(monkeypatch,
                                                              tmp_path):
    """The folder name is the only provenance a user reads in a file list. The
    dead probability knob (`p0`) is gone from it, and the ratio tier the run
    actually used is in there."""
    fc, *_rest = _make_network_fc(
        monkeypatch, tmp_path, [("S", "A", 10), ("A", "S", 8)], ["S", "A"])
    fc.weight_basis, fc.min_synapse_num = "connection_ratio", 1
    fc.min_ratio, fc.min_traversal_probability = 0.05, 0.0
    fc.FindNetwork()
    name = Path(fc.network_folder).name
    assert "p0" not in name, name
    assert "r0_05" in name, name
    assert "w1" in name, name


def test_a_zero_connection_network_gets_its_own_empty_state(
        monkeypatch, tmp_path):
    """Q5: the report used to refuse the folder ('no *_allpaths_type.csv
    found'), because a Network run with no edges is not a pathfinding run in
    the kind table. Now it writes an empty state whose wording belongs to the
    tool — and says nothing about depth or lanes, which this tool has none of.
    """
    from paths_pair_report import generate_paths_pair_report

    fc, *_rest = _make_network_fc(
        monkeypatch, tmp_path, [("S", "X", 50)], ["S", "A"])
    fc.FindNetwork()
    run_dir = Path(fc.network_folder)
    assert not list(run_dir.glob('*_allpaths_type.csv')), \
        'the fixture stopped being the zero-connection case'
    generate_paths_pair_report(run_dir, log=None)
    html = (run_dir / 'path_report.html').read_text(encoding='utf-8')
    assert 'No direct connections among the queried neurons' in html
    assert 'direct connections' in html
    # the wrong hypothesis: pathfinding advice the tool cannot act on
    for absent in ('Max Interlayer', 'corridor', 'Fast lane'):
        assert absent not in html, absent


def test_the_report_kind_separates_all_three_tools(monkeypatch, tmp_path):
    """L6: one folder per tool, three kinds — and each is a run the report
    will open, so a zero-output run gets its own tool's empty state instead of
    a refusal or a 'Pathfinding' fallback."""
    runs = {}
    for name, finder, kind in (
            ('all', 'FindAllPath', 'Complete Paths'),
            ('shortest', 'FindShortestPath', 'Shortest Paths')):
        out = tmp_path / name
        out.mkdir()
        runs[_path_run(monkeypatch, out, finder)] = kind
    fc, *_rest = _make_network_fc(
        monkeypatch, tmp_path, [("S", "A", 10), ("A", "S", 8)], ["S", "A"])
    fc.FindNetwork()
    runs[Path(fc.network_folder)] = 'Network'

    assert {run_kind(p.name) for p in runs} == set(runs.values())
    for run_dir, kind in runs.items():
        assert run_kind(run_dir.name) == kind, run_dir.name
        assert is_pathfinding_run(run_dir) is True, run_dir.name


def test_an_unrelated_folder_is_not_claimed(tmp_path):
    """The wrong hypothesis: a kind that falls through to 'Pathfinding' would
    make every folder a pathfinding run."""
    other = tmp_path / 'plot-3d_S_20261016_132625'
    other.mkdir()
    (other / 'parameters.txt').write_text('dataset: test\n', encoding='utf-8')
    assert run_kind(other.name) == 'Pathfinding'
    assert is_pathfinding_run(other) is False


def test_parameters_csv_names_the_basis_too(monkeypatch, tmp_path):
    """L2 asks for the basis to be NAMED in the run's exports — and the CSV is
    the machine-readable one every post-hoc script reads. A stamp that reaches
    parameters.txt but not data_details/parameters.csv is a fact two readers
    disagree about."""
    import pandas as pd

    for finder, want in (('FindAllPath', 'synapse'),
                         ('FindShortestPath', 'connection_ratio (bodyId)')):
        out = tmp_path / finder
        out.mkdir()
        run = _path_run(monkeypatch, out, finder,
                        **({} if want == 'synapse' else
                           {'weight_basis': 'connection_ratio',
                            'min_ratio': 0.05}))
        frame = pd.read_csv(run / 'data_details' / 'parameters.csv')
        values = dict(zip(frame['parameter'], frame['value']))
        assert values.get('weight basis') == want, (
            finder, values.get('weight basis'))
        # KNOWN GAP, recorded rather than papered over: the machine-readable
        # frame carries the THRESHOLD regime but not the two mode stamps, which
        # `_write_run_metadata` appends to parameters.txt directly (and repeats
        # in all_attributes.json). Closing it means moving `path_mode` /
        # `run mode` into parameter_dict at four call sites, which would then
        # double-write them into parameters.txt unless the explicit lines are
        # removed with it — so it is a decision, not an oversight to fix mid-
        # review. Assert the boundary stays exactly where it is documented.
        assert 'path_mode' not in values and 'run mode' not in values, (
            'the CSV gained the mode stamps — update the guides and drop this '
            'assertion, or better: fold the keys into parameter_dict and '
            'delete the explicit lines so no file states either key twice')
        text = (run / 'parameters.txt').read_text(encoding='utf-8')
        for key in ('path_mode', 'run mode'):
            assert _stamp(text, key), (finder, key)


def test_the_inactive_and_stamp_names_the_empty_knob(monkeypatch, tmp_path):
    """§10-5's second branch: a ratio-basis AND at Min Synapse Count 1 has an
    inactive AND for the OPPOSITE reason, and the folder must not blame the
    knob that is actually live."""
    fc, *_rest = _make_network_fc(
        monkeypatch, tmp_path, [("S", "A", 10), ("A", "S", 8)], ["S", "A"])
    fc.weight_basis, fc.min_synapse_num = "connection_ratio", 1
    fc.min_ratio, fc.threshold_combination = 0.05, "and"
    fc.FindNetwork()
    stamp = _stamp(_text(Path(fc.network_folder)), "threshold combination")
    assert "Min Synapse Count is 1" in stamp, stamp
    assert "ratio is 0" not in stamp, stamp


def test_a_sweep_publishes_only_the_tier_it_was_labelled(tmp_path):
    """One instance, several threshold slices (plan 2026-10-16 review).

    Two laws ride the same [2, 5] sweep, and both were broken:

    1. **One folder per slice.** ``_replay_output_folder_for_threshold`` only
       knew the cross-dataset ``minsyn_{t}`` grammar, so a standalone sweep
       (folder tail ``_L3w2_{timestamp}``) matched nothing and re-materialized
       each slice INTO the base folder. Measured pre-fix: the sweep published
       ONE ``_L3w2`` folder whose every artifact — name, parameters.txt,
       conditioned file — disagreed (``threshold: 5`` inside a ``w2`` folder),
       and a later t0 collapse would ``rmtree`` that same folder with the
       slice's only copy of the data in it.
    2. **A slice sees only its own tier.** ``_type_incoming_conditioned`` is
       an instance store exported once per slice, so a later slice only
       re-records the types its own tier still reaches; the store was never
       cleared on a tier change. Measured: the tier-5 file carried a TD row
       (mass 4) that tier 5 never recorded — TD has no edge >= 5 entering it.

    Wrong hypotheses: pre-fix (1) yields one folder whose name tier is 2 and
    (2) leaks TD into the tier-5 file. Each folder's own name, parameters.txt
    and conditioned file must state the SAME tier, and the two files must
    differ."""
    import re

    from tests.core.test_type_level_refill import (
        EDGES as REFILL_EDGES, TYPE_MAP as REFILL_TYPES, universe_types)
    from tests.core.test_pathfinding import _make_pipeline_fc
    import pandas as pd

    coana._FINDALLPATH_GRAPH_CACHE.clear()
    with universe_types(REFILL_TYPES) as shim:
        fc, _c, _l = _make_pipeline_fc(
            shim, tmp_path, edges=REFILL_EDGES, max_interlayer=3,
            min_synapse=2, source_ids=("S1", "S2"),
            target_ids=("T1", "T2"))
        fc.skip_bodyId = False
        fc.target_df = fc.target_df.assign(
            Checked=[True] * len(fc.target_df))
        fc._refill_edges_override = list(REFILL_EDGES)
        fc._refill_neuron_frame_override = pd.DataFrame({
            'bodyId': list(REFILL_TYPES),
            'type': list(REFILL_TYPES.values())})
        fc.FindAllPathMultiThreshold([2, 5])

    published = {}
    for folder in sorted(Path(tmp_path).glob("find-paths-complete_*")):
        csv = folder / 'data_details' / 'type_incoming_conditioned.csv'
        if not csv.exists():
            continue
        frame = pd.read_csv(csv)
        tiers = set(frame['threshold'].tolist())
        assert len(tiers) == 1, f"{folder.name}: two tiers in one file {tiers}"
        file_tier = tiers.pop()
        # law 1: the folder NAME states the tier its own file states
        name_tier = re.search(r'_L\d+w(\d+)(?:r[\dneg_]+)?_\d{8}_\d{6}$',
                              folder.name)
        assert name_tier is not None, (
            f'{folder.name}: no synapse tier token to check against')
        assert int(name_tier.group(1)) == file_tier, (
            f'{folder.name} is labelled w{int(name_tier.group(1))} but its '
            f'conditioned file says threshold {file_tier} — the slice was '
            f're-materialized into another tier\'s folder')
        # and the folder's own provenance agrees
        text = _text(folder)
        for key in ('requested_threshold', 'applied_threshold'):
            stamp = _stamp(text, key)
            assert str(file_tier) == str(int(float(stamp))), (
                f'{folder.name}: parameters.txt {key}={stamp!r} != file tier '
                f'{file_tier}')
        assert _stamp(text, 'run mode'), (
            f'{folder.name}: a slice folder a reader can return to must '
            f'answer the lane question like the run folder does:\n{text}')
        published[file_tier] = dict(zip(frame['type'].astype(str),
                                        frame['incoming'].astype(float)))
    # law 1 again: BOTH asked tiers survived as their own folders
    assert set(published) == {2, 5}, (
        f'the sweep published tiers {sorted(published)}, not one folder per '
        f'asked tier — a slice overwrote its sibling')
    # law 2: the tier-5 cone is the tier-2 cone minus TD, and the store reset
    # rather than accumulating
    assert 'TD' in published[2], published[2]
    assert 'TD' not in published[5], (
        f"tier 5 file carries TD {published[5].get('TD')} — the store was "
        f"not reset when the tier changed")
    assert set(published[5]) == {'TA', 'TB', 'TC', 'TT'}, published[5]
    assert published[5] != published[2], (
        'both slices published the same denominators — the later one did not '
        're-record its own tier')


def test_the_report_names_a_network_run_its_own_mode(monkeypatch, tmp_path):
    """The Analysis bar was kind-blind (review 2026-10-16): a Network run's
    1-hop table mounted as mode **Complete**, and the bar — read in a headless
    Chrome DOM dump, not inferred — captioned it

        "Mode: Complete — the only mode this run produced."

    for a folder that has no Complete mode, no lane, no budget and no tau. The
    'this tool has no Fast lane' card that was supposed to say so lived in the
    corridor template, which a Network run never selects, so no reader could
    reach it. The label is now DATA on the payload (verified where the browser
    reads it), and the unreachable card is gone.

    Wrong hypothesis: a Complete run losing its own wording — checked here too."""
    import json

    from paths_pair_report import generate_paths_pair_report
    from tests.core.test_find_network import _make_network_fc

    fc, *_rest = _make_network_fc(
        monkeypatch, tmp_path, [("S", "A", 10), ("A", "S", 8)], ["S", "A"])
    fc.FindNetwork()
    network_report = Path(generate_paths_pair_report(
        Path(fc.network_folder), log=None))

    def _run_block(report):
        text = report.read_text(encoding='utf-8')
        marker = '<script type="application/json" id="pair-data">'
        start = text.index(marker) + len(marker)
        return json.loads(text[start:text.index("</script>",
                                              start)])['run']

    block = _run_block(network_report)
    assert block['kind'] == 'Network'
    assert block['mode_labels'] == {'complete': 'Direct connections (1 hop)'}
    for phrase in ('no second mode', 'no Fast (corridor) lane',
                   'no path budget'):
        assert phrase in block['mode_note'], phrase
    html = network_report.read_text(encoding='utf-8')
    assert 'This tool has ' not in html, (
        'the corridor-template card is back: a Network run cannot mount it')
    # The Overview counts 1-hop rows, so it must not call them paths (and a
    # Network run with nothing to show must not announce 'No paths found' for
    # a tool that never searches paths).
    assert 'Direct connections (all units)' in html, html[:400]
    for phrase in ('Paths (all units)', 'No paths found in this run'):
        assert phrase not in html, phrase

    run_dir = _path_run(monkeypatch, tmp_path / 'ctl', 'FindAllPath')
    ctl_report = Path(generate_paths_pair_report(run_dir, log=None))
    ctl = _run_block(ctl_report)
    assert ctl['kind'] == 'Complete Paths'
    assert ctl['mode_labels'] == {} and ctl['mode_note'] == '', (
        'a pathfinding run picked up the Network wording')
    ctl_html = ctl_report.read_text(encoding='utf-8')
    assert 'Paths (all units)' in ctl_html, (
        'the pathfinding run lost its own Overview label')
    assert 'Direct connections (all units)' not in ctl_html


def test_a_tokenless_save_as_folder_gets_its_slice_in_a_subfolder(tmp_path):
    """A sweep into a Save-As folder whose name carries NO tier token cannot
    be restamped (plan 2026-10-16 review).

    Three behaviours were possible for that name: the old one (skip the
    redirect, so slice 5 overwrites the base tier's folder and the folder's
    own parameters.txt contradicts its contents), raising (which kills a
    legitimate multi-threshold pass mid-run after the enumeration already
    spent it), or giving the slice its own tier-named subfolder and saying so.
    The third is what the engine does now.

    Wrong hypothesis: the pre-fix skip — one folder, no `minsyn_5/`, and no
    note anywhere in the run."""
    from tests.core.test_type_level_refill import (
        EDGES as REFILL_EDGES, TYPE_MAP as REFILL_TYPES, universe_types)
    from tests.core.test_pathfinding import _make_pipeline_fc

    coana._FINDALLPATH_GRAPH_CACHE.clear()
    root = tmp_path / 'my_sweep'
    with universe_types(REFILL_TYPES) as shim:
        fc, _c, _l = _make_pipeline_fc(
            shim, root, edges=REFILL_EDGES, max_interlayer=3,
            min_synapse=2, source_ids=("S1", "S2"),
            target_ids=("T1", "T2"))
        fc.skip_bodyId = False
        fc.target_df = fc.target_df.assign(
            Checked=[True] * len(fc.target_df))
        fc.saveas = str(root)
        fc.save_folder = str(root)
        fc.allpath_folder = str(root)
        fc.FindAllPathMultiThreshold([2, 5])

    slice_dir = root / 'minsyn_5'
    assert slice_dir.is_dir(), (
        f'no per-slice subfolder: {[p.name for p in root.iterdir()]}')
    assert (slice_dir / 'parameters.txt').exists()
    text = _text(slice_dir)
    assert _stamp(text, 'requested_threshold') == '5', text
    assert _stamp(text, 'replayed_from') == '2', text
    assert _stamp(text, 'run mode'), text
    # the base tier keeps the folder itself
    base = _text(root)
    assert _stamp(base, 'requested_threshold') == '2', base
    assert _stamp(base, 'replayed_from') is None, base
    # the folder that MOVED is the one that documents the move
    notes = (slice_dir / 'user_warning_notes.txt')
    body = notes.read_text(encoding='utf-8') if notes.exists() else ''
    assert 'no threshold token' in body, (
        'the slice was moved into a subfolder without telling the reader '
        'in its own folder:\n' + body)


def test_the_network_path_table_is_ordered_like_the_other_three_tools(tmp_path,
                                                                     monkeypatch):
    """A Network run's `*_allpaths_type.csv` follows the SAME order law as the
    pathfinding tools (plan 2026-10-16 review round 2).

    The 1-hop table is the artifact the pair report mounts and whose first
    rows its Path view reads as the top paths. The Network writer ended with
    `.sort_values('path')`, so a reader got type names in alphabetical order
    presented as the run's ranking — measured on real data (R6: 95, 6, 91
    synapses in that order). It is now length ascending, `min_weight`
    descending, with `path` as the deterministic tie-break.

    Wrong hypothesis: an alphabetical fixture whose order already equals the
    strongest-first one would pass — asserted here by checking the two orders
    differ."""
    import pandas as pd

    from tests.core.test_find_network import _make_network_fc

    fc, *_rest = _make_network_fc(
        monkeypatch, tmp_path,
        [("A", "B", 5), ("B", "A", 90), ("A", "A", 50)], ["A", "B"])
    fc.FindNetwork()
    folder = Path(fc.network_folder)
    csv = next(folder.glob('*_allpaths_type.csv'))
    frame = pd.read_csv(csv)
    assert len(frame) >= 3, frame
    weights = frame['min_weight'].tolist()
    assert weights == sorted(weights, reverse=True), (
        f'{csv.name}: the 1-hop table is not strongest-first: {weights}')
    alphabetical = sorted(frame['path'].tolist())
    assert frame['path'].tolist() != alphabetical, (
        'the fixture is already alphabetical, so the law would pass on the '
        'broken sort_values("path") writer — pick different weights')
    assert list(frame['length']) == [1] * len(frame)
    # and the report mounted on this table shows the strongest row first
    from paths_pair_report import generate_paths_pair_report
    report = Path(generate_paths_pair_report(folder, log=None))
    assert frame['path'].iloc[0] in report.read_text(encoding='utf-8')


def test_a_save_as_run_is_still_named_after_its_own_tool(tmp_path):
    """L6 has to hold for EVERY folder the group writes — including one under a
    user's `Save As` name (plan 2026-10-16 review round 2).

    `run_kind` read the run-NAME grammar, so a folder called anything else
    fell back to the generic 'Pathfinding': `is_pathfinding_run` said False,
    `generate_paths_pair_report` raised FileNotFoundError instead of writing
    the kind-aware empty state, and a named Network run that did get its table
    was described with depth and lane advice it cannot act on. Measured on the
    real matrix: `runs/R9_network_zero` (a Save-As Network run with zero
    connections) produced NO report at all before the folder's own
    `path_mode: network` stamp was consulted.

    Wrong hypothesis: a folder with no path_mode line must stay generic, so
    the fallback cannot invent a kind."""
    from paths_pair_report import (
        generate_paths_pair_report, is_pathfinding_run, run_kind, run_kind_for,
    )

    head = ("weight basis:                  synapse\n"
            "max interlayer:                0\n"
            "path_mode:                     network\n"
            "run mode:                      n/a (direct connections)\n")
    named = tmp_path / 'my_saved_network_run'
    named.mkdir()
    (named / 'parameters.txt').write_text(head, encoding='utf-8')
    pd = __import__('pandas')
    pd.DataFrame([{'bodyId': ['S1'], 'type': ['TS']}]).to_csv(
        named / 'source_neurons.csv', index=False)
    pd.DataFrame([{'bodyId': ['T1'], 'type': ['TT'],
                   'Checked': [True]}]).to_csv(
        named / 'target_neurons.csv', index=False)

    assert run_kind(named.name) == 'Pathfinding', (
        'the name grammar itself changed — the fallback is supposed to be a '
        'fallback')
    assert run_kind_for(named) == 'Network'
    assert is_pathfinding_run(named) is True
    report = Path(generate_paths_pair_report(named, log=None))
    text = report.read_text(encoding='utf-8')
    assert 'direct connection' in text.lower(), text[:600]
    assert 'Max Interlayer' not in text, (
        'the Save-As Network empty state is quoting search advice again')

    # and a folder that says nothing keeps the generic reading
    silent = tmp_path / 'mystery_folder'
    silent.mkdir()
    (silent / 'parameters.txt').write_text(
        "weight basis:                  synapse\n", encoding='utf-8')
    assert run_kind_for(silent) == 'Pathfinding'
    assert is_pathfinding_run(silent) is False
    with pytest.raises(FileNotFoundError):
        generate_paths_pair_report(silent, log=None)


@pytest.mark.parametrize("overrides,want", [
    ({}, 'or'),
    (dict(min_synapse_num=5, threshold_combination='and', min_ratio=0.0),
     'and (inactive: min connection ratio is 0)'),
    (dict(min_synapse_num=5, threshold_combination='and', min_ratio=0.05),
     'and (synapse count + connection ratio)'),
    (dict(weight_basis='connection_ratio', min_ratio=0.05), 'or'),
])
def test_a_path_run_stamps_its_threshold_combination(monkeypatch, tmp_path,
                                                    overrides, want):
    """Every pathfinding folder answers the combination question, not just the
    one where AND was real (plan 2026-10-16 review round 2).

    Round-13 wrote `threshold combination` ONLY inside the active-AND branch,
    so an 'or' run and an inert-AND run printed no such line at all while every
    network folder did ('or' included) — the same drift C5 closed for
    `weight basis`, left open here. Measured on the real matrix:
    R4_complete_and_inert (AND with ratio 0) carried `min synapse number: 5`
    and `min connection ratio: 0.0` and said nothing about the regime the user
    selected.

    Wrong hypothesis: a stamp written unconditionally but with one fixed value
    — each regime's own wording is asserted, and the key must also reach
    `data_details/parameters.csv` (the CSV is written from the same dict)."""
    run_dir = _path_run(monkeypatch, tmp_path / f'r{abs(hash(want))}',
                        'FindAllPath', **overrides)
    text = _text(run_dir)
    got = _stamp(text, 'threshold combination')
    assert got == want, (want, got, text)
    csv = pd.read_csv(run_dir / 'data_details' / 'parameters.csv')
    row = csv[csv['parameter'] == 'threshold combination']
    assert len(row) == 1, (
        f'parameters.csv does not carry the regime a folder states: '
        f'{list(csv["parameter"])}')
    assert str(row.iloc[0]['value']) == want, row.iloc[0].to_dict()


def test_a_run_folder_states_its_visualization_edge_limit(monkeypatch,
                                                          tmp_path):
    """The drawing cap is a run parameter, so the folder records it (plan
    2026-10-16 §13, user 2026-10-17).

    `edgeN_limit` trims only what an HTML view draws — the note
    `[visualization edge limit]` appears when it actually bites, but a reader
    who merely opens a page showing 200 of the run's 396 pairs found nothing
    in the folder that named the cap: it was a knob the run never wrote down.
    """
    run_dir = _path_run(monkeypatch, tmp_path / 'capped', 'FindAllPath',
                        edgeN_limit=200)
    text = _text(run_dir)
    value = _stamp(text, 'visualization edge limit')
    assert value and value.startswith('200'), (value, text)
    assert 'drawing only' in value, value
    csv = pd.read_csv(run_dir / 'data_details' / 'parameters.csv')
    row = csv[csv['parameter'] == 'visualization edge limit']
    assert len(row) == 1, list(csv['parameter'])

    uncapped = _path_run(monkeypatch, tmp_path / 'uncapped', 'FindAllPath',
                         edgeN_limit=0)
    assert _stamp(_text(uncapped), 'visualization edge limit') == \
        'no drawing cap', (
        'a run with no cap must not print a limit the reader would trust')


def test_a_network_run_page_is_a_run_parameter_too(monkeypatch, tmp_path):
    """The Network tool writes its own parameters.txt, so the cap must appear
    there too — one law, four tools."""
    from tests.core.test_find_network import _make_network_fc

    fc, *_rest = _make_network_fc(
        monkeypatch, tmp_path, [("S", "A", 10), ("A", "S", 8)], ["S", "A"])
    fc.edgeN_limit = 500
    fc.FindNetwork()
    text = _text(Path(fc.network_folder))
    got = _stamp(text, 'visualization edge limit')
    assert got and got.startswith('500'), (got, text)

"""Label-mapper display compatibility for query tokens (user directive
2026-10-10): the label mapper retypes its enrolled neurons at the bodyId
level, and the settled label IS the equivalent typing — the report HTML
never displays bodyIds.

Pins the display-name chain (tier 1 label / tier 2 native type / tier 3
hidden), the per-label row collapse with the per-dataset enrollment
breakdown, the participation table's label-keyed #paths (the bodyId-keyed
lookup always read 0), the query_resolution.csv display columns, and a
digit sweep over the rendered sections.
"""

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from comparison import html_report_generator as hrg  # noqa: E402
from comparison.comparison_analyzer import ComparisonAnalyzer  # noqa: E402
from comparison.comparison_parameters import ComparisonParameters  # noqa: E402
from comparison.label_mapper import (  # noqa: E402
    LabelMapper, PRESET_FORMAT_TAG)

D1, D2 = "male-cns:v1.0", "flywire_FAFB_v783"
DATASETS = [D1, D2]


def _mapper(tmp_path, labels=None):
    """Real LabelMapper enrolling bodyId members per dataset."""
    labels = labels or {
        "custom_label": ["5th-LNv", "APDN3"],
        # per-label member lists, aligned with custom_label order
        D1: [["100001", "100002"], []],
        D2: [[], ["720575940619074049"]],
    }
    path = tmp_path / "label_map.json"
    path.write_text(
        json.dumps({"format": PRESET_FORMAT_TAG,
                    "source_mapping": labels, "target_mapping": labels}),
        encoding="utf-8")
    return LabelMapper(overall_mapping_json=str(path))


def _analyzer(tmp_path, mapper, source, target):
    params = ComparisonParameters(
        datasets=list(DATASETS),
        source_neurons=list(source),
        target_neurons=list(target),
        thresholds=[3],
        comparison_mode='path',
        max_interlayer=2,
        output_folder=str(tmp_path / 'out'),
        saveas='display_check',
    )
    return ComparisonAnalyzer(params, label_mapper=mapper, verbose=False)


def _seed_raw_results(analyzer):
    """bodyId-level frames feeding the tier-2 map."""
    analyzer.raw_results = {
        D1: {3: pd.DataFrame({
            'bodyId_pre': [300007, 300008],
            'type_pre': ['T5*', 'unknown'],
            'bodyId_post': [400001, 400002],
            'type_post': ['400001', '400002'],
        })},
        D2: {3: pd.DataFrame({
            'bodyId_pre': [],
            'type_pre': [],
            'bodyId_post': [],
            'type_post': [],
        })},
    }


# ---------------------------------------------------------------------------
# Enrichment: the display-name chain
# ---------------------------------------------------------------------------

def test_display_chain_tier1_label(tmp_path):
    an = _analyzer(tmp_path, _mapper(tmp_path),
                   source=['100001', '100002', 's-LNv'],
                   target=['DN1'])
    records = an.resolve_query_inputs()
    by_token = {}
    for r in records:
        if r['role'] == 'source':
            by_token.setdefault(r['token'], []).append(r)

    # Tier 1: enrolled bodyIds display as the settled label everywhere
    # (the label is a tier-0 identity, dataset independent)...
    for tok in ('100001', '100002'):
        assert by_token[tok], tok
        for r in by_token[tok]:
            assert r['display_name'] == '5th-LNv'
            assert r['display_role'] == 'label'
    # ...but only the HOME-dataset record carries the resolved target
    # (the bodyId exists in D1 only).
    home = [r for r in by_token['100001'] if r['dataset'] == D1][0]
    away = [r for r in by_token['100001'] if r['dataset'] == D2][0]
    assert home['target_types'] == ['5th-LNv']
    assert "Retyped to '5th-LNv'" in home['note']
    assert away['target_types'] == []

    # Ordinary type token: display name is the token itself.
    for r in by_token['s-LNv']:
        assert r['display_name'] == 's-LNv'
        assert r['display_role'] == 'type'


def test_display_chain_tier2_native_type_and_tier3_hidden(tmp_path):
    an = _analyzer(tmp_path, _mapper(tmp_path),
                   source=['300007', '300008', '999999'],
                   target=['DN1'])
    _seed_raw_results(an)
    records = [r for r in an.resolve_query_inputs() if r['role'] == 'source']
    by_token = {}
    for r in records:
        by_token.setdefault(r['token'], []).append(r)

    # Tier 2: ungoverned bodyId whose native type (T* residuals included)
    # is recoverable from the run's own frames.
    for r in by_token['300007']:
        assert r['display_name'] == 'T5*'
        assert r['display_role'] == 'type'
    home = [r for r in by_token['300007'] if r['dataset'] == D1][0]
    assert home['target_types'] == ['T5*']
    assert 'Native type recovered' in home['note']

    # Tier 3: untyped (sentinel or bodyId-fallback type) or absent from
    # every frame — hidden from the HTML, retained in the CSV.
    for r in by_token['300008'] + by_token['999999']:
        assert r['display_name'] == ''
        assert r['display_role'] == 'hidden'


def test_query_resolution_csv_gains_display_columns(tmp_path):
    an = _analyzer(tmp_path, _mapper(tmp_path),
                   source=['100001', '999999'], target=['DN1'])
    _seed_raw_results(an)
    out_dir = str(tmp_path / 'out')
    an._export_query_resolution_records(out_dir)
    csv_path = Path(out_dir) / 'comparison_report_used_data' \
        / 'query_resolution.csv'
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert {'token', 'display_name', 'display_role'} <= set(df.columns)
    src = df[df['role'] == 'source']
    governed = src[src['token'] == '100001']
    assert (governed['display_name'] == '5th-LNv').all()
    assert (governed['display_role'] == 'label').all()
    # Hidden inputs stay in the CSV verbatim.
    hidden = src[src['token'] == '999999']
    assert (hidden['display_role'] == 'hidden').all()
    assert (hidden['token'] == '999999').all()


# ---------------------------------------------------------------------------
# Rendering: collapse + no bodyIds in the HTML
# ---------------------------------------------------------------------------

def test_query_resolution_section_collapses_to_label_rows(tmp_path):
    an = _analyzer(tmp_path, _mapper(tmp_path),
                   source=['100001', '100002', 's-LNv'],
                   target=['720575940619074049'])
    section = hrg._generate_query_resolution_section(
        an, DATASETS, {D1: 'M', D2: 'F'})
    # No bodyId anywhere in the section.
    for tok in ('100001', '100002', '720575940619074049'):
        assert tok not in section, tok
    assert not re.search(r'>\s*\d{5,}\s*<', section)
    # ONE collapsed row per settled label x role — never one row per
    # bodyId x dataset (the per-record label rows are suppressed; the
    # remaining 2 ordinary rows are the s-LNv per-dataset records).
    assert section.count('<tr><td><strong>') == 4
    assert 'body_id' not in section  # no per-record body_id rows remain
    assert section.count('label_mapper</td>') == 2  # source + target rows
    assert '2 enrolled</span>' in section
    assert '1 enrolled</span>' in section
    assert 'tier 0' in section
    assert '>5th-LNv<' in section and '>APDN3<' in section
    # Breakdown: per-dataset member counts in the Dataset column.
    assert re.search(r'>M 2 · F —<', section)
    assert re.search(r'>M — · F 1<', section)
    # Ordinary type token renders as before.
    assert '>s-LNv<' in section


def test_query_resolution_section_merges_label_and_type_token(tmp_path):
    # The settled label is ALSO queried as a plain type token: the
    # ordinary rows win and the enrolled badge rides along.
    an = _analyzer(tmp_path, _mapper(tmp_path),
                   source=['100001', '5th-LNv'], target=['DN1'])
    section = hrg._generate_query_resolution_section(
        an, DATASETS, {D1: 'M', D2: 'F'})
    assert '100001' not in section
    # No collapsed duplicate row: 'label_mapper' method cell absent.
    assert 'label_mapper</td>' not in section
    # The type-token row carries the badge.
    assert '1 enrolled</span>' in section


def _participation_stub_analyzer(mapper, records, path_data):
    """Stub shaped like the coverage-file Analyzer, plus a real label
    mapper and enriched records."""

    class Mapper:
        _conflicts = []
        _loaded = True

        def get_canonical_type(self, type_name, source_dataset=None):
            return type_name

        def _get_type_mapping_key(self, dataset):
            return dataset

        def get_mapping_decision(self, source_type, source_dataset,
                                 target_dataset, include_bridges=False):
            return {'status': 'unmapped', 'source_type': source_type,
                    'target_type': None, 'target_types': [],
                    'relationship': None, 'conflicts': []}

    class Policy:
        def key_for(self, ds, name):
            return None

        def label_for_name(self, name):
            return None

        def group_by_label(self, label):
            return None

        def names_by_dataset(self, label):
            return {}

        def topology_dict(self):
            return {'summary': 'fake', 'warnings': [], 'groups': [],
                    'fan_in': {}}

    class Analyzer:
        parameters = SimpleNamespace(
            auto_type_mapping=True,
            _auto_type_mapper=Mapper(),
            threshold_mode='standard',
            thresholds=[3],
            get_dataset_nicknames=lambda: ['M', 'F'],
        )
        label_mapper = mapper

        @staticmethod
        def _merge_policy_or_none():
            return Policy()

        @staticmethod
        def resolve_query_inputs():
            return [dict(r) for r in records]

        @staticmethod
        def _collect_result_types_by_dataset():
            return {D1: {'5th-LNv'}, D2: set()}

        @staticmethod
        def _get_path_data_for_threshold(threshold):
            return path_data

    return Analyzer()


def _enriched_records():
    """body_id records as the analyzer's enrichment leaves them."""
    out = []
    for tok in ('100001', '100002'):
        for ds in DATASETS:
            out.append({
                'token': tok, 'dataset': ds, 'role': 'source',
                'status': 'body_id', 'method': 'body_id',
                'target_types': ['5th-LNv'] if ds == D1 else [],
                'display_name': '5th-LNv', 'display_role': 'label',
            })
    out.append({'token': '999999', 'dataset': D1, 'role': 'source',
                'status': 'body_id', 'method': 'body_id',
                'target_types': [], 'display_name': '',
                'display_role': 'hidden'})
    out.append({'token': 'DN1', 'dataset': D1, 'role': 'target',
                'status': 'same_name_identity', 'method': 'same_name',
                'target_types': ['DN1'], 'display_name': 'DN1',
                'display_role': 'type'})
    return out


def test_participation_table_label_rows_and_paths_fix(tmp_path):
    # Path keys are label-keyed (tier 0) — the display row must read
    # #paths off the LABEL, not the bodyId (the old bodyId-keyed lookup
    # always read 0).  Two path keys start at the label -> #paths == 2.
    path_data = pd.DataFrame(
        {D1: [2, 1], D2: [1, 0]},
        index=pd.Index(['5th-LNv -> mid -> DN1', '5th-LNv -> mid2 -> DN1'],
                       name='path_key'))
    an = _participation_stub_analyzer(
        _mapper(tmp_path), _enriched_records(), path_data)
    report = hrg._generate_type_mapping_section(an, DATASETS)

    assert '<h4>Queried types &amp; path participation</h4>' in report
    # No bodyId in the participation table.
    for tok in ('100001', '100002', '999999'):
        assert tok not in report, tok
    # ONE collapsed source row for the label, badge + label-keyed #paths.
    src_sec = report[report.find('>Queried sources</td>'):]
    src_sec = src_sec[:src_sec.find('</tbody>')]
    label_row = re.search(
        r'<tr><td><strong>5th-LNv</strong>.*?</tr>', src_sec, re.S).group(0)
    assert '2 enrolled</span>' in label_row
    assert '<td>2</td></tr>' in label_row
    assert label_row.count('<strong>5th-LNv</strong>') == 1
    # The hidden input never renders.
    assert '999999' not in src_sec
    # Ordinary target token keeps its row.
    tgt_sec = report[report.find('>Queried targets</td>'):]
    assert '<tr><td><strong>DN1</strong></td>' in tgt_sec


def test_participation_table_digit_sweep(tmp_path):
    path_data = pd.DataFrame(
        {D1: [1, 1], D2: [0, 0]},
        index=pd.Index(['5th-LNv -> mid -> DN1', '5th-LNv -> mid2 -> DN1'],
                       name='path_key'))
    an = _participation_stub_analyzer(
        _mapper(tmp_path), _enriched_records(), path_data)
    report = hrg._generate_type_mapping_section(an, DATASETS)
    section = hrg._generate_query_resolution_section(
        _participation_stub_analyzer(
            _mapper(tmp_path), _enriched_records(), path_data),
        DATASETS, {D1: 'M', D2: 'F'})
    for html_out in (report, section):
        # No digit-only cell contents or row labels anywhere.
        assert not re.search(r'>\s*\d{5,}\s*<', html_out)
        assert not re.search(r'<strong>\s*\d+\s*</strong>', html_out)


def test_type_mapping_section_float_ratio_thresholds(tmp_path):
    """Regression (2026-10-10 ratio run): ``_compared_threshold_set``
    int-truncated every sub-1 ratio tier to 0, so the raw_results type
    universe intersected with {0} and the WHOLE Type Mapping section —
    mapping grid and participation table — silently vanished on the
    connection-ratio basis."""
    frame = pd.DataFrame({
        'type_pre': ['5th-LNv'], 'type_post': ['DN1a'],
        'bodyId_pre': [100001], 'bodyId_post': [200001], 'weight': [3],
    })
    records = [
        {'token': '100001', 'dataset': D1, 'role': 'source',
         'status': 'body_id', 'method': 'body_id', 'target_types': [],
         'display_name': '5th-LNv', 'display_role': 'label'},
        {'token': 'DN1a', 'dataset': D1, 'role': 'target',
         'status': 'same_name_identity', 'method': 'same_name',
         'target_types': ['DN1a'], 'display_name': 'DN1a',
         'display_role': 'type'},
    ]
    an = _participation_stub_analyzer(
        _mapper(tmp_path), records,
        pd.DataFrame({D1: [1]}, index=pd.Index(['5th-LNv -> DN1a'])))
    an.parameters = SimpleNamespace(
        auto_type_mapping=True,
        _auto_type_mapper=an.parameters._auto_type_mapper,
        threshold_mode='standard',
        thresholds=[0.05],               # sub-1 float ratio tier
        get_dataset_nicknames=lambda: ['M', 'F'],
    )
    an.raw_results = {D1: {0.05: frame}, D2: {0.05: frame}}
    report = hrg._generate_type_mapping_section(an, DATASETS)
    assert '<h4>Queried types &amp; path participation</h4>' in report
    assert '<tr><td><strong>5th-LNv</strong>' in report


def test_conserved_paths_label_mapper_naming(tmp_path, monkeypatch):
    """Conserved-path artifact naming (user directive 2026-10-10): a node
    the label mapper governs keeps its settled label verbatim — the auto
    mapper's {canonical}({variants}) decoration applies only to its own
    remainder (auto-resolved names), which keep the style."""
    captured = {}

    class FakeVP:
        def __init__(self, path_file=None, **kw):
            captured['edges'] = path_file.copy()
            captured['kw'] = kw
            self.base_filename = ''
            import networkx
            g = networkx.DiGraph()
            for _, r in path_file.iterrows():
                g.add_edge(r['source'], r['target'])
            self.G_network = g

        def build_network(self):
            pass

        def create_network(self):
            return 'net.html'

        def create_heatmap(self):
            return 'heat.html'

    import sys
    import types
    monkeypatch.setitem(
        sys.modules, 'vispath_pkg',
        types.SimpleNamespace(VisualizePath=FakeVP))

    # A mapper governing the three labels the fixture uses (members are
    # irrelevant to the naming guard, which reads the label set only).
    gov = tmp_path / "gov.json"
    gov.write_text(json.dumps({
        "format": PRESET_FORMAT_TAG,
        "source_mapping": {"custom_label": ["5th-LNv", "DN1pC", "LNd_CRY+_ITP+"],
                           D1: [["1"], ["2"], ["3"]],
                           D2: [[], [], []]},
        "target_mapping": {"custom_label": ["5th-LNv", "DN1pC", "LNd_CRY+_ITP+"],
                           D1: [["1"], ["2"], ["3"]],
                           D2: [[], [], []]},
    }), encoding="utf-8")
    governing = LabelMapper(overall_mapping_json=str(gov))

    class AutoMapper:
        """Decorates EVERY name the way the real auto mapper decorates
        cross-dataset differences — including settled labels, which is
        exactly the behavior the fix must suppress for governed nodes."""

        def get_display_name_with_dataset_info(self, name, datasets):
            if name == 'LNd_a':
                return 'LNdX(LNd_a)', {'M': 'LNd_a'}
            if name == '5th-LNv':
                return '5thsLNv_LNd6(5th-LNv/s-LNv_a)', {'F': '5th-LNv'}
            if name == 'DN1pC':
                return 'DN1pC(SMP530)', {'B': 'SMP530'}
            return name, {}

        def get_all_dataset_short_codes(self, datasets):
            return {'male-cns:v1.0': 'M'}

    class Analyzer:
        parameters = SimpleNamespace(
            auto_type_mapping=True,
            _auto_type_mapper=AutoMapper(),
            threshold_mode='standard',
            thresholds=[3],
            separate_hemispheres=False,
            source_neurons=['5th-LNv'],
            target_neurons=['DN1pC'],
            _ensure_flat_list=lambda v: list(v),
            get_dataset_names=lambda: list(DATASETS),
            get_nickname_map=lambda: {D1: 'M', D2: 'F'},
            _sanitize_name=lambda s: s,
            get_source_neurons_for_dataset=lambda ds: ['5th-LNv'],
            get_target_neurons_for_dataset=lambda ds: ['DN1pC'],
            full_output_path=str(tmp_path),
        )
        label_mapper = governing
        raw_results = {D1: {3: pd.DataFrame()}}  # truthy: comparison "run"
        verbose = False

        @staticmethod
        def get_aligned_data(t):
            return pd.DataFrame(
                {D1: [4.0, 2.0], D2: [3.0, 2.0]},
                index=pd.Index(['5th-LNv -> LNd_a', 'LNd_CRY+_ITP+ -> DN1pC'],
                               name='edge_key'))

        @staticmethod
        def _get_edge_ratio_data_for_threshold(t):
            return pd.DataFrame()

        @staticmethod
        def _get_edge_nt_details(t):
            return {}

        def _log(self, *a, **k):
            pass

        def _log_file(self, *a, **k):
            pass

    an = Analyzer()
    ComparisonAnalyzer.visualize_conserved_paths(
        an, threshold=3, trim_dead_ends=False)

    nodes = set(captured['edges']['source']) | set(captured['edges']['target'])
    # Governed labels verbatim — never decorated.
    assert '5th-LNv' in nodes and 'DN1pC' in nodes and 'LNd_CRY+_ITP+' in nodes
    assert '5thsLNv_LNd6(5th-LNv/s-LNv_a)' not in nodes
    assert 'DN1pC(SMP530)' not in nodes
    # Auto-resolved name keeps the decoration.
    assert 'LNdX(LNd_a)' in nodes


def test_conservation_network_resolves_bodyid_orphans(tmp_path):
    """The conservation network's isolated-node pass walks the RAW query
    tokens — with bodyId queries that rendered one red bodyId node per
    enrolled neuron (685 in the clock run).  It must walk DISPLAY names:
    governed bodyIds collapse into their label's node (skipped when the
    label already has edges), unresolvable bodyIds never render."""
    aligned = pd.DataFrame(
        {D1: [3], D2: [0]},
        index=pd.Index(['5th-LNv -> DN1a'], name='edge_key'))
    records = [
        {'token': '100001', 'dataset': D1, 'role': 'source',
         'status': 'body_id', 'target_types': ['5th-LNv'],
         'display_name': '5th-LNv', 'display_role': 'label'},
        {'token': '100002', 'dataset': D1, 'role': 'source',
         'status': 'body_id', 'target_types': [],
         'display_name': '5th-LNv', 'display_role': 'label'},
        {'token': '999999', 'dataset': D1, 'role': 'source',
         'status': 'body_id', 'target_types': [],
         'display_name': '', 'display_role': 'hidden'},
        {'token': '200001', 'dataset': D2, 'role': 'target',
         'status': 'body_id', 'target_types': [],
         'display_name': 'APDN3', 'display_role': 'label'},
    ]

    class Analyzer:
        parameters = SimpleNamespace(
            auto_type_mapping=False,
            separate_hemispheres=False,
            source_neurons=['100001', '100002', '999999'],
            target_neurons=['200001'],
            _ensure_flat_list=lambda v: list(v),
        )
        label_mapper = _mapper(tmp_path)

        @staticmethod
        def _merge_policy_or_none():
            return None

        @staticmethod
        def _path_family_map():
            return {}

        @staticmethod
        def get_aligned_data_for_network(t):
            return aligned

        @staticmethod
        def _get_path_data_for_threshold(t):
            return None

        @staticmethod
        def resolve_query_inputs():
            return [dict(r) for r in records]

    net = hrg._generate_conservation_network(
        Analyzer(), DATASETS, 3, {D1: 'M', D2: 'F'})
    # No bodyId node labels/titles anywhere in the network payload.
    for tok in ('100001', '100002', '999999', '200001'):
        assert tok not in net, tok
    assert not re.search(r'"label": "\d+"', net)
    # 5th-LNv has edges (edge node, no orphan); APDN3 is queried with no
    # edges -> exactly ONE tier-0 isolated node, verbatim label.
    assert '"label": "5th-LNv"' in net
    assert net.count('"label": "APDN3"') == 1
    assert 'Role: source (isolated)' in net
    assert 'APDN3 (' not in net  # tier-0 label: no auto-mapper decoration

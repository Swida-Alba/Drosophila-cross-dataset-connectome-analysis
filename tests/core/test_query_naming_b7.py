"""Unit tests for the B7 query id/label naming scheme
(plan-query-anchored-cross-dataset-analysis.md §B7).

Vertical rows: id+label ``threshold={N}``; horizontal rows: id
``aligned_density={level:.4g}`` + explicit-threshold label.  Slug safety
(``=`` never reaches a filename), row_mode inference for BOTH the new
and the legacy prefixes, and ``.4g`` collision de-duplication.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from comparison.comparison_parameters import ComparisonParameters  # noqa: E402
from comparison.point_context import safe_point_id  # noqa: E402
from comparison.html_report_generator import _query_report_slug  # noqa: E402


def test_b7_ids_slug_safely():
    assert safe_point_id('threshold=19') == 'threshold_19'
    assert safe_point_id('aligned_density=0.7976') == 'aligned_density_0.7976'
    assert _query_report_slug('threshold=19') == 'threshold_19'
    assert _query_report_slug('aligned_density=0.7976') == \
        'aligned_density_0.7976'
    # `=` never reaches a filename / DOM id
    for raw in ('threshold=19', 'aligned_density=1.44'):
        slug = safe_point_id(raw)
        assert '=' not in slug
        assert all(c.isalnum() or c in '._-' for c in slug)


def test_install_auto_combinations_infers_row_mode_from_b7_prefixes():
    params = ComparisonParameters(
        datasets=['male-cns:v1.0', 'flywire_FAFB_v783'],
        source_neurons=['X'], target_neurons=['Y'],
    )
    params.install_auto_combinations([
        {'id': 'threshold=19', 'label': 'threshold=19',
         'thresholds': {'male-cns:v1.0': 19, 'flywire_FAFB_v783': 17}},
        {'id': 'aligned_density=0.7976',
         'label': 'aligned_density=0.7976 (MCNS 37, FAFB 34)',
         'thresholds': {'male-cns:v1.0': 37, 'flywire_FAFB_v783': 34}},
    ])
    rows = {r['id']: r for r in params.threshold_combinations}
    assert rows['threshold=19']['row_mode'] == 'vertical'
    assert rows['aligned_density=0.7976']['row_mode'] == 'horizontal'


def test_install_auto_combinations_keeps_legacy_prefixes():
    params = ComparisonParameters(
        datasets=['male-cns:v1.0', 'flywire_FAFB_v783'],
        source_neurons=['X'], target_neurons=['Y'],
    )
    params.install_auto_combinations([
        {'id': 'aligned_v_19', 'label': 'V t=19',
         'thresholds': {'male-cns:v1.0': 19, 'flywire_FAFB_v783': 19}},
        {'id': 'aligned_h_1', 'label': 'H d=0.8',
         'thresholds': {'male-cns:v1.0': 30, 'flywire_FAFB_v783': 28}},
    ])
    rows = {r['id']: r for r in params.threshold_combinations}
    assert rows['aligned_v_19']['row_mode'] == 'vertical'
    assert rows['aligned_h_1']['row_mode'] == 'horizontal'


def test_explicit_row_mode_survives_install():
    params = ComparisonParameters(
        datasets=['male-cns:v1.0', 'flywire_FAFB_v783'],
        source_neurons=['X'], target_neurons=['Y'],
    )
    params.install_auto_combinations([
        {'id': 'threshold=19', 'label': 'threshold=19',
         'row_mode': 'vertical',
         'thresholds': {'male-cns:v1.0': 19, 'flywire_FAFB_v783': 19}},
    ])
    assert params.threshold_combinations[0]['row_mode'] == 'vertical'


def test_duplicate_ids_still_rejected():
    params = ComparisonParameters(
        datasets=['male-cns:v1.0', 'flywire_FAFB_v783'],
        source_neurons=['X'], target_neurons=['Y'],
    )
    with pytest.raises(ValueError):
        params.install_auto_combinations([
            {'id': 'threshold=19', 'label': 'threshold=19',
             'thresholds': {'male-cns:v1.0': 19, 'flywire_FAFB_v783': 19}},
            {'id': 'threshold=19', 'label': 'threshold=19',
             'thresholds': {'male-cns:v1.0': 19, 'flywire_FAFB_v783': 20}},
        ])


def test_report_layout_default_is_legacy():
    """The tabbed report stays opt-in while it is refined; the backend
    default renders the original single-page report."""
    params = ComparisonParameters(
        datasets=['male-cns:v1.0', 'flywire_FAFB_v783'],
        source_neurons=['X'], target_neurons=['Y'],
    )
    assert params.report_layout == 'legacy'

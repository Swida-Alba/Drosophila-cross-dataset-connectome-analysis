"""The similarity-metric option order is a claim, so it is pinned.

The homolog benchmark draws its split metric-by-LEVEL, not with one winner
everywhere: jaccard is best per bodyId (MRR 0.9992, never degenerate), while
rank_union is best for pooled TYPE profiles (Hit@1 0.988 vs 0.963). Every
bodyId-level surface sorts by jaccard — the TM VEV ordering chain and Find
Homolog's default sort metric — so the dropdown has to lead with it too, or it
implies a default nothing at that level uses. Nothing else in the suite reads
`SIMILARITY_METRICS`, so an edit could silently restore the old order.
"""
from ui import config


def test_the_bodyid_level_metric_leads_the_dropdown():
    assert config.SIMILARITY_METRICS[0] == 'jaccard'
    assert config.DEFAULTS['similarity_metric'] == 'jaccard'


def test_rank_union_stays_offered_as_the_type_level_metric():
    # offered for sorting, just no longer presented as the default
    assert 'rank_union' in config.SIMILARITY_METRICS[1:]


def test_rank_corr_stays_excluded():
    """Shared-only Spearman is fragile at bodyId level (<3 shared types or low
    variance -> NaN), so it is not a choice the UI offers."""
    assert 'rank_corr' not in config.SIMILARITY_METRICS


def test_the_settings_surface_offers_exactly_that_order():
    spec = config.DEFAULT_SETTING_SPECS['similarity_metric']
    assert spec['options'] == config.SIMILARITY_METRICS

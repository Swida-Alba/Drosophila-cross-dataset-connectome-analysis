"""Overlay (custom_neurons) layers must stay out of the fetch phases.

The cross-dataset morphology scene injects pre-transformed FAFB neurons as
overlay layers on a NeuPrint-target scene. Two guards keep that working:

- ``_aggregate_neuprint_body_ids`` excludes overlay-layer bodyIds (the
  aggregate NeuPrint fetch cannot resolve foreign-dataset ids);
- the tree-legend loop guards ``neuron_vols[source_index]`` before the
  ``_drocat_legend_default_off`` lookup (unmatched trace names resolve to
  their raw trace position, which can sit past the neuron count once soma
  companions are interleaved — the deterministic cross-scene crash).
"""

import pandas as pd
import pytest

from visualize_skeleton import VisualizeSkeleton


def _hermetic_instance(neuron_dfs, custom_by_layer):
    inst = VisualizeSkeleton.__new__(VisualizeSkeleton)
    inst.neuron_dfs = neuron_dfs
    inst._custom_neurons_by_layer = custom_by_layer
    return inst


def test_aggregate_body_ids_excludes_overlay_layers():
    """Overlay-layer (custom_neurons) bodyIds never reach the NeuPrint
    aggregate fetch — they are injected pre-transformed objects."""
    native_df = pd.DataFrame({"bodyId": [101, 202]})
    overlay_df = pd.DataFrame({"bodyId": [720575940609627403]})

    inst = _hermetic_instance(
        [overlay_df, native_df],
        {"aMe12@FAFB_x1": [object()]},
    )
    assert inst._aggregate_neuprint_body_ids() == [101, 202]


def test_aggregate_body_ids_without_overlays_unchanged():
    inst = _hermetic_instance(
        [pd.DataFrame({"bodyId": [7, 7, 9]})],
        {},
    )
    # ordering + dedupe behavior preserved for plain scenes
    assert inst._aggregate_neuprint_body_ids() == [7, 9]


def test_aggregate_body_ids_overlay_count_mismatch_is_safe():
    """An empty overlay map must not trim any layers (defensive slice)."""
    inst = _hermetic_instance(
        [pd.DataFrame({"bodyId": [5]})],
        {},
    )
    assert inst._aggregate_neuprint_body_ids() == [5]

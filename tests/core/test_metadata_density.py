"""Feature A tests — synapse-density metadata enrichment
(threshold-alignment spec §4).

- FlyWire-style derivation: neuron tables carry ``post`` only; ``pre`` is
  derived from ``*_merged_connections.parquet`` (sum of weight grouped by
  bodyId_pre) and flagged ``pre_source: "derived_from_connections"``.
- NeuPrint-style tables carry ``pre`` + ``post`` (``pre_source:
  "neuron_table"``).
- The one-time script preserves existing metadata keys.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from comparison.metadata_density import (  # noqa: E402
    compute_synapse_density,
    merge_density_into_metadata,
)


def _flywire_fixture(tmp_path: Path):
    """3-neuron flywire-style dataset: post-only table + connections."""
    neurons = pd.DataFrame({
        "bodyId": [1, 2, 3],
        "type": ["TA", "TB", None],
        "post": [10, 0, 5],
    })
    connections = pd.DataFrame({
        "bodyId_pre": [1, 1, 2, 3],
        "bodyId_post": [2, 3, 3, 1],
        "weight": [4, 2, 7, 1],
    })
    parquet = tmp_path / "ds_v1_merged_connections.parquet"
    connections.to_parquet(parquet)
    return neurons, tmp_path


def test_flywire_pre_derived_from_connections(tmp_path):
    neurons, dataset_path = _flywire_fixture(tmp_path)
    density = compute_synapse_density(neurons, dataset_path=str(dataset_path))

    # pre: neuron 1 -> 4+2 = 6; neuron 2 -> 7; neuron 3 -> 1
    # total (pre+post): [16, 7, 6] -> median 7
    assert density["pre_source"] == "derived_from_connections"
    assert density["per_neuron_median"] == pytest.approx(7.0)
    assert density["per_neuron_median_pre"] == pytest.approx(6.0)
    assert density["per_neuron_median_post"] == pytest.approx(5.0)
    assert density["neuron_universe"] == "all_neurons"
    assert density["per_neuron_mean"] == pytest.approx((16 + 7 + 6) / 3)
    # The dataset-level presynaptic total is the derived connection mass
    # (4+2+7+1), not a literal 0 from the absent `pre` column.
    assert density["total_pre"] == pytest.approx(14.0)
    assert density["total_post"] == pytest.approx(15.0)
    assert density["total"] == pytest.approx(29.0)


def test_neuron_table_pre_post(tmp_path):
    neurons = pd.DataFrame({
        "bodyId": [1, 2, 3, 4],
        "type": ["TA", "TB", "TC", None],
        "pre": [3, 0, 5, 0],
        "post": [10, 2, 5, 0],
    })
    density = compute_synapse_density(neurons, dataset_path=str(tmp_path))
    assert density["pre_source"] == "neuron_table"
    assert density["per_neuron_median"] == pytest.approx((13 + 2 + 10 + 0) / 4 / 1) \
        or density["per_neuron_median"] == pytest.approx(6.0)
    # median of [13, 2, 10, 0] = (10 + 2) / 2 = 6
    assert density["per_neuron_median"] == pytest.approx(6.0)
    assert density["per_neuron_median_pre"] == pytest.approx(1.5)  # [3,0,5,0]
    assert density["per_neuron_median_post"] == pytest.approx(3.5)  # [10,2,5,0]


def test_derivation_failure_flagged_unavailable(tmp_path):
    neurons = pd.DataFrame({"bodyId": [1, 2], "type": ["TA", "TB"],
                            "post": [3, 4]})
    # no connections file in the folder -> pre cannot be derived
    density = compute_synapse_density(neurons, dataset_path=str(tmp_path))
    assert density["pre_source"] == "unavailable"
    assert density["per_neuron_median"] == pytest.approx(3.5)  # post only


def test_merge_preserves_existing_metadata_keys(tmp_path):
    neurons, dataset_path = _flywire_fixture(tmp_path)
    metadata = {"dataset": "ds:v1", "source": "local",
                "fetched_at": "2026-01-01T00:00:00",
                "neuron_counts": {"total": 3}}
    merged = merge_density_into_metadata(metadata, neurons, "ds:v1",
                                         str(dataset_path))
    assert merged["dataset"] == "ds:v1"
    assert merged["fetched_at"] == "2026-01-01T00:00:00"
    assert merged["neuron_counts"] == {"total": 3}
    assert "synapse_density" in merged
    assert "density_generated_at" in merged


def test_generate_script_updates_metadata_files(tmp_path):
    """End-to-end: script enriches datasets/<safe>/ in place, preserving
    keys (mirrors scripts/generate_dataset_metadata.py::enrich_dataset)."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    try:
        from generate_dataset_metadata import enrich_dataset
    finally:
        sys.path.pop(0)

    # datasets root layout: <root>/ds_v1/{metadata,neuron table,connections}
    datasets_root = tmp_path / "datasets"
    ds_dir = datasets_root / "ds_v1"
    ds_dir.mkdir(parents=True)
    neurons = pd.DataFrame({"bodyId": [1, 2, 3], "type": ["TA", "TB", None],
                            "post": [10, 0, 5]})
    connections = pd.DataFrame({"bodyId_pre": [1, 1, 2, 3],
                                "bodyId_post": [2, 3, 3, 1],
                                "weight": [4, 2, 7, 1]})
    connections.to_parquet(ds_dir / "ds_v1_merged_connections.parquet")
    neurons.to_parquet(ds_dir / "ds_v1_allneurons_neuron_df.parquet")
    metadata = {"dataset": "ds:v1", "source": "local",
                "fetched_at": "2026-08-16T18:35:25"}
    with open(ds_dir / "ds_v1_metadata.json", "w") as f:
        json.dump(metadata, f)

    assert enrich_dataset(str(datasets_root), "ds_v1") is True
    with open(ds_dir / "ds_v1_metadata.json") as f:
        updated = json.load(f)
    assert updated["fetched_at"] == "2026-08-16T18:35:25"
    assert updated["synapse_density"]["pre_source"] == \
        "derived_from_connections"

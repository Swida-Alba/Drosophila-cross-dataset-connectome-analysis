"""Release-to-release smoke test: the cross-dataset path on identical bodyIds.

Runs the real ConnectivityProfileComparer dict-query flow across two
versions of the same dataset family (male-cns v0.9 vs v1.0) and checks the
contract end to end: unique 4-char folder labels with the version suffix,
per-version display labels on both matrix axes, finite similarities, and
complete per-version neuron identity (type via the profiler backfill,
instance via the dataset tables). Skipped when the local caches are absent
so CI machines without datasets stay green.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from comparison.profile_comparator import ConnectivityProfileComparer  # noqa: E402

OLD_DS = "male-cns:v0.9"
NEW_DS = "male-cns:v1.0"
# Same physical neurons annotated in both releases (aMe12 members).
BIDS = [12211, 12740]

_REQUIRED = [
    PROJECT_ROOT / "cache" / "male-cns_v1_0" / "connections.parquet",
    PROJECT_ROOT / "cache" / "male-cns_v0_9" / "connections.parquet",
    PROJECT_ROOT / "neuron_indexes" / "male-cns_v1_0" / "neuron_index.parquet",
    PROJECT_ROOT / "neuron_indexes" / "male-cns_v0_9" / "neuron_index.parquet",
]
pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in _REQUIRED),
    reason="requires local male-cns v1.0 + v0.9 connection caches and indexes")


@pytest.fixture(scope="module")
def release_run(tmp_path_factory):
    comparer = ConnectivityProfileComparer(
        query={NEW_DS: BIDS, OLD_DS: BIDS},
        output_dir=str(tmp_path_factory.mktemp("release_smoke")),
        generate_heatmaps=False,
        verbose=False,
        use_auto_type_mapping=False,
    )
    return comparer.run(), NEW_DS, OLD_DS


def test_folder_uses_version_suffixed_labels(release_run):
    result, _, _ = release_run
    folder = Path(result["output_path"]).name
    assert folder.startswith("profiling_MCNS_v1_0_vs_MCNS_v0_9_")


def test_matrix_axes_carry_both_versions_labels(release_run):
    result, _, _ = release_run
    df = pd.read_csv(
        Path(result["output_path"]) / "cross_dataset" / "overall_jaccard.csv",
        index_col=0)
    expected = [f"{bid}_aMe12" for bid in BIDS]
    # The labels may carry the hemisphere suffix when the side resolves.
    assert len(df.index) == len(BIDS) and len(df.columns) == len(BIDS)
    for label in list(df.index) + list(df.columns):
        assert any(label.startswith(e) for e in expected), label
    values = df.to_numpy(dtype=float)
    assert np.isfinite(values).all()
    assert ((values >= 0.0) & (values <= 1.0)).all()


def test_metadata_records_both_versions(release_run):
    result, new_ds, old_ds = release_run
    meta = json.load(open(Path(result["output_path"]) / "metadata.json"))
    assert meta["datasets"] == [new_ds, old_ds]


def test_profile_jsons_have_complete_identity_per_version(release_run):
    result, new_ds, old_ds = release_run
    out = Path(result["output_path"])
    for ds in (new_ds, old_ds):
        for bid in BIDS:
            files = list((out / "profiles" / ds).glob(f"{bid}_*.json"))
            assert files, f"missing profile for {bid} in {ds}"
            data = json.load(open(files[0]))
            assert data.get("neuron_id") == str(bid)
            assert data.get("neuron_type") == "aMe12"
            assert data.get("dataset") == ds
            assert data.get("instance"), "instance should resolve from the dataset"

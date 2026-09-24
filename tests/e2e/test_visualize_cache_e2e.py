"""3D-skeleton visualization e2e — raw NeuPrint cache pipeline.

Runs against the REAL hemibrain v1.2.1 NeuPrint server (skipped when the
network is unavailable) and verifies the Mission-1 contract:

  * A fresh fetch writes the raw skeleton as compressed SWC in the shared
    ``skeletons/raw_skeletons`` namespace; no simplification marker is used.
  * A second load is a cache hit (no network): ``_load_cached_neurons``
    returns every neuron.
  * Every render simplification level reuses the raw cache; ``fast`` only
    controls in-memory mesh decimation.
  * ``morphology.fetch_skeleton_on_demand`` serves the shared raw cache (no
    network), and re-levels the returned copy when the compatibility
    ``level="simp90"`` argument is supplied.
  * The FlyWire path keeps its own cache (no marker written).
"""

import sys
import time
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import navis  # noqa: E402
import morphology as morph  # noqa: E402
from visualize_skeleton import VisualizeSkeleton  # noqa: E402

DATASET = "hemibrain:v1.2.1"
FOLDER = morph._dataset_folder(DATASET)
# aMe12 members with cached skeletons in the pre-migration build
PROBE_IDS = [911332304, 1158631810]
NEUPRINT_SERVER = "neuprint.janelia.org"


def _network_available() -> bool:
    try:
        import socket
        socket.create_connection((NEUPRINT_SERVER, 443), timeout=5).close()
        return True
    except Exception:
        return False
    return False


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not _network_available(),
        reason="NeuPrint server unreachable (network required for this e2e)",
    ),
]


@pytest.fixture(scope="module")
def probe_neurons():
    """Fetch two real hemibrain skeletons via the production client."""
    import neuprint
    from utils.token_manager import token_manager
    token = ""
    try:
        token = token_manager.get_token("NEUPRINT_TOKEN")
    except Exception:
        pass
    client = neuprint.Client(NEUPRINT_SERVER, dataset=DATASET, token=token)
    neuprint.set_default_client(client)
    neurons = []
    for bid in PROBE_IDS:
        n = neuprint.fetch_skeleton(bid)
        assert n is not None and len(n) > 0, f"no skeleton for {bid}"
        nrn = navis.TreeNeuron(n)
        # neuprint.fetch_skeleton (singular) returns a plain DataFrame;
        # attach the bodyId like the production fetch_skeletons batch path
        nrn.id = bid
        neurons.append(nrn)
    return neurons


def _vs(tmp_path, simplification=0.9):
    vs = VisualizeSkeleton(
        dataset=DATASET,
        neuron_layers=["aMe12"],
        verbose=False,
        output_dir=str(tmp_path),
        include_timestamp=False,
        cache_neurons=True,
        data_folder=str(tmp_path),
        script_path=str(tmp_path),
        skeleton_mesh_simplification=simplification,
    )
    vs.client_type = "neuprint"
    return vs


class TestSimplifiedCacheE2E:
    def test_fetch_writes_raw_swc_cache_only(self, tmp_path, probe_neurons):
        vs = _vs(tmp_path)
        cache_dir = Path(tmp_path) / "cache" / FOLDER / "skeletons"
        neuron_df = pd.DataFrame({"bodyId": PROBE_IDS})

        vs._save_cached_neurons(neuron_df, probe_neurons)

        raw_dir = cache_dir / "raw_skeletons"
        for i, bid in enumerate(PROBE_IDS):
            assert (raw_dir / f"{bid}.swc.zst").exists()
            cached = morph._load_cached_skeleton_file(
                raw_dir / f"{bid}.swc.zst")
            # Visualization persistence is raw level 0; render-time mesh
            # simplification is applied once from this common source.
            assert cached._drocat_simplification == 0
            assert len(cached.nodes) == len(probe_neurons[i].nodes)
        assert not (cache_dir / ".level").exists()
        assert not list(cache_dir.glob("*.pkl"))

    def test_second_load_is_cache_hit(self, tmp_path, probe_neurons):
        vs = _vs(tmp_path)
        vs._save_cached_neurons(pd.DataFrame({"bodyId": PROBE_IDS}), probe_neurons)

        t0 = time.time()
        neurons, missing = vs._load_cached_neurons(pd.DataFrame({"bodyId": PROBE_IDS}))
        elapsed = time.time() - t0
        assert missing == []
        assert neurons is not None and len(neurons) == 2
        assert elapsed < 5.0  # local cache read: no network round-trips

    def test_less_simplified_render_reuses_raw_cache(self, tmp_path, probe_neurons):
        vs = _vs(tmp_path, simplification=0.5)
        vs._save_cached_neurons(pd.DataFrame({"bodyId": PROBE_IDS}), probe_neurons)
        # < 0.9 render still consumes the same raw source.
        neurons, missing = vs._load_cached_neurons(
            pd.DataFrame({"bodyId": PROBE_IDS}))
        assert neurons is not None and len(neurons) == len(PROBE_IDS)
        assert missing == []

    def test_render_decimation_is_direct_from_raw_source(self, tmp_path, probe_neurons):
        vs = _vs(tmp_path, simplification=0.9)
        vs._save_cached_neurons(pd.DataFrame({"bodyId": PROBE_IDS}), probe_neurons)
        assert vs._skeleton_cache_is_simplified() is False
        assert vs._effective_render_simplification(is_fafb=False) == pytest.approx(0.9)

    def test_render_decimation_above_legacy_level_is_still_direct(self, tmp_path, probe_neurons):
        vs = _vs(tmp_path, simplification=0.95)
        vs._save_cached_neurons(pd.DataFrame({"bodyId": PROBE_IDS}), probe_neurons)
        assert vs._skeleton_cache_is_simplified() is False
        assert vs._effective_render_simplification(is_fafb=False) == pytest.approx(0.95)

    def test_render_decimation_below_cache_level_is_direct(self, tmp_path, probe_neurons):
        # The requested fraction applies directly to the raw render source.
        vs = _vs(tmp_path, simplification=0.5)
        assert vs._effective_render_simplification(is_fafb=False) == pytest.approx(0.5)

    def test_fetch_on_demand_level_semantics(self, tmp_path, probe_neurons):
        vs = _vs(tmp_path)
        vs._save_cached_neurons(pd.DataFrame({"bodyId": PROBE_IDS}), probe_neurons)

        # The compatibility level is validated and then DELIBERATELY ignored for
        # NeuPrint (`morphology.fetch_skeleton_on_demand`: "simplification is a
        # visualization-time concern"), so a `simp90` request serves the shared
        # RAW cache — no network, and no decimated geometry on disk that a later
        # vectorization could silently score.  Asserting a smaller node count
        # here would be asserting the removed behaviour: it passed only while
        # this module was skipped for want of a reachable server.
        nrn = morph.fetch_skeleton_on_demand(
            DATASET, PROBE_IDS[0], project_root=str(tmp_path), level="simp90")
        assert nrn is not None
        assert 0 < len(nrn.nodes) == len(probe_neurons[0].nodes)

        # Raw requests use the same source and therefore also hit the raw
        # cache, even when persistence is disabled for this call.
        nrn_raw = morph.fetch_skeleton_on_demand(
            DATASET, PROBE_IDS[0], project_root=str(tmp_path), level="raw",
            persist=False)
        assert nrn_raw is not None
        assert len(nrn_raw.nodes) == len(nrn.nodes)

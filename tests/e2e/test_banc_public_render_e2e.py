"""Live end-to-end tests for BANC public-bucket visualization.

Runs against the real public GCS bucket (skipped when the network is
unavailable).  Exercises the documented edge cases of the public skeleton
fetch and a full single-layer render through VisualizeSkeleton.

Run:  pytest tests/e2e/test_banc_public_render_e2e.py -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import banc_public_data  # noqa: E402
import navis  # noqa: E402


def _network_available() -> bool:
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen(
            "https://storage.googleapis.com/", timeout=5)
        return True
    except urllib.error.HTTPError:
        # Reaching the server at all means the network is up.
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not _network_available(),
                       reason="network unavailable"),
]

# l-LNv bodyIds in banc_v888 with documented bucket states.
L_LNV_L2 = "720575941442818239"        # L2 skeleton available
L_LNV_FULL_ONLY = "720575941471456733"  # full-res only (L2 404s)
# No compiled_data SWC in any namespace — but the v626-named pcg-skel set
# covers it (plan §E correction), so the fetcher reverse-crosswalks the id
# and scales the micrometre skeleton to nanometres.
L_LNV_MISSING = "720575941512418576"


class TestLiveBancFetch:
    def test_l2_fetch(self):
        neuron = banc_public_data.fetch_banc_swc("banc_v888", L_LNV_L2,
                                                 use_cache=False)
        assert neuron is not None and len(neuron.nodes) > 100
        assert "nanometer" in str(neuron.units)

    def test_l2_404_falls_back_to_full(self):
        neuron = banc_public_data.fetch_banc_swc("banc_v888", L_LNV_FULL_ONLY,
                                                 use_cache=False)
        assert neuron is not None and len(neuron.nodes) > 1000

    def test_missing_from_compiled_data_renders_via_pcg_fallback(self):
        """Plan §E correction: previously 'unrecoverable', this l-LNv now
        renders through the micrometre pcg-skel fallback (scaled to nm)
        instead of returning None."""
        neuron = banc_public_data.fetch_banc_swc(
            "banc_v888", L_LNV_MISSING, use_cache=False)
        assert neuron is not None and len(neuron.nodes) > 10
        assert "nanometer" in str(neuron.units)
        # The pcg-skel product is micrometres; the fetched neuron must sit
        # in the shared nanometre frame (unscaled µm coordinates would
        # max out near ~1,200).
        coords = neuron.nodes[["x", "y", "z"]].to_numpy()
        assert float(abs(coords).max()) > 10_000


class TestLiveBancRender:
    def test_single_layer_render(self, tmp_path):
        from visualize_skeleton import VisualizeSkeleton

        vs = VisualizeSkeleton(
            dataset="banc_v888",
            neuron_layers=[["l-LNv"]],
            output_dir=str(tmp_path),
            brain_mesh="native",
            mesh_roi=[],
            skip_synapse=True,
            show_fig=False,
            legend_mode="type",
            cache_neurons=True,
            export_views=False,
        )
        vs.plot_neurons()
        htmls = list(Path(tmp_path).rglob("*.html"))
        assert htmls, "expected an exported HTML scene"
        content = htmls[0].read_text()
        assert '"mesh3d"' in content, "expected mesh3d traces in the scene"

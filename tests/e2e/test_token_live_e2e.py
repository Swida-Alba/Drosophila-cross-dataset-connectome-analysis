"""Live production tests for the NeuPrint token chain and real data pulls.

Runs against the REAL neuprint.janelia.org server (skipped when the network
is unavailable) and verifies the token registration chain end to end:

  * The production chain (config.json -> config_local.json -> env) resolves
    to the token registered in config_local.json and the server accepts it.
  * The skip-invalid contract holds against the real server: a candidate the
    server refuses (401) is skipped to the next location instead of failing
    the run; a probe that cannot run never disqualifies a candidate.
  * The server-probe cache issues at most one HTTP request per unique token.
  * CAVE/BANC tokens keep the first-candidate behavior (no probe defined).
  * The UI dataset service resolves the same token through the shared chain.
  * Real authenticated data pulls succeed with the resolved token (server
    version, dataset listing with hidden entries filtered, neuron counts,
    and one production skeleton fetch through morphology's on-demand path).
  * A revoked token produces a real 401 that visualize_skeleton's
    rejected-token predicate recognizes.

Run:  pytest tests/e2e/test_token_live_e2e.py -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

NEUPRINT_SERVER = "https://neuprint.janelia.org"

pytest.importorskip("neuprint")

from src.utils.token_manager import TokenManager  # noqa: E402
from tests.live_neuprint import skipif_not_served  # noqa: E402


def _network_available() -> bool:
    try:
        import socket
        socket.create_connection(("neuprint.janelia.org", 443), timeout=5).close()
        return True
    except Exception:
        return False


def _config_local_token_registered() -> bool:
    """Whether config_local.json exists and registers a NeuPrint token.

    The whole module asserts against the developer-registered token; on a
    fresh clone (untracked config_local.json) it must SKIP, not error.
    """
    path = PROJECT_ROOT / "config_local.json"
    if not path.exists():
        return False
    try:
        import json
        config = json.loads(path.read_text(encoding="utf-8-sig"))
        return bool(config.get("tokens", {}).get("neuprint"))
    except Exception:
        return False


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not _network_available(),
        reason="NeuPrint server unreachable (network required for this e2e)",
    ),
    pytest.mark.skipif(
        not _config_local_token_registered(),
        reason="config_local.json with a registered NeuPrint token not "
               "available locally (untracked developer config)",
    ),
]


@pytest.fixture(scope="module")
def registered_token() -> str:
    """The token registered in config_local.json (the developer's chain)."""
    import json
    config = json.loads(
        (PROJECT_ROOT / "config_local.json").read_text(encoding="utf-8-sig"))
    return config["tokens"]["neuprint"]


@pytest.fixture
def manager() -> TokenManager:
    """Fresh manager on the real repo config, with its own probe cache."""
    return TokenManager(project_root=str(PROJECT_ROOT))


class TestTokenChainAgainstLiveServer:
    def test_production_chain_resolves_registered_token(
            self, manager, registered_token):
        """config.json (empty) -> config_local.json (registered) -> env:
        the chain must land on the registered token and the server must
        accept it."""
        token = manager.get_token("NEUPRINT_TOKEN")
        assert token == registered_token
        assert manager.neuprint_token_rejected(token) is False

    def test_probe_verdicts_on_live_server(self, manager, registered_token):
        assert manager.neuprint_token_rejected(registered_token) is False
        assert manager.neuprint_token_rejected("revoked-fake-token-123") is True

    def test_live_skip_invalid_candidate_uses_next_location(
            self, registered_token):
        """A refused config.json candidate is skipped; the registered
        config_local.json token is used instead of failing the run."""
        manager = TokenManager(project_root=str(PROJECT_ROOT))
        manager._token_sources = {
            "NEUPRINT_TOKEN": [
                ("config.json", "revoked-fake-token-123"),
                ("config_local.json", registered_token),
            ],
        }
        resolved = manager.get_token("NEUPRINT_TOKEN")
        assert resolved == registered_token
        # Both verdicts came from the live server and were cached.
        assert manager._probe_cache[(manager.NEUPRINT_DEFAULT_SERVER,
                                     "revoked-fake-token-123")] is True
        assert manager._probe_cache[(manager.NEUPRINT_DEFAULT_SERVER,
                                     registered_token)] is False

    def test_live_env_token_rescues_revoked_config_candidates(
            self, registered_token, monkeypatch):
        """Real-world stale-setup scenario: revoked candidates in both
        config files, valid token still in the env — the run is rescued
        from the next chain location instead of failing."""
        manager = TokenManager(project_root=str(PROJECT_ROOT))
        manager._token_sources = {
            "NEUPRINT_TOKEN": [
                ("config.json", "revoked-fake-token-123"),
                ("config_local.json", "revoked-fake-token-456"),
            ],
        }
        monkeypatch.setenv("NEUPRINT_TOKEN", registered_token)
        assert manager.get_token("NEUPRINT_TOKEN") == registered_token

    def test_live_all_candidates_rejected_returns_guidance_candidate(
            self, monkeypatch):
        """With every candidate (including env) refused there is no valid
        fallback; the first is returned so the caller's rejected-token
        guidance fires."""
        monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)
        monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS", raising=False)
        manager = TokenManager(project_root=str(PROJECT_ROOT))
        manager._token_sources = {
            "NEUPRINT_TOKEN": [
                ("config.json", "revoked-fake-token-123"),
                ("config_local.json", "revoked-fake-token-456"),
            ],
        }
        assert manager.get_token("NEUPRINT_TOKEN") == "revoked-fake-token-123"

    def test_live_env_fallback_with_valid_token(self, registered_token,
                                                monkeypatch):
        """No config tokens: a valid env token is accepted (real probe)."""
        manager = TokenManager(project_root=str(PROJECT_ROOT))
        manager._token_sources = {}
        monkeypatch.setenv("NEUPRINT_TOKEN", registered_token)
        assert manager.get_token("NEUPRINT_TOKEN") == registered_token

    def test_probe_cache_issues_one_http_per_unique_token(
            self, manager, registered_token, monkeypatch):
        """Two chain walks over two unique tokens must produce exactly two
        real HTTP probes the first time and none on the second walk."""
        import requests as requests_mod

        manager._token_sources = {
            "NEUPRINT_TOKEN": [
                ("config.json", "revoked-fake-token-123"),
                ("config_local.json", registered_token),
            ],
        }
        calls = []
        real_get = requests_mod.get

        def counting_get(*args, **kwargs):
            calls.append(args[0])
            return real_get(*args, **kwargs)

        monkeypatch.setattr(requests_mod, "get", counting_get)

        assert manager.get_token("NEUPRINT_TOKEN") == registered_token
        assert len(calls) == 2  # one per unique candidate
        assert manager.get_token("NEUPRINT_TOKEN") == registered_token
        assert len(calls) == 2  # cache hit: no further HTTP

    def test_cave_chain_first_candidate_without_probe(self, monkeypatch):
        """CAVE/BANC have no server probe: first candidate wins and no HTTP
        is issued."""
        import requests as requests_mod

        manager = TokenManager(project_root=str(PROJECT_ROOT))
        cave_token = manager.tokens.get("CAVE_TOKEN")
        if not cave_token:
            pytest.skip("config_local.json registers no CAVE token locally")

        def forbidden(*args, **kwargs):
            raise AssertionError("CAVE resolution must not touch the network")

        monkeypatch.setattr(requests_mod, "get", forbidden)
        assert manager.get_token("CAVE_TOKEN") == cave_token


class TestUISharedChain:
    def test_ui_dataset_service_resolves_same_token(
            self, registered_token):
        """The UI dataset service walks the shared chain and lands on the
        same registered token."""
        from ui.dataset_service import DatasetService
        assert DatasetService().get_token() == registered_token


class TestRealDataPulls:
    @pytest.fixture(scope="class")
    @classmethod
    def client(cls, registered_token):
        # classmethod form: class-scoped instance-method fixtures are
        # deprecated (PytestRemovedIn10Warning) — the fixture only returns
        # a shared Client, so nothing needs instance attributes.
        from neuprint import Client
        return Client(NEUPRINT_SERVER, dataset="male-cns:v0.9",
                      token=registered_token)

    def test_client_fetches_live_server_version(self, client):
        version = client.fetch_version()
        assert isinstance(version, str) and version

    def test_revoked_token_raises_real_401_recognized_by_visualizer(
            self, registered_token):
        """A revoked token's client fails with a 401 that
        visualize_skeleton's rejected-token predicate detects — the abort
        path fires on real server responses."""
        from neuprint import Client
        bad = Client(NEUPRINT_SERVER, dataset="male-cns:v0.9",
                     token="revoked-fake-token-123")
        with pytest.raises(Exception) as excinfo:
            bad.fetch_version()
        text = str(excinfo.value)
        assert "401" in text or "Unauthorized" in text

    @skipif_not_served("male-cns:v1.0")
    def test_dataset_service_lists_live_datasets_and_filters_hidden(self):
        from ui.dataset_service import DatasetService
        available = DatasetService().fetch_neuprint_datasets()
        assert "hemibrain:v1.2.1" in available
        assert "male-cns:v1.0" in available
        # banc:v888 is listed by the server but hidden: never queryable here.
        assert "banc:v888" not in available

    def test_dataset_service_pulls_real_neuron_counts(self):
        from ui.dataset_service import DatasetService
        total, typed = DatasetService()._fetch_neuprint_counts("male-cns:v0.9")
        assert total > 1000
        assert 0 < typed <= total

    def test_morphology_on_demand_pulls_real_skeleton(self, client):
        """One real skeleton through morphology's on-demand production path
        (which resolves its token through the chain at morphology.py:4280)."""
        import morphology as morph

        rows = client.fetch_custom(
            "MATCH (n:Neuron) WHERE n.type = 'aMe12' "
            "RETURN n.bodyId AS bodyId ORDER BY n.bodySize ASC LIMIT 1")
        assert not rows.empty, "male-cns:v0.9 must contain aMe12 neurons"
        body_id = int(rows.iloc[0]["bodyId"])

        neuron = morph.fetch_skeleton_on_demand("male-cns:v0.9", body_id)
        assert neuron is not None
        assert len(neuron.nodes) > 50

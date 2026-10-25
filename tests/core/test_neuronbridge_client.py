"""Tests for DROCAT's dependency-light NeuronBridge API client."""

from io import BytesIO

from PIL import Image
import pytest

from neuronbridge_client import Client


class FakeResponse:
    def __init__(self, *, text="", payload=None, content=b"", status=200):
        self.text = text
        self._payload = payload
        self.content = content
        self.raw = BytesIO(content)
        self.status_code = status
        self.closed = False

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, responses):
        self.responses = responses
        self.urls = []

    def get(self, url, **_kwargs):
        self.urls.append(url)
        return self.responses[url]


def _client_and_urls():
    root = "https://test-bucket.s3.us-east-1.amazonaws.com"
    version_root = f"{root}/v-test"
    config = {
        "defaultSearchLibrary": "ignored-new-api-field",
        "stores": {
            "brain": {
                "prefixes": {
                    "CDSResults": f"{version_root}/matches/",
                    "CDM": f"{version_root}/images/",
                }
            }
        },
    }
    responses = {
        f"{root}/current.txt": FakeResponse(text="v-test\n"),
        f"{version_root}/config.json": FakeResponse(payload=config),
    }
    session = FakeSession(responses)
    return Client(data_bucket="test-bucket", session=session), session, responses, version_root


def test_client_ignores_new_config_fields_and_builds_nested_objects():
    client, _session, responses, version_root = _client_and_urls()
    lookup_url = f"{version_root}/metadata/by_body/42.json"
    responses[lookup_url] = FakeResponse(payload={
        "results": [{
            "type": "EMImage",
            "id": "em-42",
            "publishedName": "hemibrain:v1.2.1:42",
            "files": {"store": "brain", "CDSResults": "42.json"},
        }]
    })

    images = client.get_em_images(42)

    assert client.version == "v-test"
    assert client.config.defaultSearchLibrary == "ignored-new-api-field"
    assert images[0].publishedName == "hemibrain:v1.2.1:42"
    assert images[0].files.CDSResults == "42.json"
    assert all(response.closed for response in responses.values())


def test_client_fetches_matches_and_resolves_store_prefixes():
    client, session, responses, version_root = _client_and_urls()
    lookup_url = f"{version_root}/metadata/by_line/LH173.json"
    match_url = f"{version_root}/matches/lh173.json"
    responses[lookup_url] = FakeResponse(payload={
        "results": [{
            "type": "LMImage",
            "id": "lm-1",
            "files": {"store": "brain", "CDSResults": "lh173.json"},
        }]
    })
    responses[match_url] = FakeResponse(payload={
        "results": [{
            "type": "CDSMatch",
            "normalizedScore": 0.91,
            "files": {"store": "brain"},
            "image": {"type": "EMImage", "id": "em-1", "files": {"store": "brain"}},
        }]
    })

    matches = client.get_cds_matches(client.get_lm_images("LH173")[0])

    assert match_url in session.urls
    assert matches[0].normalizedScore == 0.91
    assert matches[0].image.id == "em-1"


def test_client_loads_images_from_match_or_neuron_files():
    client, _session, responses, version_root = _client_and_urls()
    image_url = f"{version_root}/images/sample.png"
    buffer = BytesIO()
    Image.new("RGB", (2, 3), "red").save(buffer, format="PNG")
    responses[image_url] = FakeResponse(content=buffer.getvalue())
    match = type("Match", (), {
        "files": type("Files", (), {"store": "brain", "CDM": None})(),
        "image": type("ImageObject", (), {
            "files": type("Files", (), {"store": "brain", "CDM": "sample.png"})()
        })(),
    })()

    image = client.get_cds_image(match)

    assert image.size == (2, 3)
    assert responses[image_url].closed


def test_client_url_encodes_lookup_ids():
    client, session, responses, version_root = _client_and_urls()
    lookup_url = f"{version_root}/metadata/by_line/line%2Fwith%20space.json"
    responses[lookup_url] = FakeResponse(payload={"results": []})

    assert client.get_lm_images("line/with space") == []
    assert session.urls[-1] == lookup_url


def test_aggregate_results_polars_mixed_bodyid_columns():
    """Columns mixing python ints with numeric strings (large bodyId /
    rootId values from the API vs ints from local matches) must not crash
    the Polars conversion with ArrowInvalid."""
    import pandas as pd
    from neuronbridge_finder import NeuronBridgeFinder

    finder = object.__new__(NeuronBridgeFinder)
    finder.separate_splitgal4 = False
    finder._vprint = lambda *a, **k: None

    df = pd.DataFrame({
        "line": ["KCab-p", "KCab-p", "MBON01"],
        "score": [0.9, 0.8, 0.7],
        "source_bodyId": ["123", "456", "789"],
        "bodyId": [12345, "2881486496092082187", 999],   # mixed int/str
        "rootId": ["2881486496092082187", 42, "77"],     # mixed str/int
        "label": ["a", "b", 3],                          # mixed labels
        "huge": ["99999999999999999999999", "1", "2"],   # beyond int64
    })
    combined, line_stats = finder._aggregate_results_polars(df, "cds", False, "max")
    assert not combined.empty and len(line_stats) == 2
    # bodyId columns stay strings (explicit contract); other all-numeric
    # columns become exact int64; mixed labels stay strings; huge IDs are
    # preserved exactly (never rounded through float64).
    assert combined["source_bodyId"].dtype == object
    assert combined["bodyId"].dtype == object
    assert combined["rootId"].dtype.kind == "i"
    assert combined["label"].dtype == object
    assert combined["huge"].iloc[0] == "99999999999999999999999"
    # the aggregation still computes per-line stats
    assert set(line_stats.columns) >= {"line", "match_count", "agg_mean_score"}


def test_line_stats_sorted_by_weighted_score_for_max():
    """sort_by='max' (default) ranks lines by weighted_score
    (agg_mean_score × coverage_ratio), not by the raw mean."""
    import pandas as pd
    from neuronbridge_finder import NeuronBridgeFinder

    finder = object.__new__(NeuronBridgeFinder)
    finder.separate_splitgal4 = False
    finder._vprint = lambda *a, **k: None

    # A: 3 matches @ 0.6  -> mean 0.60, coverage 3/4, weighted 0.450
    # B: 2 matches @ 0.95 -> mean 0.95, coverage 2/4, weighted 0.475
    # 'max' must prefer B (higher weighted) even though A covers more.
    df = pd.DataFrame({
        "line": ["A", "A", "A", "B", "B"],
        "score": [0.6, 0.6, 0.6, 0.95, 0.95],
        "source_bodyId": ["q1", "q2", "q3", "q1", "q2"],
    })
    _, line_stats = finder._aggregate_results_polars(df, "cds", False, "max")
    assert line_stats["line"].tolist() == ["B", "A"]
    w = dict(zip(line_stats["line"], line_stats["weighted_score"]))
    assert w["B"] > w["A"]


def test_line_stats_sorted_by_coverage_ratio_for_completeness():
    """sort_by='completeness' ranks lines by coverage_ratio (fraction of
    query neurons matched), regardless of score."""
    import pandas as pd
    from neuronbridge_finder import NeuronBridgeFinder

    finder = object.__new__(NeuronBridgeFinder)
    finder.separate_splitgal4 = False
    finder._vprint = lambda *a, **k: None

    # Same data as above: A covers 3/4, B only 2/4 but scores higher.
    df = pd.DataFrame({
        "line": ["A", "A", "A", "B", "B"],
        "score": [0.6, 0.6, 0.6, 0.95, 0.95],
        "source_bodyId": ["q1", "q2", "q3", "q1", "q2"],
    })
    _, line_stats = finder._aggregate_results_polars(df, "cds", False, "completeness")
    assert line_stats["line"].tolist() == ["A", "B"]
    c = dict(zip(line_stats["line"], line_stats["coverage_ratio"]))
    assert c["A"] > c["B"]


def test_coverage_ratio_uses_resolved_query_set():
    import pandas as pd
    from neuronbridge_finder import NeuronBridgeFinder

    finder = object.__new__(NeuronBridgeFinder)
    finder.separate_splitgal4 = False
    finder._vprint = lambda *a, **k: None
    df = pd.DataFrame({
        "line": ["A", "A"],
        "score": [0.6, 0.6],
        "source_bodyId": ["q1", "q2"],
    })
    _, stats = finder._aggregate_results_polars(
        df, "cds", False, "completeness",
        query_bodyids={"q1", "q2", "q3"})
    assert stats.iloc[0]["coverage_ratio"] == pytest.approx(2 / 3)


def test_parse_region_from_filename_flweb_tags():
    """flweb CGI imagery embeds the region in the sample tag
    (fA01b/fA00b = brain, fA01v/fA00v = VNC); it must not fall through
    to 'Other' or the Brain/VNC region filters drop every flweb file."""
    import pandas as pd
    from neuronbridge_finder import NeuronBridgeFinder

    finder = object.__new__(NeuronBridgeFinder)
    finder.region = "Brain"
    finder._vprint = lambda *a, **k: None

    brain = "R85D07_AE_01_03-fA01b_C101223_20101223135821687_total.jpg"
    vnc = "R85D07_AE_01_01-fA01v_C101223_20101223135718562_total.jpg"
    assert finder._parse_region_from_filename(brain, "") == "Brain"
    assert finder._parse_region_from_filename(vnc, "") == "VNC"
    assert finder._parse_region_from_filename("fA00b_something.jpg", "") == "Brain"
    assert finder._parse_region_from_filename("fA00v_something.jpg", "") == "VNC"

    # the region filter now keeps the flweb brain projection for Brain mode
    class FakeFile:
        def __init__(self, filename, key=""):
            self.filename = filename
            self.key = key
            self.url = ""

    kept = finder._filter_flylight_files_by_region(
        [FakeFile(brain, brain), FakeFile(vnc, vnc)]
    )
    assert [f.filename for f in kept] == [brain]

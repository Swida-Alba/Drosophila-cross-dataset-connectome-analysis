"""Coverage-focused tests for cave_data_fetcher.

Targets the uncovered branches reported in coverage_missing.txt:
token loading, config selection, lazy cloudvolume/caveclient properties,
swc.gz / pickle cache round-trips, legacy pickle migration, mesh/skeleton
fetch error paths, CAVE query methods (synapses / connections / neuron
info / types), cache maintenance and the module-level access probe.

All network entry points (cloudvolume, caveclient) are faked through
sys.modules injection or injected fake clients; every filesystem path is
redirected to pytest tmp_path.
"""

import gzip
import io
import os
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import navis  # noqa: E402

import cave_data_fetcher as cdf  # noqa: E402


BODY_ID = 720575940596125868


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_fetcher(tmp_path, verbose=False, cache_enabled=True,
                 dataset='flywire_FAFB_v783'):
    """Build a fetcher without running __post_init__ (no real token reads)."""
    f = object.__new__(cdf.CAVEDataFetcher)
    f.dataset = dataset
    f.cave_token = 'test-token'
    f.materialization_version = None
    f.cache_enabled = cache_enabled
    f.project_root = str(tmp_path)
    f.verbose = verbose
    f._cv = None
    f._cave_client = None
    return f


def chain_swc(n=6):
    lines = []
    for i in range(1, n + 1):
        parent = -1 if i == 1 else i - 1
        lines.append(f"{i} 3 {i * 1000.0} 0.0 0.0 5.0 {parent}")
    return "\n".join(lines) + "\n"


def chain_neuron(n=6):
    return navis.read_swc(io.StringIO(chain_swc(n)))


def tetra_trimesh():
    import trimesh
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                     dtype=float) * 1000.0
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    return trimesh.Trimesh(vertices=verts, faces=faces)


class FakeMeshSource:
    def __init__(self, meshes=None, fail_ids=()):
        self.meshes = meshes or {}
        self.fail_ids = set(fail_ids)
        self.calls = []

    def get(self, body_id, **kwargs):
        self.calls.append((body_id, kwargs))
        if body_id in self.fail_ids:
            raise RuntimeError("mesh fetch failed")
        return {body_id: self.meshes[body_id]}


class FakeCloudVolume:
    def __init__(self, meshes=None, fail_ids=()):
        self.mesh = FakeMeshSource(meshes, fail_ids)


class FakeMaterialize:
    def __init__(self, view_results=None, table_results=None, raise_on=None):
        self.view_results = view_results or {}
        self.table_results = table_results or {}
        self.raise_on = raise_on or set()
        self.view_calls = []
        self.table_calls = []

    def query_view(self, view, filter_in_dict=None, **kwargs):
        key = tuple(sorted((filter_in_dict or {}).keys()))
        self.view_calls.append((view, key))
        if 'view' in self.raise_on:
            raise RuntimeError("view query failed")
        return self.view_results.get(key, pd.DataFrame())

    def query_table(self, table, filter_in_dict=None, **kwargs):
        self.table_calls.append((table, kwargs))
        if table in self.raise_on:
            raise RuntimeError(f"table query failed: {table}")
        result = self.table_results.get(table, pd.DataFrame())
        if callable(result):
            return result(filter_in_dict)
        return result.copy() if isinstance(result, pd.DataFrame) else result


class FakeCaveClient:
    def __init__(self, materialize):
        self.materialize = materialize


# ---------------------------------------------------------------------------
# construction / token / config
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_post_init_autodetects_root_and_calls_token_loader(
            self, monkeypatch):
        calls = []

        def fake_load(self, name):
            calls.append(name)
            return 'fake-token'

        monkeypatch.setattr(cdf.CAVEDataFetcher, '_load_token', fake_load)
        f = cdf.CAVEDataFetcher(cave_token=None)
        assert f.project_root  # auto-detected from module location
        assert f.cave_token == 'fake-token'
        assert calls == ['CAVE_TOKEN']

    def test_load_token_from_config_json(self, tmp_path):
        (tmp_path / 'config.json').write_text(
            '{"tokens": {"cave": "cfg-tok"}}\n', encoding='utf-8')
        f = make_fetcher(tmp_path)
        assert f._load_token('CAVE_TOKEN') == 'cfg-tok'

    def test_config_json_wins_over_legacy_token_files(self, tmp_path):
        """token_info files are deprecated; only config.json is read."""
        (tmp_path / 'token_info_local.txt').write_text(
            "CAVE_TOKEN='legacy-tok'\n", encoding='utf-8')
        (tmp_path / 'token_info.txt').write_text(
            "CAVE_TOKEN='template-tok'\n", encoding='utf-8')
        (tmp_path / 'config.json').write_text(
            '{"tokens": {"cave": "cfg-tok"}}\n', encoding='utf-8')
        f = make_fetcher(tmp_path)
        assert f._load_token('CAVE_TOKEN') == 'cfg-tok'

    def test_legacy_token_files_ignored_without_config(self, tmp_path,
                                                       monkeypatch):
        monkeypatch.delenv('CAVE_TOKEN', raising=False)
        (tmp_path / 'token_info_local.txt').write_text(
            "CAVE_TOKEN='legacy-tok'\n", encoding='utf-8')
        f = make_fetcher(tmp_path)
        assert f._load_token('CAVE_TOKEN') is None

    def test_config_json_wins_over_config_local(self, tmp_path):
        (tmp_path / 'config.json').write_text(
            '{"tokens": {"cave": "cfg-tok"}}\n', encoding='utf-8')
        (tmp_path / 'config_local.json').write_text(
            '{"tokens": {"cave": "local-tok"}}\n', encoding='utf-8')
        f = make_fetcher(tmp_path)
        assert f._load_token('CAVE_TOKEN') == 'cfg-tok'

    def test_config_local_fills_empty_config_json(self, tmp_path):
        (tmp_path / 'config.json').write_text(
            '{"tokens": {"cave": ""}}\n', encoding='utf-8')
        (tmp_path / 'config_local.json').write_text(
            '{"tokens": {"cave": "local-tok"}}\n', encoding='utf-8')
        f = make_fetcher(tmp_path)
        assert f._load_token('CAVE_TOKEN') == 'local-tok'

    def test_config_json_placeholder_ignored(self, tmp_path, monkeypatch):
        (tmp_path / 'config.json').write_text(
            '{"tokens": {"cave": "YOUR_CAVE_TOKEN_HERE"}}\n', encoding='utf-8')
        monkeypatch.delenv('CAVE_TOKEN', raising=False)
        f = make_fetcher(tmp_path)
        assert f._load_token('CAVE_TOKEN') is None

    def test_load_token_falls_back_to_env(self, tmp_path, monkeypatch):
        (tmp_path / 'config.json').write_text(
            '{"tokens": {"cave": "YOUR_CAVE_TOKEN_HERE"}}\n', encoding='utf-8')
        monkeypatch.setenv('CAVE_TOKEN', 'env-tok')
        f = make_fetcher(tmp_path)
        assert f._load_token('CAVE_TOKEN') == 'env-tok'

    def test_load_token_no_source_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.delenv('CAVE_TOKEN', raising=False)
        f = make_fetcher(tmp_path)
        assert f._load_token('CAVE_TOKEN') is None

    def test_get_config_variants(self, tmp_path):
        banc = make_fetcher(tmp_path, dataset='banc_v626')
        assert banc._get_config() is cdf.CAVEDataFetcher.FLYWIRE_BANC_CONFIG
        fafb = make_fetcher(tmp_path)
        assert fafb._get_config() is cdf.CAVEDataFetcher.FLYWIRE_FAFB_CONFIG
        unknown = make_fetcher(tmp_path, dataset='something_else')
        with pytest.raises(ValueError, match="Unknown dataset"):
            unknown._get_config()

    def test_cache_dataset_name(self, tmp_path):
        fafb = make_fetcher(tmp_path)
        assert fafb._cache_dataset_name() == 'flywire_FAFB_v783'
        banc = make_fetcher(tmp_path, dataset='flywire_BANC:v8.88')
        assert banc._cache_dataset_name() == 'banc_v8_88'

    def test_ensure_cache_dir(self, tmp_path):
        f = make_fetcher(tmp_path)
        f._ensure_cache_dir()
        base = Path(f.get_cache_path())
        assert (base / 'skeletons').is_dir()
        assert (base / 'meshes').is_dir()

        disabled = make_fetcher(tmp_path, cache_enabled=False)
        disabled._ensure_cache_dir()  # no-op, must not raise

    def test_mesh_cache_paths(self, tmp_path):
        f = make_fetcher(tmp_path)
        mesh_path = f._get_mesh_cache_path(101)
        assert mesh_path.endswith('101.pkl.zst')
        assert str(tmp_path) in mesh_path
        legacy = f._get_legacy_mesh_cache_path(101)
        assert legacy.endswith(os.path.join('API_cache', 'meshes', '101.pkl'))


# ---------------------------------------------------------------------------
# lazy clients (cloudvolume / caveclient)
# ---------------------------------------------------------------------------

class TestLazyClients:
    def test_cloudvolume_property_success(self, tmp_path, monkeypatch,
                                          capsys):
        created = {}

        class CloudVolume:
            def __init__(self, url, use_https=None, secrets=None):
                created['url'] = url
                created['secrets'] = secrets

        mod = types.ModuleType('cloudvolume')
        mod.CloudVolume = CloudVolume
        monkeypatch.setitem(sys.modules, 'cloudvolume', mod)

        f = make_fetcher(tmp_path, verbose=True)
        cv = f.cloudvolume
        assert cv is f.cloudvolume  # cached on second access
        assert created['url'].startswith('graphene://')
        assert created['secrets'] == {'token': 'test-token'}
        assert 'CloudVolume' in capsys.readouterr().out

    def test_cloudvolume_configured_for_banc(self, tmp_path, monkeypatch):
        # The BANC mesh layer is a public precomputed product now: the
        # config carries a real URL instead of the old None placeholder.
        captured = {}

        def fake_cv(url, **kwargs):
            captured['url'] = url
            captured['kwargs'] = kwargs
            return object()

        mod = types.ModuleType('cloudvolume')
        mod.CloudVolume = fake_cv
        monkeypatch.setitem(sys.modules, 'cloudvolume', mod)
        f = make_fetcher(tmp_path, dataset='banc_v626')
        _ = f.cloudvolume
        assert captured['url'].startswith('precomputed://https://')
        # A configured token is attached only when present; the public
        # layer itself needs none (no secrets key when cave_token is None).
        assert captured['kwargs'].get('use_https') is True

    def test_cloudvolume_import_error(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, 'cloudvolume', None)
        f = make_fetcher(tmp_path)
        with pytest.raises(ImportError, match="cloudvolume package required"):
            _ = f.cloudvolume

    def test_cave_client_property_success(self, tmp_path, monkeypatch,
                                          capsys):
        created = {}

        class CAVEclient:
            def __init__(self, datastack_name=None, auth_token=None,
                         write_server_cache=None):
                created['datastack'] = datastack_name
                created['token'] = auth_token
                created['write_cache'] = write_server_cache

        mod = types.ModuleType('caveclient')
        mod.CAVEclient = CAVEclient
        monkeypatch.setitem(sys.modules, 'caveclient', mod)

        f = make_fetcher(tmp_path, verbose=True)
        client = f.cave_client
        assert client is f.cave_client
        assert created == {'datastack': 'flywire_fafb_public',
                           'token': 'test-token', 'write_cache': True}
        assert 'CAVE' in capsys.readouterr().out

    def test_cave_client_import_error(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, 'caveclient', None)
        f = make_fetcher(tmp_path)
        with pytest.raises(ImportError, match="caveclient package required"):
            _ = f.cave_client


# ---------------------------------------------------------------------------
# cache load/save primitives
# ---------------------------------------------------------------------------

class TestCachePrimitives:
    def test_load_disabled_or_missing(self, tmp_path):
        f = make_fetcher(tmp_path, cache_enabled=False)
        assert f._load_from_cache(str(tmp_path / 'x.pkl')) is None
        enabled = make_fetcher(tmp_path)
        assert enabled._load_from_cache(str(tmp_path / 'nope.pkl')) is None

    def test_load_swc_gz_roundtrip(self, tmp_path):
        f = make_fetcher(tmp_path)
        path = tmp_path / '101.swc.gz'
        path.write_bytes(gzip.compress(chain_swc().encode('utf-8')))
        neuron = f._load_from_cache(str(path))
        assert isinstance(neuron, navis.TreeNeuron)
        assert neuron.id == 101

    def test_load_swc_gz_non_numeric_name(self, tmp_path):
        f = make_fetcher(tmp_path)
        path = tmp_path / 'notanumber.swc.gz'
        path.write_bytes(gzip.compress(chain_swc().encode('utf-8')))
        neuron = f._load_from_cache(str(path))
        assert isinstance(neuron, navis.TreeNeuron)

    def test_load_swc_gz_non_treeneuron_result_returns_none(self, tmp_path,
                                                            monkeypatch):
        f = make_fetcher(tmp_path)
        path = tmp_path / '42.swc.gz'
        path.write_bytes(gzip.compress(chain_swc().encode('utf-8')))
        monkeypatch.setattr(navis, 'read_swc', lambda handle: ['not',
                                                               'a neuron'])
        assert f._load_from_cache(str(path)) is None

    def test_load_swc_gz_corrupt_returns_none(self, tmp_path):
        f = make_fetcher(tmp_path)
        path = tmp_path / '77.swc.gz'
        path.write_bytes(gzip.compress(b"this is not swc at all"))
        assert f._load_from_cache(str(path)) is None

    def test_load_pickle(self, tmp_path):
        f = make_fetcher(tmp_path)
        path = tmp_path / 'obj.pkl'
        path.write_bytes(pickle.dumps({'hello': 1}))
        assert f._load_from_cache(str(path)) == {'hello': 1}

    def test_save_swc_gz_requires_treeneuron(self, tmp_path):
        f = make_fetcher(tmp_path)
        path = tmp_path / 'sub' / '99.swc.gz'
        f._save_to_cache({'not': 'a neuron'}, str(path))
        assert not path.exists()  # TypeError swallowed by warning path

    def test_save_swc_gz_roundtrip(self, tmp_path):
        f = make_fetcher(tmp_path)
        path = tmp_path / 'sub' / '12.swc.gz'
        f._save_to_cache(chain_neuron(), str(path))
        assert path.exists()
        loaded = f._load_from_cache(str(path))
        assert isinstance(loaded, navis.TreeNeuron)
        assert len(loaded.nodes) == 6

    def test_save_load_swc_zst_roundtrip(self, tmp_path):
        """The canonical raw-SWC cache form is .swc.zst with a recorded
        level header (FlyWire raw skeletons are stored at level 0)."""
        f = make_fetcher(tmp_path)
        path = tmp_path / 'sub' / '12.swc.zst'
        f._save_to_cache(chain_neuron(), str(path))
        assert path.exists()
        import zstandard as zstd
        with open(path, 'rb') as handle:
            with zstd.ZstdDecompressor().stream_reader(handle) as reader:
                content = reader.read()
        assert b'# DROCAT simpl: 0' in content
        loaded = f._load_from_cache(str(path))
        assert isinstance(loaded, navis.TreeNeuron)
        assert len(loaded.nodes) == 6
        assert loaded.id == 12
        assert loaded._drocat_simplification == 0

    def test_load_legacy_swc_gz_without_header_still_works(self, tmp_path):
        f = make_fetcher(tmp_path)
        path = tmp_path / '77.swc.gz'
        path.write_bytes(gzip.compress(chain_swc().encode('utf-8')))
        loaded = f._load_from_cache(str(path))
        assert isinstance(loaded, navis.TreeNeuron)
        assert loaded._drocat_simplification == 0

    def test_canonical_skeleton_cache_path_is_swc_zst(self, tmp_path):
        f = make_fetcher(tmp_path)
        assert f._get_skeleton_cache_path(42).endswith('42.swc.zst')

    def test_get_cache_stats_counts_swc_zst(self, tmp_path):
        f = make_fetcher(tmp_path)
        api_skel = Path(f.get_cache_path('skeletons'))
        api_skel.mkdir(parents=True, exist_ok=True)
        (api_skel / '1.swc.zst').write_bytes(b'x')
        (api_skel / '2.swc.gz').write_bytes(b'x')
        stats = f.get_cache_stats()
        assert stats['skeleton_count'] == 2

    def test_save_pickle(self, tmp_path):
        f = make_fetcher(tmp_path)
        path = tmp_path / 'deep' / 'obj.pkl'
        f._save_to_cache([1, 2, 3], str(path))
        assert pickle.loads(path.read_bytes()) == [1, 2, 3]

    def test_save_disabled(self, tmp_path):
        f = make_fetcher(tmp_path, cache_enabled=False)
        path = tmp_path / 'obj.pkl'
        f._save_to_cache([1], str(path))
        assert not path.exists()


# ---------------------------------------------------------------------------
# mesh fetch paths
# ---------------------------------------------------------------------------

class TestFetchMesh:
    def test_fetch_mesh_success_verbose(self, tmp_path, capsys):
        f = make_fetcher(tmp_path, verbose=True)
        f._cv = FakeCloudVolume(meshes={BODY_ID: tetra_trimesh()})
        mesh = f.fetch_mesh(BODY_ID)
        assert isinstance(mesh, navis.MeshNeuron)
        assert mesh.id == BODY_ID
        out = capsys.readouterr().out
        assert 'Fetched mesh' in out
        # dedup opt-out is passed through
        assert f._cv.mesh.calls[0][1]['deduplicate_chunk_boundaries'] is False

    def test_fetch_mesh_failure_returns_none(self, tmp_path, capsys):
        f = make_fetcher(tmp_path, verbose=True)
        f._cv = FakeCloudVolume(fail_ids={BODY_ID})
        assert f.fetch_mesh(BODY_ID) is None
        assert 'Failed to fetch mesh' in capsys.readouterr().out


def fresh_fake_mesh_cache(monkeypatch, loaded=None):
    cache_cls = type('FakeMeshCache', (object,), {
        'loaded': loaded,
        'saved': {},
        '__init__': lambda self, *a, **k: None,
        'load': lambda self, body_id: self.loaded,
        'save': lambda self, data: self.saved.update(data),
    })
    monkeypatch.setattr(cdf, 'FlyWireMeshCache', cache_cls)
    return cache_cls


class TestFetchFafbMesh:
    def test_cache_hit(self, tmp_path, monkeypatch, capsys):
        cached = navis.MeshNeuron(tetra_trimesh(), id=BODY_ID)
        fresh_fake_mesh_cache(monkeypatch, loaded=cached)
        f = make_fetcher(tmp_path, verbose=True)
        result = f.fetch_fafb_mesh(BODY_ID)
        assert result is cached
        assert 'Loaded prepared mesh from cache' in capsys.readouterr().out

    def test_raw_fetch_failure_returns_none(self, tmp_path, monkeypatch):
        fresh_fake_mesh_cache(monkeypatch)
        monkeypatch.setattr(cdf.CAVEDataFetcher, 'fetch_mesh',
                            lambda self, body_id, use_cache=False: None)
        f = make_fetcher(tmp_path)
        assert f.fetch_fafb_mesh(BODY_ID) is None

    def test_prepare_failure_falls_back_to_raw(self, tmp_path, monkeypatch):
        raw = navis.MeshNeuron(tetra_trimesh(), id=BODY_ID)
        cache_cls = fresh_fake_mesh_cache(monkeypatch)
        monkeypatch.setattr(cdf.CAVEDataFetcher, 'fetch_mesh',
                            lambda self, body_id, use_cache=False: raw)

        def boom(*args, **kwargs):
            raise RuntimeError("prepare failed")

        monkeypatch.setattr(cdf, 'prepare_flywire_mesh', boom)
        f = make_fetcher(tmp_path, verbose=True)
        result = f.fetch_fafb_mesh(BODY_ID)
        assert result is raw  # raw fallback returned to the caller
        assert not cache_cls.saved  # never promoted into prepared cache

    def test_prepare_success_saves_cache(self, tmp_path, monkeypatch):
        raw = navis.MeshNeuron(tetra_trimesh(), id=BODY_ID)
        prepared = navis.MeshNeuron(tetra_trimesh(), id=BODY_ID)
        cache_cls = fresh_fake_mesh_cache(monkeypatch)
        monkeypatch.setattr(cdf.CAVEDataFetcher, 'fetch_mesh',
                            lambda self, body_id, use_cache=False: raw)
        monkeypatch.setattr(
            cdf, 'prepare_flywire_mesh',
            lambda mesh, body_id, **kwargs: prepared)
        f = make_fetcher(tmp_path, verbose=True)
        result = f.fetch_fafb_mesh(BODY_ID, soma_pos=(1.0, 2.0, 3.0))
        assert result is prepared
        assert cache_cls.saved == {BODY_ID: prepared}

    def test_verbose_print_survives_prepared_without_trimesh(
            self, tmp_path, monkeypatch, capsys):
        class WeirdPrepared:  # len(prepared.trimesh.faces) raises
            pass

        fresh_fake_mesh_cache(monkeypatch)
        monkeypatch.setattr(cdf.CAVEDataFetcher, 'fetch_mesh',
                            lambda self, body_id, use_cache=False:
                            WeirdPrepared())
        monkeypatch.setattr(
            cdf, 'prepare_flywire_mesh',
            lambda mesh, body_id, **kwargs: WeirdPrepared())
        f = make_fetcher(tmp_path, verbose=True)
        result = f.fetch_fafb_mesh(BODY_ID, use_cache=False)
        assert isinstance(result, WeirdPrepared)
        assert 'Prepared FAFB mesh' in capsys.readouterr().out

    def test_fetch_fafb_meshes_mixed_results(self, tmp_path, monkeypatch,
                                             capsys):
        good = navis.MeshNeuron(tetra_trimesh(), id=1)
        outcomes = {1: good, 2: None}
        seen = []

        def fake_fetch_one(self, body_id, **kwargs):
            seen.append((body_id, kwargs.get('soma_pos')))
            return outcomes[int(body_id)]

        monkeypatch.setattr(cdf.CAVEDataFetcher, 'fetch_fafb_mesh',
                            fake_fetch_one)
        f = make_fetcher(tmp_path, verbose=True)
        result = f.fetch_fafb_meshes(
            [1, 2], soma_positions={1: (0, 0, 0), '2': (9, 9, 9)})
        assert len(result) == 1
        assert seen[0] == (1, (0, 0, 0))
        assert seen[1] == (2, (9, 9, 9))
        assert 'Failed to fetch 1/2' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# skeleton fetch paths
# ---------------------------------------------------------------------------

class TestFetchSkeleton:
    def test_legacy_pickle_migration(self, tmp_path):
        f = make_fetcher(tmp_path)
        legacy_path = Path(f._get_legacy_skeleton_cache_path(101))
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_path.write_bytes(pickle.dumps(chain_neuron()))

        neuron = f.fetch_skeleton(101, use_cache=True)
        assert isinstance(neuron, navis.TreeNeuron)
        # migrated to the canonical swc.gz location
        canonical = Path(f._get_skeleton_cache_path(101))
        assert canonical.exists()

    def test_cache_hit_with_denoise(self, tmp_path, monkeypatch):
        f = make_fetcher(tmp_path)
        path = f._get_skeleton_cache_path(55)
        f._save_to_cache(chain_neuron(), path)

        pruned = []

        def fake_prune(neuron, size=None, recursive=None):
            pruned.append(size)
            return neuron

        monkeypatch.setattr(navis, 'prune_twigs', fake_prune)
        neuron = f.fetch_skeleton(55, denoise_twigs=250.0)
        assert isinstance(neuron, navis.TreeNeuron)
        assert pruned == [250.0]

    def test_online_only_skips_cache(self, tmp_path, monkeypatch):
        f = make_fetcher(tmp_path)
        mesh = navis.MeshNeuron(tetra_trimesh(), id=7)
        monkeypatch.setattr(cdf.CAVEDataFetcher, 'fetch_mesh',
                            lambda self, body_id, use_cache=False: mesh)
        skeleton = chain_neuron(8)
        monkeypatch.setattr(navis, 'skeletonize',
                            lambda mesh, method=None: skeleton)
        result = f.fetch_skeleton(7, use_cache=False)
        assert result is skeleton
        assert not Path(f._get_skeleton_cache_path(7)).exists()

    def test_mesh_none_returns_none(self, tmp_path, monkeypatch):
        f = make_fetcher(tmp_path)
        monkeypatch.setattr(cdf.CAVEDataFetcher, 'fetch_mesh',
                            lambda self, body_id, use_cache=False: None)
        assert f.fetch_skeleton(9) is None

    def test_simplify_mesh_branch(self, tmp_path, monkeypatch, capsys):
        f = make_fetcher(tmp_path, verbose=True)
        import trimesh
        mesh = navis.MeshNeuron(
            trimesh.creation.icosphere(subdivisions=3), id=3)
        monkeypatch.setattr(cdf.CAVEDataFetcher, 'fetch_mesh',
                            lambda self, body_id, use_cache=False: mesh)
        skeleton = chain_neuron()
        monkeypatch.setattr(navis, 'skeletonize',
                            lambda mesh, method=None: skeleton)
        result = f.fetch_skeleton(3, simplify_mesh=0.5)
        assert result is skeleton
        assert 'Simplified mesh' in capsys.readouterr().out

    def test_simplify_mesh_failure_keeps_original(self, tmp_path,
                                                  monkeypatch):
        f = make_fetcher(tmp_path)
        mesh = navis.MeshNeuron(tetra_trimesh(), id=4)
        monkeypatch.setattr(cdf.CAVEDataFetcher, 'fetch_mesh',
                            lambda self, body_id, use_cache=False: mesh)
        skeleton = chain_neuron()
        monkeypatch.setattr(navis, 'skeletonize',
                            lambda mesh, method=None: skeleton)

        fake_trimesh = types.ModuleType('trimesh')

        class BrokenTrimesh:
            def __init__(self, vertices=None, faces=None):
                self._faces = np.asarray(faces)

            @property
            def faces(self):
                return self._faces

            def simplify_quadric_decimation(self, face_count=None):
                raise RuntimeError("decimation unavailable")

        fake_trimesh.Trimesh = BrokenTrimesh
        monkeypatch.setitem(sys.modules, 'trimesh', fake_trimesh)

        result = f.fetch_skeleton(4, simplify_mesh=0.5)
        assert result is skeleton

    def test_skeletonize_failure_returns_none(self, tmp_path, monkeypatch,
                                              capsys):
        f = make_fetcher(tmp_path, verbose=True)
        mesh = navis.MeshNeuron(tetra_trimesh(), id=5)
        monkeypatch.setattr(cdf.CAVEDataFetcher, 'fetch_mesh',
                            lambda self, body_id, use_cache=False: mesh)

        def boom(mesh, method=None):
            raise RuntimeError("skeletonize failed")

        monkeypatch.setattr(navis, 'skeletonize', boom)
        assert f.fetch_skeleton(5) is None
        assert 'Failed to fetch skeleton' in capsys.readouterr().out

    def test_denoise_failure_falls_back(self, monkeypatch):
        neuron = chain_neuron()

        def boom(neuron, size=None, recursive=None):
            raise RuntimeError("prune failed")

        monkeypatch.setattr(navis, 'prune_twigs', boom)
        assert cdf.CAVEDataFetcher._denoise_skeleton(neuron, 100) is neuron

    def test_fetch_skeletons_mixed(self, tmp_path, monkeypatch, capsys):
        f = make_fetcher(tmp_path, verbose=True)
        neuron = chain_neuron()
        outcomes = {1: neuron, 2: None}
        monkeypatch.setattr(cdf.CAVEDataFetcher, 'fetch_skeleton',
                            lambda self, bid, **kw: outcomes[int(bid)])
        result = f.fetch_skeletons([1, 2])
        assert len(result) == 1
        assert 'Failed to fetch 1/2' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# CAVE query methods
# ---------------------------------------------------------------------------

def synapse_frame(pre_id, post_id, n=2):
    return pd.DataFrame({
        'id': list(range(n)),
        'pre_pt_root_id': [pre_id] * n,
        'post_pt_root_id': [post_id] * n,
    })


class TestFetchSynapses:
    def test_both_directions(self, tmp_path, capsys):
        f = make_fetcher(tmp_path, verbose=True)
        mat = FakeMaterialize(view_results={
            ('pre_pt_root_id',): synapse_frame(BODY_ID, 200),
            ('post_pt_root_id',): synapse_frame(300, BODY_ID),
        })
        f._cave_client = FakeCaveClient(mat)
        result = f.fetch_synapses(BODY_ID, direction='both')
        assert result is not None
        assert len(result) == 4
        assert set(result['pre_pt_root_id']) == {str(BODY_ID), '300'}
        out = capsys.readouterr().out
        assert 'outgoing' in out and 'incoming' in out

    def test_single_direction_failure_is_tolerated(self, tmp_path):
        f = make_fetcher(tmp_path)
        post_df = synapse_frame(300, BODY_ID)
        mat = FakeMaterialize(
            view_results={('post_pt_root_id',): post_df})

        def selective(view, filter_in_dict=None, **kwargs):
            if 'pre_pt_root_id' in (filter_in_dict or {}):
                raise RuntimeError("pre query failed")
            return post_df.copy()

        mat.query_view = selective
        f._cave_client = FakeCaveClient(mat)
        result = f.fetch_synapses(BODY_ID, direction='both')
        assert result is not None
        assert set(result['pre_pt_root_id']) == {'300'}

    def test_all_queries_empty_returns_none(self, tmp_path):
        f = make_fetcher(tmp_path)
        mat = FakeMaterialize(raise_on={'view'})
        f._cave_client = FakeCaveClient(mat)
        assert f.fetch_synapses(BODY_ID) is None

    def test_normalization_failure_returns_none(self, tmp_path, monkeypatch,
                                                capsys):
        f = make_fetcher(tmp_path, verbose=True)
        mat = FakeMaterialize(view_results={
            ('pre_pt_root_id',): synapse_frame(BODY_ID, 200),
        })
        f._cave_client = FakeCaveClient(mat)

        def boom(frame, columns):
            raise ValueError("id normalization failed")

        monkeypatch.setattr(cdf, 'normalize_flywire_id_columns', boom)
        assert f.fetch_synapses(BODY_ID, direction='pre') is None
        assert 'Failed to fetch synapses' in capsys.readouterr().out


class TestFetchConnections:
    def _client_with(self, frames_by_dir, raise_all=False):
        results = {}
        if 'pre' in frames_by_dir:
            results[('pre_pt_root_id',)] = frames_by_dir['pre']
        if 'post' in frames_by_dir:
            results[('post_pt_root_id',)] = frames_by_dir['post']
        mat = FakeMaterialize(view_results=results,
                              raise_on={'view'} if raise_all else set())
        return FakeCaveClient(mat)

    def test_aggregation(self, tmp_path, capsys):
        f = make_fetcher(tmp_path, verbose=True)
        pre = synapse_frame(100, 200, n=3)
        f._cave_client = self._client_with({'pre': pre})
        result = f.fetch_connections([100], direction='pre')
        assert result is not None
        assert list(result['weight']) == [3]
        assert 'unique connections' in capsys.readouterr().out

    def test_batched_with_progress(self, tmp_path):
        f = make_fetcher(tmp_path, verbose=True)
        pre = pd.concat([synapse_frame(1, 2), synapse_frame(3, 4)],
                        ignore_index=True)
        f._cave_client = self._client_with({'pre': pre})
        result = f.fetch_connections([1, 3], direction='pre', batch_size=1,
                                     show_progress=True)
        assert result is not None
        assert len(result) == 2

    def test_batched_progress_without_tqdm(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, 'tqdm', None)
        f = make_fetcher(tmp_path, verbose=True)
        pre = synapse_frame(1, 2)
        f._cave_client = self._client_with({'pre': pre})
        result = f.fetch_connections([1, 3], direction='pre', batch_size=1,
                                     show_progress=True)
        assert result is not None

    def test_empty_batches_return_none(self, tmp_path):
        f = make_fetcher(tmp_path)
        f._cave_client = self._client_with({'pre': pd.DataFrame(),
                                            'post': pd.DataFrame()})
        assert f.fetch_connections([1], direction='both') is None

    def test_batch_query_failure_returns_none(self, tmp_path):
        f = make_fetcher(tmp_path)
        f._cave_client = self._client_with({}, raise_all=True)
        assert f.fetch_connections([1], direction='both') is None

    def test_normalization_failure_returns_none(self, tmp_path, monkeypatch,
                                                capsys):
        f = make_fetcher(tmp_path, verbose=True)
        f._cave_client = self._client_with({'pre': synapse_frame(1, 2)})

        def boom(frame, columns):
            raise ValueError("id normalization failed")

        monkeypatch.setattr(cdf, 'normalize_flywire_id_columns', boom)
        assert f.fetch_connections([1], direction='pre') is None
        assert 'Failed to fetch connections' in capsys.readouterr().out


class TestFetchNeuronInfo:
    def test_empty_input(self, tmp_path):
        f = make_fetcher(tmp_path)
        result = f.fetch_neuron_info([])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_full_pipeline(self, tmp_path, capsys):
        f = make_fetcher(tmp_path, verbose=True)
        tables = {
            'proofread_neurons': pd.DataFrame({'id': [11],
                                               'pt_root_id': [100]}),
            'hierarchical_neuron_annotations': pd.DataFrame(
                {'target_id': [11], 'classification': ['classA'],
                 'pt_root_id': [100]}),
            'neuron_information_v2': pd.DataFrame(
                {'pt_root_id': [100], 'tag': ['PPL101']}),
        }
        f._cave_client = FakeCaveClient(FakeMaterialize(table_results=tables))
        result = f.fetch_neuron_info([100])
        assert result is not None and not result.empty
        assert 'Fetched annotations' in capsys.readouterr().out

    def test_batched_with_progress(self, tmp_path):
        f = make_fetcher(tmp_path, verbose=True)
        tables = {'neuron_information_v2': pd.DataFrame(
            {'pt_root_id': [100], 'tag': ['T']})}
        f._cave_client = FakeCaveClient(FakeMaterialize(table_results=tables))
        result = f.fetch_neuron_info([100, 200], batch_size=1,
                                     show_progress=True)
        assert result is not None

    def test_batched_progress_without_tqdm(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, 'tqdm', None)
        f = make_fetcher(tmp_path, verbose=True)
        tables = {'neuron_information_v2': pd.DataFrame(
            {'pt_root_id': [100], 'tag': ['T']})}
        f._cave_client = FakeCaveClient(FakeMaterialize(table_results=tables))
        result = f.fetch_neuron_info([100, 200], batch_size=1,
                                     show_progress=True)
        assert result is not None

    def test_proofread_empty_skips_annotations(self, tmp_path, capsys):
        f = make_fetcher(tmp_path, verbose=True)
        tables = {'proofread_neurons': pd.DataFrame()}
        f._cave_client = FakeCaveClient(FakeMaterialize(table_results=tables))
        result = f.fetch_neuron_info([100])
        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert 'No annotations found' in capsys.readouterr().out

    def test_query_failure_returns_none(self, tmp_path, capsys):
        f = make_fetcher(tmp_path, verbose=True)
        f._cave_client = FakeCaveClient(
            FakeMaterialize(raise_on={'proofread_neurons'}))
        assert f.fetch_neuron_info([100]) is None
        assert 'Failed to fetch neuron info' in capsys.readouterr().out


class TestFetchNeuronsByTypes:
    def test_empty_types(self, tmp_path):
        f = make_fetcher(tmp_path)
        result = f.fetch_neurons_by_types([])
        assert list(result.columns) == ['bodyId', 'type', 'instance', 'post']
        assert result.empty

    def test_success_and_miss(self, tmp_path):
        f = make_fetcher(tmp_path)

        def query_table(table, filter_in_dict=None, **kwargs):
            regex = str(kwargs) + str(filter_in_dict)
            if 'PPL101' in regex:
                return pd.DataFrame({'pt_root_id': [100, 200],
                                     'tag': ['PPL101', 'PPL101_R']})
            return pd.DataFrame()

        mat = FakeMaterialize()
        mat.query_table = query_table
        f._cave_client = FakeCaveClient(mat)
        result = f.fetch_neurons_by_types(['PPL101', 'NOPE'],
                                          show_progress=False)
        assert len(result) == 2
        assert set(result['bodyId']) == {'100', '200'}

    def test_success_with_progress(self, tmp_path):
        f = make_fetcher(tmp_path, verbose=True)

        def query_table(table, filter_in_dict=None, **kwargs):
            return pd.DataFrame({'pt_root_id': [100], 'tag': ['A']})

        mat = FakeMaterialize()
        mat.query_table = query_table
        f._cave_client = FakeCaveClient(mat)
        result = f.fetch_neurons_by_types(['A', 'B'], show_progress=True)
        assert len(result) == 2

    def test_tags_without_tag_column(self, tmp_path):
        f = make_fetcher(tmp_path)
        mat = FakeMaterialize()
        mat.query_table = lambda table, filter_in_dict=None, **kw: (
            pd.DataFrame({'pt_root_id': [100]}))
        f._cave_client = FakeCaveClient(mat)
        result = f.fetch_neurons_by_types(['T'], show_progress=False)
        assert len(result) == 1
        assert result.loc[0, 'instance'] == ''

    def test_query_failure_returns_empty(self, tmp_path, capsys):
        f = make_fetcher(tmp_path, verbose=True)

        def boom(table, filter_in_dict=None, **kwargs):
            raise RuntimeError("tags unavailable")

        mat = FakeMaterialize()
        mat.query_table = boom
        f._cave_client = FakeCaveClient(mat)
        result = f.fetch_neurons_by_types(['T'], show_progress=False)
        assert result.empty
        assert 'Failed to fetch neurons by type' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# cache maintenance + module probe
# ---------------------------------------------------------------------------

class TestCacheMaintenance:
    def test_clear_cache_and_stats(self, tmp_path, capsys):
        f = make_fetcher(tmp_path)
        api_skel = Path(f.get_cache_path('skeletons'))
        api_mesh = Path(f.get_cache_path('meshes'))
        canonical_mesh = (Path(tmp_path) / 'cache' /
                          f._cache_dataset_name() / 'meshes')
        for d in (api_skel, api_mesh, canonical_mesh):
            d.mkdir(parents=True, exist_ok=True)
        (api_skel / '1.swc.gz').write_bytes(b'x')
        (api_skel / '2.pkl').write_bytes(b'x')
        (api_mesh / '3.pkl').write_bytes(b'x')
        (canonical_mesh / '4.pkl.zst').write_bytes(b'x')

        stats = f.get_cache_stats()
        assert stats['skeleton_count'] == 2
        assert stats['mesh_count'] == 2
        assert 'total_size_mb' in stats

        f.clear_cache('skeletons')
        assert api_skel.is_dir() and not list(api_skel.iterdir())

        f.clear_cache('meshes')
        assert api_mesh.is_dir() and not list(api_mesh.iterdir())
        assert canonical_mesh.is_dir() and not list(canonical_mesh.iterdir())

        # 'all' on already-empty dirs still works
        (api_skel / '5.swc.gz').write_bytes(b'x')
        f.clear_cache('all')
        assert not list(api_skel.iterdir())

    def test_clear_mesh_cache_tolerates_surviving_dir(
            self, tmp_path, monkeypatch):
        # Windows keeps an indexer-held directory alive through rmtree, so
        # the recreate must tolerate the path already existing.
        import shutil

        f = make_fetcher(tmp_path)
        api_mesh = Path(f.get_cache_path('meshes'))
        api_mesh.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(shutil, 'rmtree', lambda path, **kwargs: None)
        f.clear_cache('meshes')
        assert api_mesh.is_dir()

    def test_stats_with_missing_dirs(self, tmp_path):
        f = make_fetcher(tmp_path)
        stats = f.get_cache_stats()
        assert stats['skeleton_count'] == 0
        assert stats['mesh_count'] == 0


class TestModuleProbe:
    def test_fafb_access_probe(self, monkeypatch, capsys):
        mesh = navis.MeshNeuron(tetra_trimesh(), id=BODY_ID)
        skeleton = chain_neuron()

        class FakeFetcher:
            def __init__(self, dataset=None):
                pass

            def fetch_mesh(self, body_id):
                return mesh

            def fetch_skeleton(self, body_id):
                return skeleton

            def fetch_synapses(self, body_id, direction='both'):
                return synapse_frame(BODY_ID, 200)

            def get_cache_stats(self):
                return {'skeleton_count': 1}

        monkeypatch.setattr(cdf, 'CAVEDataFetcher', FakeFetcher)
        cdf.test_fafb_access()
        out = capsys.readouterr().out
        assert 'Test complete!' in out
        assert 'Mesh:' in out
        assert 'Skeleton:' in out
        assert 'Synapses:' in out

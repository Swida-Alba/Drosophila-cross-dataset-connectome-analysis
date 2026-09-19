"""
CAVE API Data Fetcher for FAFB datasets.

This module provides functionality to fetch FAFB meshes and synapses from
the CAVE/CloudVolume API.  Production mesh fetches remain ``MeshNeuron``
objects, while the explicit ``fetch_skeleton`` path wavefront-skeletonizes a
raw CAVE mesh into a ``TreeNeuron`` replacement stored in
``cache/{dataset}/skeletons/cave_skeletons/``.  The application uses that
tree path when a healed-zip skeleton is missing or fails the extrusion
check; when CAVE is unavailable, a locally pruned fix is persisted to
``cache/{dataset}/skeletons/extrusion_fixes/`` instead.

BANC is intentionally not served by this adapter. Its public-release SWCs
and prepared tables are handled by ``banc_public_data`` and the BANC
converter. The retained BANC config below is a compatibility surface for
older callers that only inspected the public mesh URL; all data-fetch methods
reject BANC before they can touch CloudVolume or CAVE.

Key features:
- Fetch meshes via CloudVolume (graphene protocol)
- Prepare and cache soma-aware meshes in
  ``cache/{dataset}/meshes/FLYWIRE_simp95_soma80_r20/``
- Keep NeuPrint's TreeNeuron/SWC cache separate
- Integration with TokenManager for authentication

Usage:
    from cave_data_fetcher import CAVEDataFetcher
    
    fetcher = CAVEDataFetcher(dataset='flywire_FAFB_v783')
    mesh = fetcher.fetch_mesh(720575940596125868)
    prepared = fetcher.fetch_fafb_mesh(720575940596125868)
    synapses = fetcher.fetch_synapses(720575940596125868)
"""

import os
import pickle
import gzip
import io
import logging
import re
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import warnings

import pandas as pd

try:
    from .utils.naming_utils import canonical_dataset_name
except ImportError:  # pragma: no cover - src laid bare on sys.path
    from utils.naming_utils import canonical_dataset_name

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover - installation is covered by requirements
    zstd = None

try:
    from .flywire_ids import (
        body_id_to_api_int,
        is_banc_dataset,
        normalize_flywire_id_columns,
    )
except ImportError:
    from flywire_ids import (
        body_id_to_api_int,
        is_banc_dataset,
        normalize_flywire_id_columns,
    )

try:
    from .flywire_mesh_cache import (
        FLYWIRE_MESH_CACHE_SIMPLIFICATION,
        FLYWIRE_MESH_CACHE_SOMA_RADIUS,
        FLYWIRE_MESH_CACHE_SOMA_SIMPLIFICATION,
        FlyWireMeshCache,
        prepare_flywire_mesh,
    )
except ImportError:
    from flywire_mesh_cache import (
        FLYWIRE_MESH_CACHE_SIMPLIFICATION,
        FLYWIRE_MESH_CACHE_SOMA_RADIUS,
        FLYWIRE_MESH_CACHE_SOMA_SIMPLIFICATION,
        FlyWireMeshCache,
        prepare_flywire_mesh,
    )

# Shared compressed-SWC provenance contract (headers, parsers, text writer).
try:
    from . import skeleton_provenance as _provenance
except ImportError:  # pragma: no cover - src laid bare on sys.path
    import skeleton_provenance as _provenance

# Suppress warnings during import
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import navis

logger = logging.getLogger(__name__)


@dataclass
class CAVEDataFetcher:
    """Fetch neuron data from CAVE/CloudVolume API for FAFB datasets."""
    
    dataset: str = 'flywire_FAFB_v783'
    """Dataset name (normally ``flywire_FAFB_v783``)."""
    
    cave_token: str = None
    """CAVE authentication token. If None, reads from config.json"""
    
    materialization_version: int = None
    """Materialization version for CAVE queries. None = latest"""
    
    cache_enabled: bool = True
    """Whether to cache fetched data to disk"""
    
    project_root: str = None
    """Project root directory. If None, auto-detected."""
    
    verbose: bool = True
    """Whether to print progress messages"""
    
    # CloudVolume config
    _cv: object = field(default=None, repr=False)
    _cave_client: object = field(default=None, repr=False)
    
    # Dataset-specific settings
    FLYWIRE_FAFB_CONFIG = {
        'datastack': 'flywire_fafb_public',
        'cloudvolume_url': 'graphene://https://prodv1.flywire-daf.com/segmentation/1.0/fly_v31',
        'synapse_table': 'synapses_nt_v1',
        'synapse_view': 'valid_synapses_nt_np_v6',
    }
    
    BANC_PUBLIC_CONFIG = {
        'datastack': 'brain_and_nerve_cord',
        # Public precomputed neuron-mesh layer (nanometre coordinates, no
        # auth).  CAVE materialization queries stay token-gated and unused:
        # BANC tables are local and skeletons come from the public bucket
        # (see banc_public_data).
        'cloudvolume_url': (
            'precomputed://https://storage.googleapis.com/'
            'lee-lab_brain-and-nerve-cord-fly-connectome/neuron_meshes'
        ),
        'synapse_table': 'synapses',
    }
    # Compatibility alias for integrations that imported the old constant.
    # BANC's active data path is in banc_public_data, never this CAVE adapter.
    FLYWIRE_BANC_CONFIG = BANC_PUBLIC_CONFIG
    
    def __post_init__(self):
        """Initialize the fetcher with tokens and paths."""
        if self.project_root is None:
            self.project_root = str(Path(__file__).parent.parent)
        
        # BANC never participates in this CAVE adapter. Do not even inspect
        # token files for it; the public BANC release has no CAVE token path.
        if is_banc_dataset(self.dataset):
            self.cave_token = None
        elif self.cave_token is None:
            self.cave_token = self._load_token('CAVE_TOKEN')
        
        # Cache creation is lazy.  In particular, a caller that performs an
        # online-only ``fetch_skeleton(..., use_cache=False)`` must not create
        # an API-cache tree just by constructing the fetcher.
    
    def _load_token(self, token_name: str) -> Optional[str]:
        """Load token from config.json, then the environment."""
        token = self._load_token_from_config(token_name)
        if token:
            return token
        # Direct environment configuration is supported by the shared token
        # manager as well; keep CAVE fetches consistent with that behavior.
        token = os.environ.get(token_name, '').strip()
        if token and not token.startswith('YOUR_'):
            return token
        return None

    def _load_token_from_config(self, token_name: str) -> Optional[str]:
        """Read a token from config.json (primary) or config_local.json."""
        import json
        config_key = token_name.lower().replace('_token', '')
        # config.json wins per key; the gitignored config_local.json only
        # fills entries that are empty in config.json.
        for filename in ('config.json', 'config_local.json'):
            config_path = os.path.join(self.project_root, filename)
            try:
                with open(config_path, 'r', encoding='utf-8-sig') as f:
                    data = json.load(f)
            except (OSError, ValueError):
                continue
            section = data.get('tokens') if isinstance(data, dict) else None
            if not isinstance(section, dict):
                continue
            value = section.get(config_key)
            if isinstance(value, str):
                value = value.strip()
                if value and not value.startswith('YOUR_'):
                    return value
        return None
    
    def _get_config(self) -> dict:
        """Get configuration for current dataset."""
        # Check BANC first: a future BANC materialization could also use a
        # version number that happens to match the historical FAFB v783.
        if 'BANC' in self.dataset.upper():
            return self.BANC_PUBLIC_CONFIG
        if 'FAFB' in self.dataset.upper() or 'v783' in self.dataset:
            return self.FLYWIRE_FAFB_CONFIG
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")
    
    def _ensure_cache_dir(self):
        """Ensure API cache directory exists."""
        if not self.cache_enabled:
            return
        cache_dir = self.get_cache_path()
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.join(cache_dir, 'skeletons'), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, 'meshes'), exist_ok=True)
    
    def get_cache_path(self, subdir: str = '') -> str:
        """Get path to API cache directory.
        
        Returns:
            Path to cache/{dataset}/API_cache/{subdir}
        """
        cache_dataset = self._cache_dataset_name()
        
        cache_path = os.path.join(self.project_root, 'cache', cache_dataset, 'API_cache')
        if subdir:
            cache_path = os.path.join(cache_path, subdir)
        return cache_path

    def _cache_dataset_name(self) -> str:
        """Return the release-specific cache namespace for this dataset.

        FAFB keeps its historical namespace for compatibility.  BANC must
        retain the requested release (for example, ``v888``) so a skeleton
        fetched for one release can never be reused by another release merely
        because both are labelled BANC.
        """
        dataset_name = str(self.dataset or '').strip()
        upper = dataset_name.upper()
        if 'BANC' not in upper and ('FAFB' in upper or 'v783' in dataset_name):
            return 'flywire_FAFB_v783'
        return canonical_dataset_name(dataset_name).replace(':', '_').replace('.', '_')
    
    @property
    def cloudvolume(self):
        """Lazy-load CloudVolume client."""
        if self._cv is None:
            try:
                from cloudvolume import CloudVolume
                config = self._get_config()
                if config['cloudvolume_url'] is None:
                    raise ValueError(f"CloudVolume not available for {self.dataset}")
                
                self._cv = CloudVolume(
                    config['cloudvolume_url'],
                    use_https=True,
                    # Public precomputed layers (BANC) need no token; only
                    # attach secrets when a token is actually configured.
                    **({'secrets': {'token': self.cave_token}}
                       if self.cave_token else {}),
                )
                if self.verbose:
                    print(f"✓ Connected to CloudVolume: {config['cloudvolume_url'][:50]}...")
            except ImportError:
                raise ImportError("cloudvolume package required. Install with: pip install cloud-volume")
        return self._cv
    
    @property
    def cave_client(self):
        """Lazy-load CAVE client."""
        if is_banc_dataset(self.dataset):
            raise RuntimeError(
                f"BANC dataset '{self.dataset}' does not use a CAVE client; "
                "use banc_public_data and the local BANC tables instead."
            )
        if self._cave_client is None:
            try:
                from caveclient import CAVEclient
                config = self._get_config()
                self._cave_client = CAVEclient(
                    datastack_name=config['datastack'],
                    auth_token=self.cave_token,
                    write_server_cache=self.cache_enabled,
                )
                if self.verbose:
                    print(f"✓ Connected to CAVE: {config['datastack']}")
            except ImportError:
                raise ImportError("caveclient package required. Install with: pip install caveclient")
        return self._cave_client
    
    def _get_skeleton_cache_path(self, body_id: int) -> str:
        """Path in the dedicated ``cave_skeletons`` replacement store.

        CAVE-skeletonized trees replace healed-zip skeletons that failed
        the extrusion check, so they are stored separately from the
        ``raw_skeletons`` legacy store and never overwrite it.
        """
        cache_dataset = self._cache_dataset_name()
        return os.path.join(
            self.project_root, 'cache', cache_dataset, 'skeletons',
            'cave_skeletons', f'{body_id}.swc.zst'
        )

    def _get_legacy_raw_store_skeleton_path(self, body_id: int) -> str:
        """Pre-unification ``raw_skeletons`` location of CAVE-skeletonized trees.

        ``fetch_skeleton`` used to cache into the shared raw store; those
        entries stay readable (level-0 only — the same directory also holds
        historical simp90 visualization writes that are not raw inputs).
        """
        cache_dataset = self._cache_dataset_name()
        return os.path.join(
            self.project_root, 'cache', cache_dataset, 'skeletons',
            'raw_skeletons', f'{body_id}.swc.zst'
        )

    def _get_legacy_skeleton_cache_path(self, body_id: int) -> str:
        """Return the former API-cache pickle path for migration reads."""
        return os.path.join(self.get_cache_path('skeletons'), f'{body_id}.pkl')
    
    def _get_mesh_cache_path(self, body_id: int) -> str:
        """Get the canonical prepared FlyWire mesh cache path."""
        return str(FlyWireMeshCache(
            self._cache_dataset_name(), project_root=self.project_root,
        ).path(body_id))

    def _get_legacy_mesh_cache_path(self, body_id: int) -> str:
        """Return the former API-cache mesh path for compatibility reads."""
        return os.path.join(self.get_cache_path('meshes'), f'{body_id}.pkl')
    
    def _load_from_cache(self, cache_path: str):
        """Load object from cache if exists."""
        if self.cache_enabled and os.path.exists(cache_path):
            try:
                if str(cache_path).endswith(('.swc.gz', '.swc.zst')):
                    content = _load_swc_text(cache_path)
                    obj = navis.read_swc(io.StringIO(content))
                    if not isinstance(obj, navis.TreeNeuron):
                        return None
                    try:
                        obj.id = int(_swc_stem(Path(cache_path).name))
                        obj.units = 'nm'
                        obj._drocat_simplification = \
                            _read_stored_simplification(content)
                        obj._drocat_source = _read_stored_source(content)
                    except Exception:
                        pass
                    return obj
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_path}: {e}")
        return None
    
    def _save_to_cache(self, obj, cache_path: str):
        """Save object to cache."""
        if self.cache_enabled:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                if str(cache_path).endswith(('.swc.gz', '.swc.zst')):
                    if not isinstance(obj, navis.TreeNeuron):
                        raise TypeError(
                            "compressed SWC cache accepts TreeNeuron only")
                    cache_path_obj = Path(cache_path)
                    with tempfile.NamedTemporaryFile(
                            suffix='.swc', dir=str(cache_path_obj.parent),
                            delete=False) as handle:
                        temp_swc = Path(handle.name)
                    try:
                        navis.write_swc(obj, temp_swc, write_meta=True)
                        payload = temp_swc.read_bytes()
                        if str(cache_path).endswith('.swc.zst'):
                            # Canonical form: zstd-19 with a recorded level
                            # (FlyWire raw skeletons are stored raw, 0).
                            _write_compressed_swc_zst(
                                cache_path_obj, payload, simplification=0)
                        else:
                            temp_gz = cache_path_obj.with_name(
                                f'.{cache_path_obj.name}.{os.getpid()}.tmp')
                            try:
                                temp_gz.write_bytes(gzip.compress(
                                    payload, compresslevel=6, mtime=0))
                                os.replace(temp_gz, cache_path_obj)
                            finally:
                                temp_gz.unlink(missing_ok=True)
                    finally:
                        temp_swc.unlink(missing_ok=True)
                    return
                with open(cache_path, 'wb') as f:
                    pickle.dump(obj, f)
            except Exception as e:
                logger.warning(f"Failed to save cache {cache_path}: {e}")
    
    def fetch_mesh(self, body_id: int, use_cache: bool = False) -> Optional['navis.MeshNeuron']:
        """Fetch a FAFB mesh for a neuron from CloudVolume.
        
        Note: Meshes are NOT cached to save disk space. Raw skeletons are
        cached separately as compressed SWC (see ``fetch_skeleton``). Each
        call fetches the mesh fresh from the API.
        
        Parameters
        ----------
        body_id : int
            The root ID of the neuron
        use_cache : bool
            Deprecated - meshes are no longer cached. Parameter kept for compatibility.
            
        Returns
        -------
        navis.MeshNeuron or None
            The mesh neuron, or None if fetch failed
        """
        if is_banc_dataset(self.dataset):
            raise RuntimeError(
                f"BANC dataset '{self.dataset}' is SWC-first and does not use "
                "CAVE/CloudVolume mesh fetching."
            )
        body_id = body_id_to_api_int(body_id)
        
        try:
            cv = self.cloudvolume
            # FAFB/CAVE Graphene meshes use variable-layered Draco. CloudVolume
            # cannot boundary-deduplicate those layers and otherwise prints a
            # warning for every mesh while returning the same undeduplicated
            # data. Opt out explicitly so API repairs do not flood the render
            # log with a misleading warning.
            mesh = cv.mesh.get(
                body_id,
                deduplicate_chunk_boundaries=False,
            )[body_id]
            
            # Convert to navis MeshNeuron
            mesh_neuron = navis.MeshNeuron(
                mesh,
                id=body_id,
                name=str(body_id),
                units='nm'
            )
            
            if self.verbose:
                print(f"  ✓ Fetched mesh: {body_id} ({len(mesh.vertices)} vertices)")
            
            return mesh_neuron
            
        except Exception as e:
            logger.error(f"Failed to fetch mesh for {body_id}: {e}")
            if self.verbose:
                print(f"  ✗ Failed to fetch mesh: {body_id}: {e}")
            return None

    def fetch_fafb_mesh(
            self,
            body_id: int,
            use_cache: bool = True,
            simplify_mesh: float = FLYWIRE_MESH_CACHE_SIMPLIFICATION,
            soma_simplification: float = FLYWIRE_MESH_CACHE_SOMA_SIMPLIFICATION,
            soma_radius: float = FLYWIRE_MESH_CACHE_SOMA_RADIUS,
            soma_pos=None,
            force_refresh: bool = False,
            ) -> Optional['navis.MeshNeuron']:
        """Fetch, prepare, and cache one FAFB ``MeshNeuron``.

        The prepared cache is intentionally mesh-native.  The default
        settings match the visualization cache: remove 95% of branch-region
        faces, remove 80% in the 20 µm soma region, and persist the resulting
        ``MeshNeuron`` as a level-1 Zstandard-compressed pickle
        (``{bodyId}.pkl.zst``).  A missing soma coordinate falls back to the
        same uniform 95% simplification used by the visualizer.

        ``use_cache=False`` is an online-only operation: it neither reads nor
        writes the local mesh cache.  ``force_refresh=True`` is the repair
        operation: it bypasses an existing prepared-mesh cache entry, fetches
        a new raw mesh from CAVE, and writes the newly prepared mesh when
        caching is enabled.  This is used for ZIP meshes with extrusion
        artifacts so a stale prepared cache entry cannot silently win.
        """
        if is_banc_dataset(self.dataset):
            raise RuntimeError(
                f"BANC dataset '{self.dataset}' uses public SWCs, not the "
                "FAFB prepared-mesh/CAVE path."
            )
        api_body_id = body_id_to_api_int(body_id)
        mesh_cache = FlyWireMeshCache(
            self._cache_dataset_name(),
            project_root=self.project_root,
            simplification=simplify_mesh,
            soma_simplification=soma_simplification,
            soma_radius=soma_radius,
        )
        cache_allowed = bool(use_cache and self.cache_enabled)

        # A repair must not reuse the prepared entry that may have been made
        # from the same extrusion-affected source.  ``force_refresh`` only
        # changes cache reads; ``use_cache`` still controls whether the fresh
        # prepared result is persisted below.
        if cache_allowed and not force_refresh:
            cached = mesh_cache.load(body_id)
            if cached is not None:
                if self.verbose:
                    print(f"  ✓ Loaded prepared mesh from cache: {body_id}")
                return cached

        # ``fetch_mesh`` is deliberately raw and never writes a mesh cache.
        mesh = self.fetch_mesh(api_body_id, use_cache=False)
        if mesh is None:
            return None
        if soma_pos is None:
            soma_pos = getattr(mesh, 'soma_pos', None)

        prepared_ok = False
        try:
            prepared = prepare_flywire_mesh(
                mesh,
                body_id,
                soma_pos=soma_pos,
                simplification=simplify_mesh,
                soma_simplification=soma_simplification,
                soma_radius=soma_radius,
            )
            prepared_ok = True
        except Exception as exc:
            logger.warning(
                "Failed to prepare FlyWire mesh %s; using the fetched mesh: %s",
                body_id, exc)
            prepared = mesh

        # Never promote an unprepared fallback into the prepared cache. The
        # online caller can still render the raw MeshNeuron, but the next
        # cache-enabled call must retry the requested 95%/80% preparation.
        if cache_allowed and prepared_ok and isinstance(prepared, navis.MeshNeuron):
            mesh_cache.save({body_id: prepared})
        if self.verbose:
            try:
                print(
                    f"  ✓ Prepared FAFB mesh: {body_id} "
                    f"({len(prepared.trimesh.faces)} faces)"
                )
            except Exception:
                print(f"  ✓ Prepared FAFB mesh: {body_id}")
        return prepared

    def fetch_fafb_meshes(
            self,
            body_ids: List[int],
            use_cache: bool = True,
            simplify_mesh: float = FLYWIRE_MESH_CACHE_SIMPLIFICATION,
            soma_simplification: float = FLYWIRE_MESH_CACHE_SOMA_SIMPLIFICATION,
            soma_radius: float = FLYWIRE_MESH_CACHE_SOMA_RADIUS,
            soma_positions: Optional[Dict[object, object]] = None,
            force_refresh: bool = False,
            ) -> 'navis.NeuronList':
        """Fetch prepared FAFB meshes without skeletonization.

        ``force_refresh`` bypasses prepared-mesh cache reads for repair
        requests while preserving normal cache writes when ``use_cache`` is
        true.  With ``use_cache=False``, both reads and writes remain
        disabled regardless of ``force_refresh``.
        """
        if is_banc_dataset(self.dataset):
            raise RuntimeError(
                f"BANC dataset '{self.dataset}' uses public SWCs, not FAFB "
                "prepared-mesh fetching."
            )
        from tqdm import tqdm

        positions = soma_positions or {}
        meshes = []
        failed = []
        iterator = body_ids
        if self.verbose:
            iterator = tqdm(body_ids, desc="Fetching FAFB meshes")
        for body_id in iterator:
            key = str(body_id)
            soma_pos = positions.get(body_id, positions.get(key))
            mesh = self.fetch_fafb_mesh(
                body_id,
                use_cache=use_cache,
                force_refresh=force_refresh,
                simplify_mesh=simplify_mesh,
                soma_simplification=soma_simplification,
                soma_radius=soma_radius,
                soma_pos=soma_pos,
            )
            if mesh is not None:
                meshes.append(mesh)
            else:
                failed.append(body_id)
        if failed and self.verbose:
            print(f"  ⚠️  Failed to fetch {len(failed)}/{len(body_ids)} meshes")
        return navis.NeuronList(meshes)
    
    def fetch_skeleton(self, body_id: int, use_cache: bool = True,
                       simplify_mesh: float = 0.0,
                       denoise_twigs: Optional[float] = None) -> Optional['navis.TreeNeuron']:
        """Fetch a FAFB skeleton for a neuron by skeletonizing its mesh.

        Since FAFB doesn't have L2 cache for pcg_skel, we fetch the mesh and
        skeletonize it using navis. The raw mesh is skeletonized without
        pre-decimation: wavefront on the raw surface reproduces the healed
        bundle's node density (measured ×1.01–1.14 across l-LNv, 1–8 s per
        neuron), while any pre-decimation is slower end-to-end and drops
        thin processes below wavefront's resolution.

        Trees are cached in the dedicated ``cave_skeletons`` store with a
        ``# DROCAT source: cave_mesh_wavefront`` header — they replace
        healed-zip skeletons that failed the extrusion check and must
        never overwrite the ``raw_skeletons`` store.

        Parameters
        ----------
        body_id : int
            The root ID of the neuron
        use_cache : bool
            Whether to use cached skeleton if available
        simplify_mesh : float
            Legacy mesh preprocessing factor before skeletonization (0.0-1.0).
            Raw DROCAT fetches use the default ``0.0`` (measured optimal);
            visualization simplification is applied after the raw skeleton
            is fetched.
        denoise_twigs : float or None
            Optional length threshold (nm) for transient terminal-twig
            pruning after skeletonization. It is never written to the raw
            cache. ``None`` (the default) leaves the fetched skeleton raw.

        Returns
        -------
        navis.TreeNeuron or None
            The skeleton, or None if fetch failed
        """
        if is_banc_dataset(self.dataset):
            raise RuntimeError(
                f"BANC dataset '{self.dataset}' uses public SWCs, not CAVE "
                "mesh skeletonization."
            )
        body_id = body_id_to_api_int(body_id)
        cache_path = self._get_skeleton_cache_path(body_id)

        # ``use_cache`` is a complete read/write policy for this operation:
        # false means online-only and must not inspect or populate the local
        # API skeleton cache, even when the fetcher itself has caching enabled
        # for other calls.
        cache_allowed = bool(use_cache and self.cache_enabled)

        # Try cache first: the dedicated cave_skeletons store, then the
        # pre-unification raw-store location (level-0 entries only).
        if cache_allowed:
            cached = self._load_from_cache(cache_path)
            if cached is None:
                cached = self._load_pre_unification_raw_store_skeleton(
                    body_id)
                if cached is not None and self.verbose:
                    print(f"  ✓ Migrated CAVE skeleton from raw store: "
                          f"{body_id}")
            if cached is None and str(cache_path).endswith(
                    ('.swc.gz', '.swc.zst')):
                # Read the former API-cache pickle once for a non-destructive
                # migration, then converge on canonical raw SWC. The legacy
                # file is intentionally retained for recoverability.
                legacy_path = self._get_legacy_skeleton_cache_path(body_id)
                cached = self._load_from_cache(legacy_path)
                if cached is not None:
                    self._save_to_cache(cached, cache_path)
            if cached is not None:
                if denoise_twigs:
                    cached = self._denoise_skeleton(cached, denoise_twigs)
                if self.verbose:
                    print(f"  ✓ Loaded skeleton from CAVE cache: {body_id}")
                return cached
        
        try:
            # Fetch mesh first
            mesh = self.fetch_mesh(body_id, use_cache=use_cache)
            if mesh is None:
                return None
            
            # Simplify mesh for faster skeletonization
            if simplify_mesh > 0:
                import trimesh
                tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
                
                # simplify_mesh is the fraction to REMOVE (0.9 = remove 90% = keep 10%)
                # trimesh.simplify_quadric_decimation takes percent= fraction to KEEP
                keep_fraction = 1.0 - simplify_mesh
                target_faces = max(int(len(tm.faces) * keep_fraction), 1000)
                
                try:
                    # Use face_count parameter instead of percent for more predictable results
                    tm_simplified = tm.simplify_quadric_decimation(face_count=target_faces)
                    mesh = navis.MeshNeuron(tm_simplified, id=body_id, name=str(body_id), units='nm')
                    if self.verbose:
                        print(f"  ℹ️  Simplified mesh: {len(tm_simplified.faces)} faces (from {len(tm.faces)})")
                except Exception as e:
                    logger.warning(f"Mesh simplification failed: {e}")
            
            # Skeletonize
            if self.verbose:
                print(f"  ⏳ Skeletonizing mesh for {body_id}...")
            
            skeleton = navis.skeletonize(mesh, method='wavefront')
            skeleton.id = body_id
            skeleton.name = str(body_id)
            skeleton.units = 'nm'
            
            # Persist the raw skeleton before any optional transient
            # denoising. This keeps one representation in the cache while
            # allowing callers to request a temporary cleanup.
            if cache_allowed:
                self._save_cave_skeleton(skeleton, cache_path)

            if denoise_twigs:
                skeleton = self._denoise_skeleton(skeleton, denoise_twigs)
            
            if self.verbose:
                print(f"  ✓ Generated skeleton: {body_id} ({len(skeleton.nodes)} nodes)")
            
            return skeleton
            
        except Exception as e:
            logger.error(f"Failed to fetch skeleton for {body_id}: {e}")
            if self.verbose:
                print(f"  ✗ Failed to fetch skeleton: {body_id}: {e}")
            return None
    
    def _save_cave_skeleton(self, skeleton, cache_path: str,
                            source: str = "cave_mesh_wavefront") -> None:
        """Persist a replacement tree with a provenance header.

        The ``# DROCAT source:`` line (``cave_mesh_wavefront`` for CAVE
        mesh-skeletonized replacements, ``local_extrusion_fix`` for locally
        pruned trees) marks the tree as a replacement for an
        extrusion-flagged healed skeleton, so readers can tell it apart
        from bundle-derived entries even outside the dedicated stores.
        """
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            if not isinstance(skeleton, navis.TreeNeuron):
                raise TypeError(
                    "cave_skeletons store accepts TreeNeuron only")
            cache_path_obj = Path(cache_path)
            with tempfile.NamedTemporaryFile(
                    suffix='.swc', dir=str(cache_path_obj.parent),
                    delete=False) as handle:
                temp_swc = Path(handle.name)
            try:
                navis.write_swc(skeleton, temp_swc, write_meta=True)
                payload = temp_swc.read_bytes()
                source_line = (f"# {_SOURCE_HEADER} {source}\n"
                               .encode("ascii"))
                _write_compressed_swc_zst(
                    cache_path_obj, source_line + payload, simplification=0)
            finally:
                temp_swc.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_path}: {e}")

    def load_cached_skeleton(self, body_id):
        """Read a cached CAVE-skeletonized tree without any network access.

        Returns the ``TreeNeuron`` from the ``cave_skeletons`` store, or
        ``None`` on a miss. Unlike :meth:`fetch_skeleton` this never fetches
        online, so callers can probe the store before deciding to spend a
        network round-trip.
        """
        try:
            body_id = body_id_to_api_int(body_id)
        except Exception:
            return None
        cached = self._load_from_cache(self._get_skeleton_cache_path(body_id))
        return cached if isinstance(cached, navis.TreeNeuron) else None

    def _get_fix_cache_path(self, body_id: int) -> str:
        """Path in the dedicated ``extrusion_fixes`` local-repair store.

        Locally pruned trees replace extrusion-flagged healed skeletons
        when the CAVE replacement is unavailable; like ``cave_skeletons``
        they live outside ``raw_skeletons`` and never overwrite it.
        """
        cache_dataset = self._cache_dataset_name()
        return os.path.join(
            self.project_root, 'cache', cache_dataset, 'skeletons',
            'extrusion_fixes', f'{body_id}.swc.zst'
        )

    def save_extrusion_fix_skeleton(self, body_id, skeleton) -> bool:
        """Persist a locally pruned extrusion fix with an explicit header.

        Writes ``# DROCAT source: local_extrusion_fix`` so next runs serve
        the repaired tree directly instead of re-running the prune or the
        CAVE fetch. Returns True when the tree was written.
        """
        try:
            body_id = body_id_to_api_int(body_id)
            self._save_cave_skeleton(
                skeleton, self._get_fix_cache_path(body_id),
                source=LOCAL_EXTRUSION_FIX_SOURCE)
            return True
        except Exception as e:
            logger.warning(f"Failed to save extrusion fix for {body_id}: {e}")
            return False

    def load_cached_fix_skeleton(self, body_id):
        """Read a locally pruned extrusion fix without any network access.

        Returns the ``TreeNeuron`` from the ``extrusion_fixes`` store, or
        ``None`` on a miss.
        """
        try:
            body_id = body_id_to_api_int(body_id)
        except Exception:
            return None
        cached = self._load_from_cache(self._get_fix_cache_path(body_id))
        return cached if isinstance(cached, navis.TreeNeuron) else None

    def _load_pre_unification_raw_store_skeleton(self, body_id: int):
        """Read a CAVE tree from the pre-unification raw-store location.

        Only entries whose recorded simplification level is 0 are accepted:
        the shared ``raw_skeletons`` directory also holds historical simp90
        visualization writes that are not valid raw inputs. Level-0 hits
        stay readable so a pre-unification CAVE fetch is not re-downloaded.
        """
        legacy_path = self._get_legacy_raw_store_skeleton_path(body_id)
        if not (self.cache_enabled and os.path.exists(legacy_path)):
            return None
        if str(legacy_path).endswith(('.swc.gz', '.swc.zst')):
            try:
                content = _load_swc_text(legacy_path)
            except Exception:
                return None
            if _read_stored_simplification(content) != 0:
                return None
        return self._load_from_cache(legacy_path)

    @staticmethod
    def _denoise_skeleton(neuron, threshold_nm: float) -> 'navis.TreeNeuron':
        """Prune terminal twigs shorter than ``threshold_nm`` (recursive).

        Mesh-derived skeletons carry short twig artifacts (small tracing
        noise); removing them makes morphology features more stable. Falls
        back to the input when pruning fails.
        """
        try:
            import navis
            return navis.prune_twigs(neuron, size=float(threshold_nm), recursive=True)
        except Exception:
            return neuron
    
    def fetch_skeletons(self, body_ids: List[int], use_cache: bool = True,
                        simplify_mesh: float = 0.0,
                        denoise_twigs: Optional[float] = None) -> 'navis.NeuronList':
        """Fetch skeletons for multiple neurons.
        
        Parameters
        ----------
        body_ids : list of int
            List of root IDs
        use_cache : bool
            Whether to use cached skeletons if available
        simplify_mesh : float
            Legacy mesh preprocessing factor (0.0-1.0). Raw fetches default
            to 0.0; render-time simplification is separate.
        denoise_twigs : float or None
            Optional transient twig-pruning threshold passed to
            ``fetch_skeleton``. The raw cache is not pruned.
            
        Returns
        -------
        navis.NeuronList
            List of skeletons (may be fewer than requested if some fail)
        """
        if is_banc_dataset(self.dataset):
            raise RuntimeError(
                f"BANC dataset '{self.dataset}' uses public SWCs, not CAVE "
                "skeleton fetching."
            )
        from tqdm import tqdm
        
        neurons = []
        failed = []
        
        iterator = body_ids
        if self.verbose:
            iterator = tqdm(body_ids, desc="Fetching skeletons")
        
        for bid in iterator:
            skeleton = self.fetch_skeleton(bid, use_cache=use_cache, simplify_mesh=simplify_mesh,
                                           denoise_twigs=denoise_twigs)
            if skeleton is not None:
                neurons.append(skeleton)
            else:
                failed.append(bid)
        
        if failed and self.verbose:
            print(f"  ⚠️  Failed to fetch {len(failed)}/{len(body_ids)} skeletons")
        
        return navis.NeuronList(neurons)
    
    def fetch_synapses(self, body_id: int, direction: str = 'both') -> Optional[pd.DataFrame]:
        """Fetch FAFB synapses for a neuron from CAVE.
        
        Parameters
        ----------
        body_id : int
            The root ID of the neuron
        direction : str
            'pre' for outgoing synapses, 'post' for incoming, 'both' for all
            
        Returns
        -------
        pd.DataFrame or None
            Synapse data with columns: pre_pt_root_id, post_pt_root_id, 
            pre_pt_position, post_pt_position, etc.
        """
        if is_banc_dataset(self.dataset):
            raise RuntimeError(
                f"BANC dataset '{self.dataset}' has no CAVE synapse path; "
                "use its public-release tables."
            )
        body_id = body_id_to_api_int(body_id)
        config = self._get_config()
        
        try:
            client = self.cave_client
            
            dfs = []
            
            if direction in ['pre', 'both']:
                # Outgoing synapses (this neuron is presynaptic)
                try:
                    pre_df = client.materialize.query_view(
                        config['synapse_view'],
                        filter_in_dict={'pre_pt_root_id': [body_id]},
                        materialization_version=self.materialization_version
                    )
                    dfs.append(pre_df)
                    if self.verbose:
                        print(f"  ✓ Fetched {len(pre_df)} outgoing synapses")
                except Exception as e:
                    logger.warning(f"Failed to fetch outgoing synapses: {e}")
            
            if direction in ['post', 'both']:
                # Incoming synapses (this neuron is postsynaptic)
                try:
                    post_df = client.materialize.query_view(
                        config['synapse_view'],
                        filter_in_dict={'post_pt_root_id': [body_id]},
                        materialization_version=self.materialization_version
                    )
                    dfs.append(post_df)
                    if self.verbose:
                        print(f"  ✓ Fetched {len(post_df)} incoming synapses")
                except Exception as e:
                    logger.warning(f"Failed to fetch incoming synapses: {e}")
            
            if dfs:
                result = pd.concat(dfs, ignore_index=True)
                # drop_duplicates fails on array columns, use subset of hashable columns
                id_cols = [c for c in ['id', 'pre_pt_root_id', 'post_pt_root_id'] if c in result.columns]
                if id_cols:
                    result = result.drop_duplicates(subset=id_cols)
                normalize_flywire_id_columns(
                    result, ['pre_pt_root_id', 'post_pt_root_id']
                )
                return result
            return None
            
        except Exception as e:
            logger.error(f"Failed to fetch synapses for {body_id}: {e}")
            if self.verbose:
                print(f"  ✗ Failed to fetch synapses: {body_id}: {e}")
            return None
    
    def fetch_connections(self, body_ids: List[int], direction: str = 'both', 
                          batch_size: int = 200, show_progress: bool = True) -> Optional[pd.DataFrame]:
        """Fetch FAFB connections (aggregated synapse counts) from CAVE.
        
        This fetches synapses and aggregates them into connection weights.
        For large numbers of neurons, batches requests with a progress bar.
        
        Parameters
        ----------
        body_ids : list of int
            List of root IDs
        direction : str
            'pre' for outgoing connections, 'post' for incoming, 'both' for all
        batch_size : int
            Number of neurons per API request (default 200)
        show_progress : bool
            Whether to show progress bar for batched requests
            
        Returns
        -------
        pd.DataFrame or None
            Connections with columns: pre_pt_root_id, post_pt_root_id, weight
        """
        if is_banc_dataset(self.dataset):
            raise RuntimeError(
                f"BANC dataset '{self.dataset}' has no CAVE connection path; "
                "use its locally prepared public-release tables."
            )
        body_ids = [body_id_to_api_int(bid) for bid in body_ids]
        config = self._get_config()
        
        try:
            client = self.cave_client
            
            # Batch large requests
            n_neurons = len(body_ids)
            n_batches = (n_neurons + batch_size - 1) // batch_size
            
            # Only show progress bar for multiple batches
            use_progress = show_progress and n_batches > 1 and self.verbose
            
            if use_progress:
                try:
                    from tqdm import tqdm
                    batches = tqdm(
                        range(n_batches), 
                        desc="  🌐 CAVE API", 
                        unit="batch",
                        ncols=80,
                        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
                    )
                except ImportError:
                    use_progress = False
                    batches = range(n_batches)
            else:
                batches = range(n_batches)
            
            dfs = []
            
            for i in batches:
                batch_start = i * batch_size
                batch_end = min((i + 1) * batch_size, n_neurons)
                batch_ids = body_ids[batch_start:batch_end]
                
                if direction in ['pre', 'both']:
                    # Outgoing connections
                    try:
                        pre_df = client.materialize.query_view(
                            config['synapse_view'],
                            filter_in_dict={'pre_pt_root_id': batch_ids},
                            materialization_version=self.materialization_version
                        )
                        if not pre_df.empty:
                            dfs.append(pre_df)
                    except Exception as e:
                        logger.warning(f"Failed to fetch outgoing connections for batch {i+1}: {e}")
                
                if direction in ['post', 'both']:
                    # Incoming connections
                    try:
                        post_df = client.materialize.query_view(
                            config['synapse_view'],
                            filter_in_dict={'post_pt_root_id': batch_ids},
                            materialization_version=self.materialization_version
                        )
                        if not post_df.empty:
                            dfs.append(post_df)
                    except Exception as e:
                        logger.warning(f"Failed to fetch incoming connections for batch {i+1}: {e}")
            
            if dfs:
                # Combine and aggregate
                all_synapses = pd.concat(dfs, ignore_index=True)
                
                # Aggregate to connection weights
                connections = all_synapses.groupby(
                    ['pre_pt_root_id', 'post_pt_root_id']
                ).size().reset_index(name='weight')
                normalize_flywire_id_columns(
                    connections, ['pre_pt_root_id', 'post_pt_root_id']
                )
                
                if self.verbose:
                    print(f"  ✓ Fetched {len(connections):,} unique connections for {n_neurons} neurons")
                
                return connections
            return None
            
        except Exception as e:
            logger.error(f"Failed to fetch connections: {e}")
            if self.verbose:
                print(f"  ✗ Failed to fetch connections: {e}")
            return None
    
    def fetch_neuron_info(self, body_ids: List[int], batch_size: int = 500, 
                          show_progress: bool = True) -> Optional[pd.DataFrame]:
        """Fetch FAFB neuron annotations/info from CAVE by root ID.

        ``hierarchical_neuron_annotations`` references
        ``proofread_neurons`` by its internal row ID, so the two API tables
        are queried in sequence.  This avoids the invalid ``pt_root_id``
        filter that the annotation table itself does not accept.
        """
        if is_banc_dataset(self.dataset):
            raise RuntimeError(
                f"BANC dataset '{self.dataset}' has no CAVE annotation path; "
                "use its locally prepared public-release tables."
            )
        body_ids = [body_id_to_api_int(bid) for bid in body_ids]
        if not body_ids:
            return pd.DataFrame()
        n_neurons = len(body_ids)
        n_batches = (n_neurons + batch_size - 1) // batch_size
        
        try:
            client = self.cave_client
            
            # Only show progress bar for multiple batches
            use_progress = show_progress and n_batches > 1 and self.verbose
            
            if use_progress:
                try:
                    from tqdm import tqdm
                    batches = tqdm(
                        range(n_batches), 
                        desc="  📋 Neuron info", 
                        unit="batch",
                        ncols=80,
                        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
                    )
                except ImportError:
                    use_progress = False
                    batches = range(n_batches)
            else:
                batches = range(n_batches)
            
            dfs = []
            for i in batches:
                batch_start = i * batch_size
                batch_end = min((i + 1) * batch_size, n_neurons)
                batch_ids = body_ids[batch_start:batch_end]
                
                # Resolve root IDs to proofread-neuron reference IDs first.
                proofread = client.materialize.query_table(
                    'proofread_neurons',
                    filter_in_dict={'pt_root_id': batch_ids},
                    metadata=False,
                    merge_reference=False,
                    materialization_version=self.materialization_version,
                )
                if proofread is not None and not proofread.empty and 'id' in proofread.columns:
                    target_ids = proofread['id'].dropna().astype(int).tolist()
                    if target_ids:
                        annotations = client.materialize.query_table(
                            'hierarchical_neuron_annotations',
                            filter_in_dict={'target_id': target_ids},
                            metadata=False,
                            merge_reference=True,
                            materialization_version=self.materialization_version,
                        )
                        if annotations is not None and not annotations.empty:
                            dfs.append(annotations)

                # User tags are the online source for named cell types such as
                # PPL101/aMe26 when the hierarchy only exposes broad classes.
                tags = client.materialize.query_table(
                    'neuron_information_v2',
                    filter_in_dict={'pt_root_id': batch_ids},
                    metadata=False,
                    merge_reference=False,
                    materialization_version=self.materialization_version,
                )
                if tags is not None and not tags.empty:
                    dfs.append(tags)
            
            if dfs:
                result = pd.concat(dfs, ignore_index=True)
                normalize_flywire_id_columns(
                    result,
                    ['bodyId', 'body_id', 'pt_root_id', 'root_id'],
                )
                if self.verbose:
                    print(f"  ✓ Fetched annotations for {len(result):,}/{n_neurons:,} neurons")
                return result
            else:
                if self.verbose:
                    print(f"  ℹ️ No annotations found for {n_neurons} neurons")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Failed to fetch neuron info: {e}")
            if self.verbose:
                print(f"  ✗ Failed to fetch neuron info: {e}")
            return None

    def fetch_neurons_by_types(
        self, types: List[str], show_progress: bool = True
    ) -> pd.DataFrame:
        """Resolve named FAFB types from online annotation tags.

        The public CAVE materialization exposes user tags by root ID.  Query
        each requested type as a regex in that table and return a normalized
        ``bodyId/type/instance/post`` frame for DROCAT.
        """
        if is_banc_dataset(self.dataset):
            raise RuntimeError(
                f"BANC dataset '{self.dataset}' has no CAVE type path; use its "
                "locally prepared public-release tables."
            )
        if not types:
            return pd.DataFrame(columns=['bodyId', 'type', 'instance', 'post'])

        try:
            client = self.cave_client
            frames = []
            iterator = types
            if show_progress and self.verbose and len(types) > 1:
                try:
                    from tqdm import tqdm
                    iterator = tqdm(types, desc='  📋 FlyWire types', unit='type')
                except ImportError:
                    pass

            for neuron_type in iterator:
                tags = client.materialize.query_table(
                    'neuron_information_v2',
                    filter_regex_dict={'tag': re.escape(str(neuron_type))},
                    metadata=False,
                    merge_reference=False,
                    materialization_version=self.materialization_version,
                )
                if tags is None or tags.empty or 'pt_root_id' not in tags.columns:
                    continue
                instances = (
                    tags['tag'].fillna('').astype(str)
                    if 'tag' in tags.columns
                    else pd.Series('', index=tags.index)
                )
                result = pd.DataFrame({
                    'bodyId': tags['pt_root_id'],
                    'type': str(neuron_type),
                    'instance': instances,
                    'post': 0,
                })
                normalize_flywire_id_columns(result, ['bodyId'])
                frames.append(result)

            if not frames:
                return pd.DataFrame(columns=['bodyId', 'type', 'instance', 'post'])
            return pd.concat(frames, ignore_index=True).drop_duplicates(
                subset=['bodyId', 'type']
            )
        except Exception as exc:
            logger.error(f"Failed to fetch neurons by type: {exc}")
            if self.verbose:
                print(f"  ✗ Failed to fetch neurons by type: {exc}")
            return pd.DataFrame(columns=['bodyId', 'type', 'instance', 'post'])
    
    def clear_cache(self, cache_type: str = 'all'):
        """Clear API and canonical dataset caches.
        
        Parameters
        ----------
        cache_type : str
            'skeletons', 'meshes', or 'all'
        """
        import shutil
        
        if cache_type in ['skeletons', 'all']:
            skeleton_dirs = [
                Path(self.get_cache_path('skeletons')),
                Path(self.project_root) / 'cache' / self._cache_dataset_name()
                / 'skeletons',
            ]
            seen = set()
            for skeleton_dir in skeleton_dirs:
                resolved = str(skeleton_dir.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                if skeleton_dir.exists():
                    shutil.rmtree(skeleton_dir)
                skeleton_dir.mkdir(parents=True, exist_ok=True)
                print(f"✓ Cleared skeleton cache: {skeleton_dir}")
        
        if cache_type in ['meshes', 'all']:
            mesh_dirs = [
                Path(self.get_cache_path('meshes')),
                Path(self.project_root) / 'cache' / self._cache_dataset_name()
                / 'meshes',
            ]
            seen = set()
            for mesh_dir_path in mesh_dirs:
                mesh_dir = str(mesh_dir_path)
                if mesh_dir in seen or not os.path.exists(mesh_dir):
                    continue
                seen.add(mesh_dir)
                shutil.rmtree(mesh_dir)
                os.makedirs(mesh_dir, exist_ok=True)
                print(f"✓ Cleared mesh cache: {mesh_dir}")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about the cache.
        
        Returns
        -------
        dict
            Dictionary with cache statistics
        """
        skeleton_dirs = (
            Path(self.get_cache_path('skeletons')),
            Path(self.project_root) / 'cache' / self._cache_dataset_name()
            / 'skeletons',
        )
        api_mesh_dir = Path(self.get_cache_path('meshes'))
        canonical_mesh_dir = (
            Path(self.project_root) / 'cache' / self._cache_dataset_name()
            / 'meshes'
        )

        skel_files = []
        seen_skeleton_files = set()
        for skel_dir in skeleton_dirs:
            if not skel_dir.exists():
                continue
            for path in skel_dir.rglob('*'):
                if not (path.is_file() and (
                        path.name.endswith('.pkl')
                        or path.name.endswith('.pkl.zst')
                        or path.name.endswith('.swc.gz')
                        or path.name.endswith('.swc.zst'))):
                    continue
                resolved = path.resolve()
                if resolved in seen_skeleton_files:
                    continue
                seen_skeleton_files.add(resolved)
                skel_files.append(path)
        mesh_files = []
        for mesh_dir in (api_mesh_dir, canonical_mesh_dir):
            if not mesh_dir.exists():
                continue
            mesh_files.extend(
                path for path in mesh_dir.rglob('*')
                if path.is_file()
                and (path.name.endswith('.pkl')
                     or path.name.endswith('.pkl.zst'))
            )
        mesh_files = list({path.resolve() for path in mesh_files})

        skel_count = len(skel_files)
        mesh_count = len(mesh_files)
        
        skel_size = sum(path.stat().st_size for path in skel_files)
        mesh_size = sum(path.stat().st_size for path in mesh_files)
        
        return {
            'skeleton_count': skel_count,
            'mesh_count': mesh_count,
            'skeleton_size_mb': round(skel_size / 1024 / 1024, 2),
            'mesh_size_mb': round(mesh_size / 1024 / 1024, 2),
            'total_size_mb': round((skel_size + mesh_size) / 1024 / 1024, 2),
        }


# Header lines recording the simplification level and the producing pipeline
# of a compressed-SWC cache file. The contract lives in
# :mod:`skeleton_provenance`; these names stay importable here for existing
# callers/tests.
_SIMPLIFICATION_HEADER = _provenance.SIMPLIFICATION_HEADER
_SOURCE_HEADER = _provenance.SOURCE_HEADER

# Provenance values written by the two replacement stores. CAVE
# mesh-skeletonized replacements live in ``cave_skeletons``; locally pruned
# extrusion fixes live in ``extrusion_fixes``.
CAVE_SKELETON_SOURCE = _provenance.CAVE_MESH_WAVEFRONT
LOCAL_EXTRUSION_FIX_SOURCE = _provenance.LOCAL_EXTRUSION_FIX


def _read_stored_source(text) -> str:
    """Parse ``# DROCAT source: NAME`` from a compressed-SWC header."""
    return _provenance.read_stored_source(text)


def _read_stored_simplification(text) -> int:
    """Parse ``# DROCAT simpl: N`` from a compressed-SWC header (0 = raw)."""
    return _provenance.read_stored_simplification(text)


def _swc_stem(name: str) -> str:
    """BodyId stem of a ``.swc.zst`` / ``.swc.gz`` cache filename."""
    for suffix in (".swc.zst", ".swc.gz"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return Path(name).stem


def _load_swc_text(path: str) -> str:
    """Read a compressed-SWC cache file (.swc.zst or .swc.gz) as text."""
    if str(path).endswith(".swc.zst"):
        if zstd is None:
            raise ImportError("zstandard is required to read .swc.zst caches")
        with open(path, "rb") as handle:
            with zstd.ZstdDecompressor().stream_reader(handle) as reader:
                content = reader.read()
        return content.decode("utf-8", "replace")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return handle.read()


def _write_compressed_swc_zst(path: str, text: bytes, simplification: int = 0
                              ) -> None:
    """Atomically write compressed SWC as zstd-19 with a recorded level.

    Thin wrapper over the shared :mod:`skeleton_provenance` writer: the level
    (percent of nodes removed; FlyWire raw skeletons are stored raw, 0) is
    recorded in the first SWC header line so later loads can re-level.
    """
    _provenance.write_compressed_swc_zst(path, text, simplification)


def load_repaired_skeletons(dataset, body_ids, project_root=None, log=None):
    """Serve previously repaired FAFB skeletons from the local repair caches.

    Network-free probe driven by the repair status recorded in
    ``extrusion_check_results.parquet``:

    1. ``api_repaired`` bodies are served from the ``cave_skeletons`` store.
    2. ``local_fallback`` bodies probe ``cave_skeletons`` first (a CAVE
       replacement may have arrived since the local prune), then the
       ``extrusion_fixes`` store with the persisted local fix.

    Returns the repaired trees keyed by ``int`` body id. Flagged ids that
    still need the repair chain (pending / api_failed statuses, or a
    recorded repair whose cache entry has vanished) are simply absent, so
    callers route them through the normal load + repair stages.
    """
    if is_banc_dataset(dataset):
        return {}
    from fafb_utils import (
        EXTRUSION_REPAIR_API_REPAIRED,
        EXTRUSION_REPAIR_LOCAL_FALLBACK,
        load_extrusion_repair_status,
    )

    fetcher = CAVEDataFetcher(
        dataset=dataset, cave_token="", cache_enabled=True,
        project_root=project_root, verbose=False)
    statuses = load_extrusion_repair_status(
        str(fetcher.project_root), fetcher._cache_dataset_name())
    served: Dict[int, object] = {}
    for body_id in body_ids or ():
        try:
            key = int(body_id)
        except (TypeError, ValueError):
            continue
        status = statuses.get(str(key))
        if status not in (EXTRUSION_REPAIR_API_REPAIRED,
                          EXTRUSION_REPAIR_LOCAL_FALLBACK):
            continue
        neuron = fetcher.load_cached_skeleton(key)
        if neuron is None and status == EXTRUSION_REPAIR_LOCAL_FALLBACK:
            neuron = fetcher.load_cached_fix_skeleton(key)
        if neuron is None:
            continue
        try:
            neuron.id = body_id_to_api_int(key)
            neuron.name = str(key)
            neuron.units = "nm"
        except Exception:
            pass
        served[key] = neuron
    if served and log:
        log(f"FAFB skeletons: served {len(served)} previously repaired "
            "tree(s) from the local repair caches.")
    return served


def test_fafb_access():
    """Test FAFB access with a sample neuron."""
    print("=" * 60)
    print("Testing CAVE API Data Fetcher for FlyWire FAFB")
    print("=" * 60)
    
    fetcher = CAVEDataFetcher(dataset='flywire_FAFB_v783')
    
    # Test with a known FAFB neuron
    test_id = 720575940596125868
    
    print(f"\n--- Testing with neuron: {test_id} ---")
    
    # Test mesh fetch
    print("\n1. Fetching mesh...")
    mesh = fetcher.fetch_mesh(test_id)
    if mesh:
        print(f"   Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Test skeleton fetch
    print("\n2. Fetching skeleton (via mesh skeletonization)...")
    skeleton = fetcher.fetch_skeleton(test_id)
    if skeleton:
        print(f"   Skeleton: {len(skeleton.nodes)} nodes")
    
    # Test synapse fetch
    print("\n3. Fetching synapses...")
    synapses = fetcher.fetch_synapses(test_id, direction='pre')
    if synapses is not None:
        print(f"   Synapses: {len(synapses)} outgoing")
    
    # Cache stats
    print("\n4. Cache statistics:")
    stats = fetcher.get_cache_stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == '__main__':
    test_fafb_access()

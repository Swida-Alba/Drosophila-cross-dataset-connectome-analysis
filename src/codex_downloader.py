"""FlyWire Codex bulk downloads for the FAFB dataset.

Server contract, measured against codex.flywire.ai (2026-10-01):
``GET /api/download_resource?data_product=...&dataset=fafb&api_token=...``
answers 401 without a token, 404 JSON for unknown products, honours Range
for resumable downloads and IGNORES If-Range (a wrong ETag still returns
206), so a resume pins the ETag locally in a sidecar and restarts from byte
0 whenever the server's ETag changed. Downloads happen only when the user
has provided a FlyWire Codex API token (``tokens.flywire_codex`` /
``FLYWIRE_CODEX_TOKEN``); the public GCS mirror of the same files is
deliberately not used.
"""

from __future__ import annotations

import http.client
import json
import os
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from .utils.token_manager import TokenManager
except ImportError:  # pragma: no cover - src laid bare on sys.path
    from utils.token_manager import TokenManager

try:
    from .utils.flywire_readiness import dataset_folder
except ImportError:  # pragma: no cover - src laid bare on sys.path
    from utils.flywire_readiness import dataset_folder

CODEX_DOWNLOAD_URL = 'https://codex.flywire.ai/api/download_resource'
CODEX_DATASET = 'fafb'
CODEX_TOKEN_ENV = 'FLYWIRE_CODEX_TOKEN'
CODEX_PORTAL_URL = 'https://codex.flywire.ai/api/download?dataset=fafb'
CODEX_ACCOUNT_URL = 'https://codex.flywire.ai/account'
DEFAULT_DATASET = 'flywire_FAFB_v783'

PROBE_ATTEMPTS = 3
DOWNLOAD_ATTEMPTS = 12
CHUNK_BYTES = 4 * 1024 * 1024
USER_AGENT = 'DROCAT/codex-downloader'

LEVEL_NECESSARY = 'necessary'
LEVEL_SYNAPSE = 'synapse'
LEVEL_SKELETON = 'skeleton'
LEVEL_ORDER = (LEVEL_NECESSARY, LEVEL_SYNAPSE, LEVEL_SKELETON)


@dataclass(frozen=True)
class CodexProduct:
    key: str
    filename: str
    size: int
    level: str
    description: str


PRODUCT_CATALOG: Dict[str, CodexProduct] = {p.key: p for p in [
    CodexProduct('classification', 'classification.csv.gz', 934_402,
                 LEVEL_NECESSARY, 'neuron table core (converter-required)'),
    CodexProduct('connections_princeton_no_threshold',
                 'connections_princeton_no_threshold.csv.gz', 275_679_780,
                 LEVEL_NECESSARY,
                 'unthresholded connections (converter-required)'),
    CodexProduct('names', 'names.csv.gz', 1_181_576, LEVEL_NECESSARY,
                 'neuron instance names'),
    CodexProduct('coordinates', 'coordinates.csv.gz', 5_314_546,
                 LEVEL_NECESSARY, 'soma coordinates'),
    CodexProduct('neurons', 'neurons.csv.gz', 1_679_884, LEVEL_NECESSARY,
                 'neuron metadata incl. neurotransmitter'),
    CodexProduct('cell_stats', 'cell_stats.csv.gz', 2_526_548,
                 LEVEL_NECESSARY, 'per-neuron stats'),
    CodexProduct('consolidated_cell_types', 'consolidated_cell_types.csv.gz',
                 901_707, LEVEL_NECESSARY, 'consolidated cell types'),
    CodexProduct('synapse_table', 'fafb_v783_princeton_synapse_table.csv.gz',
                 2_695_106_039, LEVEL_SYNAPSE,
                 'per-synapse table for synapse visualization'),
    CodexProduct('skeleton_swc_files', 'sk_lod1_783_healed.zip',
                 13_873_645_070, LEVEL_SKELETON,
                 'healed skeleton bundle (139,273 per-neuron SWCs)'),
]}


def level_products(level: str) -> List[CodexProduct]:
    return [p for p in PRODUCT_CATALOG.values() if p.level == level]


def level_bytes(level: str) -> int:
    return sum(p.size for p in level_products(level))


class CodexTokenRequired(RuntimeError):
    """No FlyWire Codex API token is configured."""


class CodexAuthError(RuntimeError):
    """The Codex server refused the configured token (HTTP 401)."""


class CodexProductUnavailable(RuntimeError):
    """The product is missing from the server or changed its size."""


class CodexDownloadCancelled(RuntimeError):
    """The user cancelled; the .part file stays on disk for a later resume."""


def codex_token_instructions() -> str:
    return (
        "A FlyWire Codex API token is required for FAFB downloads.\n"
        f"  1. Open {CODEX_PORTAL_URL}\n"
        "  2. Sign in with a Google account\n"
        f"  3. Copy your API token from the Account page "
        f"({CODEX_ACCOUNT_URL})\n"
        "  4. Paste it into Settings > API Tokens (FlyWire Codex), or set\n"
        f"     {CODEX_TOKEN_ENV}, or add tokens.flywire_codex to "
        "config_local.json.\n"
        "The CAVE token is contributor-gated and is NOT needed for downloads."
    )


def get_codex_token(project_root=None) -> Optional[str]:
    return TokenManager(project_root=project_root).get_token(CODEX_TOKEN_ENV)


def require_codex_token(project_root=None) -> str:
    token = get_codex_token(project_root)
    if not token:
        raise CodexTokenRequired(codex_token_instructions())
    return token


def build_download_url(product_key: str, token: str) -> str:
    query = urllib.parse.urlencode({
        'data_product': product_key,
        'dataset': CODEX_DATASET,
        'api_token': token,
    })
    return f'{CODEX_DOWNLOAD_URL}?{query}'


def _open(product_key: str, token: str, headers=None):
    request = urllib.request.Request(
        build_download_url(product_key, token),
        headers={'User-Agent': USER_AGENT, **(headers or {})})
    return urllib.request.urlopen(request, timeout=180)


def _raise_for_http(exc: urllib.error.HTTPError, product_key: str):
    if exc.code == 401:
        raise CodexAuthError(
            'The Codex server refused the API token (HTTP 401); it may have '
            'expired. Re-copy it from https://codex.flywire.ai/account and '
            'update Settings > API Tokens (FlyWire Codex).') from exc
    if exc.code == 404:
        raise CodexProductUnavailable(
            f"The Codex server does not serve '{product_key}' for dataset "
            f"'{CODEX_DATASET}'.") from exc
    raise exc


def _probe(product_key: str, token: str):
    """Return (etag, total) via a 0-byte ranged GET."""
    last_error = None
    for attempt in range(PROBE_ATTEMPTS):
        try:
            with _open(product_key, token, {'Range': 'bytes=0-0'}) as response:
                content_range = response.headers.get('Content-Range', '')
                if '/' in content_range:
                    total = int(content_range.rsplit('/', 1)[1])
                else:
                    total = int(response.headers.get('Content-Length', 0))
                return response.headers.get('ETag'), total
        except urllib.error.HTTPError as exc:
            _raise_for_http(exc, product_key)
        except (urllib.error.URLError, http.client.HTTPException,
                OSError, ValueError) as exc:
            last_error = exc
            time.sleep(min(10.0, 2.0 * (attempt + 1)))
    raise RuntimeError(
        f'Could not reach the Codex server for {product_key}: {last_error}')


def verify_codex_token(project_root=None, token=None):
    """(ok, message) for the Settings 'Test Codex' button.

    *token* bypasses the config chain so an unsaved value typed into the
    Settings field can be tested before saving.
    """
    if token is not None and token.strip():
        token = token.strip()
    else:
        token = get_codex_token(project_root)
    if not token:
        return False, 'No FlyWire Codex token is configured.'
    try:
        _probe('consolidated_cell_types', token)
        return True, 'Codex token accepted.'
    except CodexAuthError:
        return False, ('The Codex server refused this token (HTTP 401); '
                       're-copy it from the Account page.')
    except CodexProductUnavailable as exc:
        return False, f'Token reached the server, but the probe failed: {exc}'
    except RuntimeError as exc:
        return False, f'Could not reach codex.flywire.ai: {exc}'


def downloads_dir_for(dataset: str = DEFAULT_DATASET,
                      project_root=None) -> Path:
    root = Path(project_root) if project_root is not None \
        else Path(__file__).resolve().parents[1]
    return root / 'datasets' / dataset_folder(dataset) / 'downloads'


def product_status(product_key: str, downloads_dir) -> dict:
    product = PRODUCT_CATALOG[product_key]
    path = Path(downloads_dir) / product.filename
    local = path.stat().st_size if path.exists() else 0
    return {
        'key': product.key,
        'filename': product.filename,
        'expected': product.size,
        'local': local,
        'complete': local == product.size,
    }


def synapse_parquet_path(dataset: str = DEFAULT_DATASET,
                         project_root=None) -> Path:
    """Local converted synapse table the visualization reads."""
    folder = dataset_folder(dataset)
    return _root_path(project_root) / 'datasets' / folder / \
        f'{folder}_synapse_table.parquet'


def synapse_table_ready(dataset: str = DEFAULT_DATASET,
                        project_root=None) -> bool:
    return synapse_parquet_path(dataset, project_root).exists()


def _sidecar_path(dest: Path) -> Path:
    return dest.with_name(dest.name + '.part.json')


def _write_sidecar(sidecar: Path, etag, total: int):
    sidecar.write_text(json.dumps({'etag': etag, 'total': total}),
                       encoding='utf-8')


def _sidecar_matches(sidecar: Path, etag, total: int) -> bool:
    try:
        data = json.loads(sidecar.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    return data.get('etag') == etag and data.get('total') == total


def download_product(product_key: str, downloads_dir, project_root=None,
                     progress_callback=None, cancel_event=None,
                     attempts: int = DOWNLOAD_ATTEMPTS,
                     chunk_bytes: int = CHUNK_BYTES) -> Path:
    """Download one product into *downloads_dir*, resuming any partial.

    A partial transfer lives at ``{name}.part`` with a sidecar pinning the
    ETag and total observed at probe time. Because the server ignores
    If-Range, a resume is only continued when the sidecar ETag equals the
    current server ETag; otherwise the download restarts from byte 0.
    """
    product = PRODUCT_CATALOG[product_key]
    dest = Path(downloads_dir) / product.filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    token = require_codex_token(project_root)
    if dest.exists() and dest.stat().st_size == product.size:
        return dest

    part = dest.with_name(dest.name + '.part')
    sidecar = _sidecar_path(dest)
    etag, total = _probe(product_key, token)
    if total != product.size:
        raise CodexProductUnavailable(
            f"'{product_key}' now serves {total:,} bytes, but DROCAT pins "
            f'the {product.size:,}-byte v783 release measured on '
            '2026-10-01; refusing to download a changed product.')

    start = 0
    if part.exists():
        if _sidecar_matches(sidecar, etag, total):
            start = min(part.stat().st_size, total)
        if start == 0:
            part.unlink(missing_ok=True)
            sidecar.unlink(missing_ok=True)

    last_error = None
    for attempt in range(attempts):
        try:
            _write_sidecar(sidecar, etag, total)
            pos = start
            with open(part, 'ab' if pos else 'wb') as out:
                while pos < total:
                    if cancel_event is not None and cancel_event.is_set():
                        raise CodexDownloadCancelled(
                            f'{product.filename} cancelled after '
                            f'{pos:,}/{total:,} bytes; the partial file '
                            'resumes on the next attempt.')
                    end = min(pos + chunk_bytes, total) - 1
                    with _open(product_key, token,
                               {'Range': f'bytes={pos}-{end}'}) as response:
                        chunk = response.read()
                    out.write(chunk)
                    pos += len(chunk)
                    if progress_callback is not None:
                        progress_callback(pos, total)
            if part.stat().st_size != total:
                raise OSError(
                    f'incomplete download: {part.stat().st_size}/{total}')
            sidecar.unlink(missing_ok=True)
            os.replace(part, dest)
            return dest
        except urllib.error.HTTPError as exc:
            _raise_for_http(exc, product_key)
        except (urllib.error.URLError, http.client.HTTPException,
                OSError, ConnectionError) as exc:
            last_error = exc
        start = part.stat().st_size if part.exists() else 0
        if start > total:
            part.unlink(missing_ok=True)
            sidecar.unlink(missing_ok=True)
            start = 0
        if attempt + 1 < attempts:
            time.sleep(min(30.0, 3.0 * (attempt + 1)))
    raise RuntimeError(
        f'Failed to download {product.filename} after {attempts} attempts: '
        f'{last_error}')


# ---------------------------------------------------------------------------
# Lazy per-neuron skeletons from the skeleton_swc_files bundle
#
# The bundle is a zip (139,273 `{rootId}.swc` members, zip64), so a client
# can Range-read its central directory once (~10.9 MB, cached ETag-pinned)
# and then fetch any single neuron with one small ranged GET. Verified
# byte-identical to the local healed bundle on 2026-10-01.
# ---------------------------------------------------------------------------

SKELETON_PRODUCT_KEY = 'skeleton_swc_files'
# Name+extra bound of a local file header inside the bundle; the per-neuron
# ranged GET covers header + 4 KB slack + the compressed member.
LOCAL_HEADER_WINDOW = 4096
BUNDLE_TAIL_BYTES = 4096


def _root_path(project_root=None) -> Path:
    return Path(project_root) if project_root is not None \
        else Path(__file__).resolve().parents[1]


def bundle_cache_dir(dataset: str = DEFAULT_DATASET,
                     project_root=None) -> Path:
    return _root_path(project_root) / 'cache' / dataset_folder(dataset) / \
        'skeletons'


def parse_bundle_tail(tail: bytes):
    """(entry_count, cd_size, cd_offset) from the zip64 EOCD in *tail*."""
    marker = tail.rfind(b'PK\x06\x06')
    if marker >= 0:
        count, cd_size, cd_offset = struct.unpack(
            '<QQQ', tail[marker + 32:marker + 56])
        return count, cd_size, cd_offset
    eocd = tail.rfind(b'PK\x05\x06')
    if eocd < 0:
        raise CodexProductUnavailable(
            'The skeleton product is not a zip archive.')
    cd_size, cd_offset = struct.unpack('<II', tail[eocd + 12:eocd + 20])
    if cd_offset == 0xFFFFFFFF:
        raise CodexProductUnavailable(
            'The skeleton bundle lacks a zip64 end record.')
    return None, cd_size, cd_offset


def parse_bundle_central_directory(cd: bytes) -> Dict[str, dict]:
    """``{member: {method, comp, uncomp, crc, offset}}`` (zip64-aware)."""
    entries: Dict[str, dict] = {}
    pos = 0
    while pos + 46 <= len(cd) and cd[pos:pos + 4] == b'PK\x01\x02':
        method, = struct.unpack('<H', cd[pos + 10:pos + 12])
        crc, = struct.unpack('<I', cd[pos + 16:pos + 20])
        comp, uncomp = struct.unpack('<II', cd[pos + 20:pos + 28])
        nlen, elen, clen = struct.unpack('<HHH', cd[pos + 28:pos + 34])
        offset, = struct.unpack('<I', cd[pos + 42:pos + 46])
        name = cd[pos + 46:pos + 46 + nlen].decode('utf-8', 'replace')
        extra = cd[pos + 46 + nlen:pos + 46 + nlen + elen]
        if 0xFFFFFFFF in (comp, uncomp, offset):
            cursor = 0
            while cursor + 4 <= len(extra):
                header_id, size = struct.unpack(
                    '<HH', extra[cursor:cursor + 4])
                if header_id == 0x0001:
                    value_pos = cursor + 4
                    if uncomp == 0xFFFFFFFF:
                        uncomp, = struct.unpack(
                            '<Q', extra[value_pos:value_pos + 8])
                        value_pos += 8
                    if comp == 0xFFFFFFFF:
                        comp, = struct.unpack(
                            '<Q', extra[value_pos:value_pos + 8])
                        value_pos += 8
                    if offset == 0xFFFFFFFF:
                        offset, = struct.unpack(
                            '<Q', extra[value_pos:value_pos + 8])
                    break
                cursor += 4 + size
        entries[name] = {
            'method': method,
            'comp': comp,
            'uncomp': uncomp,
            'crc': crc & 0xFFFFFFFF,
            'offset': offset,
        }
        pos += 46 + nlen + elen + clen
    return entries


def _write_json_atomic(final: Path, payload: dict):
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(final.name + '.tmp')
    try:
        tmp.write_text(json.dumps(payload), encoding='utf-8')
        os.replace(tmp, final)
    finally:
        tmp.unlink(missing_ok=True)


def ensure_bundle_index(dataset: str = DEFAULT_DATASET, project_root=None,
                        progress_callback=None,
                        cancel_event=None) -> Dict[str, dict]:
    """The parsed bundle central-directory index, cached and ETag-pinned.

    The ~10.9 MB central directory is fetched once into
    ``cache/<dataset>/skeletons/`` and re-fetched only when the server's
    ETag changes (the server ignores If-Range, so the pin lives here).
    """
    token = require_codex_token(project_root)
    cache_dir = bundle_cache_dir(dataset, project_root)
    index_path = cache_dir / 'codex_bundle_index.json'
    meta_path = cache_dir / 'codex_bundle_meta.json'
    product = PRODUCT_CATALOG[SKELETON_PRODUCT_KEY]
    etag, total = _probe(SKELETON_PRODUCT_KEY, token)
    if total != product.size:
        raise CodexProductUnavailable(
            f"'{SKELETON_PRODUCT_KEY}' now serves {total:,} bytes, but "
            f'DROCAT pins the {product.size:,}-byte v783 release measured '
            'on 2026-10-01; refusing to index a changed bundle.')
    try:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        meta = {}
    if meta.get('etag') == etag and index_path.exists():
        try:
            cached = json.loads(index_path.read_text(encoding='utf-8'))
            if cached:
                return cached
        except (OSError, ValueError):
            pass

    tail_start = max(total - BUNDLE_TAIL_BYTES, 0)
    with _open(SKELETON_PRODUCT_KEY, token,
               {'Range': f'bytes={tail_start}-{total - 1}'}) as response:
        tail = response.read()
    entry_count, cd_size, cd_offset = parse_bundle_tail(tail)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cd_path = cache_dir / 'codex_bundle_central_directory.bin'
    cd_part = cd_path.with_name(cd_path.name + '.part')
    pos = 0
    with open(cd_part, 'wb') as out:
        while pos < cd_size:
            if cancel_event is not None and cancel_event.is_set():
                raise CodexDownloadCancelled(
                    'Bundle index download cancelled; it resumes on the '
                    'next attempt.')
            end = min(pos + CHUNK_BYTES, cd_size) - 1
            with _open(SKELETON_PRODUCT_KEY, token,
                       {'Range': f'bytes={cd_offset + pos}-'
                                 f'{cd_offset + end}'}) as response:
                chunk = response.read()
            out.write(chunk)
            pos += len(chunk)
            if progress_callback is not None:
                progress_callback(pos, cd_size)
    if cd_part.stat().st_size != cd_size:
        cd_part.unlink(missing_ok=True)
        raise OSError(f'incomplete bundle index: {pos}/{cd_size}')
    entries = parse_bundle_central_directory(cd_part.read_bytes())
    os.replace(cd_part, cd_path)
    if entry_count is not None and len(entries) != entry_count:
        raise CodexProductUnavailable(
            f'The bundle index parsed {len(entries):,} of {entry_count:,} '
            'entries; refusing to use a partial index.')
    _write_json_atomic(index_path, entries)
    _write_json_atomic(meta_path, {
        'etag': etag, 'total': total, 'entries': len(entries),
        'product': SKELETON_PRODUCT_KEY})
    return entries


def fetch_skeleton_swcs(body_ids, dataset: str = DEFAULT_DATASET,
                        project_root=None, index=None,
                        progress_callback=None,
                        cancel_event=None) -> Tuple[dict, list]:
    """Fetch healed SWC texts per neuron: ``({bodyId: text}, [missing])``.

    Each neuron costs one ranged GET against the bundle; the member CRC32
    from the central directory is verified before the text is returned.
    Body ids absent from the bundle land in *missing* instead of raising.
    """
    token = require_codex_token(project_root)
    if index is None:
        index = ensure_bundle_index(dataset, project_root,
                                    cancel_event=cancel_event)
    result = {}
    missing = []
    total = len(body_ids)
    for position, body_id in enumerate(body_ids):
        if cancel_event is not None and cancel_event.is_set():
            raise CodexDownloadCancelled(
                f'Skeleton fetch cancelled after {position}/{total}.')
        name = f'{body_id}.swc'
        entry = index.get(name)
        if entry is None:
            missing.append(body_id)
            continue
        start = entry['offset']
        end = start + 30 + LOCAL_HEADER_WINDOW + entry['comp'] - 1
        with _open(SKELETON_PRODUCT_KEY, token,
                   {'Range': f'bytes={start}-{end}'}) as response:
            blob = response.read()
        name_len, extra_len = struct.unpack('<HH', blob[26:30])
        data_start = 30 + name_len + extra_len
        payload = blob[data_start:data_start + entry['comp']]
        if entry['method'] == 0:
            raw = payload
        elif entry['method'] == 8:
            raw = zlib.decompressobj(-15).decompress(payload)
        else:
            missing.append(body_id)
            continue
        if len(raw) != entry['uncomp'] or \
                (zlib.crc32(raw) & 0xFFFFFFFF) != entry['crc']:
            raise CodexProductUnavailable(
                f'CRC/size mismatch for {name}; the remote bundle changed '
                'under the cached index. Delete '
                'cache/<dataset>/skeletons/codex_bundle_index.json and '
                'retry.')
        result[body_id] = raw.decode('utf-8', 'replace')
        if progress_callback is not None:
            progress_callback(position + 1, total)
    return result, missing

"""
FlyLight Downloader Module

This module provides programmatic access to FlyLight imagery data from multiple sources:
    1. S3 bucket (janelia-flylight-imagery): Gen1 MCFO, Split-GAL4 lines (R-lines)
    2. HTTP CDN (flimg.janelia.org): VT GAL4 lines (standard expression patterns)
    3. Gen1 MCFO CDN (gen1mcfo.janelia.org): VT MCFO lines (sparse labeling clones)

Key Features:
    - Search by driver line name (e.g., 'R10A06', 'VT037867', 'SS00731')
    - Automatic source detection (S3 for R-lines, HTTP CDN for VT lines)
    - Filter by file format: png, jpg, h5j, lsm, mp4, json, or all
    - Filter by image type: mip, cdm, aligned, unaligned, translation, signals, etc.
    - Filter by collection: Gen1 GAL4, Gen1 MCFO, Split-GAL4, VT GAL4, etc.
    - List available files without downloading
    - Automatic directory structure preservation
    - Image summary generation (PDF/PPTX) with configurable layout and styling

Usage:
    >>> from flylight_downloader import FlyLightDownloader
    >>> # Basic download
    >>> downloader = FlyLightDownloader(output_dir='./downloads')
    >>> downloader.download('R10A06')
    
    >>> # Download with PPTX summary of 4x2 images on black background
    >>> downloader = FlyLightDownloader(output_dir='./downloads', verbose='pbar')
    >>> downloader.download(
    ...     'SS00731', 
    ...     generate_summary='pptx', 
    ...     summary_images_per_page=(4, 2),
    ...     background_color='black'
    ... )

Data Sources:

1. S3 Bucket (janelia-flylight-imagery/) - R-lines:
    ├── Annotator Gen1 MCFO/
    │   ├── R10A06/
    │   │   ├── R10A06-...-metadata.json
    │   │   ├── R10A06-...-aligned_stack.h5j
    │   │   ├── R10A06-...-CDM_1.png
    │   │   └── ...
    │   └── ...
    ├── Split-GAL4 Omnibus/
    │   └── ...
    └── ...

2. HTTP CDN (flimg.janelia.org) - VT GAL4 Lines:
    ├── projections/  - JPEG projection images
    ├── translations/ - MP4 fly-through movies
    └── (LSM stacks via CGI download)

3. Gen1 MCFO for VT Lines (discovered via gen1mcfo.janelia.org):
    VT lines with MCFO data are hosted in the S3 bucket under 'Gen1 MCFO'
    and 'Annotator Gen1 MCFO' collections, but are not easily discoverable
    via direct S3 listing. This module parses the gen1mcfo.janelia.org viewer
    page to find all available S3-hosted MCFO images for VT lines.

**VT Line Search Order:**
When searching for VT line images, this module searches in priority order:
    1. VT GAL4 (flimg.janelia.org) - Standard expression patterns
    2. VT MCFO (S3 via gen1mcfo.janelia.org) - Sparse labeling clones (included automatically)

This ensures that lines like VT000770, which have no GAL4 images but have
MCFO data, will still return results.

File Types:
    - *-metadata.json: Specimen metadata
    - *-aligned_stack.h5j: Aligned 3D stack (H5J format)
    - *-unaligned_stack.h5j: Unaligned 3D stack
    - *-CDM_*.png: Color Depth Mask images (for NeuronBridge)
    - *-mip.png: Maximum Intensity Projection
    - *-signals_mip.png: Signal channel MIP
    - *-multichannel_mip.png: Multichannel MIP
    - *-translation.mp4: Translation video
    - *.lsm.bz2: Raw confocal data (compressed)
    - VT lines: *_total.jpg, *_ch2_total.jpg (projections), *.t.mp4 (translations)

Author: DROCAT Project
"""

import os
import re
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False


# S3 bucket configuration
FLYLIGHT_BUCKET = 'janelia-flylight-imagery'
FLYLIGHT_REGION = 'us-east-1'

# VT line HTTP CDN configuration
VT_CDN_BASE = 'https://flimg.janelia.org/flylight-image/external-data/adult/secdata'
VT_VIEW_CGI = 'https://flweb.janelia.org/cgi-bin/view_flew_imagery.cgi'
VT_DOWNLOAD_CGI = 'https://flweb.janelia.org/cgi-bin/download.cgi'

# Gen1 MCFO configuration (for VT lines MCFO data)
# VT lines have MCFO data at gen1mcfo.janelia.org, not in the S3 bucket
GEN1_MCFO_VIEW_CGI = 'https://gen1mcfo.janelia.org/cgi-bin/view_gen1mcfo_imagery.cgi'
GEN1_MCFO_SEARCH_CGI = 'https://gen1mcfo.janelia.org/cgi-bin/gen1mcfo.cgi'
GEN1_MCFO_CDN_BASE = 'https://gen1mcfo.janelia.org/imagery'

# User-Agent for HTTP requests to Janelia servers (they block Python's default User-Agent)
HTTP_USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'


def _create_request(url: str, method: str = 'GET') -> urllib.request.Request:
    """
    Create an HTTP request with proper headers for Janelia servers.
    
    The Janelia servers (flweb.janelia.org, flimg.janelia.org, gen1mcfo.janelia.org)
    block requests without proper User-Agent headers.
    
    Args:
        url: The URL to request
        method: HTTP method (GET, HEAD, POST, etc.)
        
    Returns:
        urllib.request.Request object with proper headers
    """
    req = urllib.request.Request(url, method=method)
    req.add_header('User-Agent', HTTP_USER_AGENT)
    req.add_header('Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8')
    req.add_header('Accept-Language', 'en-US,en;q=0.5')
    return req

# Known collections in the S3 bucket (top-level folders)
# Note: VT GAL4 lines are NOT in this S3 bucket - they use the HTTP CDN above
FLYLIGHT_COLLECTIONS = [
    'Annotator Gen1 MCFO',  # R-lines with MCFO stochastic labeling
    'Gen1 MCFO',            # More R-lines with MCFO
    'Gen1',                 # Gen1 CDM images (R-lines)
    'Split-GAL4 Omnibus Broad',    # Split-GAL4 broad collection
    'Split-GAL4 Omnibus Rescreen', # Split-GAL4 rescreen collection
    # Paper-specific Split-GAL4 collections:
    'Ascending Neurons 2023',
    'Aso 2021',
    'Aso&Rubin 2016',
    'Descending Neurons 2018',
    'Descending Neurons 2025',
    'Dorsal VNC 2023',
    'Hulse et al. 2021',           # Central Complex
    'Isaacson et al 2023',
    'Lateral Horn 2019',
    'Lillvis 2022',
    'Lillvis 2023',
    'MB Paper 2014',
    'Namiki 2021',
    'Nern et al 2023',
    'Nern et al 2024',             # Optic lobe Split-GAL4 lines (SS27796, etc.)
    'Rubin & Aso 2023',
    'Rubin et al 2025',
    'Schretter et al 2024',
    'Schretter et al. 2020',
    'SEZ 2021',
    'Wolff 2018',
    'Wolff et al 2024',
]

# Collection categories for easier filtering
# Maps category name (case-insensitive) to list of collection folder names
COLLECTION_CATEGORIES = {
    'GAL4/LEXA': [
        'Gen1',                  # Gen1 GAL4/LexA R-lines (CDM images)
        # VT GAL4 lines are handled separately via HTTP CDN
    ],
    'SPLITGAL4': [
        'Split-GAL4 Omnibus Broad',    # SS-lines broad collection
        'Split-GAL4 Omnibus Rescreen', # SS-lines rescreen
        # Paper-specific Split-GAL4 collections:
        'Ascending Neurons 2023',
        'Aso 2021',
        'Aso&Rubin 2016',
        'Descending Neurons 2018',
        'Descending Neurons 2025',
        'Dorsal VNC 2023',
        'Hulse et al. 2021',           # Central Complex
        'Isaacson et al 2023',
        'Lateral Horn 2019',
        'Lillvis 2022',
        'Lillvis 2023',
        'MB Paper 2014',
        'Namiki 2021',
        'Nern et al 2023',
        'Nern et al 2024',             # Optic lobe Split-GAL4 lines
        'Rubin & Aso 2023',
        'Rubin et al 2025',
        'Schretter et al 2024',
        'Schretter et al. 2020',
        'SEZ 2021',
        'Wolff 2018',
        'Wolff et al 2024',
    ],
    'MCFO': [
        'Annotator Gen1 MCFO',  # R-lines with MCFO labeling
        'Gen1 MCFO',            # More MCFO data
    ],
    'RAWIMAGES': [
        # Collections with raw LSM confocal data
        'Annotator Gen1 MCFO',  # Has .lsm.bz2 files
        'Gen1 MCFO',            # Has .lsm.bz2 files
    ],
    'ALL': FLYLIGHT_COLLECTIONS,  # All collections
}

# Normalized lookup for case-insensitive access
_CATEGORY_LOOKUP = {k.upper().replace('-', '').replace('_', ''): k for k in COLLECTION_CATEGORIES.keys()}

# Reverse mapping: collection folder -> category
COLLECTION_TO_CATEGORY = {}
for category, collections in COLLECTION_CATEGORIES.items():
    if category.upper() != 'ALL':  # Skip 'ALL' to avoid overwriting specific categories
        for coll in collections:
            COLLECTION_TO_CATEGORY[coll] = category

# Cache for dynamically discovered collections
_DISCOVERED_COLLECTIONS_CACHE: Optional[List[str]] = None


def discover_s3_collections(refresh: bool = False, verbose: bool = False) -> List[str]:
    """
    Dynamically discover all available collections from the FlyLight S3 bucket.
    
    This function queries the S3 bucket to get all top-level folders (collections)
    instead of relying on a hardcoded list. Results are cached for efficiency.
    
    Args:
        refresh: If True, force a refresh of the cache
        verbose: Print discovery progress
        
    Returns:
        List of collection names (folder prefixes without trailing slash)
    """
    global _DISCOVERED_COLLECTIONS_CACHE
    
    if _DISCOVERED_COLLECTIONS_CACHE is not None and not refresh:
        return _DISCOVERED_COLLECTIONS_CACHE
    
    collections = []
    
    # Try boto3 first
    if HAS_BOTO3:
        try:
            s3 = boto3.client(
                's3',
                region_name=FLYLIGHT_REGION,
                config=Config(signature_version=UNSIGNED)
            )
            
            result = s3.list_objects_v2(
                Bucket=FLYLIGHT_BUCKET,
                Delimiter='/'
            )
            
            for prefix in result.get('CommonPrefixes', []):
                folder = prefix['Prefix'].rstrip('/')
                # Skip non-collection folders
                if folder not in ['content']:
                    collections.append(folder)
            
            if verbose:
                print(f"✅ Discovered {len(collections)} collections from S3")
                
        except Exception as e:
            if verbose:
                print(f"⚠️  Failed to discover collections via boto3: {e}")
    
    # Fallback to HTTP if boto3 failed or not available
    if not collections:
        try:
            url = f"https://s3.amazonaws.com/{FLYLIGHT_BUCKET}?delimiter=/"
            with urllib.request.urlopen(url, timeout=30) as response:
                content = response.read().decode('utf-8')
            
            # Parse XML response
            root = ET.fromstring(content)
            ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
            
            for prefix in root.findall('.//s3:CommonPrefixes/s3:Prefix', ns):
                if prefix.text:
                    folder = prefix.text.rstrip('/')
                    if folder not in ['content']:
                        collections.append(folder)
            
            if verbose:
                print(f"✅ Discovered {len(collections)} collections via HTTP")
                
        except Exception as e:
            if verbose:
                print(f"⚠️  Failed to discover collections via HTTP: {e}")
            # Fall back to hardcoded list
            collections = FLYLIGHT_COLLECTIONS.copy()
            if verbose:
                print(f"   Using hardcoded list of {len(collections)} collections")
    
    _DISCOVERED_COLLECTIONS_CACHE = collections
    return collections


def update_collection_categories(discovered: List[str]) -> Dict[str, List[str]]:
    """
    Update COLLECTION_CATEGORIES with newly discovered collections.
    
    Attempts to categorize new collections based on naming patterns.
    Collections that can't be categorized are added to SPLITGAL4 by default
    (most paper-specific collections are Split-GAL4).
    
    Args:
        discovered: List of discovered collection names
        
    Returns:
        Updated categories dict
    """
    global COLLECTION_TO_CATEGORY
    
    known = set(FLYLIGHT_COLLECTIONS)
    new_collections = [c for c in discovered if c not in known]
    
    if not new_collections:
        return COLLECTION_CATEGORIES
    
    # Add new collections to appropriate categories
    for coll in new_collections:
        coll_lower = coll.lower()
        
        # Determine category based on naming patterns
        if 'mcfo' in coll_lower:
            if coll not in COLLECTION_CATEGORIES['MCFO']:
                COLLECTION_CATEGORIES['MCFO'].append(coll)
            category = 'MCFO'
        elif 'gen1' in coll_lower and 'mcfo' not in coll_lower:
            if coll not in COLLECTION_CATEGORIES['GAL4/LEXA']:
                COLLECTION_CATEGORIES['GAL4/LEXA'].append(coll)
            category = 'GAL4/LEXA'
        else:
            # Default: assume Split-GAL4 (most paper collections are Split-GAL4)
            if coll not in COLLECTION_CATEGORIES['SPLITGAL4']:
                COLLECTION_CATEGORIES['SPLITGAL4'].append(coll)
            category = 'SPLITGAL4'
        
        # Update reverse mapping
        COLLECTION_TO_CATEGORY[coll] = category
    
    # Update ALL category
    COLLECTION_CATEGORIES['ALL'] = discovered
    
    return COLLECTION_CATEGORIES


# File format extensions
FILE_FORMATS = {
    'png': ['.png'],
    'jpg': ['.jpg', '.jpeg'],
    'h5j': ['.h5j'],
    'lsm': ['.lsm.bz2', '.lsm'],
    'mp4': ['.mp4'],
    'json': ['.json'],
    'all': ['.png', '.jpg', '.jpeg', '.h5j', '.lsm.bz2', '.lsm', '.mp4', '.json'],
}

# Image type patterns
IMAGE_TYPES = {
    'mip': r'_(mip|signals_mip|multichannel_mip)\.png$',
    'cdm': r'-CDM_\d+\.png$',
    'aligned': r'-aligned_stack\.(h5j|png)$',
    'unaligned': r'-unaligned_stack\.(h5j|png)$',
    'translation': r'(_translation\.mp4$|\.t\.mp4$)',  # S3 and VT formats
    'signals': r'[_-]signals',
    'multichannel': r'[_-]multichannel',
    'metadata': r'-metadata\.json$',
    'raw': r'\.lsm(\.bz2)?$',
    # VT line specific patterns
    'projection': r'_total\.jpg$',  # VT line main projections
    'pattern': r'_ch2_.*\.jpg$',  # VT line pattern channel
    'substack': r'_\d{2}\.jpg$',  # VT line substacks
    'vt': r'(_(total|ch2_).*\.jpg$|\.t\.mp4$)',  # All VT file types
    'all': r'.*',
}


@dataclass
class FlyLightFile:
    """Represents a file in the FlyLight S3 bucket or HTTP CDN."""
    key: str
    size: int
    last_modified: str
    collection: str = ''
    line_name: str = ''
    source: str = 's3'  # 's3' or 'http'
    http_url: str = ''  # Direct URL for HTTP sources
    
    @property
    def filename(self) -> str:
        """Get the filename from the S3 key."""
        return os.path.basename(self.key)
    
    @property
    def extension(self) -> str:
        """Get the file extension."""
        name = self.filename.lower()
        if name.endswith('.lsm.bz2'):
            return '.lsm.bz2'
        return os.path.splitext(name)[1]
    
    @property
    def size_mb(self) -> float:
        """Get the file size in MB."""
        return self.size / (1024 * 1024)
    
    @property
    def url(self) -> str:
        """Get the URL for this file."""
        if self.source == 'http' and self.http_url:
            return self.http_url
        encoded_key = urllib.parse.quote(self.key)
        return f"https://s3.amazonaws.com/{FLYLIGHT_BUCKET}/{encoded_key}"


@dataclass
class VTSampleInfo:
    """Information about a VT line sample (brain or VNC)."""
    sample_id: str
    line_name: str
    region: str  # 'brain' or 'vnc'
    date: str
    sample_path: str  # Full sample path for URL construction
    session_id: str = ''  # For LSM download
    
    # Metadata from the page
    vector: str = ''
    landing_site: str = ''
    age: str = ''
    gender: str = ''
    reporter: str = ''


@dataclass
class FlyLightDownloader:
    """
    Download FlyLight imagery data from the Janelia S3 bucket.
    
    Attributes:
        output_dir: Directory to save downloaded files
        collections: List of collections to search (None = all, or use collection_category)
        collection_category: Category to filter ('GAL4', 'SplitGAL4', 'MCFO', 'All')
        formats: File formats to include ('png', 'h5j', 'lsm', 'mp4', 'json', 'all')
        image_types: Image types to include ('mip', 'cdm', 'aligned', etc.)
        region: Anatomical region filter ('Brain', 'VNC', or 'All')
        max_workers: Number of parallel download threads
        verbose: Print progress messages. 
                 - True: full output
                 - 'pbar': use tqdm progress bar and condensed output
                 - 'simple': simplified text output
                 - False: no output
        use_boto3: Use boto3 for downloads (faster) if available
        simple_mode: Apply filename filtering to reduce download volume
        
    Collection Categories (case-insensitive):
        - 'GAL4/LEXA': Gen1 GAL4/LexA R-lines (+ VT lines via HTTP CDN if include_vt_lines=True)
        - 'SplitGAL4': SS-lines from Split-GAL4 Omnibus and paper-specific collections
        - 'MCFO': MCFO stochastic labeling collections
        - 'RawImages': Collections with raw LSM confocal data
        - 'All': All available collections
        
    Region Filtering:
        - 'Brain': Only files with brain imagery (excludes VNC)
        - 'VNC': Only files with ventral nerve cord imagery
        - 'All': All regions (default)
        
    Simple Mode Filtering:
        When simple_mode=True, applies filename-based filtering to reduce download volume:
        - Split-GAL4 collections: only files with '20x' AND 'multichannel' in filename,
          excluding files with 'image1' or 'image2' (duplicates)
        - VT GAL4 lines: only files with 'total' in filename
        - Gen1 R-lines: keep CDM and MIP files (they don't have 'total' variants)
        - Other collections (MCFO, etc.): keep all files
        
    Can also pass a list of categories: ['GAL4/LEXA', 'SplitGAL4']
        
    Note: If both `collections` and `collection_category` are specified,
          `collections` takes precedence.
    """
    output_dir: str = './flylight_downloads'
    collections: Optional[List[str]] = None
    collection_category: Optional[Union[str, List[str]]] = None  # 'GAL4/LEXA', 'SplitGAL4', 'MCFO', 'RawImages', 'All'
    formats: Union[str, List[str]] = field(default_factory=lambda: ['png', 'jpg'])
    image_types: Union[str, List[str]] = 'all'
    region: str = 'All'  # 'Brain', 'VNC', or 'All' - filter by anatomical region
    max_workers: int = 4
    verbose: Union[bool, str] = 'pbar'
    use_boto3: bool = True
    include_vt_lines: bool = True  # Also search VT lines via HTTP CDN
    simple_mode: bool = True  # Apply filename filtering to reduce download volume
    auto_discover: bool = True  # Auto-discover collections from S3 bucket
    simple_mode_min_files: int = 2  # Minimum files to return per line when simple_mode filters all
    automatic_fallback: bool = True  # Try MCFO, then RawImages when primary categories have no matches
    
    # Internal state
    _s3_client: Any = field(default=None, repr=False, init=False)
    _file_cache: Dict[str, List[FlyLightFile]] = field(default_factory=dict, repr=False, init=False)
    _vt_sample_cache: Dict[str, List[VTSampleInfo]] = field(default_factory=dict, repr=False, init=False)
    _resolved_collections: List[str] = field(default=None, repr=False, init=False)
    _fallback_downloader_cache: Dict[str, Any] = field(default_factory=dict, repr=False, init=False)
    
    def __post_init__(self):
        """Initialize S3 client and normalize parameters."""
        # Normalize formats
        if isinstance(self.formats, str):
            self.formats = [self.formats]
        
        # Normalize image types
        if isinstance(self.image_types, str):
            self.image_types = [self.image_types]
        
        # Resolve collections from category if not explicitly set
        self._resolve_collections()
        
        # Initialize boto3 client if available
        if self.use_boto3 and HAS_BOTO3:
            try:
                self._s3_client = boto3.client(
                    's3',
                    region_name=FLYLIGHT_REGION,
                    config=Config(signature_version=UNSIGNED)
                )
                if self.verbose:
                    print("✅ Using boto3 for S3 access")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  boto3 initialization failed: {e}")
                    print("   Falling back to HTTP access")
                self._s3_client = None
        else:
            self._s3_client = None
            if self.verbose:
                if not HAS_BOTO3:
                    print("ℹ️  boto3 not installed, using HTTP access")
                    print("   Install boto3 for faster downloads: pip install boto3")
    
    def _resolve_collections(self):
        """Resolve collections from category if not explicitly provided."""
        # Auto-discover collections if enabled
        if self.auto_discover:
            discovered = discover_s3_collections(verbose=self.verbose)
            if discovered:
                # Update categories with newly discovered collections
                update_collection_categories(discovered)
        
        if self.collections is not None:
            # Use explicitly specified collections
            self._resolved_collections = self.collections
        elif self.collection_category:
            # Handle list of categories or single category
            categories = self.collection_category if isinstance(self.collection_category, list) else [self.collection_category]
            
            resolved = []
            for cat in categories:
                # Normalize for case-insensitive lookup
                normalized = cat.upper().replace('-', '').replace('_', '').replace(' ', '')
                
                # Find matching category
                if normalized in _CATEGORY_LOOKUP:
                    cat_key = _CATEGORY_LOOKUP[normalized]
                    resolved.extend(COLLECTION_CATEGORIES[cat_key])
                    if self.verbose:
                        print(f"📁 Category '{cat}' → {len(COLLECTION_CATEGORIES[cat_key])} collections")
                else:
                    if self.verbose:
                        print(f"⚠️ Unknown category: {cat}")
            
            if resolved:
                # Remove duplicates while preserving order
                seen = set()
                self._resolved_collections = [x for x in resolved if not (x in seen or seen.add(x))]
            else:
                if self.verbose:
                    print(f"⚠️ No valid categories found, using all collections")
                self._resolved_collections = discover_s3_collections() if self.auto_discover else FLYLIGHT_COLLECTIONS
        else:
            # Default: all collections (use discovered if available)
            self._resolved_collections = discover_s3_collections() if self.auto_discover else None
    
    def get_collection_category(self, collection_name: str) -> str:
        """Get the category for a given collection name."""
        return COLLECTION_TO_CATEGORY.get(collection_name, 'Other')
    
    @staticmethod
    def list_categories() -> Dict[str, List[str]]:
        """Return available collection categories and their collections."""
        return COLLECTION_CATEGORIES.copy()
    
    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if not self.verbose and self.verbose != 'False':
            return
        
        # 'pbar' mode: condense output (suppress file-level logs)
        if self.verbose == 'pbar':
            # Suppress "Downloading..." messages or other file-level details
            if "Downloading" in message or "Saved:" in message:
                return
            # Allow summary/error messages
        
        # 'simple' mode: standard output for now, maybe condensed later
        
        print(message)

    def _progress(self, step: int, total: int, label: str = ""):
        """Emit a structured step-progress event consumed by the web UI.

        The line ``[DROCAT][progress] <step>/<total> <label>`` drives the
        determinate progress bar + step label in the results panel; it is a
        control event, never shown in the execution log.  Printed in both
        'pbar' and full-verbose modes (the UI runs with ``verbose='pbar'``).
        """
        if not self.verbose or self.verbose == 'False':
            return
        msg = f"[DROCAT][progress] {int(step)}/{int(total)} {label}".rstrip()
        print(msg, flush=True)
    
    def _is_vt_line(self, line_name: str) -> bool:
        """Check if a line name is a VT line (served from HTTP CDN)."""
        return bool(re.match(r'^VT\d+', line_name, re.IGNORECASE))
    
    def _is_r_line(self, line_name: str) -> bool:
        """Check if a line name is an R-line (GMR collection, served from flweb CGI).
        
        R-lines follow the pattern R##X## (e.g., R78H08, R14A01) where:
        - R = prefix
        - ## = two digits
        - X = single letter (A-Z)
        - ## = two digits
        """
        return bool(re.match(r'^R\d{2}[A-Z]\d{2}$', line_name, re.IGNORECASE))

    def _requested_category_keys(self) -> set:
        """Return normalized collection-category names for the current query.

        R-lines are served by a mixture of CGI and S3 endpoints, so their
        category handling cannot rely on the generic S3 collection loop alone.
        Keep the normalization here in one place so category-specific R-line
        lookups do not accidentally return GAL4 images for a Split-GAL4 query.
        """
        if self.collection_category is None:
            if self.collections is None:
                return {'ALL'}
            categories = self.collections
        else:
            categories = (
                self.collection_category
                if isinstance(self.collection_category, list)
                else [self.collection_category]
            )

        normalized = set()
        for category in categories:
            normalized.add(
                str(category).upper().replace('-', '').replace('_', '')
                .replace(' ', '').replace('/', '')
            )
        return normalized

    def _automatic_fallback_categories(self) -> List[str]:
        """Return fallback categories for an exhausted primary-category query.

        A direct FlyLight download that explicitly asks for GAL4/LexA and/or
        Split-GAL4 should still find MCFO and raw products when those primary
        collections are empty. Explicit MCFO/RawImages/All queries already
        include their intended sources, so they must not recurse into this
        fallback path.
        """
        if (
            not self.automatic_fallback
            or self.collection_category is None
            or self.collections is not None
        ):
            return []

        requested = self._requested_category_keys()
        primary_categories = {'GAL4LEXA', 'GAL4', 'LEXA', 'SPLITGAL4'}
        if not requested or 'ALL' in requested or not requested.issubset(primary_categories):
            return []

        return ['MCFO', 'RawImages']

    def _get_fallback_downloader(self, category: str) -> "FlyLightDownloader":
        """Create a quiet, non-recursive downloader for one fallback category."""
        if category not in self._fallback_downloader_cache:
            fallback = FlyLightDownloader(
                output_dir=self.output_dir,
                collection_category=category,
                formats=list(self.formats),
                image_types=list(self.image_types),
                region=self.region,
                max_workers=self.max_workers,
                verbose=False,
                use_boto3=False,
                include_vt_lines=self.include_vt_lines,
                simple_mode=self.simple_mode,
                auto_discover=False,
                simple_mode_min_files=self.simple_mode_min_files,
                automatic_fallback=False,
            )
            # Reuse the parent S3 client so the fallback does not create a
            # second connection or change the selected access method.
            fallback._s3_client = self._s3_client
            self._fallback_downloader_cache[category] = fallback

        return self._fallback_downloader_cache[category]
    
    def _parse_vt_page(self, line_name: str) -> tuple[List[VTSampleInfo], str]:
        """
        Parse the VT line view page to extract sample information.
        
        Returns:
            Tuple of (list of VTSampleInfo, session_id)
        """
        url = f"{VT_VIEW_CGI}?line={line_name}"
        samples = []
        session_id = ''
        
        try:
            req = _create_request(url)
            with urllib.request.urlopen(req, timeout=30) as response:
                html = response.read().decode('utf-8')
        except Exception as e:
            self._log(f"❌ Error fetching VT page: {e}")
            return samples, session_id
        
        # Extract session ID from download URLs
        sid_match = re.search(r'sid=([a-f0-9]+)', html)
        if sid_match:
            session_id = sid_match.group(1)
        
        # Parse sample sections (Brain, VNC)
        # Pattern: download.cgi?id={sample_id} and projection URLs
        sample_pattern = re.compile(
            r"download\.cgi\?id=(\d+).*?"
            r"<h3>([^<]+)</h3>.*?"  # Section title (Brain, VNC)
            r"projections/(\d+)/"  # Date folder
            r"([^/]+)/"  # Sample path
            r"([^\"]+)\.jpg",  # First image filename
            re.DOTALL
        )
        
        # Simplified pattern for extracting projection URLs
        projection_pattern = re.compile(
            r'https://flimg\.janelia\.org/flylight-image/external-data/adult/secdata/projections/'
            r'(\d+)/'  # date
            r'([^/]+)/'  # sample folder
            r'([^\"\']+\.jpg)',  # filename
            re.IGNORECASE
        )
        
        # Find all projection URLs
        projection_matches = projection_pattern.findall(html)
        
        # Group by sample folder to identify unique samples
        sample_folders = {}
        for date, folder, filename in projection_matches:
            if folder not in sample_folders:
                sample_folders[folder] = {'date': date, 'files': []}
            sample_folders[folder]['files'].append(filename)
        
        # Extract sample IDs from download links
        download_pattern = re.compile(r"download\.cgi\?id=(\d+)")
        sample_ids = download_pattern.findall(html)
        
        # Determine region from folder name 
        # fA00b = brain, fA00v = VNC (case-insensitive search)
        for i, (folder, info) in enumerate(sample_folders.items()):
            folder_lower = folder.lower()
            if '-fa00b' in folder_lower or 'fa00b' in folder_lower:
                region = 'brain'
            elif '-fa00v' in folder_lower or 'fa00v' in folder_lower:
                region = 'vnc'
            else:
                # Try to infer from the sample path structure
                region = 'brain' if 'brain' in folder_lower else 'vnc' if 'vnc' in folder_lower else 'unknown'
            
            sample_id = sample_ids[i] if i < len(sample_ids) else ''
            
            samples.append(VTSampleInfo(
                sample_id=sample_id,
                line_name=line_name,
                region=region,
                date=info['date'],
                sample_path=folder,
                session_id=session_id
            ))
        
        return samples, session_id
    
    def _get_vt_files(self, line_name: str) -> List[FlyLightFile]:
        """
        Get available files for a VT line from the HTTP CDN.
        
        VT lines have a different structure than S3 files:
        - Projections: JPG images at flimg.janelia.org
        - Translations: MP4 movies
        - LSM: Full confocal stacks (requires CGI download)
        """
        if line_name in self._vt_sample_cache:
            samples = self._vt_sample_cache[line_name]
        else:
            samples, session_id = self._parse_vt_page(line_name)
            self._vt_sample_cache[line_name] = samples
        
        files = []
        
        for sample in samples:
            base_url = f"{VT_CDN_BASE}/projections/{sample.date}/{sample.sample_path}"
            translation_base = f"{VT_CDN_BASE}/translations/{sample.date}"
            
            # Construct filenames based on VT naming convention
            # Main projection: {line}_{sample_path}_total.jpg
            filename_base = f"{line_name}_{sample.sample_path.split('_', 1)[1] if '_' in sample.sample_path else sample.sample_path}"
            
            # Standard VT files (these exist for most samples):
            vt_file_types = [
                ('_total.jpg', 'projection'),
                ('_ch2_total.jpg', 'pattern_projection'),
            ]
            
            # Substack files (01-10 typically)
            for i in range(1, 11):
                vt_file_types.append((f'_{i:02d}.jpg', f'substack_{i:02d}'))
                vt_file_types.append((f'_ch2_{i:02d}.jpg', f'pattern_substack_{i:02d}'))
            
            for suffix, file_type in vt_file_types:
                filename = f"{filename_base}{suffix}"
                url = f"{base_url}/{filename}"
                
                files.append(FlyLightFile(
                    key=f"VT GAL4/{line_name}/{sample.region}/{filename}",
                    size=0,  # Unknown until we HEAD the URL
                    last_modified='',
                    collection='VT GAL4',
                    line_name=line_name,
                    source='http',
                    http_url=url
                ))
            
            # Translation video
            video_filename = f"{filename_base}.t.mp4"
            video_url = f"{translation_base}/{video_filename}"
            files.append(FlyLightFile(
                key=f"VT GAL4/{line_name}/{sample.region}/{video_filename}",
                size=0,
                last_modified='',
                collection='VT GAL4',
                line_name=line_name,
                source='http',
                http_url=video_url
            ))
        
        return files
    
    def _get_r_line_files(self, line_name: str) -> List[FlyLightFile]:
        """
        Get available files for an R-line from the FlyLight web interface.
        
        R-lines (GMR collection) have expression pattern images available at
        flweb.janelia.org. Unlike VT lines, R-lines use a different naming
        convention and may have images in multiple formats (GAL4, LexA).
        
        Args:
            line_name: R-line name (e.g., 'R78H08')
            
        Returns:
            List of FlyLightFile objects for available images
        """
        url = f"{VT_VIEW_CGI}?line={line_name}"
        files = []
        
        try:
            req = _create_request(url)
            with urllib.request.urlopen(req, timeout=30) as response:
                html = response.read().decode('utf-8')
        except Exception as e:
            self._log(f"❌ Error fetching R-line page: {e}")
            return files
        
        # Extract all unique flimg URLs from the page (without query params)
        # Pattern: https://flimg.janelia.org/flylight-image/external-data/adult/secdata/projections/.../*.jpg
        url_pattern = re.compile(
            r'https://flimg\.janelia\.org/flylight-image/external-data/adult/secdata/'
            r'(projections|translations)/'
            r'([^/]+)/'  # date or line folder
            r'([^/]+)/'  # sample folder
            r'([^"\?\']+\.(jpg|mp4))',  # filename with extension
            re.IGNORECASE
        )
        
        # Find all matches and deduplicate
        seen_urls = set()
        for match in url_pattern.finditer(html):
            image_type, folder, sample_path, filename, ext = match.groups()
            # Construct clean URL without query params
            clean_url = f"https://flimg.janelia.org/flylight-image/external-data/adult/secdata/{image_type}/{folder}/{sample_path}/{filename}"
            
            if clean_url in seen_urls:
                continue
            seen_urls.add(clean_url)
            
            # Determine region from sample path
            sample_lower = sample_path.lower()
            if '-fa' in sample_lower and 'b' in sample_lower.split('-fa')[-1][:3]:
                region = 'brain'
            elif '-fa' in sample_lower and 'v' in sample_lower.split('-fa')[-1][:3]:
                region = 'vnc'
            else:
                region = 'unknown'
            
            # Determine if it's GAL4 or LexA from the path/filename
            driver = 'LexA' if '_LJ_' in sample_path or '_L_' in filename or 'LexA' in filename else 'GAL4'
            collection = f'Gen1 {driver}'
            
            files.append(FlyLightFile(
                key=f"{collection}/{line_name}/{region}/{filename}",
                size=0,  # Unknown until HEAD request
                last_modified='',
                collection=collection,
                line_name=line_name,
                source='http',
                http_url=clean_url
            ))
        
        self._log(f"   Found {len(files)} files from flweb.janelia.org")
        return files
    
    def _get_gen1_mcfo_files(self, line_name: str) -> List[FlyLightFile]:
        """
        Get Gen1 MCFO images for a driver line from gen1mcfo.janelia.org.
        
        Both VT and R lines can have MCFO data on the viewer page. The actual
        images are hosted in the S3 bucket (janelia-flylight-imagery) under
        ``Gen1 MCFO`` and ``Annotator Gen1 MCFO`` collections, but the viewer
        is the most reliable way to discover the image keys for a line.
        
        Args:
            line_name: Driver line name (e.g., 'VT000770' or 'R96A08')
            
        Returns:
            List of FlyLightFile objects from the Gen1 MCFO collection on S3
        """
        import ssl
        files = []
        
        # Try both hosts. The flweb mirror is useful when the dedicated
        # gen1mcfo host has a transient TLS problem.
        endpoints = [
            f"{GEN1_MCFO_VIEW_CGI}?line={urllib.parse.quote(str(line_name), safe='')}",
            f"https://flweb.janelia.org/cgi-bin/view_gen1mcfo_imagery.cgi?line={urllib.parse.quote(str(line_name), safe='')}",
        ]
        
        html = None
        for url in endpoints:
            try:
                # Use SSL context to avoid SSL handshake errors
                ctx = ssl.create_default_context()
                req = _create_request(url)
                with urllib.request.urlopen(req, timeout=30, context=ctx) as response:
                    html = response.read().decode('utf-8')
                break  # Success - stop trying endpoints
            except Exception as e:
                self._log(f"   ⚠️ Error fetching MCFO page from {url.split('/')[2]}: {e}")
                continue
        
        if html is None:
            self._log(f"   ⚠️ Could not fetch Gen1 MCFO page for {line_name} from any endpoint")
            return files
        
        # Parse S3 image URLs from the page
        # Gen1 MCFO images are hosted on S3: https://s3.amazonaws.com/janelia-flylight-imagery/...
        # Pattern: src="https://s3.amazonaws.com/janelia-flylight-imagery/Gen1+MCFO/VT000770/..."
        # Also: "Annotator+Gen1+MCFO" collection
        
        # Find all S3 image URLs (PNG and JPG). The page currently uses
        # ``s3.amazonaws.com`` URLs with ``+`` for spaces, but accept the
        # regional host too so a page-side URL change does not break fallback.
        s3_pattern = re.compile(
            r'https://(?:s3\.amazonaws\.com|s3-[^./]+\.amazonaws\.com)/janelia-flylight-imagery/'
            r'((?:Gen1\+MCFO|Annotator\+Gen1\+MCFO)/[^"\'?]+\.(?:png|jpe?g))',
            re.IGNORECASE
        )
        
        matches = s3_pattern.findall(html)
        seen_keys = set()
        
        for key in matches:
            # Normalize the URL path (the viewer uses + for spaces).
            key = urllib.parse.unquote_plus(key)
            
            # Skip duplicates
            if key in seen_keys:
                continue
            seen_keys.add(key)
            
            # Extract filename and collection
            parts = key.split('/')
            collection = parts[0] if parts else 'Gen1 MCFO'
            filename = parts[-1] if parts else key
            
            # Skip thumbnails and query-string variants
            if 'thumbnail' in filename.lower() or '_thumb' in filename.lower():
                continue
            
            files.append(FlyLightFile(
                key=key,
                size=0,
                last_modified='',
                collection=collection,
                line_name=line_name,
                source='s3',  # These are S3-hosted files
                http_url=(
                    f"https://s3.amazonaws.com/{FLYLIGHT_BUCKET}/"
                    f"{urllib.parse.quote(key, safe='/')}"
                )
            ))
        
        if files:
            self._log(f"   📦 Found {len(files)} Gen1 MCFO images for {line_name}")
        
        return files

    # Backward-compatible name used by older callers and documentation.
    def _get_vt_mcfo_files(self, line_name: str) -> List[FlyLightFile]:
        return self._get_gen1_mcfo_files(line_name)
    
    def _verify_vt_file_exists(self, file: FlyLightFile) -> bool:
        """Check if a VT file URL is accessible."""
        try:
            req = _create_request(file.http_url, method='HEAD')
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status == 200
        except Exception:
            return False
    
    def list_vt_files(
        self, 
        line_name: str, 
        verify: bool = False,
        include_mcfo: bool = True
    ) -> List[FlyLightFile]:
        """
        List available files for a VT line.
        
        VT lines have images from two sources:
        1. GAL4 images from flimg.janelia.org (standard projections)
        2. MCFO (Multi-Color Flip-Out) images from gen1mcfo.janelia.org
        
        The search order follows the FlyLight category priority:
        1. GAL4/LEXA - Standard expression patterns (flimg.janelia.org)
        2. MCFO - Sparse labeling clones (gen1mcfo.janelia.org)
        
        Args:
            line_name: VT line name (e.g., 'VT037867', 'VT000770')
            verify: If True, verify each URL exists (slower but accurate)
            include_mcfo: If True, also search Gen1 MCFO collection (default: True)
            
        Returns:
            List of FlyLightFile objects from both GAL4 and MCFO sources
        """
        self._log(f"🔍 Searching VT line files for '{line_name}'...")
        
        # 1. Get standard VT GAL4 files from flimg.janelia.org
        files = self._get_vt_files(line_name)
        self._log(f"   Found {len(files)} GAL4 files from flimg.janelia.org")
        
        # 2. Get MCFO files from gen1mcfo.janelia.org
        if include_mcfo:
            mcfo_files = self._get_vt_mcfo_files(line_name)
            if mcfo_files:
                self._log(f"   Found {len(mcfo_files)} MCFO files from gen1mcfo.janelia.org")
                files.extend(mcfo_files)
        
        if verify:
            self._log("   Verifying file URLs (this may take a moment)...")
            verified_files = []
            for f in files:
                if self._verify_vt_file_exists(f):
                    verified_files.append(f)
            files = verified_files
            self._log(f"   Found {len(files)} verified files")
        else:
            self._log(f"   Found {len(files)} total potential files (use verify=True to confirm)")
        
        return files

    def _get_format_extensions(self) -> List[str]:
        """Get list of file extensions to include."""
        extensions = []
        for fmt in self.formats:
            fmt = fmt.lower()
            if fmt in FILE_FORMATS:
                extensions.extend(FILE_FORMATS[fmt])
            else:
                # Treat as raw extension
                if not fmt.startswith('.'):
                    fmt = '.' + fmt
                extensions.append(fmt)
        return list(set(extensions))
    
    def _get_image_type_patterns(self) -> List[re.Pattern]:
        """Get compiled regex patterns for image types."""
        patterns = []
        for img_type in self.image_types:
            img_type = img_type.lower()
            if img_type in IMAGE_TYPES:
                patterns.append(re.compile(IMAGE_TYPES[img_type], re.IGNORECASE))
            else:
                # Treat as raw pattern
                patterns.append(re.compile(img_type, re.IGNORECASE))
        return patterns
    
    def _matches_filters(self, file: FlyLightFile) -> bool:
        """Check if a file matches the format, image type, and region filters."""
        # Check format
        extensions = self._get_format_extensions()
        if not any(file.filename.lower().endswith(ext) for ext in extensions):
            return False
        
        # Check image type
        patterns = self._get_image_type_patterns()
        if not any(p.search(file.filename) for p in patterns):
            return False
        
        # Check region filter
        if self.region and self.region.lower() != 'all':
            filename_lower = file.filename.lower()
            key_lower = file.key.lower() if file.key else ''
            region_lower = self.region.lower()
            
            # Detect region from filename or key patterns
            # S3 files: '-brain-' or '-ventral_nerve_cord-' or '-vnc-'
            # R-line/VT files: 'fA01b' or 'fA00b' (brain), 'fA01v' or 'fA00v' (VNC)
            combined = filename_lower + ' ' + key_lower
            is_brain = ('-brain-' in combined or 
                       '_brain' in combined or
                       'fa01b' in combined or 
                       'fa00b' in combined)
            is_vnc = ('-ventral_nerve_cord-' in combined or 
                      '-vnc-' in combined or 
                      '_vnc' in combined or
                      'fa01v' in combined or
                      'fa00v' in combined)
            
            if region_lower == 'brain':
                # Only brain files, or files without clear region (e.g., VT GAL4 projections)
                if is_vnc:
                    return False
            elif region_lower == 'vnc':
                # Only VNC files
                if is_brain or (not is_vnc and not is_brain):
                    return False
        
        return True
    
    def _list_bucket_http(self, prefix: str = '', marker: str = '') -> List[FlyLightFile]:
        """List bucket contents using HTTP (no boto3 required)."""
        files = []
        
        # Build request URL
        params = {'prefix': prefix}
        if marker:
            params['marker'] = marker
        
        url = f"https://s3.amazonaws.com/{FLYLIGHT_BUCKET}"
        if params:
            url += '?' + urllib.parse.urlencode(params)
        
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                xml_data = response.read()
        except Exception as e:
            self._log(f"❌ Error listing bucket: {e}")
            return files
        
        # Parse XML response
        root = ET.fromstring(xml_data)
        ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
        
        for content in root.findall('s3:Contents', ns):
            key = content.find('s3:Key', ns).text
            size = int(content.find('s3:Size', ns).text)
            last_modified = content.find('s3:LastModified', ns).text
            
            # Extract collection and line name from key
            # Handle structure: Collection/LineName/... or Collection/CDM/LineName/...
            parts = key.split('/')
            collection = parts[0] if len(parts) > 1 else ''
            
            # Check for CDM subfolder structure (Gen1/CDM/R21B12/...)
            if len(parts) > 2 and parts[1].upper() == 'CDM':
                line_name = parts[2] if len(parts) > 3 else ''
            else:
                line_name = parts[1] if len(parts) > 2 else ''
            
            files.append(FlyLightFile(
                key=key,
                size=size,
                last_modified=last_modified,
                collection=collection,
                line_name=line_name
            ))
        
        # Check for truncation
        is_truncated = root.find('s3:IsTruncated', ns)
        if is_truncated is not None and is_truncated.text == 'true':
            # Get more results
            next_marker = files[-1].key if files else marker
            files.extend(self._list_bucket_http(prefix, next_marker))
        
        return files
    
    def _list_bucket_boto3(self, prefix: str = '') -> List[FlyLightFile]:
        """List bucket contents using boto3."""
        files = []
        
        paginator = self._s3_client.get_paginator('list_objects_v2')
        
        try:
            for page in paginator.paginate(Bucket=FLYLIGHT_BUCKET, Prefix=prefix):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    parts = key.split('/')
                    collection = parts[0] if len(parts) > 1 else ''
                    
                    # Check for CDM subfolder structure (Gen1/CDM/R21B12/...)
                    if len(parts) > 2 and parts[1].upper() == 'CDM':
                        line_name = parts[2] if len(parts) > 3 else ''
                    else:
                        line_name = parts[1] if len(parts) > 2 else ''
                    
                    files.append(FlyLightFile(
                        key=key,
                        size=obj['Size'],
                        last_modified=str(obj['LastModified']),
                        collection=collection,
                        line_name=line_name
                    ))
        except Exception as e:
            self._log(f"❌ Error listing bucket: {e}")
        
        return files
    
    def list_files(self, line_name: str, use_cache: bool = True) -> List[FlyLightFile]:
        """
        List available files for a driver line.
        
        Automatically detects whether to use S3 (R-lines) or HTTP CDN (VT lines).
        
        Args:
            line_name: Driver line name (e.g., 'R10A06', 'VT037867')
            use_cache: Use cached file list if available
            
        Returns:
            List of FlyLightFile objects matching the line name
        """
        # Check cache
        if use_cache and line_name in self._file_cache:
            return self._file_cache[line_name]
        
        self._log(f"🔍 Searching for files matching '{line_name}'...")
        
        all_files = []
        
        # Check if this is a VT line - use HTTP CDN
        if self._is_vt_line(line_name) and self.include_vt_lines:
            self._log(f"   Detected VT line - searching HTTP CDN...")
            
            # Determine whether to include MCFO based on collection category
            # MCFO should be included if:
            # - No specific category is set (default: search all)
            # - Category is 'All' or 'MCFO'
            # - Category list includes 'MCFO' or 'All'
            include_mcfo = True  # Default: include MCFO
            include_gal4 = True  # Default: include GAL4
            
            if self.collection_category:
                categories = self.collection_category if isinstance(self.collection_category, list) else [self.collection_category]
                categories_normalized = [c.upper().replace('-', '').replace('_', '').replace(' ', '').replace('/', '') for c in categories]
                
                # Check if MCFO should be included
                mcfo_keywords = {'MCFO', 'ALL'}
                include_mcfo = any(c in mcfo_keywords for c in categories_normalized)
                
                # Check if GAL4/LEXA should be included
                gal4_keywords = {'GAL4LEXA', 'GAL4', 'LEXA', 'ALL'}
                include_gal4 = any(c in gal4_keywords for c in categories_normalized)
            
            # Get VT files based on category
            if include_gal4 and include_mcfo:
                # Get both GAL4 and MCFO
                all_files = self.list_vt_files(line_name, verify=False, include_mcfo=True)
            elif include_gal4:
                # GAL4 only - skip MCFO
                self._log(f"   Category filter: GAL4/LEXA only (no MCFO)")
                all_files = self._get_vt_files(line_name)
                self._log(f"   Found {len(all_files)} GAL4 files from flimg.janelia.org")
            elif include_mcfo:
                # MCFO only - skip GAL4
                self._log(f"   Category filter: MCFO only (no GAL4)")
                all_files = self._get_gen1_mcfo_files(line_name)
                self._log(f"   Found {len(all_files)} MCFO files from gen1mcfo.janelia.org")
            else:
                # No matching category - return empty
                self._log(f"   ⚠️ No matching VT collections for category")
                all_files = []
        # Check if this is an R-line - use HTTP viewer for expression patterns
        elif self._is_r_line(line_name):
            requested_categories = self._requested_category_keys()
            search_all = 'ALL' in requested_categories
            requested_collection_names = {
                str(category).lower() for category in requested_categories
            }

            # ``collection_category`` normally contains normalized category
            # names, while explicit ``collections`` contains S3 folder names.
            # Support both forms without making an R-line query leak files
            # from a different collection category.
            include_gal4 = (
                search_all
                or bool(requested_categories & {'GAL4LEXA', 'GAL4', 'LEXA'})
                or any(
                    'gen1' in collection and 'mcfo' not in collection
                    for collection in requested_collection_names
                )
            )
            include_mcfo = (
                search_all
                or 'MCFO' in requested_categories
                or any('mcfo' in collection for collection in requested_collection_names)
            )
            include_raw = search_all or 'RAWIMAGES' in requested_categories

            # Keep the ordered fallback categories independent: GAL4/LexA
            # calls this branch first, then MCFO and RawImages are requested
            # by their own downloader instances when needed.
            if include_gal4:
                self._log(f"   Detected R-line (GMR collection) - searching flweb.janelia.org...")

                # Get expression pattern images from flweb.janelia.org.
                all_files.extend(self._get_r_line_files(line_name))

            # Gen1 GAL4/LexA R-lines expose CDM files below Gen1/CDM/<line>.
            self._log(f"   Also searching S3 for CDM images...")
            collections_to_search = self._resolved_collections or FLYLIGHT_COLLECTIONS
            if include_gal4:
                for collection in collections_to_search:
                    collection_lower = collection.lower()
                    if 'gen1' in collection_lower and 'mcfo' not in collection_lower:
                        self._log(f"   Searching {collection}...")

                        prefix_cdm = f"{collection}/CDM/{line_name}/"
                        if self._s3_client:
                            files = self._list_bucket_boto3(prefix_cdm)
                        else:
                            files = self._list_bucket_http(prefix_cdm)

                        if files:
                            all_files.extend(files)
                            self._log(f"   Found {len(files)} CDM files in {collection}")

            # The Gen1 MCFO CGI is the authoritative discovery path for R-line
            # MCFO data (for example R96A08). It returns direct S3 image URLs,
            # even when the normal GAL4/Split-GAL4 collections are empty.
            if include_mcfo:
                self._log(f"   Searching Gen1 MCFO viewer for {line_name}...")
                mcfo_files = self._get_gen1_mcfo_files(line_name)
                seen_keys = {file.key for file in all_files}
                for file in mcfo_files:
                    if file.key not in seen_keys:
                        all_files.append(file)
                        seen_keys.add(file.key)

            # RawImages uses the direct collection/line layout. Keep this
            # separate from MCFO so callers can enforce MCFO → RawImages
            # fallback order.
            if include_raw:
                for collection in collections_to_search:
                    collection_lower = collection.lower()
                    if 'mcfo' not in collection_lower:
                        continue

                    self._log(f"   Searching {collection} for raw files...")
                    prefix = f"{collection}/{line_name}/"
                    if self._s3_client:
                        files = self._list_bucket_boto3(prefix)
                    else:
                        files = self._list_bucket_http(prefix)

                    seen_keys = {file.key for file in all_files}
                    for file in files:
                        if file.key not in seen_keys:
                            all_files.append(file)
                            seen_keys.add(file.key)

            # Do not query the flweb CGI or Gen1 GAL4 collection for an
            # R-line when only Split-GAL4/RawImages was requested.
            if not include_gal4 and not include_mcfo and not include_raw:
                self._log(f"   ⚠️ No R-line source matches the requested categories")
        else:
            # Search S3 bucket for Split-GAL4, SS-lines, etc.
            collections_to_search = self._resolved_collections or FLYLIGHT_COLLECTIONS
            
            # Collections that commonly overlap - always search all of them
            # MCFO collections: always search both as they commonly overlap
            # Split-GAL4 collections: search all when simple_mode is enabled, since 
            # different collections may have different resolution images (20x vs 63x)
            mcfo_collections = {'Annotator Gen1 MCFO', 'Gen1 MCFO'}
            splitgal4_collections = {c for c in COLLECTION_CATEGORIES.get('SPLITGAL4', [])}
            
            for i, collection in enumerate(collections_to_search):
                self._log(f"   Searching {collection}...")
                
                # Try direct path first: Collection/LineName/
                prefix = f"{collection}/{line_name}/"
                
                if self._s3_client:
                    files = self._list_bucket_boto3(prefix)
                else:
                    files = self._list_bucket_http(prefix)
                
                # If no files found, try nested CDM path: Collection/CDM/LineName/
                # (Gen1 collection has this structure)
                if not files:
                    prefix_cdm = f"{collection}/CDM/{line_name}/"
                    if self._s3_client:
                        files = self._list_bucket_boto3(prefix_cdm)
                    else:
                        files = self._list_bucket_http(prefix_cdm)
                
                all_files.extend(files)
                
                # Early exit logic - skip remaining collections if we have files,
                # BUT always check overlapping collection types:
                # - MCFO collections commonly overlap
                # - Split-GAL4 collections when simple_mode is enabled (different resolutions)
                if all_files and len(collections_to_search) > 3:
                    remaining = collections_to_search[i+1:]
                    # Check if any remaining collection is MCFO
                    next_mcfo = [c for c in remaining if c in mcfo_collections]
                    # Check if any remaining collection is Split-GAL4 AND we're in simple_mode
                    next_splitgal4 = [c for c in remaining if c in splitgal4_collections] if self.simple_mode else []
                    
                    if next_mcfo or next_splitgal4:
                        continue  # Keep searching overlapping collections
                    self._log(f"   ✓ Found files in {collection}, skipping remaining collections")
                    break
        
        # Cache results
        self._file_cache[line_name] = all_files
        
        self._log(f"   Found {len(all_files)} total files")
        
        return all_files
    
    def apply_simple_mode_filter(
        self, 
        files: List[FlyLightFile], 
        collection_name: Optional[str] = None
    ) -> List[FlyLightFile]:
        """
        Filter files based on simple_mode rules.
        
        Simple mode reduces download volume by selecting only representative files:
        - Split-GAL4 collections: only '20x' AND 'multichannel' files, excluding 'image1'/'image2'
        - GAL4/LexA collections: only 'total' files
        - VT lines: only 'total' files
        - Other collections (MCFO, etc.): keep all files
        
        Args:
            files: List of FlyLightFile objects to filter
            collection_name: Collection name for determining filter rules.
                           If None, uses file.collection attribute.
                           
        Returns:
            List of filtered FlyLightFile objects
        """
        if not self.simple_mode or not files:
            return files
        
        filtered = []
        
        for f in files:
            # Determine collection to use for filtering
            coll = collection_name or f.collection or ''
            coll_lower = coll.lower()
            filename_lower = f.filename.lower()
            
            # Get the category for this collection to determine filter rules
            collection_category = COLLECTION_TO_CATEGORY.get(coll, 'Other').upper()
            
            # Split-GAL4: only include '20x' AND 'multichannel', exclude 'image1'/'image2'
            # Check both category mapping and string patterns for backward compatibility
            is_splitgal4 = (collection_category == 'SPLITGAL4' or
                           any(x in coll_lower for x in ['splitgal4', 'split-gal4', 'split_gal4', 'omnibus']))
            if is_splitgal4:
                if '20x' in filename_lower and 'multichannel' in filename_lower:
                    # Exclude duplicate images (image1, image2, etc.)
                    if 'image1' not in filename_lower and 'image2' not in filename_lower:
                        filtered.append(f)
            # VT GAL4 lines (from HTTP CDN): only include 'total'
            elif coll_lower.startswith('vt') or 'vt gal4' in coll_lower:
                if 'total' in filename_lower:
                    filtered.append(f)
            # Gen1 R-lines (from HTTP or S3): keep 'total' and CDM files
            elif 'gen1' in coll_lower and 'mcfo' not in coll_lower:
                # HTTP images: keep 'total' (projection and pattern projection)
                if 'total' in filename_lower:
                    filtered.append(f)
                # S3 CDM images - keep them all (they're already filtered)
                elif 'cdm' in filename_lower or 'mip' in filename_lower:
                    filtered.append(f)
            else:
                # For other collections (MCFO, etc.), keep all files
                filtered.append(f)
        
        if self.verbose and len(filtered) != len(files):
            self._log(f"   Simple mode: {len(files)} → {len(filtered)} files")
        
        return filtered
    
    def get_filtered_files(self, line_name: Union[str, List[str]], apply_simple_mode: Optional[bool] = None, max_files_per_line: Optional[int] = None) -> List[FlyLightFile]:
        """
        Get files matching the configured format and image type filters.
        
        Args:
            line_name: Driver line name(s). Can be:
                - Single string: 'R10A06'
                - Comma-separated: 'R10A06,VT037867'
                - List: ['R10A06', 'VT037867']
            apply_simple_mode: Override class simple_mode setting. If None, uses class setting.
            max_files_per_line: Maximum number of files per line. Applied after simple_mode filtering.
            
        Returns:
            List of FlyLightFile objects matching filters
        """
        # Normalize input to list of line names
        if isinstance(line_name, str):
            if ',' in line_name:
                line_names = [name.strip() for name in line_name.split(',')]
            else:
                line_names = [line_name]
        else:
            line_names = line_name
        
        # Collect files from all lines, tracking per-line for fallback
        from collections import defaultdict
        files_by_line_pre_simple = defaultdict(list)  # Before simple_mode filter
        fallback_categories = self._automatic_fallback_categories()

        for name in line_names:
            all_files = self.list_files(name)
            filtered = [f for f in all_files if self._matches_filters(f)]

            if not filtered and fallback_categories:
                requested = self.collection_category
                if isinstance(requested, list):
                    requested_label = ', '.join(str(category) for category in requested)
                else:
                    requested_label = str(requested)
                self._log(
                    f"   ℹ️ No files from {requested_label} for '{name}'; "
                    "trying MCFO → RawImages fallback..."
                )

                for fallback_category in fallback_categories:
                    fallback_downloader = self._get_fallback_downloader(fallback_category)
                    fallback_files = fallback_downloader.get_filtered_files(
                        name,
                        apply_simple_mode=False,
                    )
                    if fallback_files:
                        filtered = fallback_files
                        self._log(
                            f"   ✅ Found {len(filtered)} matching files in "
                            f"{fallback_category} fallback for '{name}'"
                        )
                        break

            self._log(f"   {len(filtered)} files match format/type filters for '{name}'")
            files_by_line_pre_simple[name] = filtered
        
        # Apply simple_mode filtering if enabled
        use_simple = apply_simple_mode if apply_simple_mode is not None else self.simple_mode
        
        all_filtered = []
        lines_with_fallback = []  # Track lines that needed fallback
        
        if use_simple:
            for name, pre_files in files_by_line_pre_simple.items():
                if not pre_files:
                    continue
                    
                # Apply simple_mode filter
                post_files = self.apply_simple_mode_filter(pre_files)
                
                # Fallback: if simple_mode filtered out ALL files, include some anyway
                if not post_files and pre_files and self.simple_mode_min_files > 0:
                    # Prioritize multichannel MIP files, then any MIP, then any file
                    fallback = []
                    
                    # First priority: multichannel_mip files (any resolution)
                    for f in pre_files:
                        if 'multichannel_mip' in f.filename.lower():
                            fallback.append(f)
                            if len(fallback) >= self.simple_mode_min_files:
                                break
                    
                    # Second priority: any MIP files
                    if len(fallback) < self.simple_mode_min_files:
                        for f in pre_files:
                            if f not in fallback and '_mip' in f.filename.lower():
                                fallback.append(f)
                                if len(fallback) >= self.simple_mode_min_files:
                                    break
                    
                    # Last resort: any file
                    if len(fallback) < self.simple_mode_min_files:
                        for f in pre_files:
                            if f not in fallback:
                                fallback.append(f)
                                if len(fallback) >= self.simple_mode_min_files:
                                    break
                    
                    post_files = fallback
                    lines_with_fallback.append(name)
                
                all_filtered.extend(post_files)
            
            # Warn user about fallback
            if lines_with_fallback and self.verbose:
                self._log(f"   ⚠️ Simple mode filter too strict for {len(lines_with_fallback)} line(s): "
                         f"{', '.join(lines_with_fallback[:5])}{'...' if len(lines_with_fallback) > 5 else ''}")
                self._log(f"      → Included {self.simple_mode_min_files} fallback file(s) per line (may not match 20x/multichannel criteria)")
        else:
            # No simple_mode filtering
            for pre_files in files_by_line_pre_simple.values():
                all_filtered.extend(pre_files)
        
        # Apply max_files_per_line limit if specified
        if max_files_per_line:
            # Group files by line name and limit each group
            files_by_line = defaultdict(list)
            for f in all_filtered:
                files_by_line[f.line_name].append(f)
            
            # Limit each line and reassemble
            limited_files = []
            for line, files in files_by_line.items():
                limited_files.extend(files[:max_files_per_line])
            all_filtered = limited_files
        
        return all_filtered
    
    def _download_file_http(self, file: FlyLightFile, local_path: Path, max_retries: int = 5) -> bool:
        """
        Download a file using HTTP with retry logic for transient errors.
        
        Args:
            file: FlyLightFile object to download
            local_path: Local path to save the file
            max_retries: Maximum number of retry attempts (default: 5)
            
        Returns:
            True if successful, False otherwise
        """
        import time
        import ssl
        
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        for attempt in range(max_retries):
            try:
                urllib.request.urlretrieve(file.url, str(local_path))
                return True
            except ssl.SSLError as e:
                # SSL errors (e.g., UNEXPECTED_EOF_WHILE_READING) - retry with backoff
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                    self._log(f"⚠️  SSL error downloading {file.filename}, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    self._log(f"❌ SSL error downloading {file.filename} after {max_retries} attempts: {e}")
                    return False
            except urllib.error.URLError as e:
                # Network errors - retry with backoff
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    self._log(f"⚠️  Network error downloading {file.filename}, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    self._log(f"❌ Network error downloading {file.filename} after {max_retries} attempts: {e}")
                    return False
            except Exception as e:
                # Other errors - don't retry
                self._log(f"❌ Error downloading {file.filename}: {e}")
                return False
        
        return False
    
    def _download_file_boto3(self, file: FlyLightFile, local_path: Path) -> bool:
        """Download a file using boto3."""
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self._s3_client.download_file(FLYLIGHT_BUCKET, file.key, str(local_path))
            return True
        except Exception as e:
            self._log(f"❌ Error downloading {file.filename}: {e}")
            return False
    
    def download_file(self, file: FlyLightFile, output_dir: Optional[str] = None, flat_structure: bool = False) -> Optional[Path]:
        """
        Download a single file.
        
        Args:
            file: FlyLightFile object to download
            output_dir: Override output directory (optional)
            flat_structure: If True, save to {line_name}/{filename} instead of preserving S3 key structure
            
        Returns:
            Path to downloaded file, or None if failed
        """
        output_path = Path(output_dir or self.output_dir)
        
        if flat_structure and file.line_name:
            # Save as {output_dir}/{line_name}/{filename}
            local_path = output_path / file.line_name / file.filename
        else:
            # Preserve directory structure
            local_path = output_path / file.key
        
        size_str = f"({file.size_mb:.1f} MB)" if file.size > 0 else ""
        self._log(f"⬇️  Downloading {file.filename} {size_str}...")
        
        # Use HTTP for VT files (source='http'), otherwise S3
        if file.source == 'http':
            success = self._download_file_http(file, local_path)
        elif self._s3_client:
            success = self._download_file_boto3(file, local_path)
        else:
            success = self._download_file_http(file, local_path)
        
        if success:
            self._log(f"   ✅ Saved: {local_path}")
            return local_path
        return None
    
    def download(
        self,
        line_name: Union[str, List[str]],
        output_dir: Optional[str] = None,
        max_files: Optional[int] = None,
        dry_run: bool = False,
        on_file_downloaded: Optional[Callable[[Path, str], None]] = None,
        flat_structure: bool = False,
        files: Optional[List] = None,
        generate_summary: Union[bool, str, List[str]] = None,
        summary_images_per_page: tuple = (3, 2),
        add_timestamp: bool = True,
        background_color: Union[str, tuple] = 'black',
    ) -> List[Path]:
        """
        Download files for driver line(s).
        
        Args:
            line_name: Driver line name(s). Can be:
                - Single string: 'R10A06'
                - Comma-separated: 'R10A06,VT037867'
                - List: ['R10A06', 'VT037867']
                With ``add_timestamp=True``, output folders are prefixed
                ``flylight-downloads_`` followed by the combined line name
                and timestamp.
            output_dir: Override output directory (optional)
            max_files: Maximum number of files to download **per line** (optional).
                When downloading multiple lines, this limit applies to each line separately.
            dry_run: If True, only list files without downloading
            on_file_downloaded: Optional callback called after each file download
                               Signature: callback(file_path, line_name)
            flat_structure: If True, save to {line_name}/{filename} instead of preserving S3 key structure
            files: Optional pre-filtered list of FlyLightFile objects. If provided, skips get_filtered_files() call.
            generate_summary: Generate image summary after download.
                Options: 'pdf', 'pptx', ['pdf', 'pptx'], False/None to disable
            summary_images_per_page: (columns, rows) for summary layout. Default: (5, 3)
            add_timestamp: If True, create a timestamped
                ``flylight-downloads_...`` output folder. Default: True
            background_color: Background color for summary slides/pages. Default: 'black'
            
        Returns:
            List of paths to downloaded files
        """
        # Normalize line_name input
        if isinstance(line_name, str):
            if ',' in line_name:
                line_names = [name.strip() for name in line_name.split(',')]
            else:
                line_names = [line_name]
        else:
            line_names = line_name
        
        # Create combined name for folder if multiple lines
        # Use _etc{N} format if more than 3 lines to avoid long names
        if len(line_names) == 1:
            combined_name = line_names[0]
        elif len(line_names) <= 3:
            combined_name = '_'.join(line_names)
        else:
            combined_name = '_'.join(line_names[:3]) + f'_etc{len(line_names)}'
        
        # Step protocol: the summary phase is optional (driven by
        # generate_summary), so the total varies between 3 and 4 steps.
        total_steps = 4 if generate_summary else 3
        self._progress(1, total_steps, "Resolving collections and image filters")
        
        if files is None:
            # Pass max_files as per-line limit
            self._progress(2, total_steps, "Listing and filtering FlyLight files")
            files = self.get_filtered_files(line_name, max_files_per_line=max_files)
        
        if not files:
            self._log(f"⚠️  No files found for '{combined_name}' matching filters")
            return []
        
        # Calculate total size
        total_size = sum(f.size for f in files)
        self._log(f"\n📦 {len(files)} files to download ({total_size / (1024*1024):.1f} MB total)")
        
        if dry_run:
            self._log("\n📋 Files (dry run - not downloading):")
            for f in files:
                self._log(f"   {f.filename} ({f.size_mb:.1f} MB)")
            return []
        
        # Create output path with optional timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if add_timestamp:
            output_path = Path(output_dir or self.output_dir) / f"flylight-downloads_{combined_name}_{timestamp}"
        else:
            output_path = Path(output_dir or self.output_dir)
        
        output_path.mkdir(parents=True, exist_ok=True)
        # The UI runner uses this marker to identify the per-run folder under
        # the user-selected output directory. Without it, the results panel
        # cannot distinguish this run from older sibling folders.
        self._log(f"📁 Output folder: {output_path}")
        downloaded = []
        self._progress(3, total_steps, "Downloading selected images")
        
        if self.max_workers > 1 and len(files) > 1:
            # Parallel download
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.download_file, f, str(output_path), flat_structure): f
                    for f in files
                }
                
                iterator = as_completed(futures)
                if self.verbose == 'pbar' and tqdm:
                    iterator = tqdm(iterator, total=len(files), desc=f"Downloading {combined_name}", unit="file")
                
                for future in iterator:
                    result = future.result()
                    if result:
                        downloaded.append(result)
                        if on_file_downloaded:
                            on_file_downloaded(result, combined_name)
        else:
            # Sequential download
            iterator = files
            if self.verbose == 'pbar' and tqdm:
                iterator = tqdm(files, desc=f"Downloading {combined_name}", unit="file")
                
            for f in iterator:
                result = self.download_file(f, str(output_path), flat_structure)
                if result:
                    downloaded.append(result)
                    if on_file_downloaded:
                        on_file_downloaded(result, combined_name)
        
        self._log(f"\n✅ Downloaded {len(downloaded)}/{len(files)} files")
        
        # Generate summary if requested
        if generate_summary and downloaded:
            self._progress(4, 4, "Generating summaries and saving metadata")
            self._generate_image_summary(
                downloaded_files=downloaded,
                output_dir=output_path,
                line_name=combined_name,
                timestamp=timestamp,
                generate_summary=generate_summary,
                images_per_page=summary_images_per_page,
                background_color=background_color
            )
        
        return downloaded
    
    def _generate_image_summary(
        self,
        downloaded_files: List[Path],
        output_dir: Path,
        line_name: str,
        timestamp: str,
        generate_summary: Union[bool, str, List[str]],
        images_per_page: tuple = (5, 3),
        background_color: Union[str, tuple] = 'black'
    ) -> None:
        """
        Generate PDF/PPTX image summary from downloaded files.
        
        Args:
            downloaded_files: List of downloaded file paths
            output_dir: Output directory for summary files
            line_name: Driver line name for summary title
            timestamp: Timestamp string for filename
            generate_summary: Format(s) to generate: 'pdf', 'pptx', ['pdf', 'pptx']
            images_per_page: (columns, rows) layout for summary
            background_color: Background color for summary. Default: 'black'
        """
        # Filter image files only
        image_files = [f for f in downloaded_files if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        
        if not image_files:
            self._log(f"⚠️  No image files to summarize")
            return
        
        self._log(f"\n📊 Generating image summary ({len(image_files)} images)...")
        
        # Normalize generate_summary to list
        if isinstance(generate_summary, str):
            formats = [generate_summary]
        elif isinstance(generate_summary, list):
            formats = generate_summary
        elif generate_summary is True:
            formats = ['pdf', 'pptx']
        else:
            return
        
        summary_title = f"{line_name} - {timestamp}"
        
        for fmt in formats:
            if fmt.lower() == 'pdf':
                try:
                    from neuronbridge_finder import create_image_pdf
                    pdf_path = output_dir / f"{line_name}_{timestamp}_summary.pdf"
                    create_image_pdf(
                        images_dir=str(output_dir),
                        output_pdf=str(pdf_path),
                        images_per_page=images_per_page,
                        title_font_size=24,
                        margin=0.3,
                        verbose=self.verbose,
                        background_color=background_color
                    )
                    self._log(f"   ✅ PDF summary: {pdf_path.name}")
                except ImportError:
                    self._log(f"   ⚠️  PDF generation requires neuronbridge_finder module")
                except Exception as e:
                    self._log(f"   ⚠️  PDF generation failed: {e}")
            
            elif fmt.lower() == 'pptx':
                try:
                    try:
                        from utils.report_utils import img2pptx
                    except ImportError:
                        # Fallback for when running directly or different path structure
                        from src.utils.report_utils import img2pptx
                        
                    pptx_path = output_dir / f"{line_name}_{timestamp}_summary.pptx"
                    # Use directory path with subfolder grouping to separate lines on different pages
                    # This matches the PDF behavior where each line starts on a new page
                    img2pptx(
                        input_path=str(output_dir),
                        output_pptx=str(pptx_path),
                        images_per_slide=images_per_page,
                        slide_title="{subfolder}",  # Use line name as title
                        label_fontsize=12,
                        title_fontsize=24,
                        include_subfolders=True,
                        group_by_subfolder=True,
                        background_color=background_color,
                        font_color=(255, 255, 255) if background_color in ['black', '#000000', (0,0,0)] else (0,0,0)
                    )
                    self._log(f"   ✅ PPTX summary: {pptx_path.name}")
                except ImportError as e:
                    self._log(f"   ⚠️  PPTX generation failed (ImportError): {e}")
                    self._log(f"      Please ensure 'python-pptx' is installed and src/utils/report_utils.py is accessible.")
                except Exception as e:
                    self._log(f"   ⚠️  PPTX generation failed: {e}")
    
    def get_metadata(self, line_name: str) -> List[Dict[str, Any]]:
        """
        Get metadata for a driver line without downloading full files.
        
        Args:
            line_name: Driver line name
            
        Returns:
            List of metadata dictionaries
        """
        # Temporarily override formats to get only JSON
        original_formats = self.formats
        original_types = self.image_types
        
        self.formats = ['json']
        self.image_types = ['metadata']
        
        files = self.get_filtered_files(line_name)
        
        # Restore original filters
        self.formats = original_formats
        self.image_types = original_types
        
        metadata_list = []
        
        for f in files:
            try:
                # Use proper headers for Janelia servers
                if 'janelia.org' in f.url:
                    req = _create_request(f.url)
                    with urllib.request.urlopen(req, timeout=30) as response:
                        data = json.loads(response.read())
                else:
                    with urllib.request.urlopen(f.url, timeout=30) as response:
                        data = json.loads(response.read())
                metadata_list.append(data)
            except Exception as e:
                self._log(f"⚠️  Error fetching metadata from {f.filename}: {e}")
        
        return metadata_list
    
    def search_lines(self, pattern: str) -> List[str]:
        """
        Search for driver lines matching a pattern.

        Args:
            pattern: Regex pattern to match line names

        Returns:
            List of matching line names
        """
        self._log(f"🔍 Searching for lines matching '{pattern}'...")

        matching_lines = set()
        regex = re.compile(pattern, re.IGNORECASE)
        collections_to_search = self._resolved_collections or FLYLIGHT_COLLECTIONS

        # Efficient path: line-name folders live at {collection}/{line}/ or
        # {collection}/CDM/{line}/ (Gen1 R-lines). Asking for CommonPrefixes
        # with the pattern as the path prefix returns every matching line
        # folder in ONE request (S3 prefix matching is literal, so this covers
        # the common prefix-style patterns such as 'R10A0' or 'SS01').
        def _collect(prefix: str) -> int:
            found = 0
            if self._s3_client:
                try:
                    result = self._s3_client.list_objects_v2(
                        Bucket=FLYLIGHT_BUCKET,
                        Prefix=prefix,
                        Delimiter='/'
                    )
                    for prefix_obj in result.get('CommonPrefixes', []):
                        leaf = prefix_obj['Prefix'].rstrip('/').split('/')[-1]
                        if regex.search(leaf):
                            matching_lines.add(leaf)
                            found += 1
                except Exception as e:
                    self._log(f"   ⚠️  Error: {e}")
            else:
                # HTTP fallback: walk the prefix and match parsed line names
                for f in self._list_bucket_http(prefix):
                    if f.line_name and regex.search(f.line_name):
                        matching_lines.add(f.line_name)
                        found += 1
            return found

        for collection in collections_to_search:
            self._log(f"   Searching {collection}...")
            _collect(f"{collection}/{pattern}")
            _collect(f"{collection}/CDM/{pattern}")

        # Regex patterns (e.g. 'R10A0[0-9]') cannot use the literal prefix
        # trick. Fall back to a bounded two-level walk for collections with a
        # small number of subfolders (e.g. Gen1 -> CDM -> line names); flat
        # collections with hundreds of level-1 folders are skipped to keep
        # the search cheap.
        if not matching_lines:
            self._log("   No prefix matches; walking shallow collections...")
            for collection in collections_to_search:
                if self._s3_client is None:
                    continue
                try:
                    level1 = self._s3_client.list_objects_v2(
                        Bucket=FLYLIGHT_BUCKET,
                        Prefix=f"{collection}/",
                        Delimiter='/'
                    ).get('CommonPrefixes', [])
                except Exception as e:
                    self._log(f"   ⚠️  Error: {e}")
                    continue
                if len(level1) > 30:
                    continue  # flat collection - lines live at level 1
                for sub in level1:
                    _collect(sub['Prefix'])

        result = sorted(matching_lines)
        self._log(f"   Found {len(result)} matching lines")
        return result


def download_flylight_images(
    line_name: str,
    output_dir: str = './flylight_downloads',
    formats: Union[str, List[str]] = 'png',
    image_types: Union[str, List[str]] = 'mip',
    max_files: Optional[int] = None,
    simple_mode: bool = False,
    verbose: Union[bool, str] = True
) -> List[Path]:
    """
    Convenience function to download FlyLight images for a driver line.
    
    Args:
        line_name: Driver line name (e.g., 'R10A06')
        output_dir: Directory to save files
        formats: File formats ('png', 'h5j', 'lsm', 'mp4', 'json', 'all')
        image_types: Image types ('mip', 'cdm', 'aligned', 'metadata', 'all')
        max_files: Maximum files to download
        simple_mode: Apply filename filtering to reduce download volume:
            - Split-GAL4: only '20x' AND 'multichannel' files, excluding 'image1'/'image2'
            - GAL4/LexA: only 'total' files
        verbose: Print progress (True, False, 'pbar', 'simple')
        
    Returns:
        List of paths to downloaded files
        
    Example:
        >>> download_flylight_images('R10A06', formats='png', image_types='cdm')
        >>> download_flylight_images('VT037867', formats=['png', 'json'], image_types='all')
        >>> download_flylight_images('SS01015', simple_mode=True, verbose='pbar')
    """
    downloader = FlyLightDownloader(
        output_dir=output_dir,
        formats=formats,
        image_types=image_types,
        simple_mode=simple_mode,
        verbose=verbose
    )
    
    return downloader.download(line_name, max_files=max_files)


def list_flylight_files(
    line_name: str,
    formats: Union[str, List[str]] = 'all',
    image_types: Union[str, List[str]] = 'all',
    simple_mode: bool = False,
    verbose: Union[bool, str] = True
) -> List[FlyLightFile]:
    """
    List available FlyLight files for a driver line without downloading.
    
    Args:
        line_name: Driver line name
        formats: File formats to filter
        image_types: Image types to filter
        simple_mode: Apply filename filtering to reduce file list:
            - Split-GAL4: only '20x' AND 'multichannel' files, excluding 'image1'/'image2'
            - GAL4/LexA: only 'total' files
        verbose: Print progress (True, False, 'pbar', 'simple')
        
    Returns:
        List of FlyLightFile objects
    """
    downloader = FlyLightDownloader(
        formats=formats,
        image_types=image_types,
        simple_mode=simple_mode,
        verbose=verbose
    )
    
    return downloader.get_filtered_files(line_name)

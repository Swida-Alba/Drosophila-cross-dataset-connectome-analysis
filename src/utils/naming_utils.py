"""
Shared output-folder naming helpers for DROCAT.

Every main function creates one top-level, timestamped run folder under the
user's output directory, named:

    {tool}_{dataset_abbreviation}_{detail}_{timestamp}

Examples:
    find-paths-complete_MCNS_aMe12_to_aMe10_L2w3_20260801_183000
    finddirect_MCNS_aMe12_to_aMe10_L2w3r0p0_20260801_183005
    profiling_MCNS_aMe12_aMe10_aMe9_20260801_183010
    homologs_MCNS_to_HEMI_aMe12_20260801_183015
    similar-morphology_MCNS_aMe12_20260801_183018
    similar-connectivity_MCNS_to_HEMI_aMe12_20260801_183019
    NB-find-lines_MCNS_aMe12_20260801_183020
    plot-3d_MCNS_aMe12_20260801_183025
"""

from collections import Counter
import json
import re
from functools import lru_cache
from pathlib import Path


DATASET_ABBREVIATIONS = {
    "male-cns": "MCNS",
    "male_cns": "MCNS",
    "hemibrain": "HEMI",
    "optic-lobe": "OL",
    "optic_lobe": "OL",
    "manc": "MANC",
    "banc": "BANC",
    "fib19": "FIB",
    "mushroombody": "MB",
    "flywire_fafb": "FAFB",
    "fafb": "FAFB",
    "flywire_banc": "BANC",
    # bare flywire identifiers refer to the FAFB dataset in DROCAT
    "flywire": "FAFB",
}


_DATASET_VERSION_SUFFIX_RE = re.compile(
    r"(?:^|[:_\-\s])v?(\d+(?:[._]\d+)*)$",
    re.IGNORECASE,
)

# Legacy BANC identifiers carried the ``flywire_`` prefix while BANC was
# handled as a FlyWire release.  BANC is now analyzed from its own public
# release data (not through FlyWire), so the canonical names drop the
# prefix; the legacy spellings stay accepted as aliases everywhere a
# dataset name enters the app.
_BANC_LEGACY_NAME_RE = re.compile(
    # Keep the hidden NeuPrint spelling ``banc:v888`` intact; only legacy
    # FlyWire-prefixed colon forms (and canonical underscore forms) are
    # local-release identifiers that should fold into a cache namespace.
    r"^(?:(?:flywire[_-]?)banc(?:[_:-](v\d+(?:[._]\d+)*))?|"
    r"banc(?:_(v\d+(?:[._]\d+)*))?)$",
    re.IGNORECASE,
)


def canonical_dataset_name(dataset) -> str:
    """Return the canonical dataset identifier for *dataset*.

    Legacy ``flywire_BANC_v626``-style names map to ``banc_v626`` /``banc_v888``
    (per release).  Bare ``flywire_BANC`` / ``banc`` pin to the historical
    default BANC release ``banc_v626`` — the same pin the cross-dataset type
    mapper applies — so the unversioned alias can never straddle the two
    BANC releases (v626 and v888 are distinct datasets with distinct id
    spaces).  Every other identifier — including the hidden NeuPrint
    ``banc:v888`` colon form and the FAFB release — passes through unchanged.
    """
    text = str(dataset or "").strip()
    match = _BANC_LEGACY_NAME_RE.match(text)
    if match:
        version = next((group for group in match.groups() if group), None)
        return f"banc_{version.lower()}" if version else "banc_v626"
    return text


@lru_cache(maxsize=1024)
def dataset_version(dataset) -> str | None:
    """Return a normalized version token from a dataset identifier.

    Examples:
        ``male-cns:v1.0`` -> ``v1.0``
        ``male_cns_v0_9`` -> ``v0.9``
        ``flywire_BANC_v888`` -> ``v888``

    Dataset versions are intentionally extracted from the original identifier;
    callers can therefore distinguish releases that share a family abbreviation.

    Memoized: the type mapper calls this per dataset-name lookup, and one
    expanded search touched it ~1M times for ~a dozen unique names
    (2026-09-10 profile).
    """
    if not dataset:
        return None

    match = _DATASET_VERSION_SUFFIX_RE.search(str(dataset).strip())
    if not match:
        return None
    return f"v{match.group(1).replace('_', '.')}"


def make_unique_dataset_labels(datasets, labels=None) -> list[str]:
    """Make display labels unique without discarding dataset identity.

    The normal label remains unchanged when it is unique.  When two selected
    datasets share the same family label (for example ``MCNS`` or ``BANC``),
    their version is appended using a filename-safe separator:
    ``MCNS_v1_0`` and ``MCNS_v0_9``.

    ``datasets`` may contain strings or objects exposing a ``dataset``
    attribute.  ``labels`` is optional and defaults to :func:`dataset_abbrev`.
    """
    dataset_names = [
        getattr(dataset, "dataset", str(dataset))
        for dataset in (datasets or [])
    ]
    base_labels = []
    for index, dataset_name in enumerate(dataset_names):
        label = labels[index] if labels is not None and index < len(labels) else None
        label = str(label).strip() if label is not None else ""
        base_labels.append(label or dataset_abbrev(dataset_name))

    counts = Counter(label.casefold() for label in base_labels)
    result = []
    used = set()

    for index, (dataset_name, base_label) in enumerate(zip(dataset_names, base_labels)):
        candidate = base_label
        if counts[base_label.casefold()] > 1:
            version = dataset_version(dataset_name)
            if version:
                candidate = f"{base_label}_{version.replace('.', '_')}"
            else:
                candidate = f"{base_label}_{index + 1}"

        # Protect against duplicate release identifiers or user-provided labels
        # that still collide after the version suffix is added.
        stem = candidate
        suffix = 2
        while candidate.casefold() in used:
            candidate = f"{stem}_{suffix}"
            suffix += 1

        result.append(candidate)
        used.add(candidate.casefold())

    return result


def dataset_abbrev(dataset) -> str:
    """Return a short, folder-safe abbreviation for a dataset identifier."""
    if not dataset:
        return "UNKN"
    ds = str(dataset).lower()
    for key, abbrev in DATASET_ABBREVIATIONS.items():
        if key in ds:
            return abbrev
    letters = "".join(c for c in ds.split(":")[0] if c.isalpha())
    return (letters[:4] or "DS").upper()


# --------------------------------------------------------------------------
# Brain-mesh selection tokens shared by the renderer and the UI.
#
# 'native' renders in the dataset's own template space; 'BANC' / 'FAFB' /
# 'male-cns' move the whole scene into that template's coordinates and draw
# its outline; 'none' hides the outline.
BRAIN_MESH_OPTIONS = ["native", "BANC", "FAFB", "male-cns", "none"]

# Case restoration for the two capitalized options: they are lowercase words
# in most inputs (a CLI flag, a saved setting, a hand-typed value) and the
# option list is exact-match. The retired 'template'/'whole'/'mcns' spellings
# are deliberately absent -- they now fail validation like any unknown token.
_BRAIN_MESH_CASE_FOLDS = {
    "fafb": "FAFB",
    "banc": "BANC",
}


def normalize_brain_mesh_choice(value) -> str:
    """Lowercase a brain-mesh choice and restore the capitalized spellings."""
    v = str(value or "").strip().lower()
    return _BRAIN_MESH_CASE_FOLDS.get(v, v)


_HEMI_SUFFIXES = ('_L', '_R', '_U')


def split_hemi_suffix(label) -> tuple:
    """Split a hemisphere suffix (_L/_R/_U) from a label.

    Returns ``(base, suffix)`` where suffix includes the leading
    underscore; non-string inputs return ``(label, '')``. Mirrors
    ``comparison.label_mapper.LabelMapper._split_hemi_suffix`` — the
    canonical shared version (plan R1-a).
    """
    if not isinstance(label, str):
        return label, ''
    for suffix in _HEMI_SUFFIXES:
        if label.endswith(suffix):
            return label[:-len(suffix)], suffix
    return label, ''


# --------------------------------------------------------------------------
# Per-run output-folder classification (Settings → Storage + ui/runner).
#
# Every main function creates one timestamped run folder named
# ``{tool}_{dataset_abbrev}_{detail}_{YYYYMMDD_HHMMSS}`` (see the module
# docstring).  This table is the single source of truth for "does this
# folder name look like a tool run folder": ui/runner's scan-dir heuristic
# and the storage utility's run-folder classifier both derive from it, so
# the two can never drift apart.
RUN_FOLDER_PREFIXES = (
    # Current tool prefixes (hyphenated / multi-word spellings).
    "find-paths-complete",
    "find-paths-shortest",
    "find-network",
    "cross-dataset",
    "type-map-validation",
    "plot-3d",
    "plot-network",
    "homologs",
    "similar-morphology",
    "similar-connectivity",
    "similar",
    "profiling",
    "morphology_comparison",
    "morph_cross",
    # 'expanded' must precede the bare prefix: the regex anchors an
    # underscore directly after the alternative, so 'NB-find-lines' alone
    # never matched 'NB-find-lines-expanded_...' names.
    "NB-find-lines-expanded",
    "NB-find-lines",
    "NB-find-neurons",
    "NB-colabeling",
    # Legacy pre-reorg one-word spellings still found in old output roots
    # (and the historical 'flylignt' typo, kept so those folders classify).
    "flylight-downloads",
    "flylignt-downloads",
    "findpath",
    "findallpath",
    "findshortestpath",
    "findnetwork",
    "finddirect",
    "findhomologs",
    "interdataset",
    "plot3d",
    "plotpath",
    "colabel",
    "findlines",
    "findneuron",
    "findsimilar",
)

RUN_FOLDER_PREFIX_RE = re.compile(
    r"^(?:" + "|".join(RUN_FOLDER_PREFIXES) + r")_"
)

_RUN_FOLDER_TIMESTAMP_RE = re.compile(r"_(\d{8}_\d{6})$")


def is_run_folder_name(name) -> bool:
    """True when *name* is a per-run output folder name.

    A run folder = known tool prefix AND the embedded
    ``_YYYYMMDD_HHMMSS`` timestamp.  The prefix alone is not sufficient:
    shared roots such as ``morph_cross_dataset/`` also match the
    ``morph_cross`` prefix but hold many runs, so callers descend into
    them instead of treating them as single runs.
    """
    text = str(name or "")
    if not RUN_FOLDER_PREFIX_RE.match(text):
        return False
    return bool(_RUN_FOLDER_TIMESTAMP_RE.search(text))


def run_folder_timestamp(name) -> str:
    """Return the embedded ``YYYYMMDD_HHMMSS`` stamp, or '' when absent."""
    match = _RUN_FOLDER_TIMESTAMP_RE.search(str(name or ""))
    return match.group(1) if match else ""


def dataset_key_candidates(dataset) -> set:
    """A dataset's own name plus the safe-name variant used as a mapping key
    (colons and dots replaced by underscores)."""
    return {dataset,
            canonical_dataset_name(dataset).replace(':', '_').replace('.', '_')}


def group_member_rows(rows):
    """One preset group: strings become single-member lists, digit-looking
    identifiers become ints, everything else stays a stripped string."""
    if isinstance(rows, str):
        rows = [rows]
    processed = []
    for value in (rows or []):
        text = str(value).strip()
        processed.append(int(text) if text.isdigit() else text)
    return processed


def load_labelmapper_source_groups(mapping_path, dataset, log=None):
    """Read the source-side groups of a LabelMapper preset JSON.

    Preset format (the overall JSON exported by the Settings tab's mapping
    presets)::

        {"source_mapping": {
            "custom_label": ["grp1", "grp2"],
            "hemibrain:v1.2.1": [["aMe12", "aMe12_R"], ["aMe12_L"]]}}

    Each ``custom_label``/``std_label`` row is one group whose members are the
    identifiers listed under the requested dataset; a dataset may appear under
    its own name or its safe-name variant. Groups with no members in that
    dataset are dropped.

    Args:
        mapping_path: Path to the LabelMapper preset JSON.
        dataset: The dataset whose column supplies the members.
        log: Optional callable for the progress/warning lines.

    Returns:
        ``(groups, group_names, side)`` — ``groups`` is a list parallel to
        ``group_names``, each group a list of bodyIds/type names (the same
        contract as the profiling group-map CSV); ``side`` is the raw
        ``source_mapping`` dict, so callers that need other datasets' columns
        do not re-read the file. Unreadable or invalid JSON raises ValueError;
        an absent dataset column yields ``([], [], side)`` rather than raising,
        because both comparers treat "no groups here" as a warn-and-continue.
    """
    say = log if callable(log) else (lambda _msg: None)

    try:
        data = json.loads(Path(mapping_path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Could not read custom mapping file {mapping_path}: {exc}")

    side = data.get('source_mapping') or {}
    labels = side.get('custom_label') or side.get('std_label') or []
    if not labels:
        say("Warning: mapping file has no source groups "
            "(custom_label/std_label) — custom groups skipped")
        return [], [], side

    candidates = dataset_key_candidates(dataset)
    ds_key = next(
        (k for k in side
         if k not in ('custom_label', 'std_label') and k in candidates),
        None
    )
    if ds_key is None:
        say(f"Warning: mapping file has no groups for dataset "
            f"'{dataset}' — custom groups skipped")
        return [], [], side

    groups = []
    group_names = []
    group_rows = side.get(ds_key) or []
    for index, label in enumerate(labels):
        processed = group_member_rows(
            group_rows[index] if index < len(group_rows) else None)
        if not processed:
            continue
        group_names.append(str(label))
        groups.append(processed)

    if not group_names:
        say(f"Warning: no groups with members in '{dataset}' "
            f"found in mapping file")
        return [], [], side

    say(f"Loaded {len(group_names)} custom groups from mapping file:")
    for name, ids in zip(group_names, groups):
        say(f"  {name}: {len(ids)} identifiers")

    return groups, group_names, side

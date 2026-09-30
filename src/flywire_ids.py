"""Canonical FAFB/BANC body-ID handling.

FlyWire root/body IDs are identifiers, not measurements.  They are commonly
larger than JavaScript's safe-integer range, so DROCAT keeps them as exact
decimal strings everywhere inside the application.  The CAVE/CloudVolume
adapters are the only consumers that receive integers, and they use
``body_id_to_api_int`` at that narrow boundary.
"""

from __future__ import annotations

import math
import numbers
import re
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    from .utils.naming_utils import canonical_dataset_name
except ImportError:  # pragma: no cover - src laid bare on sys.path
    from utils.naming_utils import canonical_dataset_name


JS_SAFE_INTEGER = 2**53 - 1
SIGNED_INT64_MAX = 2**63 - 1

# FlyWire/FAFB body ids are "720575940" + a 9-digit short id.  A bare 9-digit
# id is the truncated short form and must be completed to its full form before
# it can be matched against already-full ids in another frame.
FLYWIRE_SHORT_ID_PREFIX = "720575940"
SHORT_ID_DIGITS = 9

_DIGITS = re.compile(r"^[0-9]+$")
_INTEGRAL_DECIMAL = re.compile(r"^([0-9]+)\.0+$")


class FlyWireBodyIdError(ValueError):
    """Raised when a FAFB/BANC body ID cannot be preserved exactly."""


def is_banc_dataset(dataset: object) -> bool:
    """Return whether *dataset* identifies a BANC release."""

    return "banc" in str(dataset or "").strip().lower()


def is_fafb_dataset(dataset: object) -> bool:
    """Return whether *dataset* identifies a FAFB release.

    FAFB is matched by the ``fafb`` token (the real releases are spelled
    ``flywire_FAFB_v783``), or by the bare aliases ``flywire`` / ``fafb``.
    A generic ``flywire_*`` prefix is *not* enough: the prefix is historical
    and standalone BANC once carried it, so it is explicitly excluded even
    for a legacy ``flywire_BANC_*`` alias.  Callers that need either local
    release must use :func:`is_local_connectome_dataset`.
    """

    normalized = canonical_dataset_name(str(dataset or "").strip()).lower()
    if is_banc_dataset(normalized):
        return False
    return normalized in {"flywire", "fafb"} or "fafb" in normalized


def is_flywire_dataset(dataset: object) -> bool:
    """Return whether *dataset* is the FAFB release.

    The function name is retained for API compatibility.  It is no longer a
    family predicate: standalone BANC identifiers return ``False`` and
    callers that need either local release must use
    :func:`is_local_connectome_dataset`.
    """

    return is_fafb_dataset(dataset)


def is_local_connectome_dataset(dataset: object) -> bool:
    """Return whether *dataset* is a local FAFB or standalone BANC release."""

    return is_fafb_dataset(dataset) or is_banc_dataset(dataset)


def dataset_folder(dataset: object) -> str:
    """Map a dataset identifier to the repository folder convention.

    Legacy BANC identifiers canonicalize first, so the legacy
    ``flywire_BANC_v888`` resolves to the same folder as ``banc_v888``
    in ``datasets/`` and ``cache/``.
    """

    return canonical_dataset_name(dataset).replace(":", "_").replace(".", "_")


_NEUPRINT_FOLDER_NAME_RE = re.compile(r"^(.+?)_v(\d+(?:_\d+)*)$")


def neuprint_dataset_name(dataset: object) -> str:
    """Map a dataset identifier to the spelling the NeuPrint server accepts.

    ``dataset_folder`` is one-way: a NeuPrint release lives under
    ``hemibrain_v1_2_1`` on disk but only ``hemibrain:v1.2.1`` exists on the
    server, and the underscore form resolves happily against the local tables,
    profile caches and folder names — so a run launched with it loses every
    remote fetch while nothing local complains.  Use this ONLY at the NeuPrint
    boundary: FAFB and standalone BANC are analyzed from local releases and
    must keep their own identifiers.
    """

    text = canonical_dataset_name(dataset).strip()
    if ":" in text or is_local_connectome_dataset(text):
        return text
    match = _NEUPRINT_FOLDER_NAME_RE.match(text)
    if not match:
        return text
    return f"{match.group(1).replace('_', '-')}:v{match.group(2).replace('_', '.')}"


def resolve_flywire_dataset_dir(
    project_root: str | Path, dataset: object
) -> Optional[Path]:
    """Resolve an existing FlyWire dataset directory without FAFB fallback.

    The exact name is tried first, followed by the repository-safe spelling
    used for identifiers containing ``:`` or ``.``.  In particular, a missing
    BANC directory never silently resolves to the FAFB directory.
    """

    root = Path(project_root) / "datasets"
    raw_name = str(dataset or "").strip()
    candidates = []
    for name in (raw_name, dataset_folder(raw_name)):
        if name and name not in candidates:
            candidates.append(name)
    for name in candidates:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


def _canonical_digits(digits: str) -> str:
    """Canonicalize decimal digits while preserving their exact integer."""

    canonical = digits.lstrip("0")
    return canonical or "0"


def _complete_short_id(text: str) -> str:
    """Complete a 9-digit FlyWire short id to its full ``720575940`` + id form.

    Ids are exact decimal strings; a 9-digit value is the short-form suffix of
    a FlyWire body id, so the fixed namespace prefix is prepended.  Longer ids
    (already full) are returned unchanged after canonicalization.  This unifies
    the short-form handling used by the FAFB synapse and BANC connection
    converters with the canonical ``normalize_flywire_body_id`` path.
    """

    if _DIGITS.fullmatch(text) and len(text) == SHORT_ID_DIGITS:
        return FLYWIRE_SHORT_ID_PREFIX + text
    return _canonical_digits(text)


def normalize_flywire_body_id(value: Any, *, field: str = "bodyId") -> str:
    """Return an exact canonical decimal string for one FlyWire body ID.

    Integral Python/numpy integers are safe.  Small integral floats and
    legacy strings such as ``"123.0"`` are accepted.  A float-like value
    above JavaScript's safe-integer limit is rejected because its original
    digits may already have been rounded and cannot be recovered.
    """

    if value is None or isinstance(value, bool):
        raise FlyWireBodyIdError(f"{field} is missing or boolean: {value!r}")

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise FlyWireBodyIdError(f"{field} is empty")
        if _DIGITS.fullmatch(text):
            return _complete_short_id(text)
        match = _INTEGRAL_DECIMAL.fullmatch(text)
        if match:
            digits = _canonical_digits(match.group(1))
            if int(digits) > JS_SAFE_INTEGER:
                raise FlyWireBodyIdError(
                    f"{field} looks like a rounded float ({value!r}); "
                    "the exact FlyWire ID cannot be recovered"
                )
            return digits
        raise FlyWireBodyIdError(
            f"{field} must be an exact decimal integer string: {value!r}"
        )

    if isinstance(value, numbers.Integral):
        integer = int(value)
    elif isinstance(value, Decimal):
        if not value.is_finite() or value != value.to_integral_value():
            raise FlyWireBodyIdError(f"{field} is not integral: {value!r}")
        integer = int(value)
    elif isinstance(value, numbers.Real):
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise FlyWireBodyIdError(f"{field} is not a finite integer: {value!r}")
        if abs(numeric) > JS_SAFE_INTEGER:
            raise FlyWireBodyIdError(
                f"{field} arrived as an unsafe float ({value!r}); "
                "read FlyWire IDs as strings before parsing"
            )
        integer = int(numeric)
    else:
        # Decimal-like third-party values occasionally expose an exact string
        # representation.  Do not accept scientific notation or arbitrary
        # coercions here; identifiers must remain lossless.
        try:
            text = str(value).strip()
        except Exception as exc:  # pragma: no cover - defensive boundary
            raise FlyWireBodyIdError(f"invalid {field}: {value!r}") from exc
        if _DIGITS.fullmatch(text):
            return _complete_short_id(text)
        raise FlyWireBodyIdError(f"invalid {field}: {value!r}")

    if integer < 0:
        raise FlyWireBodyIdError(f"{field} must be non-negative: {value!r}")
    digits = str(integer)
    if len(digits) == SHORT_ID_DIGITS:
        digits = FLYWIRE_SHORT_ID_PREFIX + digits
    return digits


def normalize_flywire_body_ids(values: Iterable[Any], *, field: str = "bodyId") -> list[str]:
    """Normalize an iterable of FlyWire body IDs to canonical strings."""

    return [normalize_flywire_body_id(value, field=field) for value in values]


def body_id_to_api_int(value: Any, *, field: str = "bodyId") -> int:
    """Convert one canonical FlyWire ID to the checked API integer form."""

    canonical = normalize_flywire_body_id(value, field=field)
    integer = int(canonical)
    if integer > SIGNED_INT64_MAX:
        raise FlyWireBodyIdError(
            f"{field} exceeds the signed 64-bit API range: {canonical}"
        )
    return integer


def normalize_flywire_id_columns(frame, columns: Iterable[str]):
    """Normalize present identifier columns in a pandas-like frame in place."""

    for column in columns:
        if column in frame.columns:
            frame[column] = frame[column].map(
                lambda value, _column=column: normalize_flywire_body_id(
                    value, field=_column
                )
            ).astype("string")
    return frame


def canonicalize_flywire_id_expr(column: str):
    """Polars column expression mirroring :func:`normalize_flywire_body_id`.

    Completes a 9-digit short id to its full ``720575940`` + id form, then
    canonicalizes (strip, drop integral ``.0``, strip leading zeros) exactly
    like the rest of DROCAT.  Used by the FAFB synapse and BANC connection
    converters so the short-form rule lives in one place.
    """

    import polars as pl

    # Strip FIRST (a padded '123456789 ' must not miss the completion,
    # an embedded-space ' 12345678' must not become an invalid id), and
    # gate the short-id completion on pure digits — mirroring
    # normalize_flywire_body_id's strip-then-fullmatch order.
    s = pl.col(column).cast(pl.Utf8).str.strip_chars()
    s = pl.when(s.str.contains(r"^\d+$") & (s.str.len_chars() == SHORT_ID_DIGITS)).then(
        pl.concat_str(pl.lit(FLYWIRE_SHORT_ID_PREFIX), s)
    ).otherwise(s)
    s = s.str.replace(r"^([0-9]+)\.0+$", "${1}")
    s = s.str.strip_chars_start("0")
    return pl.when(s.str.len_chars() == 0).then(pl.lit("0")).otherwise(s)

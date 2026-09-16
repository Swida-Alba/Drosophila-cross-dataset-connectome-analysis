"""Unified morph-qualification bars (floors v3).

One bar engine for every morph-admission decision, shared by the TM VEV
validation pipeline and the Find Homolog qualification option (plan:
``_plan/plan-unified-morph-qualification-bars.md``).

Currencies
----------
- **native** (Track B): similarities inside the target dataset — the branch's
  matched+verified reference pool.  The native floor is BINDING whenever the
  branch has >= 2 reference neurons with native vectors.
- **Track A** (transformed query): the source neuron transformed into the
  target render space.  Two Track-A levels exist on one dial:

  - candidate bar  = ``pool_track_a_baseline - track_a_offset``  (B_b - Δ)
  - suspicious bar = ``pool_track_a_baseline - suspicious_level * Δ``
    (aggressive-mode deep window only)

  with the run null (``null_bar`` / ``null_bar_lo``) as the fallback basis when
  a branch has no scored pool pairs, and an optional ``random_null_floor``
  clamp so a thin branch can never admit chance-level similarity.

Kinds
-----
A :class:`BarSet` carries the *kind* of its candidate bar — ``'native'``,
``'track_a_backup'`` or ``'null'`` — and of its suspicious bar
(``'track_a_suspicious'`` / ``'null_lo'``).  Callers must compare the matching
currency: native bars compare ``morph_pool_ref``, every Track-A bar compares
``morph_v2_similarity``.  :func:`candidate_qualified` /
:func:`suspicious_qualified` do that pairing so the CSV and the scene cannot
disagree.
"""
from dataclasses import dataclass
from typing import Optional

CANDIDATE_NATIVE = 'native'
CANDIDATE_BACKUP = 'track_a_backup'
CANDIDATE_NULL = 'null'
SUSPICIOUS_TRACK_A = 'track_a_suspicious'
SUSPICIOUS_NULL_LO = 'null_lo'


@dataclass
class BarSet:
    """Per-branch morph-admission bars (all LOWER bounds on a score)."""

    candidate_kind: str = CANDIDATE_NULL
    native_floor: Optional[float] = None
    backup_floor: Optional[float] = None
    null_bar: Optional[float] = None
    suspicious_kind: str = SUSPICIOUS_NULL_LO
    suspicious_floor: Optional[float] = None
    # provenance / audit
    pool_track_a_baseline: Optional[float] = None   # B_b
    n_native_refs: int = 0
    n_scored_pool: int = 0
    track_a_offset: float = 0.05
    native_margin: float = 0.05
    suspicious_level: int = 3
    random_null_floor: Optional[float] = None
    clamp_applied: bool = False

    def candidate_bar_value(self) -> Optional[float]:
        return {'native': self.native_floor,
                'track_a_backup': self.backup_floor,
                'null': self.null_bar}.get(self.candidate_kind)


def _clamp(value: Optional[float],
           random_null_floor: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if random_null_floor is not None and value < random_null_floor:
        return float(random_null_floor)
    return float(value)


def compute_branch_bars(
        ref_pool_sim_mean: Optional[float] = None,
        pool_track_a_baseline: Optional[float] = None,
        n_native_refs: int = 0,
        n_scored_pool: int = 0,
        null_bar: Optional[float] = None,
        null_bar_lo: Optional[float] = None,
        random_null_floor: Optional[float] = None,
        track_a_offset: float = 0.05,
        native_margin: float = 0.05,
        suspicious_level: int = 3,
) -> BarSet:
    """Compute one branch's admission bars.

    ``ref_pool_sim_mean`` is the mean pairwise NATIVE similarity among the
    branch's matched+verified reference neurons; it only yields a native floor
    when ``n_native_refs >= 2`` (a self-similarity baseline needs at least a
    pair).  ``pool_track_a_baseline`` (B_b) is the mean Track-A morph of the
    branch's scored m+v pool pairs and backs the backup floor when at least one
    pair was scored.  ``null_bar`` (run null p95) and ``null_bar_lo``
    (run null p50) take over when neither exists.
    """
    native_floor = None
    if ref_pool_sim_mean is not None and n_native_refs >= 2:
        native_floor = _clamp(ref_pool_sim_mean - native_margin,
                              random_null_floor)

    backup_floor = None
    if native_floor is None and pool_track_a_baseline is not None \
            and n_scored_pool >= 1:
        backup_floor = _clamp(pool_track_a_baseline - track_a_offset,
                              random_null_floor)

    if native_floor is not None:
        candidate_kind = CANDIDATE_NATIVE
    elif backup_floor is not None:
        candidate_kind = CANDIDATE_BACKUP
    else:
        candidate_kind = CANDIDATE_NULL

    suspicious_floor = None
    suspicious_kind = SUSPICIOUS_NULL_LO
    if pool_track_a_baseline is not None and n_scored_pool >= 1:
        suspicious_floor = _clamp(
            pool_track_a_baseline - suspicious_level * track_a_offset,
            random_null_floor)
        suspicious_kind = SUSPICIOUS_TRACK_A
    elif null_bar_lo is not None:
        suspicious_floor = _clamp(null_bar_lo, random_null_floor)

    clamp_applied = random_null_floor is not None and (
        (native_floor is not None and native_floor == random_null_floor)
        or (backup_floor is not None and backup_floor == random_null_floor)
        or (suspicious_floor is not None
            and suspicious_floor == random_null_floor))

    return BarSet(
        candidate_kind=candidate_kind,
        native_floor=native_floor,
        backup_floor=backup_floor,
        null_bar=null_bar,
        suspicious_kind=suspicious_kind,
        suspicious_floor=suspicious_floor,
        pool_track_a_baseline=pool_track_a_baseline,
        n_native_refs=n_native_refs,
        n_scored_pool=n_scored_pool,
        track_a_offset=track_a_offset,
        native_margin=native_margin,
        suspicious_level=suspicious_level,
        random_null_floor=random_null_floor,
        clamp_applied=clamp_applied,
    )


def _finite(v) -> bool:
    return v is not None and not (isinstance(v, float) and v != v)


def candidate_qualified(bars: BarSet, pool_ref, track_a) -> bool:
    """Candidate-bar admission; compares the currency the bar kind names."""
    if bars.candidate_kind == CANDIDATE_NATIVE:
        return _finite(pool_ref) and pool_ref >= bars.native_floor
    if bars.candidate_kind == CANDIDATE_BACKUP:
        return _finite(track_a) and track_a >= bars.backup_floor
    return _finite(track_a) and _finite(bars.null_bar) \
        and track_a >= bars.null_bar


def suspicious_qualified(bars: BarSet, track_a) -> bool:
    """Suspicious-bar admission (aggressive deep window); Track-A currency."""
    if bars.suspicious_floor is None:
        return False
    return _finite(track_a) and track_a >= bars.suspicious_floor


def synthesize_barset(native_floor: Optional[float] = None,
                      track_a_bar: Optional[float] = None,
                      null_bar: Optional[float] = None) -> BarSet:
    """Floors-v2 compatibility adapter: build a BarSet from the legacy
    (floor, threshold, null_bar) triple with the v2 precedence — the
    native floor binds when present, else the null bar, else the
    factor-x-mean threshold.  Used by fixtures and any caller still
    holding v2 fields; production runs build BarSets via
    :func:`compute_branch_bars` instead."""
    bars = BarSet(null_bar=null_bar)
    if native_floor is not None:
        bars.candidate_kind = CANDIDATE_NATIVE
        bars.native_floor = native_floor
    elif null_bar is not None:
        bars.candidate_kind = CANDIDATE_NULL
    elif track_a_bar is not None:
        bars.candidate_kind = CANDIDATE_BACKUP
        bars.backup_floor = track_a_bar
    else:
        bars.candidate_kind = CANDIDATE_NULL
    return bars

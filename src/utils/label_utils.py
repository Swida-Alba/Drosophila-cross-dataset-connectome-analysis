"""
Shared neuron-label predicates for DROCAT.

This module is the single source of truth for the "untyped neuron"
definition used by pathfinding (``FindNeuronConnection(drop_untyped=...)``)
and Cross-Dataset Comparison (``ComparisonParameters.drop_untyped``).
Both surfaces must agree on which resolved type labels count as untyped so
a neuron dropped in one tool would be dropped in the other.
"""


class UntypedLabelPolicy:
    """Single owner of the untyped sentinel vocabulary (plan-untyped-labels-and-drop-hardening §4.1): every surface routes its untyped test
through this policy so the spellings can never drift apart again."""

    SENTINELS = frozenset({'unknown', 'nan', 'none', 'null',
                           '<na>', '<null>'})

    @classmethod
    def is_untyped(cls, value) -> bool:
        """True when a resolved type label means 'untyped' (the
        :func:`is_untyped_type_label` contract)."""
        s = str(value).strip()

        def _fallback(text: str) -> bool:
            return text.isdigit() or (
                text.lower().startswith('hb') and text[2:].isdigit())

        while True:
            if not s or s.lower() in cls.SENTINELS or _fallback(s):
                return True
            if s.endswith(("_L", "_R", "_U")):
                s = s[:-2].strip()
                continue
            if s.startswith("(") and s.endswith(")"):
                s = s[1:-1].strip()
                continue
            return False

    @classmethod
    def normalize(cls, value) -> str:
        """'' for an untyped label, else the stripped label."""
        return '' if cls.is_untyped(value) else str(value).strip()


def is_untyped_type_label(value) -> bool:
    """True when a resolved type label means 'untyped'.

    A label is untyped when it is
    - empty after whitespace stripping,
    - one of the explicit sentinel strings ``Unknown`` / ``None`` / ``NaN``
      / ``Null`` / ``<NA>`` / ``<null>`` (case-insensitive), or
    - the numeric bodyId fallback (the neuron's own id used as its type
      when no name resolved — an all-digit string),
    - or a hemisphere-suffixed form of any of the above
      (``Unknown_L`` / ``None_R`` / ``nan_U`` …): ``separate_hemispheres``
      suffixes type labels *before* the untyped filter runs, so the
      predicate must look through the suffix to keep hemi runs dropping
      the same edges plain runs drop.

    Accepts any scalar; values are stringified first, so pandas ``NaN``
    floats stringify to ``'nan'`` and are caught by the sentinel branch.
    The ``null``/``<na>``/``<null>`` spellings match the viewer's display
    normalization (``neuron_search._display_value``) so a label hidden as
    missing there is never kept as typed here.
    """
    return UntypedLabelPolicy.is_untyped(value)


def untyped_side(pre_untyped: bool, post_untyped: bool) -> str:
    """Record flag naming which side(s) of an edge are untyped."""
    if pre_untyped and post_untyped:
        return 'pre+post'
    if pre_untyped:
        return 'pre'
    if post_untyped:
        return 'post'
    return ''


# ---------------------------------------------------------------------------
# BodyId display labels
#
# The canonical bodyId label rule is ``VisualizeSkeleton._tree_neuron_label``
# (visualization legend): NeuPrint-style datasets identify a neuron by its
# instance name, local FAFB/BANC releases by type + hemisphere. The helpers
# below apply the same rule to bodyId-level exports (similarity matrices,
# heatmaps, reports) so a row never reads as a bare numeric id.


def _local_connectome_dataset(dataset: object) -> bool:
    """True for FAFB / standalone BANC releases (type+hemisphere labels)."""
    try:
        from flywire_ids import is_local_connectome_dataset
    except ImportError:  # pragma: no cover - direct src/ execution
        try:
            from src.flywire_ids import is_local_connectome_dataset
        except ImportError:
            return False
    try:
        return bool(is_local_connectome_dataset(dataset))
    except Exception:
        return False


def _hemisphere_code(side_map_value=None, instance=None):
    """'_L'/'_R' for one neuron, mirroring ``_neuron_hemisphere_code``.

    The side map (somaSide / hemisphere column) wins; the instance's
    ``_L``/``_R`` suffix is the fallback.
    """
    text = str(side_map_value or "").strip().lower()
    if text in ("l", "left", "lhs", "left hemisphere"):
        return "L"
    if text in ("r", "right", "rhs", "right hemisphere"):
        return "R"
    inst = str(instance or "").strip()
    if inst.endswith("_R"):
        return "R"
    if inst.endswith("_L"):
        return "L"
    return None


def _map_get(mapping, body_id):
    """Map lookup tolerant to int/str key spellings of bodyId."""
    if not mapping:
        return None
    for key in (body_id, str(body_id)):
        if key in mapping:
            return mapping[key]
    text = str(body_id).strip()
    if text.isdigit():
        try:
            return mapping.get(int(text))
        except (TypeError, ValueError):
            return None
    return None


def build_body_id_label(dataset, body_id,
                        type_map=None, instance_map=None,
                        side_map=None, fallback_type=None) -> str:
    """Display label for one bodyId, suffixed per the tree-legend rule.

    Local FAFB/BANC releases: ``'{bodyId}_{type}_{L|R}'`` (hemisphere from
    *side_map* — bodyId -> 'left'/'right'/'L'/'R' — or the instance suffix).
    NeuPrint-style datasets: ``'{bodyId}_{instance}'``. Whenever the instance
    is unavailable the type is used instead (with the hemisphere appended);
    *fallback_type* (e.g. the comparison's resolved type label) is the last
    named resort and the bare bodyId is final, so the label is never empty.
    """
    bid = str(body_id).strip()
    ntype = str(_map_get(type_map, body_id) or "").strip()
    inst = str(_map_get(instance_map, body_id) or "").strip()
    side = _hemisphere_code(_map_get(side_map, body_id), inst)
    local = _local_connectome_dataset(dataset)

    if local:
        suffix = ntype or inst
    else:
        suffix = inst or ntype
    type_based = bool(suffix) and suffix == ntype
    if not suffix:
        suffix = str(fallback_type or "").strip()
        type_based = bool(suffix)
    if not suffix:
        return bid
    label = f"{bid}_{suffix}"
    # The hemisphere is appended to type-based suffixes only; instance
    # names usually carry the side themselves (e.g. 'aMe4_L').
    if type_based and side in ("L", "R"):
        label = f"{label}_{side}"
    return label


def body_id_label_map(dataset, body_ids, project_root=None,
                      type_map=None, instance_map=None,
                      side_map=None) -> dict:
    """``{str(canonical bodyId): display label}`` for *body_ids*.

    Uses the dataset's type/instance/side maps (neuron table, then neuron
    index — the same sources the vector cache merges) and formats every id
    with :func:`build_body_id_label`. Preloaded maps may be passed to avoid
    re-reading; unprovided maps are loaded once. Ids that cannot be
    canonicalized keep their raw spelling as the key.
    """
    try:
        from morphology import (
            _canonical_dataset_body_id,
            _dataset_soma_side_map,
            _load_neuron_type_map,
        )
    except ImportError:  # pragma: no cover - package import
        from src.morphology import (
            _canonical_dataset_body_id,
            _dataset_soma_side_map,
            _load_neuron_type_map,
        )

    if type_map is None or instance_map is None:
        type_map, instance_map = _load_neuron_type_map(dataset, project_root)
    if side_map is None:
        side_map = {}
        try:
            raw_sides = _dataset_soma_side_map(dataset, project_root) or {}
            for bid, side in raw_sides.items():
                try:
                    key = str(_canonical_dataset_body_id(dataset, bid))
                except Exception:
                    key = str(bid)
                side_map[key] = side
        except Exception:
            side_map = {}

    labels: dict = {}
    for bid in body_ids:
        try:
            canon = _canonical_dataset_body_id(dataset, bid)
        except Exception:
            canon = bid
        labels[str(canon)] = build_body_id_label(
            dataset, canon, type_map, instance_map, side_map)
    return labels

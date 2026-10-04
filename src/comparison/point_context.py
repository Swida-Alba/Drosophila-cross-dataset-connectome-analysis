"""Normalized comparison-point identities shared by exports and reports.

Standard comparisons use one scalar threshold as their comparison point.
Custom combination comparisons use one stable query row.  Raw execution keys
remain dataset/threshold pairs; this module only describes the point that
consumes those raw cells.
"""

from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional


def safe_point_id(value: Any, fallback: str = "query") -> str:
    """Return a deterministic filesystem/DOM-safe identifier."""
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or fallback))
    text = text.strip(" ._-")
    return text or fallback


def _cell_value(value: Any):
    """Value-preserving threshold cast: ints stay ints (synapse identity),
    float ratio tiers stay exact floats (a bare ``int()`` would truncate
    every sub-1 tier to 0)."""
    number = float(value)
    return int(number) if number.is_integer() else number


def threshold_file_stem(value: Any, weight_basis: str = "synapse") -> str:
    """Export-file stem for one threshold value — the delegate folder
    grammar (``minsyn_{int}`` synapse, ``minratio_{decimal}`` ratio with
    '.'→'_' per the coana folder convention)."""
    if weight_basis == "connection_ratio":
        return "minratio_" + str(float(value)).replace("-", "neg").replace(".", "_")
    return f"minsyn_{int(value)}"


@dataclass(frozen=True)
class ComparisonPoint:
    """One Standard scalar point or one Custom combination query."""

    point_id: str
    label: str
    mode: str
    thresholds_by_dataset: Dict[str, Any]
    display_order: int = 1
    weight_basis: str = "synapse"

    @property
    def query_id(self) -> Optional[str]:
        return self.point_id if self.mode == "combinations" else None

    @property
    def raw_thresholds(self) -> List[Any]:
        return sorted({_cell_value(value)
                       for value in self.thresholds_by_dataset.values()})

    @property
    def file_stem(self) -> str:
        if self.mode == "combinations":
            return f"query_{safe_point_id(self.point_id)}"
        value = next(iter(self.thresholds_by_dataset.values()), self.point_id)
        return threshold_file_stem(value, self.weight_basis)

    @property
    def display_label(self) -> str:
        if self.mode != "combinations":
            return self.label
        return f"{self.point_id}: {self.label}"

    @classmethod
    def from_query(
        cls,
        query: Mapping[str, Any],
        dataset_order: Iterable[str],
        display_order: int = 1,
        weight_basis: str = "synapse",
    ) -> "ComparisonPoint":
        query_id = str(query.get("id") or query.get("query_id") or "query")
        thresholds = query.get("thresholds") or query.get(
            "thresholds_by_dataset", {}
        )
        ordered = {
            dataset: _cell_value(thresholds[dataset])
            for dataset in dataset_order
            if dataset in thresholds
        }
        label = str(query.get("label") or query_id)
        return cls(query_id, label, "combinations", ordered, display_order,
                   weight_basis)

    @classmethod
    def from_standard(
        cls,
        threshold: Any,
        dataset_order: Iterable[str],
        display_order: int = 1,
        weight_basis: str = "synapse",
    ) -> "ComparisonPoint":
        threshold = _cell_value(threshold)
        ordered = {dataset: threshold for dataset in dataset_order}
        return cls(
            f"threshold_{threshold}",
            f"N={threshold}",
            "standard",
            ordered,
            display_order,
            weight_basis,
        )


def points_from_parameters(parameters: Any) -> List[ComparisonPoint]:
    """Build stable points without changing the parameter model."""
    datasets = list(parameters.get_dataset_names())
    basis = str(getattr(parameters, "weight_basis", "synapse"))
    if getattr(parameters, "threshold_mode", "standard") == "combinations":
        getter = getattr(parameters, "get_threshold_queries", None)
        if callable(getter):
            queries = getter()
        else:
            # Lightweight parameter doubles and older integrations may only
            # expose the serialized combination rows.  Keep point identity
            # available for those callers as well.
            queries = []
            for index, row in enumerate(
                    getattr(parameters, "threshold_combinations", []) or [],
                    start=1):
                if not isinstance(row, Mapping):
                    continue
                thresholds = row.get("thresholds") or row.get(
                    "thresholds_by_dataset", {})
                queries.append({
                    "id": row.get("id", f"query_{index:03d}"),
                    "label": row.get("label", row.get(
                        "id", f"query_{index:03d}")),
                    "thresholds": thresholds,
                })
        return [
            ComparisonPoint.from_query(
                query, datasets, index, weight_basis=basis)
            for index, query in enumerate(queries, start=1)
        ]
    thresholds = list(getattr(parameters, "thresholds", []) or [])
    return [
        ComparisonPoint.from_standard(
            threshold, datasets, index, weight_basis=basis)
        for index, threshold in enumerate(thresholds, start=1)
    ]


def point_from_value(
    value: Any,
    parameters: Any,
    points: Optional[List[ComparisonPoint]] = None,
) -> ComparisonPoint:
    """Resolve a point object, query ID, or Standard scalar threshold."""
    if isinstance(value, ComparisonPoint):
        return value
    points = points if points is not None else points_from_parameters(parameters)
    if isinstance(value, Mapping):
        query_id = value.get("id") or value.get("query_id")
        for point in points:
            if point.point_id == str(query_id):
                return point
    if isinstance(value, str):
        for point in points:
            if point.point_id == value:
                return point
        if getattr(parameters, "threshold_mode", "standard") == "standard":
            try:
                value = float(value)
            except ValueError:
                pass
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        threshold = _cell_value(value)
        for point in points:
            if point.mode == "standard" and next(
                iter(point.thresholds_by_dataset.values()), None
            ) == threshold:
                return point
    raise KeyError(f"Unknown comparison point: {value!r}")

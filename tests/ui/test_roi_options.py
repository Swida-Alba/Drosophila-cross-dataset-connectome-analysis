"""Tests for dataset-backed skeleton ROI option lists."""

import json
from pathlib import Path

from ui.roi_options import (
    BUNDLED_ROI_CATALOG,
    load_roi_catalog,
    roi_options_for_mode,
)


def _write_metadata(root: Path, dataset: str, metadata: dict) -> Path:
    folder = dataset.replace(":", "_").replace(".", "_")
    directory = root / "datasets" / folder
    directory.mkdir(parents=True)
    path = directory / f"{folder}_metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return directory


def test_primary_options_use_metadata_and_can_expand_bilateral_names(tmp_path):
    dataset = "test:v1.0"
    _write_metadata(
        tmp_path,
        dataset,
        {
            "primaryRois": ["A(L)", "A(R)", "M", "NotPrimary"],
        },
    )

    assert roi_options_for_mode(
        dataset, project_root=tmp_path
    ) == ["A", "M"]
    assert roi_options_for_mode(
        dataset, include_lr=True, project_root=tmp_path
    ) == ["A(L)", "A(R)", "M"]


def test_primary_subprimary_and_all_modes_use_available_inventory(tmp_path):
    dataset = "test:v1.0"
    _write_metadata(
        tmp_path,
        dataset,
        {"roi_coverage": {"roi_list": ["A(L)", "A(R)", "M"]}},
    )
    cache_dir = tmp_path / "cache" / "test_v1_0"
    cache_dir.mkdir(parents=True)
    (cache_dir / "available_rois.json").write_text(
        json.dumps(["A(L)", "A(R)", "A-sub(L)", "container", "M"]),
        encoding="utf-8",
    )

    assert roi_options_for_mode(
        dataset,
        include_subprimary=True,
        project_root=tmp_path,
    ) == ["A", "M", "A-sub", "container"]
    assert roi_options_for_mode(
        dataset, "all available", project_root=tmp_path
    ) == ["A(L)", "A(R)", "A-sub(L)", "container", "M"]


def test_roi_count_table_fills_all_mode_when_cache_is_missing(tmp_path):
    import polars as pl

    dataset = "test:v1.0"
    directory = _write_metadata(
        tmp_path,
        dataset,
        {"roi_coverage": {"roi_list": ["A(L)", "A(R)"]}},
    )
    pl.DataFrame({"roi": ["A(L)", "A(R)", "A-sub(L)", "NotPrimary"]}).write_parquet(
        directory / "test_v1_0_allneurons_roi_count_df.parquet"
    )

    catalog = load_roi_catalog(dataset, project_root=tmp_path)
    assert catalog["primary"] == ["A(L)", "A(R)"]
    assert catalog["available"] == ["A(L)", "A(R)", "A-sub(L)"]


def test_skeleton_tab_starts_with_no_meshes_and_exposes_roi_controls():
    from nicegui import Client
    from nicegui.page import page
    from ui.tabs.visualization import create_skeleton_tab

    client = Client(page("/skeleton-roi-controls"))
    with client:
        create_skeleton_tab()

    meshes = next(
        element for element in client.elements.values()
        if getattr(element, "_props", {}).get("label") == "Mesh ROIs"
    )
    labels = [
        getattr(element, "text", "")
        for element in client.elements.values()
        if getattr(element, "text", "")
    ]

    assert "ROI Selection Mode" not in labels
    assert "Include L/R variants" in labels
    assert "Include sub-primary ROIs" in labels
    assert meshes.value == []
    assert "outlined" in meshes._props


def _bundled_entry(folder: str) -> dict:
    data = json.loads(BUNDLED_ROI_CATALOG.read_text(encoding="utf-8"))
    return data[folder]


def _first_subprimary(entry: dict) -> str:
    primary = set(entry["primary"])
    return next(name for name in entry["available"] if name not in primary)


def test_fresh_install_offers_bundled_primary_and_subprimary(tmp_path):
    """An empty project root (a new clone has no datasets/ or cache/) must
    still surface the shipped primary list and its sub-primary entries."""
    dataset = "hemibrain:v1.2.1"
    entry = _bundled_entry("hemibrain_v1_2_1")

    catalog = load_roi_catalog(dataset, project_root=tmp_path)
    assert catalog["primary"] == entry["primary"]
    assert set(catalog["available"]) == set(entry["available"])
    assert len(catalog["available"]) > len(catalog["primary"])

    primary_opts = roi_options_for_mode(
        dataset, include_lr=True, project_root=tmp_path
    )
    sub_opts = roi_options_for_mode(
        dataset, include_lr=True, include_subprimary=True, project_root=tmp_path
    )
    assert len(sub_opts) > len(primary_opts)
    sub_name = _first_subprimary(entry)
    assert sub_name in sub_opts
    assert sub_name not in primary_opts


def test_live_cache_overrides_the_bundled_snapshot(tmp_path):
    """A backend available_rois.json cache is authoritative: the snapshot's
    sub-primary names must not merge into a live inventory."""
    dataset = "hemibrain:v1.2.1"
    folder = "hemibrain_v1_2_1"
    entry = _bundled_entry(folder)
    cache_dir = tmp_path / "cache" / folder
    cache_dir.mkdir(parents=True)
    sentinel = "ZZZ-live-sentinel"
    (cache_dir / "available_rois.json").write_text(
        json.dumps([sentinel]), encoding="utf-8"
    )

    catalog = load_roi_catalog(dataset, project_root=tmp_path)
    assert sentinel in catalog["available"]
    # No local metadata, so the primary list still comes from the snapshot.
    assert catalog["primary"] == entry["primary"]
    assert _first_subprimary(entry) not in catalog["available"]


def test_metadata_only_restores_bundled_subprimary(tmp_path):
    """A downloaded primary sidecar without an available-ROI cache still gets
    the snapshot's sub-primary breadth (the fresh-install-with-metadata case)."""
    dataset = "hemibrain:v1.2.1"
    entry = _bundled_entry("hemibrain_v1_2_1")
    _write_metadata(
        tmp_path, dataset, {"roi_coverage": {"roi_list": entry["primary"]}}
    )

    catalog = load_roi_catalog(dataset, project_root=tmp_path)
    assert catalog["primary"] == entry["primary"]
    assert _first_subprimary(entry) in catalog["available"]

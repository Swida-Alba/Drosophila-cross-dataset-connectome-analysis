"""Grouped Recommended/Available sections in the dataset dropdowns.

Every standard dataset selector (single and multi) builds with
``group_recommended=True``: the dropdown opens with a Recommended section
(fixed order) followed by an Available section grouping every dataset by
source (NeuPrint, FAFB, BANC) and sorted by name within each source, the
recommended ones repeated in place.  Grouping rides on the palette-picker
technique — an update hook re-applies enriched option rows after every
NiceGUI options rebuild, and an ``option`` slot renders the group headers.
The NeuronBridge Find Lines and Co-Labeling tabs keep their flat "(all)"
lists by passing no flag.
"""

import ui.dataset_service as dataset_service_module
from nicegui import Client
from nicegui.page import page


def _build_multi(**kwargs):
    from ui.components.common import dataset_multi_selector

    client = Client(page(f"/grouped-multi-dataset-{next(_build_multi._routes)}"))
    with client:
        return dataset_multi_selector(show_local_status=False, **kwargs)


_build_multi._routes = iter(range(100, 199))


def _build_single(**kwargs):
    from ui.components.common import dataset_selector

    client = Client(page(f"/grouped-single-dataset-{next(_build_single._routes)}"))
    with client:
        return dataset_selector(show_local_status=False, **kwargs)


_build_single._routes = iter(range(200, 299))


def test_recommended_lead_and_available_orders_by_source_then_name():
    datasets = [
        "banc_v626", "manc:v1.2.3", "male-cns:v1.0",
        "banc_v888", "flywire_FAFB_v783", "hemibrain:v1.1",
    ]
    selector = _build_multi(datasets=datasets, group_recommended=True)

    assert list(selector.options) == [
        "male-cns:v1.0",
        "flywire_FAFB_v783",
        "banc_v888",
        "hemibrain:v1.1",
        "manc:v1.2.3",
        "banc_v626",
    ]


def test_dropdown_rows_have_recommended_and_available_sections():
    datasets = ["banc_v626", "banc_v888", "male-cns:v1.0", "flywire_FAFB_v783"]
    selector = _build_multi(datasets=datasets, group_recommended=True)
    # Canonical order: male-cns:v1.0(0), flywire_FAFB_v783(1),
    # banc_v888(2), banc_v626(3).
    rows = selector._props["options"]

    assert [row["group"] for row in rows] == (
        ["Recommended"] * 3 + ["Available"] * 4
    )
    assert rows[0] == {
        "value": 0,
        "label": "male-cns:v1.0",
        "group": "Recommended",
        "first_of_group": True,
    }
    assert rows[2]["value"] == 2 and rows[2]["first_of_group"] is False
    assert rows[3]["first_of_group"] is True
    # Available groups by source (NeuPrint, FAFB, BANC) and repeats the
    # recommended datasets in place; the copies carry the canonical index
    # and label, so Quasar highlights and toggles the same selection from
    # either row.
    assert [row["label"] for row in rows[3:]] == [
        "male-cns:v1.0",
        "flywire_FAFB_v783",
        "banc_v626",
        "banc_v888",
    ]
    assert [row["value"] for row in rows[3:]] == [0, 1, 3, 2]
    assert rows[6] == {
        "value": 2,
        "label": "banc_v888",
        "group": "Available",
        "first_of_group": False,
    }


def test_grouping_survives_option_updates_and_label_changes():
    datasets = ["banc_v888", "male-cns:v1.0"]
    selector = _build_multi(datasets=datasets, group_recommended=True)
    assert "option" in selector.slots

    selector.set_options(
        {ds: f"{ds} relabeled" for ds in selector.options},
        value=selector.value,
    )
    rows = selector._props["options"]

    assert [row["group"] for row in rows] == (
        ["Recommended", "Recommended", "Available", "Available"]
    )
    assert rows[0]["label"] == "male-cns:v1.0 relabeled"
    assert rows[2]["label"] == "male-cns:v1.0 relabeled"
    assert rows[1]["label"] == "banc_v888 relabeled"
    assert rows[3]["label"] == "banc_v888 relabeled"


def test_other_tabs_keep_the_flat_list():
    datasets = ["banc_v888", "male-cns:v1.0", "manc:v1.2.3"]
    selector = _build_multi(datasets=datasets)

    assert list(selector.options) == datasets
    assert "option" not in selector.slots
    assert all("group" not in row for row in selector._props["options"])


def test_partial_recommended_subset_renders_both_sections():
    selector = _build_multi(datasets=["manc:v1.0", "banc_v888"], group_recommended=True)
    rows = selector._props["options"]

    assert [(row["group"], row["first_of_group"]) for row in rows] == [
        ("Recommended", True),
        ("Available", True),
        ("Available", False),
    ]
    assert [row["value"] for row in rows] == [0, 1, 0]
    assert rows[1]["label"] == "manc:v1.0"
    assert rows[2]["label"] == "banc_v888"


def test_subset_without_recommended_datasets_has_a_single_section():
    selector = _build_multi(
        datasets=["manc:v1.0", "fib19:v1.0"], group_recommended=True
    )
    rows = selector._props["options"]

    assert [(row["group"], row["first_of_group"]) for row in rows] == [
        ("Available", True),
        ("Available", False),
    ]


def test_full_dataset_list_groups_recommended_first(monkeypatch, tmp_path):
    from ui.components.common import dataset_multi_selector
    from ui.dataset_service import DatasetService

    service = DatasetService()
    service._datasets_dir = tmp_path / "datasets"
    service._cache_dir = tmp_path / "cache"
    monkeypatch.setattr(dataset_service_module, "_dataset_service", service)

    client = Client(page("/grouped-multi-dataset-full-list"))
    with client:
        selector = dataset_multi_selector(group_recommended=True)

    ids = list(selector.options)
    assert ids[:3] == ["male-cns:v1.0", "flywire_FAFB_v783", "banc_v888"]
    # Available-tail canonical order: NeuPrint sorted, then FAFB, then BANC
    # (the recommended FAFB/BANC entries live only in the first three slots).
    assert ids[3:] == [
        "fib19:v1.0",
        "hemibrain:v1.1",
        "hemibrain:v1.2.1",
        "male-cns:v0.9",
        "manc:v1.0",
        "manc:v1.2.1",
        "manc:v1.2.3",
        "mushroombody",
        "optic-lobe:v1.0.1",
        "optic-lobe:v1.1",
        "banc_v626",
    ]
    assert set(ids) == set(service.get_all_datasets())


def test_fresh_page_default_prefers_the_recommended_pair(monkeypatch):
    import ui.components.common as common_module

    datasets = ["manc:v1.0", "flywire_FAFB_v783", "male-cns:v1.0"]
    monkeypatch.setattr(common_module, "get_user_default", lambda key: None)
    selector = _build_multi(datasets=datasets, group_recommended=True)

    assert selector.value == ["male-cns:v1.0", "flywire_FAFB_v783"]


def test_status_refresh_updates_labels_and_keeps_sections(monkeypatch, tmp_path):
    from ui.components.common import (
        dataset_multi_selector,
        refresh_dataset_selector_statuses,
    )
    from ui.dataset_service import DatasetService

    service = DatasetService()
    service._datasets_dir = tmp_path / "datasets"
    service._cache_dir = tmp_path / "cache"
    monkeypatch.setattr(dataset_service_module, "_dataset_service", service)

    client = Client(page("/grouped-multi-dataset-status-refresh"))
    with client:
        selector = dataset_multi_selector(
            datasets=["manc:v1.2.3", "banc_v888"], group_recommended=True
        )

    rows = selector._props["options"]
    assert [(row["group"], row["first_of_group"]) for row in rows] == [
        ("Recommended", True),
        ("Available", True),
        ("Available", False),
    ]

    cache_dir = service._cache_dir / "manc_v1_2_3"
    cache_dir.mkdir(parents=True)
    (cache_dir / "connections.parquet").touch()
    refresh_dataset_selector_statuses(service)

    assert "◐ cached" in selector.options["manc:v1.2.3"]
    rows = selector._props["options"]
    assert [(row["group"], row["first_of_group"]) for row in rows] == [
        ("Recommended", True),
        ("Available", True),
        ("Available", False),
    ]
    assert rows[1]["label"] == selector.options["manc:v1.2.3"]


def test_status_refresh_preserves_synthetic_entry_labels(monkeypatch, tmp_path):
    from ui.components.common import (
        dataset_multi_selector,
        refresh_dataset_selector_statuses,
    )
    from ui.dataset_service import DatasetService

    service = DatasetService()
    service._datasets_dir = tmp_path / "datasets"
    service._cache_dir = tmp_path / "cache"
    monkeypatch.setattr(dataset_service_module, "_dataset_service", service)

    client = Client(page("/grouped-multi-dataset-synthetic-label"))
    with client:
        selector = dataset_multi_selector(
            datasets=["(all)", "manc:v1.2.3"],
            default=["(all)"],
        )
        selector.set_options(
            {"(all)": "(all) — search everywhere", "manc:v1.2.3": "manc:v1.2.3  [NP]"},
            value=selector.value,
        )

    cache_dir = service._cache_dir / "manc_v1_2_3"
    cache_dir.mkdir(parents=True)
    (cache_dir / "connections.parquet").touch()
    refresh_dataset_selector_statuses(service)

    assert selector.options["(all)"] == "(all) — search everywhere"
    assert "◐ cached" in selector.options["manc:v1.2.3"]


def test_single_selector_groups_recommended_and_available():
    datasets = ["banc_v626", "manc:v1.0", "flywire_FAFB_v783", "male-cns:v1.0"]
    selector = _build_single(datasets=datasets, group_recommended=True)

    assert list(selector.options) == [
        "male-cns:v1.0",
        "flywire_FAFB_v783",
        "manc:v1.0",
        "banc_v626",
    ]
    assert "option" in selector.slots
    rows = selector._props["options"]
    assert [(row["group"], row["first_of_group"]) for row in rows] == [
        ("Recommended", True),
        ("Recommended", False),
        ("Available", True),
        ("Available", False),
        ("Available", False),
        ("Available", False),
    ]
    # Available groups by source then name: male-cns (NP), manc (NP),
    # flywire (FAFB), banc_v626 (BANC).
    assert [(row["label"], row["value"]) for row in rows[2:]] == [
        ("male-cns:v1.0", 0),
        ("manc:v1.0", 2),
        ("flywire_FAFB_v783", 1),
        ("banc_v626", 3),
    ]


def test_single_selector_default_still_resolves(monkeypatch):
    import ui.components.common as common_module

    datasets = ["manc:v1.0", "flywire_FAFB_v783", "male-cns:v1.0"]
    monkeypatch.setattr(common_module, "get_user_default", lambda key: None)
    selector = _build_single(datasets=datasets, group_recommended=True)

    assert selector.value == "male-cns:v1.0"


def test_single_selector_keeps_the_flat_list_without_the_flag():
    datasets = ["banc_v888", "male-cns:v1.0"]
    selector = _build_single(datasets=datasets)

    assert list(selector.options) == datasets
    assert "option" not in selector.slots
    assert all("group" not in row for row in selector._props["options"])

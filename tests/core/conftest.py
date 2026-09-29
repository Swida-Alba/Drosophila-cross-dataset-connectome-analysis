"""Environment gating for the core suite: real data, tokens, the pull memo.

Why this file exists
--------------------
A fresh checkout on a machine with no NeuPrint token and no downloaded
``datasets/`` tables used to report ~110 core failures from *one* missing
prerequisite: the first pull attempt failed, and ``statvis``'s process-global
``_FAILED_DATASET_DOWNLOADS`` memo then made every later test fail fast with
``FileNotFoundError: Dataset '...' is still missing local data files after a
previous pull attempt`` — a message that hides the real cause (no token) and
pins the outcome to test order.

Three pieces fix the infrastructure half of that:

* :func:`local_dataset_available` — session-scoped verdict on "can this test
  get real local data?" (a NeuPrint token is configured *or* the dataset's
  tables are already on disk).
* :func:`_reset_dataset_pull_memo` — autouse, clears the pull memo around
  every test so one test's failed pull cannot change another's outcome.
* the ``requires_data`` / ``requires_token`` markers — a data-dependent test
  SKIPs with an explicit reason instead of ERRORing when the prerequisite is
  missing (same idea as the ad-hoc ``pytest.mark.skipif`` gates in
  ``test_type_mapper_real_datasets.py``, ``test_release_comparison_smoke.py``
  and ``test_banc_client_type.py``).

Missing *data* is never "fixed" here: the gate only changes how an
environment gap is reported, and the underlying pull error still propagates
unchanged for unmarked tests so the gap stays loud.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_PROJECT_ROOT / "src")
if _SRC not in sys.path:
    # The runtime import layout, same as tests/conftest.py and ``pythonpath``.
    sys.path.insert(0, _SRC)
# The repository root is only *appended* (so ``from tests.core import conftest``
# works in tests): its datasets//ui//cache/ folders must never shadow an
# installed distribution.
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

#: Dataset assumed by the un-argumented ``requires_data`` marker; matches the
#: default of ``statvis.getNeurons`` / ``statvis._ensure_local_dataset_files``.
DEFAULT_DATASET = "hemibrain:v1.2.1"

#: NeuPrint datasets keep their colon/dot version in the folder name folded to
#: underscores (``statvis._get_dataset_path_body``).
DATASETS_DIR = _PROJECT_ROOT / "datasets"

_TOKEN_ENV_VARS = ("NEUPRINT_APPLICATION_CREDENTIALS", "NEUPRINT_TOKEN")


# ---------------------------------------------------------------------------
# availability primitives (import-free and offline: usable at collection time)
# ---------------------------------------------------------------------------

def _normalize_dataset(dataset: str) -> str:
    """Fold a dataset id into its local folder/file name.

    Mirrors ``statvis._get_dataset_path_body``; ``canonical_dataset_name`` is
    imported from the lightweight ``utils.naming_utils`` rather than from
    ``statvis`` so this module stays cheap to import.
    """
    try:
        from utils.naming_utils import canonical_dataset_name
        name = str(canonical_dataset_name(dataset))
    except Exception:  # pragma: no cover - naming_utils is part of the suite
        name = str(dataset)
    return name.replace(":", "_").replace(".", "_")


def dataset_table_paths(dataset: str = DEFAULT_DATASET) -> tuple[Path, Path]:
    """The exact local files ``_ensure_local_dataset_files`` requires.

    Derived from ``src/statvis.py`` (``_get_dataset_path_body``,
    ``roi_count_table_path`` and the ``neuron_csv``/``roi_table`` checks in
    ``_ensure_local_dataset_files``), not from convention:

    * neuron table: ``datasets/<folder>/<folder>_allneurons_neuron_df.csv``
    * ROI table: ``datasets/<folder>/<folder>_allneurons_roi_count_df.parquet``
      (``.csv`` is accepted for pre-parquet pulls), and the flat
      ``datasets/<folder>_allneurons*`` spelling while the folder is absent.
    """
    normalized = _normalize_dataset(dataset)
    folder = DATASETS_DIR / normalized
    body = folder / f"{normalized}_allneurons"
    if not folder.exists():
        body = DATASETS_DIR / f"{normalized}_allneurons"
    # ``roi_count_table_path`` resolves to the parquet when it exists and to
    # the CSV name otherwise, so a dataset with neither table reads as absent
    # either way.
    roi = body.parent / (body.name + "_roi_count_df.parquet")
    if not roi.exists():
        roi = body.parent / (body.name + "_roi_count_df.csv")
    return body.parent / (body.name + "_neuron_df.csv"), roi


def dataset_files_present(dataset: str = DEFAULT_DATASET) -> bool:
    """True when this machine already holds the dataset's local tables."""
    neuron_csv, roi_table = dataset_table_paths(dataset)
    return neuron_csv.exists() and roi_table.exists()


def _usable_token(value) -> bool:
    """A candidate token that is neither empty nor a committed placeholder."""
    text = str(value or "").strip()
    return bool(text) and not text.startswith("YOUR_")


def token_configured() -> bool:
    """True when the DROCAT NeuPrint token chain yields a usable value.

    Deliberately probe-free: ``TokenManager.get_token()`` verifies candidates
    against the server, which is exactly what is unavailable here, and a
    collection-time gate must not hit the network.  ``tests/conftest.py``
    already seeds the canonical env var from the full chain for a machine that
    *can* resolve a token, so the env check is the primary answer and the
    config-file values (no probe) are the fallback.  Committed placeholders
    such as ``YOUR_NEUPRINT_TOKEN`` never count as configured.
    """
    for var in _TOKEN_ENV_VARS:
        if _usable_token(os.environ.get(var)):
            return True
    try:
        from utils.token_manager import token_manager
        # ``tokens`` is the probe-free config-file view TokenManager keeps for
        # callers/tests; the env vars above are still the authoritative chain.
        return _usable_token(token_manager.tokens.get("NEUPRINT_TOKEN"))
    except Exception:  # pragma: no cover - unreadable config is "no token"
        return False


class DatasetAvailability:
    """Callable verdict handed out by :func:`local_dataset_available`."""

    def __init__(self, token_available: bool | None = None):
        self.token_available = (token_configured()
                                if token_available is None else bool(token_available))

    def __call__(self, dataset: str = DEFAULT_DATASET) -> bool:
        """Can this test obtain real local data for ``dataset``?"""
        return bool(self.token_available or dataset_files_present(dataset))

    def tables_present(self, dataset: str = DEFAULT_DATASET) -> bool:
        return dataset_files_present(dataset)

    def paths(self, dataset: str = DEFAULT_DATASET) -> tuple[Path, Path]:
        return dataset_table_paths(dataset)

    def explain(self, dataset: str = DEFAULT_DATASET) -> str:
        neuron_csv, roi_table = dataset_table_paths(dataset)
        return (
            f"no NeuPrint token configured and no local tables for {dataset!r} "
            f"(expected {neuron_csv.name} + {roi_table.name} in {neuron_csv.parent})"
        )

    def require(self, dataset: str = DEFAULT_DATASET) -> tuple[Path, Path]:
        """``pytest.skip`` unless real data for ``dataset`` is reachable."""
        if not self(dataset):
            pytest.skip(self.explain(dataset))
        return dataset_table_paths(dataset)


def _datasets_from_marker(marker) -> list[str]:
    args = list(marker.args) + [marker.kwargs.get("dataset")]
    datasets: list[str] = []
    for arg in args:
        if arg is None:
            continue
        if isinstance(arg, (list, tuple, set)):
            datasets.extend(str(item) for item in arg)
        else:
            datasets.append(str(arg))
    return datasets or [DEFAULT_DATASET]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def local_dataset_available() -> DatasetAvailability:
    """Session verdict on whether real dataset tables can be obtained here.

    Usage::

        def test_x(local_dataset_available):
            local_dataset_available.require("male-cns:v1.0")   # skips if not
            if not local_dataset_available("hemibrain:v1.2.1"):
                pytest.skip("needs data")

    A token counts as available because ``statvis`` can pull the tables with
    it; the tables on disk count because no pull is needed at all.
    """
    return DatasetAvailability()


def _clear_dataset_pull_memo() -> None:
    """Reset ``statvis._FAILED_DATASET_DOWNLOADS`` in every loaded copy.

    Uses ``sys.modules`` rather than importing ``statvis``: the memo only
    matters once statvis has been imported, and a 2-second import in an
    autouse fixture would tax every core test.  Both the ``statvis`` and
    ``src.statvis`` spellings are cleared because the two module objects keep
    separate registries.
    """
    for name in ("statvis", "src.statvis"):
        module = sys.modules.get(name)
        if module is None:
            continue
        clear = getattr(module, "clear_failed_dataset_downloads", None)
        if callable(clear):
            try:
                clear()
            except Exception:  # pragma: no cover - never fail a test on reset
                pass


@pytest.fixture(autouse=True)
def _reset_dataset_pull_memo():
    """Keep the process-global pull memo from leaking between tests."""
    _clear_dataset_pull_memo()
    yield
    _clear_dataset_pull_memo()


# ---------------------------------------------------------------------------
# marker -> skip
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(config, items):
    """Turn ``requires_data`` / ``requires_token`` into skips up front.

    Resolved at collection so the reason names the missing prerequisite; the
    gate is env/filesystem-only, so it behaves identically offline.
    """
    has_token = token_configured()
    for item in items:
        data_marker = item.get_closest_marker("requires_data")
        if data_marker is not None:
            # Round-7 R7-3: local data presence is the gate.  A token alone
            # no longer satisfies ``requires_data`` — with a token but no
            # local tables the marked tests PULLED real datasets mid-suite
            # (~94 MB on the round-7 Windows host), breaking the "run §H on
            # a tree nothing has touched" rule from the inside.  A test
            # that genuinely needs the server declares ``requires_token``.
            for dataset in _datasets_from_marker(data_marker):
                if not dataset_files_present(dataset):
                    item.add_marker(pytest.mark.skip(
                        reason=("marker 'requires_data': local dataset "
                                f"tables for {dataset} not present "
                                "(token alone no longer satisfies this "
                                "marker — round-7 R7-3)")))
                    break
        if (item.get_closest_marker("requires_token") is not None
                and not has_token):
            item.add_marker(pytest.mark.skip(
                reason="marker 'requires_token': no NeuPrint token in the "
                       "environment or config.json/config_local.json"))

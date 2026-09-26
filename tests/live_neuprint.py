"""Gate for tests that assert against the LIVE NeuPrint dataset catalog.

The catalog is authoritative over which dataset names the server serves, and
it moves underneath us: as of 2026-09-26 NeuPrint no longer answers
``male-cns:v1.0`` — it lists ``manc:v1.0``/``manc:v1.2.1``/``manc:v1.2.3``
where that name used to be, and whether those are the same release renamed is
the open decision tracked as task #83.  Until it lands, a test that only
passes while a particular name is served carries
:func:`skipif_not_served`, so the withdrawal reads as a skip naming its cause
instead of a red suite.

The gate asks the product's own question —
:meth:`ui.dataset_service.DatasetService.fetch_neuprint_datasets` is the list
the UI builds from the server — so it cannot disagree with a guarded test
about what is served.  A missing token or an unreachable server yields an
empty catalog, i.e. a skip: those tests cannot pass without the network
anyway.
"""
from typing import Set

import pytest

_SERVED: dict = {}


def served_datasets() -> Set[str]:
    """The dataset names the live server currently serves (cached per process)."""
    if 'names' not in _SERVED:
        try:
            from ui.dataset_service import DatasetService
            names = DatasetService().fetch_neuprint_datasets()
        except Exception:
            names = []
        _SERVED['names'] = set(names or [])
    return _SERVED['names']


def serves(dataset: str) -> bool:
    return dataset in served_datasets()


def skipif_not_served(dataset: str):
    """``pytest.mark.skipif`` for a test that names a live-server dataset."""
    return pytest.mark.skipif(
        not serves(dataset),
        reason=(f"the live NeuPrint catalog does not serve {dataset!r} "
                f"(observed 2026-09-26; see task #83)"),
    )

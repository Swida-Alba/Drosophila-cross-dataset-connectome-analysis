"""Session guards for the UI test suite.

`ui.config` persists its editable settings through a module-level path
(`LOCAL_CONFIG_FILE`) and an mtime-keyed cache, so any test that reaches a
config setter without redirecting that path writes the developer's real
`ui/local_config.json` - and because `load_local_config` re-reads on mtime
change, a live app picks the test's values up mid-session. Individual tests
already patch the path (see `test_config_local.py`), but the guard has to hold
for tests that forget to.
"""

import pytest

import ui.config as cfg


@pytest.fixture(autouse=True)
def hermetic_local_config(tmp_path, monkeypatch):
    """Point `ui.config` at a per-test config file instead of the real one."""
    monkeypatch.setattr(cfg, "LOCAL_CONFIG_FILE", tmp_path / "local_config.json")
    cfg._LOCAL_CONFIG_CACHE["mtime_ns"] = None
    cfg._LOCAL_CONFIG_CACHE["data"] = {}
    yield
    # Drop whatever the test cached so a later test cannot read it through a
    # path that no longer exists.
    cfg._LOCAL_CONFIG_CACHE["mtime_ns"] = None
    cfg._LOCAL_CONFIG_CACHE["data"] = {}

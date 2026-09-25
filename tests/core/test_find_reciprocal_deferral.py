"""find_reciprocal parameter deferral (2026-09-25 fix round 3).

The dataclass documents that ``find_reciprocal=True`` enriches FindAllPath,
and ``_find_paths_core`` contains defer-to-field logic keyed on ``None`` —
but every public wrapper defaulted the parameter to ``False``, so the
constructor field was silently ignored (the UI runner even carried a
workaround passing ``find_reciprocal=fc.find_reciprocal`` explicitly).
These tests pin the restored contract: omitted parameter defers to the
field, explicit values always win, and the run's warning note reports what
actually ran.
"""

import inspect
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import coana  # noqa: E402
from coana import FindNeuronConnection  # noqa: E402


@pytest.fixture
def fnc(monkeypatch):
    """A bare instance whose pipeline call is captured, not executed."""
    fc = FindNeuronConnection.__new__(FindNeuronConnection)
    fc.find_reciprocal = False
    captured = {}

    def fake_core(path_mode, find_bodyId_path=True, forward_only=True,
                  exclude_searched_neurons=None, use_graph_cache=True,
                  find_reciprocal=None):
        captured['path_mode'] = path_mode
        # the same defer the real core performs
        if find_reciprocal is None:
            find_reciprocal = fc.find_reciprocal
        captured['find_reciprocal'] = find_reciprocal
        return 'core-result'

    monkeypatch.setattr(fc, '_find_paths_core', fake_core)
    fc._captured = captured
    return fc


def test_findallpath_omitted_defers_to_field(fnc):
    fnc.find_reciprocal = True
    fnc.FindAllPath()
    assert fnc._captured['find_reciprocal'] is True


def test_findallpath_explicit_false_overrides_field(fnc):
    fnc.find_reciprocal = True
    fnc.FindAllPath(find_reciprocal=False)
    assert fnc._captured['find_reciprocal'] is False


def test_findallpath_explicit_true_overrides_field_false(fnc):
    fnc.find_reciprocal = False
    fnc.FindAllPath(find_reciprocal=True)
    assert fnc._captured['find_reciprocal'] is True


def test_findshortestpath_defers_to_field(fnc):
    fnc.find_reciprocal = True
    fnc.FindShortestPath()
    assert fnc._captured['find_reciprocal'] is True
    assert fnc._captured['path_mode'] == 'shortest'


def test_signatures_default_to_none_deferral():
    """The wrappers must not forward a hard False over the constructor
    field: their defaults are None so the core's defer fires."""
    for name in ('FindAllPath', 'FindShortestPath', '_find_paths_core',
                 'FindAllPathMultiThreshold'):
        sig = inspect.signature(getattr(FindNeuronConnection, name))
        default = sig.parameters['find_reciprocal'].default
        assert default is None, (name, default)


def test_warning_note_reports_what_ran(tmp_path):
    """The reciprocal note keys on the run-truth flag when the pipeline
    defines it — a field-True run overridden to False must NOT claim the
    enrichment happened."""
    fc = FindNeuronConnection.__new__(FindNeuronConnection)
    fc._warn_notes = []
    fc.find_reciprocal = True
    fc._find_reciprocal_ran = False          # pipeline ran, reciprocal off
    fc._write_user_warning_notes(str(tmp_path))
    assert not (tmp_path / 'user_warning_notes.txt').exists() or \
        'find_reciprocal' not in (tmp_path / 'user_warning_notes.txt').read_text()

    fc2 = FindNeuronConnection.__new__(FindNeuronConnection)
    fc2._warn_notes = []
    fc2.find_reciprocal = True
    fc2._find_reciprocal_ran = True          # reciprocal actually ran
    fc2._write_user_warning_notes(str(tmp_path))
    assert 'find_reciprocal=True' in (tmp_path / 'user_warning_notes.txt').read_text()


def test_coana_imports_without_vispath_subprocess():
    """The wheel no longer hard-requires vispath_pkg at import time: a
    subprocess with the module blocked must still import coana, while
    FastGraph/VisualizePath resolve lazily with the install hint."""
    import subprocess
    import sys
    import textwrap
    code = textwrap.dedent('''
        import sys
        sys.path.insert(0, %r)
        class _Block:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.startswith("vispath_pkg"):
                    raise ImportError("blocked for test")
                return None
        sys.meta_path.insert(0, _Block())
        import coana
        import core.fast_graph  # module import must not require vispath
        # placeholders: attribute access is fine, USE raises the hint
        for attr in ("FastGraph", "VisualizePath"):
            try:
                getattr(coana, attr)()
                raise SystemExit(f"{attr} construction did not raise")
            except ImportError as exc:
                assert "vispath-subproject" in str(exc), exc
        try:
            core.fast_graph.DiGraph()
            raise SystemExit("DiGraph construction did not raise")
        except ImportError as exc:
            assert "vispath-subproject" in str(exc), exc
        print("lazy-import contract OK")
    ''' % str(SRC))
    result = subprocess.run([sys.executable, '-c', code],
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'lazy-import contract OK' in result.stdout

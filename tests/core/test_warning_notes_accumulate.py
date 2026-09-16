"""user_warning_notes.txt must accumulate, never truncate.

Regression for the review finding: the auto-type-mapping writer used mode
'w' and ran after the comparability / untyped-drop / query-resolution notes
were appended, silently erasing them.
"""

import os
from pathlib import Path
from types import SimpleNamespace

from comparison.comparison_analyzer import ComparisonAnalyzer


class _Mapper:
    _loaded = True
    _conflicts = []

    def build_user_warning_notes(self, type_names, dataset_names):
        return ['auto mapping expanded X -> Y']


def _analyzer(tmp_path):
    a = object.__new__(ComparisonAnalyzer)
    a.verbose = False
    a._log = lambda *x, **k: None
    a.parameters = SimpleNamespace(
        full_output_path=str(tmp_path),
        _auto_type_mapper=_Mapper(),
    )
    return a


def test_mapping_notes_append_and_preserve_existing(tmp_path):
    notes = Path(tmp_path) / 'user_warning_notes.txt'
    # Simulate notes appended earlier in the run.
    notes.write_text(
        'User warning notes\n==================\n\n'
        '- [threshold comparability] only 30 is comparable\n'
        '- [untyped dropped] ds @ t=3: 5 edges\n',
        encoding='utf-8')
    a = _analyzer(tmp_path)
    a._write_user_warning_notes_for_mapping(_Mapper(), ['X'], ['a'])
    text = notes.read_text(encoding='utf-8')
    assert 'threshold comparability' in text
    assert 'untyped dropped' in text
    assert 'auto mapping expanded X -> Y' in text
    # Header appears exactly once.
    assert text.count('User warning notes') == 1


def test_mapping_notes_creates_header_when_new(tmp_path):
    a = _analyzer(tmp_path)
    a._write_user_warning_notes_for_mapping(_Mapper(), ['X'], ['a'])
    text = (Path(tmp_path) / 'user_warning_notes.txt').read_text(
        encoding='utf-8')
    assert text.startswith('User warning notes')
    assert 'auto mapping expanded X -> Y' in text

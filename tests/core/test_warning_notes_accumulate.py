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


class TestThresholdFamilyIsStateOwned:
    """`- [threshold comparability]` describes the CURRENT run, so its writer
    replaces its own family (N warnings in, N out) and no other writer may
    decide how many copies survive.

    Found 2026-10-14 on one folder across three re-exports: two copies after
    one, one after a later one. The count moved because the bullet has no
    `[tag]` header line of its own, so the only thing that ever removed a prior
    copy was the state pass swallowing it as a continuation of whichever
    `[STATE]` header preceded it.
    """

    HEADER = 'User warning notes\n==================\n\n'

    def _analyzer(self, tmp_path):
        from comparison.comparison_analyzer import ComparisonAnalyzer

        a = object.__new__(ComparisonAnalyzer)
        a.verbose = False
        a._log = lambda *x, **k: None
        a.parameters = SimpleNamespace(full_output_path=str(tmp_path))
        return a

    def _emit(self, a, warnings):
        a._write_threshold_view_and_notes(
            {}, {'warnings': list(warnings), 'warning_level': 'warning'})

    def _text(self, tmp_path):
        return (tmp_path / 'user_warning_notes.txt').read_text(
            encoding='utf-8')

    def test_reexport_replaces_n_with_n_not_n_plus_n(self, tmp_path):
        a = self._analyzer(tmp_path)
        self._emit(a, ['cutoff 1 differs', 'cutoff 2 differs',
                       'cutoff 3 differs'])
        assert self._text(tmp_path).count(
            '- [threshold comparability]') == 3
        self._emit(a, ['only 3 is comparable', 'only 4 is comparable'])
        text = self._text(tmp_path)
        assert text.count('- [threshold comparability]') == 2, (
            'the family accumulated instead of being replaced: '
            + repr(text))
        assert 'cutoff 1 differs' not in text
        assert text.count('User warning notes') == 1

    def test_foreign_writers_leave_the_bullet_count_alone(self, tmp_path):
        """The write-order half: a state pass that does not own the family may
        neither erase the bullet nor decide how many there are."""
        a = self._analyzer(tmp_path)
        (tmp_path / 'user_warning_notes.txt').write_text(
            self.HEADER
            + '[user label mapper] active: 1 label(s).\n'
            '- DN1a\n'
            '- [threshold comparability] only 3 is comparable\n'
            '- [untyped dropped] ds @ t=3: 5 edges\n',
            encoding='utf-8')
        kept = a._replace_state_notes(
            ['[user label mapper] active: 2 label(s).\n- DN1a\n- APDN3'])
        a._append_user_warning_notes(str(tmp_path), kept)
        text = self._text(tmp_path)
        assert text.count('- [threshold comparability]') == 1
        assert '- [untyped dropped]' in text

    def test_its_own_writer_removes_only_its_own_family(self, tmp_path):
        a = self._analyzer(tmp_path)
        (tmp_path / 'user_warning_notes.txt').write_text(
            self.HEADER
            + '[merge policy] stale counts\n'
            '- [merge fan-in] belongs to that header\n'
            '- [threshold comparability] stale warning\n'
            '- [untyped dropped] keep me\n',
            encoding='utf-8')
        self._emit(a, ['fresh warning'])
        text = self._text(tmp_path)
        assert 'stale warning' not in text
        assert '- [threshold comparability] fresh warning' in text
        # Untouched: another writer's family, and the always-surviving bullet.
        assert '[merge policy] stale counts' in text
        assert '- [merge fan-in] belongs to that header' in text
        assert '- [untyped dropped] keep me' in text

    def test_the_tag_reader_spells_both_block_forms(self):
        from comparison.comparison_analyzer import ComparisonAnalyzer

        read = ComparisonAnalyzer._note_block_tag
        assert read('[merge policy] header form') == '[merge policy]'
        assert read('- [threshold comparability] bullet form') \
            == '[threshold comparability]'
        # An untagged bullet or a prose body opens no family.
        assert read('- DN1a') is None
        assert read('Auto type mapping changed how some names matched') is None


class TestDeclaredBulletFamiliesCannotBeErased:
    """Every family whose note is one bullet line is declared, so a state pass
    ends its block at the bullet instead of swallowing it as a continuation.

    Measured while fixing the threshold line: `- [query resolution]`,
    `- [auto threshold]` and `- [density]` were undeclared, so whoever happened
    to write a `[STATE]` header before them could delete them. Declaring them is
    the whole protection — and the copy-count side is handled by the append
    pass never writing a note twice, because `- [auto threshold]` has TWO
    writers (the degraded/partial note and `_append_density_warnings`), so a
    replace pass on that shared tag would let one writer delete the other's
    note in the same run.
    """

    HEADER = 'User warning notes\n==================\n\n'

    def _analyzer(self, tmp_path):
        from comparison.comparison_analyzer import ComparisonAnalyzer

        a = object.__new__(ComparisonAnalyzer)
        a.verbose = False
        a._log = lambda *x, **k: None
        a.parameters = SimpleNamespace(full_output_path=str(tmp_path))
        return a

    def test_the_three_families_are_declared(self):
        from comparison.comparison_analyzer import ComparisonAnalyzer as CA

        for tag in ('[untyped dropped]', '[threshold comparability]',
                    '[query resolution]', '[auto threshold]', '[density]'):
            assert tag in CA._BULLET_HEADED_TAGS, tag
        # …and only the single-writer families are state-owned.
        owned = {tag for family in CA._STATE_NOTE_FAMILIES for tag in family}
        assert '[query resolution]' in owned
        assert '[threshold comparability]' in owned
        assert '[auto threshold]' not in owned
        assert '[density]' not in owned

    def test_a_state_pass_leaves_every_declared_bullet_alone(self, tmp_path):
        a = self._analyzer(tmp_path)
        notes = Path(tmp_path) / 'user_warning_notes.txt'
        notes.write_text(
            self.HEADER
            + '[merge policy] stale counts\n'
            '- [merge fan-in] belongs to that header\n'
            '- [auto threshold] auto mode did NOT resolve: nothing aligned\n'
            '  • no density capture: ds_a\n'
            '- [density] y-axis is E(t)/N with a t-independent N\n'
            '- [query resolution] Unresolved mapping conflicts: X\n'
            '- [untyped dropped] keep me\n'
            '- [threshold comparability] keep me too\n',
            encoding='utf-8')
        kept = a._replace_state_notes(['[merge policy] fresh counts'])
        a._append_user_warning_notes(str(tmp_path), kept)
        text = notes.read_text(encoding='utf-8')
        # the owned family's own continuation bullet goes with it …
        assert '[merge policy] stale' not in text
        assert '- [merge fan-in]' not in text
        # … and every declared bullet of another family survives, with its
        # untagged continuation line.
        for keep in ('- [auto threshold] auto mode did NOT resolve',
                     '  • no density capture: ds_a',
                     '- [density] y-axis is E(t)/N',
                     '- [query resolution] Unresolved mapping conflicts: X',
                     '- [untyped dropped] keep me',
                     '- [threshold comparability] keep me too'):
            assert keep in text, keep

    def test_query_resolution_replaces_its_own_notes_n_for_n(self, tmp_path):
        """Its single writer routes through the replace pass, so a re-export
        cannot stack a second copy of a stale token list."""
        a = self._analyzer(tmp_path)
        notes = Path(tmp_path) / 'user_warning_notes.txt'
        a.resolve_query_inputs = lambda: [
            {'token': 'X', 'status': 'same_name_fallback', 'dataset': 'ds'},
            {'token': 'Y', 'status': 'conflict', 'dataset': 'ds'},
        ]
        a._export_query_resolution_records(str(tmp_path))
        text = notes.read_text(encoding='utf-8')
        assert text.count('- [query resolution]') == 2, text

        a.resolve_query_inputs = lambda: [
            {'token': 'Z', 'status': 'conflict', 'dataset': 'ds'}]
        a._export_query_resolution_records(str(tmp_path))
        text = notes.read_text(encoding='utf-8')
        assert text.count('- [query resolution]') == 1, text
        assert 'X' not in text and 'Z' in text
        assert text.count('User warning notes') == 1


class TestProfilingNotesWriterAppends:
    """`profile_comparator.append_warning_note_blocks` creates or appends.

    The profiling lane opened the file with mode 'w' — harmless only while it
    was that folder's only writer, and the exact precondition this file exists
    for on the comparison side.
    """

    def test_an_existing_bullet_survives_the_profiling_write(self, tmp_path):
        from comparison.profile_comparator import append_warning_note_blocks

        notes = Path(tmp_path) / 'user_warning_notes.txt'
        notes.write_text(
            'User warning notes\n==================\n\n'
            '- [threshold comparability] only 3 is comparable\n'
            '- [untyped dropped] ds @ t=3: 5 edges\n', encoding='utf-8')
        append_warning_note_blocks(notes, ['Auto type mapping changed things:',
                                           '', '- expanded A -> B'])
        text = notes.read_text(encoding='utf-8')
        assert '- [threshold comparability] only 3 is comparable' in text
        assert '- [untyped dropped] ds @ t=3: 5 edges' in text
        assert '- expanded A -> B' in text
        assert text.count('User warning notes') == 1

    def test_it_creates_the_header_when_the_file_is_new(self, tmp_path):
        from comparison.profile_comparator import append_warning_note_blocks

        notes = Path(tmp_path) / 'user_warning_notes.txt'
        append_warning_note_blocks(notes, ['- a note'])
        assert notes.read_text(encoding='utf-8').startswith(
            'User warning notes')

    def test_a_headerless_legacy_file_is_healed_not_duplicated(self, tmp_path):
        from comparison.profile_comparator import append_warning_note_blocks

        notes = Path(tmp_path) / 'user_warning_notes.txt'
        notes.write_text('loose older note\n', encoding='utf-8')
        append_warning_note_blocks(notes, ['- a note'])
        text = notes.read_text(encoding='utf-8')
        assert 'loose older note' in text
        assert text.count('User warning notes') == 1
        assert text.startswith('User warning notes')


class TestAnIdenticalNoteIsNeverWrittenTwice:
    """The append pass skips a block already in the file.

    Measured on a re-exported run folder (2026-10-14): the original notes had
    `- [auto threshold]` x1 and `- [density]` x2; after ONE re-export they were
    x2 and x4. Those families cannot be made state-owned (two writers share the
    `[auto threshold]` tag, so a replace pass would let each delete the other's
    note within one run), and a repeated sentence carries no information —
    skipping the exact duplicate fixes the noise without touching published tags.
    """

    HEADER = 'User warning notes\n==================\n\n'

    def _analyzer(self, tmp_path):
        from comparison.comparison_analyzer import ComparisonAnalyzer

        a = object.__new__(ComparisonAnalyzer)
        a.verbose = False
        a._log = lambda *x, **k: None
        a.parameters = SimpleNamespace(full_output_path=str(tmp_path))
        return a

    def test_the_measured_duplication_cannot_recur(self, tmp_path):
        a = self._analyzer(tmp_path)
        notes = Path(tmp_path) / 'user_warning_notes.txt'
        notes.write_text(self.HEADER, encoding='utf-8')
        blocks = [
            '- [auto threshold] density-aligned mode: measured per-dataset '
            'windows\n  • vertical (same-threshold) rows: 3; horizontal '
            '(same-density) rows: 2.',
            '- [density] y-axis is E(t)/N with a t-independent N.',
        ]
        for _round in range(3):
            a._append_user_warning_notes(str(tmp_path), blocks)
        text = notes.read_text(encoding='utf-8')
        assert text.count('- [auto threshold]') == 1, text
        assert text.count('- [density]') == 1, text
        assert text.count('User warning notes') == 1

    def test_a_different_count_still_appears_beside_the_old_one(self, tmp_path):
        """Dedupe is EXACT-match only: a changed disclosure must not be hidden.

        The state families exist because a stale count beside a fresh one is
        worse than a duplicate; this keeps the two properties from being
        confused. The `[auto threshold]` pair below is what a shared-tag family
        looks like across two re-exports with different measurements.
        """
        a = self._analyzer(tmp_path)
        notes = Path(tmp_path) / 'user_warning_notes.txt'
        a._append_user_warning_notes(
            str(tmp_path), ['- [auto threshold] dataset ds_a has 0 paths in '
                            'the aligned window'])
        a._append_user_warning_notes(
            str(tmp_path), ['- [auto threshold] dataset ds_a has 3 paths in '
                            'the aligned window'])
        text = notes.read_text(encoding='utf-8')
        assert 'has 0 paths' in text and 'has 3 paths' in text

    def test_a_multiline_block_matches_as_one_unit(self, tmp_path):
        """Whitespace and blank-line spacing are not part of a block's identity,
        but its lines are: one bullet re-indented is the same note, a new
        bullet under the same header is a different one."""
        a = self._analyzer(tmp_path)
        notes = Path(tmp_path) / 'user_warning_notes.txt'
        a._append_user_warning_notes(
            str(tmp_path),
            ['- [auto threshold] head line\n  • first detail'])
        a._append_user_warning_notes(
            str(tmp_path),
            ['- [auto threshold] head line\n  • first detail'])
        a._append_user_warning_notes(
            str(tmp_path),
            ['- [auto threshold] head line\n  • second detail'])
        text = notes.read_text(encoding='utf-8')
        assert text.count('- [auto threshold] head line') == 2, text
        assert 'second detail' in text

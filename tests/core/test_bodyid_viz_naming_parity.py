"""Plan 2026-09-07 Phase D: bodyId visualization filename parity.

Runs the relocation logic against synthetic type-level and bodyId-level
artifacts and asserts identical ``Network_`` / ``Heatmap_`` / ``Sankey_`` /
``visualization_data/`` naming rules, plus the level-specific selected-path
companion files.
"""

import pandas as pd

from coana import FindNeuronConnection


RUN = 'find-paths-complete_TEST_src_to_tgt_L2w3_20260907_000000'


def _fc(tmp_path):
    fc = object.__new__(FindNeuronConnection)
    fc.allpath_folder = str(tmp_path)
    fc._vprint = lambda *a, **k: None
    return fc


def _write_raw_artifacts(folder, base):
    folder.joinpath(f'{base}_network.html').write_text('<html>n</html>')
    folder.joinpath(f'{base}_Sankey.html').write_text('<html>s</html>')
    folder.joinpath(f'{base}_heatmap.html').write_text('<html>h</html>')
    folder.joinpath(f'{base}_data_connections.csv').write_text('a\n')
    folder.joinpath(f'{base}_data_original_paths.csv').write_text('b\n')


_COMPANION = pd.DataFrame({'path': ['A->B->C'], 'length': [2]})


def test_type_level_naming(tmp_path):
    run = tmp_path / RUN
    run.mkdir()
    fc = _fc(run)
    _write_raw_artifacts(run, RUN)

    fc._relocate_viz_outputs(
        input_df=_COMPANION, input_filename='type_paths_visualized.csv')

    viz = run / 'visualization'
    for prefix in ('Network', 'Heatmap', 'Sankey'):
        assert (viz / f'{prefix}_{RUN}.html').exists(), prefix
    data = viz / 'visualization_data'
    assert (data / f'{RUN}_data_connections.csv').exists()
    assert (data / f'{RUN}_data_original_paths.csv').exists()
    assert (data / 'type_paths_visualized.csv').exists()
    # raw names no longer at the run root
    assert not (run / f'{RUN}_network.html').exists()


def test_bodyId_level_naming_mirrors_type_level(tmp_path):
    run = tmp_path / RUN
    run.mkdir()
    fc = _fc(run)
    bodyid_dir = run / 'bodyId_visualization'
    bodyid_dir.mkdir()
    _write_raw_artifacts(bodyid_dir, 'bodyId_visualization')

    fc._relocate_bodyid_viz_outputs(
        str(bodyid_dir), run_folder=str(run), input_df=_COMPANION)

    # identical naming rules, level-specific location
    for prefix in ('Network', 'Heatmap', 'Sankey'):
        assert (bodyid_dir / f'{prefix}_{RUN}.html').exists(), prefix
        assert not (bodyid_dir / f'bodyId_visualization_{prefix.lower()}.html')\
            .exists()
    data = bodyid_dir / 'visualization_data'
    assert (data / f'{RUN}_data_connections.csv').exists()
    assert (data / f'{RUN}_data_original_paths.csv').exists()
    # level-specific selected-path companion
    assert (data / 'bodyId_paths_visualized.csv').exists()
    saved = pd.read_csv(data / 'bodyId_paths_visualized.csv')
    assert list(saved.columns) == ['path', 'length']


def test_bodyId_naming_without_companion_and_direct_layout(tmp_path):
    """Missing artifact types stay absent; the run folder for naming can
    differ from allpath_folder (direct-connection layout)."""
    fc = _fc(tmp_path)
    run = tmp_path / RUN
    run.mkdir()
    direct = tmp_path / 'finddirect_TEST_src_to_tgt'
    direct.mkdir()
    bodyid_dir = direct / 'bodyId_visualization'
    bodyid_dir.mkdir()
    # network only (direct connections draw the same three, but a preview
    # style network-only folder must gain nothing)
    bodyid_dir.joinpath('bodyId_visualization_network.html').write_text('x')

    fc._relocate_bodyid_viz_outputs(
        str(bodyid_dir), run_folder=str(direct))

    # naming uses the DIRECT run folder, not allpath_folder
    assert (bodyid_dir / 'Network_finddirect_TEST_src_to_tgt.html').exists()
    assert not (bodyid_dir / f'Network_{RUN}.html').exists()
    assert not (bodyid_dir / 'Sankey_finddirect_TEST_src_to_tgt.html').exists()
    assert not (bodyid_dir / 'Heatmap_finddirect_TEST_src_to_tgt.html').exists()


def test_early_preview_folder_uses_same_network_prefix(tmp_path):
    fc = _fc(tmp_path)
    run = tmp_path / RUN
    run.mkdir()
    preview = run / 'network_early_bodyId'
    preview.mkdir()
    preview.joinpath('network_early_bodyId_network.html').write_text('x')

    fc._organize_vispath_artifacts(
        source_dir=str(preview), viz_dir=str(preview), run_name=RUN)

    assert (preview / f'Network_{RUN}.html').exists()
    assert not (preview / 'network_early_bodyId_network.html').exists()


def test_plotly_library_co_locates_with_moved_heatmap(tmp_path):
    """The shared plotly.min.js follows the heatmap html into viz_dir;
    the in-place layout (source IS viz_dir) stays a no-op, and a
    pre-existing viz copy turns the source copy into a dropped orphan
    (both were written from the same plotly package file)."""
    run = tmp_path / RUN
    run.mkdir()
    fc = _fc(run)
    run.joinpath(f'{RUN}_heatmap.html').write_text('<html>h</html>')
    run.joinpath('plotly.min.js').write_text('root-copy')

    fc._relocate_viz_outputs()
    viz = run / 'visualization'
    assert (viz / 'plotly.min.js').read_text() == 'root-copy'
    assert not run.joinpath('plotly.min.js').exists()

    # in-place: the library already sits beside the html — untouched
    preview = run / 'network_early_bodyId'
    preview.mkdir()
    preview.joinpath('plotly.min.js').write_text('in-place')
    fc._organize_vispath_artifacts(
        source_dir=str(preview), viz_dir=str(preview), run_name=RUN)
    assert preview.joinpath('plotly.min.js').read_text() == 'in-place'

    # duplicate: viz copy wins, the source copy is dropped
    other = run / 'other_preview'
    other.mkdir()
    other.joinpath('plotly.min.js').write_text('stale-source')
    fc._organize_vispath_artifacts(
        source_dir=str(other), viz_dir=str(viz), run_name=RUN)
    assert viz.joinpath('plotly.min.js').read_text() == 'root-copy'
    assert not other.joinpath('plotly.min.js').exists()

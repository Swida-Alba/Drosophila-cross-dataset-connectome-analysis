"""Render sample outputs of the profiler report machinery (equivalence harness).

Run before and after the report_kit extraction; the two outputs must be
byte-identical so the profiling export is provably unchanged.
"""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from comparison.profile_comparator import ConnectivityProfileComparer  # noqa: E402

out_dir = Path(sys.argv[1])
out_dir.mkdir(parents=True, exist_ok=True)

comparer = ConnectivityProfileComparer.__new__(ConnectivityProfileComparer)
comparer.verbose = False
comparer.show_figures = False
comparer.direction = 'both'
comparer.datasets = ['male-cns:v1.0']

parts = []
parts.append(comparer._report_css())
parts.append(comparer._report_script())

pos = pd.DataFrame([[1.0, 0.5], [0.5, np.nan]],
                   index=['aMe12', 'aMe10'], columns=['aMe12', 'aMe10'])
signed = pd.DataFrame([[-1.0, 0.0, 0.7]],
                      index=['r'], columns=['n', 'z', 'p'])

frag_pos, cl_pos = comparer._plotly_heatmap_fragment(
    pos, 'Intra · jaccard', 'jaccard', 'x', 'y', include_plotlyjs=False,
    square_cells=True)
frag_sig, cl_sig = comparer._plotly_heatmap_fragment(
    signed, 'Intra · rank', 'rank_corr_union', 'x', 'y')
parts.append(f'CLUSTERED_POS={cl_pos}')
parts.append(frag_pos or 'NONE')
parts.append(f'CLUSTERED_SIG={cl_sig}')
parts.append(frag_sig or 'NONE')

lines = []
comparer._append_report_tab_group(
    lines, 'grp', [('overall', 'Overall'), ('up', 'Upstream')],
    lambda key, pid: lines.append(f'<p>panel {key} {pid}</p>'))
comparer._append_report_metric_grid(
    lines, out_dir,
    {'jaccard': pos, 'rank_corr_union': signed},
    'Intra-dataset · male-cns:v1.0 · Type level · Overall',
    {'jaccard': 'intra_dataset/x/results/similarity_overall_jaccard.csv',
     'rank_corr_union': 'intra_dataset/x/results/similarity_overall_rank_corr_union.csv'},
    {'jaccard': 'intra_dataset/x/visualization/heatmap_intra_overall_jaccard.html',
     'rank_corr_union': None},
    'Neuron / type', 'Neuron / type', {'include_plotlyjs': False},
    square_cells=True)
parts.append('\n'.join(lines))

(out_dir / 'machinery_sample.txt').write_text('\n@@@SPLIT@@@\n'.join(parts),
                                              encoding='utf-8')

# Standalone VisPath heatmap (falls back to interactive_heatmap when absent)
saved = {'heatmaps_generated': []}
viz = out_dir / 'visualization'
comparer._generate_heatmaps_vispath(
    {'overall': {'jaccard': pos, 'rank_corr_union': signed}},
    viz, saved, prefix='inter')
names = sorted(p.name for p in viz.glob('*.html'))
(out_dir / 'vispath_files.txt').write_text('\n'.join(names), encoding='utf-8')
for name in names:
    (viz / name).rename(out_dir / f'viz_{name}')
print('sample written to', out_dir)

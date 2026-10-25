"""Multi-dataset morphology comparison: intra + cross in ONE run folder.

The morphology analogue of ``ConnectivityProfileComparer``'s multi-dataset
mode. Until this module existed the two Similarity > Morphology > Comparison
lanes disagreed with the connectivity tab: one dataset ran the intra
comparer, two or more switched the whole panel to the cross comparer, and a
three-volume deliverable therefore needed one run folder per volume (and,
because the method was a constructor choice, one MORE folder per method)
plus a separate cross folder. The connectivity tab writes every volume's
intra matrix and the cross overview into one folder; this class does the
same for morphology.

Layout (under ``output_dir``)::

    morphology_comparison_{DATASETS}_{query}_{ts}/
      intra_dataset/{MCNS,FAFB,ABNC}/   one self-contained intra comparison
        group_level/ or type_level/      (both methods side by side when
        bodyid_level/                    method='both')
        members.csv
        report.html, parameters.json, README.txt
      cross_dataset/                   ordered pairs + null baselines
        {SRC}_to_{TGT}/results/
        overview.csv, members_summary.csv, report.html, parameters.json
      report.html                      hub: links both halves + key tables
      parameters.json                  combined provenance
      README.txt
      user_warning_notes.txt           combined disclosures (when any)

The intra and cross phases score the SAME resolved population: this class
resolves the query through the cross comparer's type/preset lanes, applies
the total-neuron cap per dataset, and hands the result to both backends
(``resolved_members`` on the intra comparer, ``members_override`` on the
cross comparer). A custom-group run filters the preset's groups to that
same bodyId set, so no volume is compared against a larger pool in one half
than in the other.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:  # package import (runner scripts / tests)
    from .morph_cross_dataset import (
        CrossDatasetMorphComparer,
        _dataset_abbrev,
        _safe_name,
    )
    from . import report_kit
except ImportError:  # direct src/ execution
    from morph_cross_dataset import (  # type: ignore
        CrossDatasetMorphComparer,
        _dataset_abbrev,
        _safe_name,
    )
    import report_kit  # type: ignore

try:  # top-level module (src is on sys.path for both callers)
    from morphology_comparison import MorphologyProfileComparer
except ImportError:  # pragma: no cover - direct src/ package execution
    from ..morphology_comparison import MorphologyProfileComparer  # type: ignore

DEFAULT_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

_AGGREGATION = {
    "type": "type",
    "bodyid": "bodyid",
    "custom": "custom",
    "custom group": "custom",
}


class MultiDatasetMorphComparer:
    """Run every selected volume's intra comparison plus the ordered cross
    pairs, written as one folder — the morphology equivalent of a
    multi-dataset connectivity profiling run."""

    def __init__(
        self,
        datasets: Optional[Sequence[str]] = None,
        query: Any = None,
        method: str = "vector_v2",
        aggregation_level: str = "type",
        custom_mapping_file: Optional[str] = None,
        label_mapping_file: Optional[str] = None,
        max_members_per_type: int = 25,
        max_total_neurons: int = 200,
        null_k: int = 200,
        reference_template: Optional[str] = None,
        scene_members_per_type: int = 3,
        visualize: bool = False,
        intra_visualize: bool = False,
        intra_visualization_settings: Optional[Dict[str, object]] = None,
        fetch_online: bool = True,
        intra_fetch_online: Optional[bool] = None,
        use_cache: bool = True,
        intra_use_cache: Optional[bool] = None,
        use_auto_type_mapping: bool = True,
        generate_heatmaps: bool = True,
        show_figures: bool = False,
        output_dir: Optional[str] = None,
        saveas: Optional[str] = None,
        project_root: Optional[str] = None,
        verbose: bool = True,
        n_workers: int = 8,
    ):
        self.datasets = [str(d) for d in (datasets or [])]
        if len(self.datasets) < 2:
            raise ValueError(
                "The multi-dataset morphology comparison needs at least two "
                "datasets; use MorphologyProfileComparer for one dataset.")
        if len({str(d) for d in self.datasets}) != len(self.datasets):
            raise ValueError("Duplicate dataset in the comparison selection.")
        self.query = query
        self.method = str(method or "vector_v2").strip().lower()
        if self.method not in ("vector_v2", "nblast", "both"):
            raise ValueError(
                f"Invalid method: {method!r} (vector_v2|nblast|both)")
        self.methods = (["vector_v2", "nblast"] if self.method == "both"
                        else [self.method])
        level = _AGGREGATION.get(str(aggregation_level or "type").strip())
        if level is None:
            raise ValueError(
                f"Invalid aggregation level: {aggregation_level!r} "
                "(type|bodyid|custom)")
        self.aggregation_level = level
        self.custom_mapping_file = str(custom_mapping_file or "").strip()
        self.label_mapping_file = str(label_mapping_file or "").strip()
        if level == "custom" and not self.custom_mapping_file:
            raise ValueError(
                "The custom group level needs a grouping preset "
                "(custom_mapping_file).")
        self.max_members_per_type = max(1, int(max_members_per_type))
        self.max_total_neurons = max(2, int(max_total_neurons))
        self.null_k = max(1, int(null_k))
        self.reference_template = reference_template
        self.scene_members_per_type = max(1, int(scene_members_per_type))
        self.visualize = bool(visualize)
        self.intra_visualize = bool(intra_visualize)
        self.intra_visualization_settings = dict(
            intra_visualization_settings or {})
        self.fetch_online = bool(fetch_online)
        self.intra_fetch_online = (
            self.fetch_online if intra_fetch_online is None
            else bool(intra_fetch_online))
        self.use_cache = bool(use_cache)
        self.intra_use_cache = (
            self.use_cache if intra_use_cache is None
            else bool(intra_use_cache))
        self.use_auto_type_mapping = bool(use_auto_type_mapping)
        self.generate_heatmaps = bool(generate_heatmaps)
        self.show_figures = bool(show_figures)
        self.output_dir = output_dir
        self.saveas = str(saveas or "").strip()
        self.project_root = str(Path(project_root or DEFAULT_PROJECT_ROOT))
        self.verbose = bool(verbose)
        self.n_workers = max(1, int(n_workers))
        self.warnings: List[str] = []

    # ------------------------------------------------------------------ log
    def _log(self, message: str) -> None:
        if self.verbose:
            print(str(message), flush=True)

    # --------------------------------------------------------------- tokens
    def _tokens(self) -> List[str]:
        value = self.query
        if value is None:
            return []
        if isinstance(value, (str, int)):
            value = [value]
        out: List[str] = []
        for item in value:
            text = str(item or "").strip()
            if text and text not in out:
                out.append(text)
        if not out:
            raise ValueError(
                "The multi-dataset morphology comparison needs a query: "
                "enter at least one neuron, type, or pattern (or the custom "
                "group labels).")
        return out

    # ------------------------------------------------------------- resolve
    def _cross_resolver(self, tokens: Sequence[str]) -> CrossDatasetMorphComparer:
        """A resolution-only cross comparer (it is not run for outputs)."""
        return CrossDatasetMorphComparer(
            datasets=list(self.datasets), query=list(tokens),
            max_members_per_type=self.max_members_per_type,
            null_k=self.null_k,
            reference_template=self.reference_template,
            scene_members_per_type=self.scene_members_per_type,
            visualize=False, generate_heatmaps=False,
            fetch_online=self.fetch_online,
            use_auto_type_mapping=self.use_auto_type_mapping,
            custom_mapping_file=self.custom_mapping_file or None,
            label_mapping_file=self.label_mapping_file or None,
            use_cache=self.use_cache, verbose=False,
            n_workers=self.n_workers, project_root=self.project_root)

    def _resolve_shared_members(
            self, tokens: Sequence[str]
    ) -> "tuple[Dict[str, Dict[str, List[int]]], List[str]]":
        """Resolve the query per dataset once, then apply the total cap."""
        resolver = self._cross_resolver(tokens)
        resolved: Dict[str, Dict[str, List[int]]] = {}
        notes: List[str] = []
        for dataset in self.datasets:
            rows, ds_notes = resolver._resolve_members(tokens, dataset)
            resolved[dataset] = rows
            notes.extend(ds_notes)
        return resolved, notes

    def _trim_total(
            self, rows: Dict[str, List[int]], dataset: str
    ) -> "tuple[Dict[str, List[int]], int, bool]":
        total = sum(len(ids) for ids in rows.values())
        if total <= self.max_total_neurons:
            return rows, total, False
        trimmed: Dict[str, List[int]] = {}
        kept = 0
        for label, ids in rows.items():
            if kept >= self.max_total_neurons:
                break
            take = ids[:self.max_total_neurons - kept]
            if take:
                trimmed[label] = take
                kept += len(take)
        self.warnings.append(
            f"{dataset}: max_total_neurons={self.max_total_neurons} kept "
            f"{kept} of {total} resolved neuron(s) in row order; the intra "
            "and cross phases both use the kept set.")
        self._log(self.warnings[-1])
        return trimmed, kept, True

    def _custom_intra_members(
            self, dataset: str, allowed: set
    ) -> "tuple[Dict[str, List[object]], int]":
        """The preset's groups for one dataset, filtered to *allowed*.

        Uses the intra comparer's own preset resolution (groups may list
        types, instances or bodyIds), so the custom rows match what a
        single-dataset custom run would have produced; the filter keeps the
        cross phase's population authoritative.
        """
        resolver = MorphologyProfileComparer(
            dataset=dataset, query=None, aggregation_level="custom",
            custom_mapping_file=self.custom_mapping_file,
            label_mapping_file=self.label_mapping_file,
            max_members_per_type=self.max_members_per_type,
            max_total_neurons=10 ** 9,
            fetch_online=self.fetch_online, use_cache=self.use_cache,
            generate_heatmaps=False, verbose=False,
            project_root=self.project_root, n_workers=self.n_workers)
        rows = resolver._resolve_members()
        filtered: Dict[str, List[object]] = {}
        missing = 0
        for label, ids in rows.items():
            keep = [bid for bid in ids
                    if int(resolver._body_id(bid)) in allowed]
            if keep:
                filtered[str(label)] = keep
            else:
                missing += 1
        if missing:
            self.warnings.append(
                f"{dataset}: {missing} custom group(s) have no member in the "
                "shared comparison population and were omitted.")
        return filtered, sum(len(v) for v in filtered.values())

    def _intra_members(
            self, dataset: str, shared_rows: Dict[str, List[int]]
    ) -> Dict[str, List[object]]:
        if self.aggregation_level != "custom":
            return {label: list(ids) for label, ids in shared_rows.items()}
        allowed = {int(b) for ids in shared_rows.values() for b in ids}
        rows, _total = self._custom_intra_members(dataset, allowed)
        if not rows:
            raise ValueError(
                f"{dataset}: none of the custom groups has a member in the "
                "shared comparison population.")
        return rows

    # ------------------------------------------------------------- outputs
    def _run_path(
            self, tokens: Sequence[str],
            intra_members: Dict[str, Dict[str, List[object]]]
    ) -> Path:
        base = (Path(self.output_dir) if self.output_dir
                else Path(self.project_root) / "local_data"
                / "morphology_comparison")
        base.mkdir(parents=True, exist_ok=True)
        if self.saveas:
            return base / self.saveas
        ds_part = "_".join(_dataset_abbrev(d) for d in self.datasets)
        first = intra_members.get(self.datasets[0]) or {}
        if self.aggregation_level == "bodyid":
            source = [str(t) for t in tokens]
        else:
            source = [str(k) for k in first]
        query = "_".join(source[:4])
        if len(source) > 4:
            query += "_etc"
        query = _safe_name(query, 60) or "comparison"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return base / f"morphology_comparison_{ds_part}_{query}_{stamp}"

    def _run_intra(
            self, dataset: str, members: Dict[str, List[object]],
            run_path: Path, total: int, total_capped: bool
    ) -> Dict[str, Any]:
        abbr = _dataset_abbrev(dataset)
        self._log(
            f"[{dataset}] intra comparison: {len(members)} row(s) / "
            f"{total} neuron(s), methods {'+'.join(self.methods)}, "
            f"{self.aggregation_level} level")
        comparer = MorphologyProfileComparer(
            dataset=dataset,
            query=None if self.aggregation_level == "custom" else list(
                self._tokens()),
            method=self.method,
            aggregation_level=self.aggregation_level,
            custom_mapping_file=self.custom_mapping_file or None,
            label_mapping_file=self.label_mapping_file or None,
            resolved_members=members,
            max_members_per_type=self.max_members_per_type,
            # The suite already applied the shared cap; a second cap here
            # would trim the two phases differently.
            max_total_neurons=max(self.max_total_neurons, total),
            fetch_online=self.intra_fetch_online,
            use_cache=self.intra_use_cache,
            output_dir=str(run_path / "intra_dataset"),
            saveas=abbr,
            generate_heatmaps=self.generate_heatmaps,
            show_figures=self.show_figures,
            verbose=self.verbose,
            n_workers=self.n_workers,
            project_root=self.project_root,
            visualize=self.intra_visualize,
            visualization_settings=self.intra_visualization_settings)
        result = comparer.run()
        return {
            "dataset": dataset,
            "run_folder": str(Path(result["output_folder"]).relative_to(
                run_path)),
            "method": result.get("method"),
            "methods": list(result.get("methods") or self.methods),
            "aggregation_level": result.get("aggregation_level"),
            "rows_compared": result.get("rows_compared"),
            "neurons_compared": result.get("neurons_compared"),
            "neurons_compared_by_method": result.get(
                "neurons_compared_by_method", {}),
            "total_capped": bool(total_capped),
            "report": str(Path(result["output_folder"]).relative_to(
                run_path) / "report.html"),
        }

    def _run_cross(
            self, tokens: Sequence[str],
            shared: Dict[str, Dict[str, List[int]]], run_path: Path
    ) -> Dict[str, Any]:
        self._log(
            f"cross comparison: {len(self.datasets)} datasets, "
            f"{len(self.datasets) * (len(self.datasets) - 1)} ordered pairs")
        comparer = CrossDatasetMorphComparer(
            datasets=list(self.datasets), query=list(tokens),
            output_dir=str(run_path), saveas="cross_dataset",
            max_members_per_type=self.max_members_per_type,
            null_k=self.null_k,
            reference_template=self.reference_template,
            scene_members_per_type=self.scene_members_per_type,
            visualize=self.visualize,
            generate_heatmaps=self.generate_heatmaps,
            fetch_online=self.fetch_online,
            use_auto_type_mapping=self.use_auto_type_mapping,
            label_mapping_file=self.label_mapping_file or None,
            members_override=shared,
            use_cache=self.use_cache, verbose=self.verbose,
            n_workers=self.n_workers, project_root=self.project_root)
        result = comparer.run()
        cross_path = Path(result["output_folder"])
        pairs = sorted(p.name for p in cross_path.iterdir()
                       if p.is_dir() and "_to_" in p.name)
        return {
            "run_folder": str(cross_path.relative_to(run_path)),
            "pairs": pairs,
            "report": str(cross_path.relative_to(run_path) / "report.html"),
            "overview": str(cross_path.relative_to(run_path) / "overview.csv"),
            "members_summary": str(cross_path.relative_to(run_path)
                                   / "members_summary.csv"),
            "warnings": list(result.get("warnings") or []),
        }

    def _write_hub(self, run_path: Path, intra: Dict[str, Dict[str, Any]],
                   cross: Dict[str, Any], tokens: Sequence[str]) -> None:
        from html import escape

        def link(path: str, label: str) -> str:
            return f"<a href='{escape(path, quote=True)}'>{escape(label)}</a>"

        dataset_cards = []
        for dataset in self.datasets:
            abbr = _dataset_abbrev(dataset)
            info = intra[abbr]
            aggregate = ("group_level/group_similarity"
                         if self.aggregation_level == "custom"
                         else "type_level/type_similarity")
            rows = []
            for method in info["methods"]:
                body_csv = (f"{info['run_folder']}/bodyid_level/"
                            f"bodyid_similarity_{method}.csv")
                if self.aggregation_level == "bodyid":
                    # At the bodyId level the bodyId matrix IS the comparison;
                    # no aggregate-level CSV exists to link.
                    rows.append(
                        "<tr><td>" + escape(method) + "</td>"
                        f"<td>{link(body_csv, 'bodyId CSV')}</td></tr>")
                else:
                    agg_csv = (f"{info['run_folder']}/{aggregate}"
                               f"_{method}.csv")
                    rows.append(
                        "<tr><td>" + escape(method) + "</td>"
                        f"<td>{link(agg_csv, 'aggregate CSV')}</td>"
                        f"<td>{link(body_csv, 'bodyId CSV')}</td></tr>")
            dataset_cards.append(
                "<article class='section-card'>"
                f"<h2 class='section-heading'>{escape(dataset)}</h2>"
                f"<p>{link(info['report'], 'Open the intra comparison report')} "
                f"· {info['rows_compared']} row(s) · "
                f"{info['neurons_compared']} neuron(s) compared</p>"
                "<table class='mapping-table'><thead><tr><th>method</th>"
                + ("<th>bodyId matrix</th>"
                   if self.aggregation_level == "bodyid" else
                   "<th>aggregate matrix</th><th>bodyId matrix</th>")
                + f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
                "</article>")
        cross_rows = "".join(
            f"<tr><td>{escape(pair)}</td></tr>" for pair in cross["pairs"])
        html = [
            '<!DOCTYPE html>',
            "<html><head><meta charset='utf-8'>",
            '<title>Multi-dataset morphology comparison</title>',
            report_kit.report_css(),
            '</head><body><main class="report-shell">',
            '<header class="report-hero">',
            '<div class="report-kicker">DROCAT · Multi-dataset morphology</div>',
            '<h1 class="report-title">Morphology comparison</h1>',
            '<p class="report-subtitle">One run folder: every selected '
            "volume's intra-dataset comparison plus all ordered "
            'cross-dataset pairs, scored in the same resolved population. '
            'Use the per-volume and cross reports for the heatmaps and '
            'source links.</p>',
            '<div class="report-meta">',
            f"<div class='meta-chip'><span>Datasets</span><strong>"
            f"{escape(', '.join(self.datasets))}</strong></div>",
            f"<div class='meta-chip'><span>Intra methods</span><strong>"
            f"{escape(' + '.join(self.methods))}</strong></div>",
            "<div class='meta-chip'><span>Cross method</span><strong>"
            "vector_v2</strong></div>",
            f"<div class='meta-chip'><span>Intra level</span><strong>"
            f"{escape(self.aggregation_level)}</strong></div>",
            '</div></header>',
            f"<p class='report-note'>Query: {escape(', '.join(tokens))}</p>",
            *dataset_cards,
            "<article class='section-card'>"
            "<h2 class='section-heading'>Cross-dataset comparison</h2>"
            f"<p>{link(cross['report'], 'Open the cross comparison report')} "
            f"· {link(cross['overview'], 'overview.csv')} · "
            f"{link(cross['members_summary'], 'members_summary.csv')}</p>"
            "<table class='mapping-table'><thead><tr><th>ordered pair</th>"
            f"</tr></thead><tbody>{cross_rows}</tbody></table></article>",
            '</main>', report_kit.report_script(), '</body></html>',
        ]
        (run_path / "report.html").write_text(
            "\n".join(html), encoding="utf-8")

    def _write_readme(self, run_path: Path, intra: Dict[str, Dict[str, Any]],
                      cross: Dict[str, Any]) -> None:
        lines = [
            "MULTI-DATASET MORPHOLOGY COMPARISON",
            "=" * 70,
            f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
            f"Datasets: {', '.join(self.datasets)}",
            f"Intra methods: {', '.join(self.methods)}",
            f"Intra aggregation level: {self.aggregation_level}",
            "Cross method: vector_v2 (the only score defined across datasets)",
            "",
            "Output layout:",
            "  intra_dataset/<DATASET>/        one self-contained intra run",
            "    group_level/ or type_level/   aggregate matrix per method",
            "    bodyid_level/                 bodyId x bodyId matrix per method",
            "    members.csv, report.html, parameters.json, README.txt",
            "  cross_dataset/                  ordered dataset pairs",
            "    <SRC>_to_<TGT>/results/       LabelMapper label matrix "
            "(type-level comparison), matched-label CSV, native type matrix "
            "(diagnostic), bodyId matrices/scores, null baseline",
            "    overview.csv                  21 labels x 6 pairs, SAME label "
            "in source and target only (like connectivity)",
            "    members_summary.csv, report.html, parameters.json",
            "  report.html                     this hub",
            "  parameters.json                 combined provenance",
            "",
            "The intra and cross phases score the SAME resolved population: "
            "the query is resolved once, the total-neuron cap is applied "
            "once, and both halves receive that member set. A custom-group "
            "run filters the preset's groups to the same bodyIds.",
            "",
            "Per-volume reports:",
        ]
        for dataset in self.datasets:
            info = intra[_dataset_abbrev(dataset)]
            lines.append(
                f"  {dataset}: {info['run_folder']}/report.html "
                f"({info['rows_compared']} rows, "
                f"{info['neurons_compared']} neurons)")
        lines += [
            "",
            f"Cross report: {cross['run_folder']}/report.html",
            f"Ordered pairs: {', '.join(cross['pairs'])}",
            "",
        ]
        (run_path / "README.txt").write_text(
            "\n".join(lines), encoding="utf-8")

    def _write_parameters(self, run_path: Path, tokens: Sequence[str],
                          intra: Dict[str, Dict[str, Any]],
                          cross: Dict[str, Any],
                          shared: Dict[str, Dict[str, List[int]]]) -> None:
        cross_params_path = run_path / cross["run_folder"] / "parameters.json"
        cross_params = {}
        if cross_params_path.is_file():
            try:
                cross_params = json.loads(
                    cross_params_path.read_text(encoding="utf-8"))
            except ValueError:
                cross_params = {}
        params = {
            "comparison": "multi-dataset morphology (intra per dataset + "
                          "ordered cross pairs)",
            "datasets": list(self.datasets),
            "query": [str(t) for t in tokens],
            "method": self.method,
            "methods_intra": list(self.methods),
            "aggregation_level_intra": self.aggregation_level,
            "method_cross": "vector_v2",
            "max_members_per_type": self.max_members_per_type,
            "max_total_neurons": self.max_total_neurons,
            "null_k": self.null_k,
            "reference_template": self.reference_template,
            "visualize_cross": self.visualize,
            "visualize_intra": self.intra_visualize,
            "fetch_online": self.fetch_online,
            "use_cache": self.use_cache,
            "use_auto_type_mapping": self.use_auto_type_mapping,
            "generate_heatmaps": self.generate_heatmaps,
            "resolved_population": {
                dataset: {
                    "rows": len(rows),
                    "neurons": sum(len(ids) for ids in rows.values()),
                }
                for dataset, rows in shared.items()},
            "intra_dataset": intra,
            "cross_dataset": cross,
            "cross_warnings": list(cross.get("warnings") or []),
            "warnings": list(self.warnings),
            "cross_parameters": cross_params,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        if self.custom_mapping_file:
            params["custom_mapping_file"] = self.custom_mapping_file
        if self.label_mapping_file:
            params["label_mapping_file"] = self.label_mapping_file
        (run_path / "parameters.json").write_text(
            json.dumps(params, indent=2, default=str), encoding="utf-8")

    def _write_warnings(self, run_path: Path,
                        intra: Dict[str, Dict[str, Any]],
                        cross: Dict[str, Any]) -> None:
        lines: List[str] = []
        for warning in self.warnings:
            lines.append(f"- {warning}")
        for warning in cross.get("warnings") or []:
            lines.append(f"- [cross] {warning}")
        for dataset in self.datasets:
            info = intra[_dataset_abbrev(dataset)]
            notes = run_path / info["run_folder"] / "user_warning_notes.txt"
            if notes.is_file():
                body = notes.read_text(encoding="utf-8").splitlines()
                for line in body:
                    if line.startswith("- "):
                        lines.append(f"- [{dataset}] {line[2:]}")
        if not lines:
            return
        (run_path / "user_warning_notes.txt").write_text(
            "User warning notes\n==================\n\n"
            + "\n".join(lines) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ run
    def run(self) -> Dict[str, Any]:
        started = time.time()
        tokens = self._tokens()
        self._log("=" * 60)
        self._log("MULTI-DATASET MORPHOLOGY COMPARISON")
        self._log("=" * 60)
        self._log(f"Datasets: {', '.join(self.datasets)}")
        self._log(f"Query: {', '.join(tokens)}")
        self._log(f"Intra methods: {'+'.join(self.methods)}; "
                  f"aggregation level: {self.aggregation_level}")

        resolved, notes = self._resolve_shared_members(tokens)
        for note in notes:
            self._log(note)
        shared: Dict[str, Dict[str, List[int]]] = {}
        capped_by_dataset: Dict[str, bool] = {}
        for dataset in self.datasets:
            trimmed, _kept, capped = self._trim_total(
                resolved[dataset], dataset)
            if not trimmed:
                raise ValueError(
                    f"{dataset}: the query resolved no neurons; the "
                    "comparison cannot publish this dataset.")
            shared[dataset] = trimmed
            capped_by_dataset[dataset] = capped
        intra_members = {
            dataset: self._intra_members(dataset, shared[dataset])
            for dataset in self.datasets}
        run_path = self._run_path(tokens, intra_members)
        run_path.mkdir(parents=True, exist_ok=True)
        self._log(f"📁 Output folder: {run_path}")

        intra: Dict[str, Dict[str, Any]] = {}
        for dataset in self.datasets:
            members = intra_members[dataset]
            total = sum(len(ids) for ids in members.values())
            intra[_dataset_abbrev(dataset)] = self._run_intra(
                dataset, members, run_path, total,
                capped_by_dataset[dataset])

        cross = self._run_cross(tokens, shared, run_path)
        self._write_hub(run_path, intra, cross, tokens)
        self._write_readme(run_path, intra, cross)
        self._write_parameters(run_path, tokens, intra, cross, shared)
        self._write_warnings(run_path, intra, cross)

        files = [str(p) for p in sorted(run_path.rglob("*")) if p.is_file()]
        self._log(f"Output: {run_path}")
        return {
            "output_folder": str(run_path),
            "datasets": list(self.datasets),
            "query": list(tokens),
            "method": self.method,
            "methods": list(self.methods),
            "aggregation_level": self.aggregation_level,
            "intra": intra,
            "cross": cross,
            "warnings": list(self.warnings),
            "files": files,
            "duration_s": round(time.time() - started, 1),
        }

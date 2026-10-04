"""
DROCAT exported run user guide.

After every successful UI run, a ``_UserGuide_please_read_me`` file is
written into the run's output folder. It describes every file in the run
folder, every column of the exported tables (via a shared, score/metric
directed column glossary), and renders the run's ``user_warning_notes.txt``
content when present.

One content model (``TOOL_GUIDE_SPECS`` + ``COLUMN_GLOSSARY``) drives three
renderers (HTML, Markdown, plain text), so the descriptions never diverge
between formats. The format is configurable in Settings → Default Settings
("Run Guide Format"): ``html`` (default), ``txt``, ``markdown`` or
``disabled``. The ``DROCAT_RUN_GUIDE_FORMAT`` environment variable overrides
the setting (used by tests and scripts for determinism).
"""

import fnmatch
import csv
import html
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

GUIDE_BASENAME = "_UserGuide_please_read_me"
GUIDE_FORMAT_ENV = "DROCAT_RUN_GUIDE_FORMAT"
GUIDE_FORMATS = ("html", "txt", "markdown", "disabled")
GUIDE_EXTENSIONS = {"html": ".html", "txt": ".txt", "markdown": ".md"}

WARNING_FILENAME = "user_warning_notes.txt"
NO_WARNINGS_TEXT = "No warnings were recorded for this run."


# =============================================================================
# Column glossary — every repeated column is described exactly once.
# Values are (description, value type/range).
# =============================================================================

COLUMN_GLOSSARY = {
    # ratio-lane disclosure file (plan-connection-ratio-pathfinding §13.2)
    'bodyId_post': ('Post-synaptic bodyId of a ratio-lane cone edge.',
                    'string'),
    'total_incoming': ('ALL-post incoming synapse mass of the post '
                       'neuron at min_weight=1 (the F9 threshold-free '
                       'denominator).', 'number'),
    'implied_syn_cutoff': ('Per-neuron synapse cutoff the ratio threshold '
                           'implies: max(1, ceil(t_r x total_incoming)) — '
                           'a sub-synapse cutoff would admit every edge.',
                           'integer'),
    'kept_in_edges': ('Cone edges into this post neuron that survived '
                      'the ratio threshold.', 'integer'),
    'kept_syn_min': ('Weakest kept-edge synapse count into this post '
                     'neuron.', 'number'),
    'kept_syn_max': ('Strongest kept-edge synapse count into this post '
                     'neuron.', 'number'),
    # --- Identifiers ---------------------------------------------------------
    "bodyId": ("Unique numeric neuron identifier.", "integer"),
    "queried_type": ("Neuron type as entered in the comparison query (one row per queried type).", "text"),
    "morph_v2": ("Production vector_v2 similarity (block-weighted whitened cosine; no NBLAST), cross- or intra-dataset.", "number 0-1"),
    "morph_null_p95": ("Per-query null bar: the p95 of this query's vector_v2 scores against ~200 seeded random target neurons.", "number"),
    "morph_z": ("morph_v2 expressed in null-distribution standard deviations above the null median.", "number"),
    "morph_bar_kind": ("Which rule gated this row. The pooling tables name them 'native' (the source's own reference-pool floor), 'track_a' (its pool baseline B_b minus the offset) or 'null_bar' (no reference bar was available); the supervised bins name the same ladder 'native' / 'track_a_backup' / 'null'; an out-map or source-candidate row is always 'null' (the run null bar).", "text"),
    "morph_pool_ref": ("A candidate's similarity to the source's own reference pool (native Track B). When morph_bar_kind is 'native' THIS is the number morph_bar was applied to — read the verdict as morph_pool_ref >= morph_bar; on other kinds the graded score is the row's Track-A column.", "number"),
    "morph_bar": ("The bar this row's score was compared against, so its ✓/✗ is recomputable from the row. Apply it to `morph_pool_ref` when the kind is 'native', otherwise to `morph_v2_similarity`.", "number"),
    "morph_null_level": ("The null percentile the bar sits at when morph_bar_kind is 'null_bar' (default 95).", "integer"),
    "morph_qualified": ("Whether the visualized candidate passed the morph bar (morph_v2 >= morph_bar). Failing candidates are excluded from the rendered scenes but keep their result rows.", "boolean"),
    "layer": ("One-based visualization layer assignment used by the reusable layer map.", "integer or text"),
    "neuron": ("Neuron identifier resolved by the Skeleton layer-map parser.", "text or integer"),
    "color": ("Effective neuron display color.", "CSS color"),
    "synapse_color": ("Effective connector color for synapses whose pre-neuron is this row.", "CSS color"),
    "pre_synaptic_color": ("Effective color for this neuron's pre-synaptic sites.", "CSS color"),
    "post_synaptic_color": ("Effective color for this neuron's post-synaptic sites.", "CSS color"),
    "bodyId_pre": ("Pre-synaptic (source) neuron body id of a synapse.", "integer"),
    "bodyId_post": ("Post-synaptic (target) neuron body id of a synapse.", "integer"),
    "instance": ("Instance (individual) name of the neuron.", "text"),
    "row": ("Matrix row this neuron feeds: the neuron type at the type "
            "level, the neuron's own display label at the bodyId level, or "
            "the group label at the custom group level. `type` beside it "
            "stays the neuron's real type at every level.", "text"),
    "type": ("Neuron type name.", "text"),
    "type_pre": ("Presynaptic (source) neuron type of the edge.", "text"),
    "type_post": ("Postsynaptic (target) neuron type of the edge.", "text"),
    "source": ("Edge source neuron type.", "text"),
    "target": ("Edge target neuron type.", "text"),
    "edge_key": ("Canonical edge identifier: 'source -> target'.", "text"),
    "group": ("Custom query group the neuron belongs to.", "text"),
    "dataset": ("Dataset the record comes from.", "text"),
    "token": ("Raw input query token the row resolves.", "text"),
    "role": ("Query side of the token: source or target.", "text"),
    "status": ("Resolution outcome (same_name_identity, taxonomy, "
               "taxonomy_mapped, mapped, bridged, valid_split, "
               "evidence_only, same_name_fallback, conflict, body_id, "
               "group, pattern, unmatched).", "text"),
    "method": ("Resolution mechanism that produced the answer.", "text"),
    "evidence": ("For identity rows: confirmed, contradicted or none — "
                 "what the cross-dataset relation says about the identity.",
                 "text"),
    "target_types": ("Resolved type name(s) queried in the dataset.", "text"),
    "confidence": ("Resolution confidence tier (higher = more trustworthy).",
                   "number"),
    "query_id": ("Stable identifier for one cross-dataset threshold query row.", "text"),
    "query_label": ("Human-readable label for one cross-dataset threshold query row.", "text"),
    # type_resolution_union.csv: one (type, dataset) coverage record per query.
    "present": ("Whether this type was found in THIS dataset's own searched "
                "graph, so no cross-dataset name resolution was needed. False "
                "means it is absent here and `resolution_status` says why.",
                "boolean"),
    "resolved_type": ("The name this type resolves to in THIS dataset's "
                      "namespace (merged-display alternates such as "
                      "\"CB0937(CB2577)\" are normalized). Equals `type` when "
                      "`present` is true, and is empty when the type could not "
                      "be resolved at all.", "text"),
    "resolution_status": ("Why this (type, dataset) cell does or does not carry "
                          "coverage: present, below_threshold, not_recruited, "
                          "no_edges, not_in_dataset, resolved_absent (the "
                          "diagnosis inputs themselves were unavailable, so "
                          "absence is unproven rather than established), or the "
                          "resolver's own verdicts unmapped, conflict, "
                          "evidence_only, claimed, mapper_unavailable.",
                          "text"),
    "detail": ("The evidence behind `resolution_status` — e.g. the largest "
               "connected edge weight for a below_threshold type, or the "
               "reason a resolution was refused.", "text"),
    "threshold_mode": ("Threshold query mode: standard same-threshold rows or explicit combinations.", "text"),
    "threshold_scope": ("Whether a row is a scalar threshold grid cell, a Custom query comparison cell, or a raw-run schedule diagnostic.", "text"),
    "path_mode": ("Path enumeration mode for the run: all paths or per-pair shortest.", "text"),
    "comparison_mode": ("Comparison engine mode: pathfinding or direct edge comparison.", "text"),
    "threshold_": ("Prefix for one requested-threshold column per dataset (e.g. threshold_banc_v888).", "text"),
    "direction": ("Synaptic direction relative to the query: upstream or downstream.", "text"),
    "partner_type": ("Partner neuron type in a connectivity profile.", "text"),
    "neuron_type": ("Neuron type owning the profile row.", "text"),
    "query": ("Query neuron/type the row belongs to.", "text"),
    "line": ("Driver line name.", "text"),
    "source_line": ("Driver line that produced the match.", "text"),
    # --- Synapse counts / edge scores ----------------------------------------
    "pre": ("Number of presynaptic sites of the neuron.", "integer"),
    "post": ("Number of postsynaptic sites of the neuron.", "integer"),
    "downstream": ("Downstream synapse count (NeuPrint synapse category).", "integer"),
    "upstream": ("Upstream synapse count (NeuPrint synapse category).", "integer"),
    "synweight": ("Synapse weight of the neuron (post + downstream).", "number"),
    "weight": ("Synapse count of the connection.", "integer"),
    "weights": ("Per-edge synapse counts along the path.", "list of integers"),
    "weight_a": ("Partner weight in profile A (query).", "number"),
    "weight_b": ("Partner weight in profile B (candidate).", "number"),
    "weight_L": ("Edge weight on the left hemisphere (L-L or L-R pairing).", "integer"),
    "weight_R": ("Edge weight on the right hemisphere (R-R or R-L pairing).", "integer"),
    "weight_LR": ("Weight of the left-to-right counterpart edge.", "integer"),
    "weight_RL": ("Weight of the right-to-left counterpart edge.", "integer"),
    "diff": ("Absolute weight difference between the paired hemisphere edges.", "integer"),
    "ratio": ("Strength ratio of the paired hemisphere edges.", "number"),
    "present_LR": ("Left-to-right edge present.", "boolean"),
    "present_RL": ("Right-to-left edge present.", "boolean"),
    "total_weight": ("Total connection weight of the layer/path group.", "number"),
    # --- Connection scores ----------------------------------------------------
    "connection_ratio": (
        "Fraction of the postsynaptic neuron's input coming from this "
        "partner: $w_{ij} / \\sum_k w_{kj}$ over the postsynaptic total "
        "input $D_t$.", "0-1"),
    "ratios": ("Per-edge connection ratios along the path.", "list of 0-1"),
    "min_ratio": ("Smallest edge connection ratio along the path.", "0-1"),
    "traversal_probability": (
        "Probability that a signal traverses the edge: "
        "$\\min(1.0,\\ connection\\_ratio/0.3)$.", "0-1"),
    "probabilities": ("Per-edge traversal probabilities along the path.", "list of 0-1"),
    "block_probability": ("Probability the signal is blocked: "
                         "$1 - \\text{traversal\\_probability}$.", "0-1"),
    "path": ("Path as a neuron-type sequence, e.g. aMe12 → SMP238 → PPL101.", "text"),
    # --- Pair report (post-hoc, scripts/PathsPairReport.py) -----------------
    "unit": ("Pathfinding unit the row came from: the raw folder name relative "
             "to the run root ('' = run root; cross-dataset delegates look "
             "like dataset/minsyn_N, e.g. flywire_FAFB_v783/minsyn_3).", "text"),
    "rank_in_pair_length": ("1-based rank of the path within its (pair, "
                            "length) group, bottleneck-first: min_weight desc, "
                            "path_prob desc, path asc.", "integer"),
    "paths_in_pair": ("Total paths found for this (unit, source, target) pair "
                      "— the denominator of every capped view.", "integer"),
    "source_bodyid_coverage": ("bodyId-level n/N for the pair's SOURCE type: "
                               "bodyIds on found paths (isInPath) / all "
                               "enrolled bodyIds of the type, from "
                               "source_neurons.csv. Empty when the file is "
                               "missing.", "text"),
    "target_bodyid_coverage": ("bodyId-level n/N for the pair's TARGET type: "
                               "bodyIds reached (Checked) / all resolved "
                               "bodyIds of the type, from "
                               "target_neurons.csv. Empty when the file is "
                               "missing.", "text"),
    "intermediate": ("Intermediate (non-source, non-target) node on a path.", "text"),
    "n_paths_using": ("How many paths of the pair pass through this "
                      "intermediate.", "integer"),
    "classification": ("shared = the intermediate lies on >=2 paths of the "
                       "pair (full path set); unique = exactly 1.", "shared | unique"),
    "min_hop_position": ("Earliest hop index (1-based) at which the "
                         "intermediate appears across the pair's paths; also "
                         "its column in the report's layered network.", "integer"),
    # --- Type-level refill (post-hoc, scripts/TypeLevelRefill.py --in-run) --
    "emitted_weight": ("The type pair's weight as exported by the run "
                       "itself (connection_type.csv summed across its "
                       "conn_layer rows) — what the budgets left in.",
                       "integer"),
    "refill_weight": ("BodyId-pair weight the budgets cut that the refill "
                      "recovered for this emitted type pair: pairs on "
                      "asked-threshold paths over the involved types' "
                      "induced subgraph, minus the re-derived emitted set.",
                      "integer"),
    "refilled_total": ("emitted_weight + refill_weight: the refilled "
                       "type-level connection strength.", "integer"),
    "emitted_pair_count": ("BodyId pairs behind the emitted weight "
                           "(re-derived).", "integer"),
    "refill_pair_count": ("BodyId pairs behind the refill weight.", "integer"),
    "refilled_connection_ratio": ("refilled_total over the post type's "
                                  "ALL-post incoming weight (threshold-free "
                                  "F9 denominator, min synapse = 1).",
                                  "0-1"),
    "refilled_traversal_probability": ("The type pair's traversal "
                                       "probability over the UNION of "
                                       "emitted + refilled bodyId pairs, "
                                       "folded by the run's aggregate "
                                       "method (product = 1 - "
                                       "prod(1 - p_pair)).", "0-1"),
    "split_status": ("How the emitted/refill split was reconstructed: "
                     "rederived_floor (Edge Budget), rederived_topn "
                     "(StrongestFirst bite), rederived_floor+topn (both).",
                     "text"),
    "traversal_count": ("How many re-exploration paths use this bodyId "
                        "pair.", "integer"),
    "min_hop": ("Smallest 0-based hop index of the pair's pre endpoint "
                "across re-exploration paths.", "integer"),
    "max_hop": ("Largest 0-based hop index of the pair's pre endpoint "
                "across re-exploration paths.", "integer"),
    "path_prob": ("Overall path probability = product of edge traversal "
                   "probabilities $\\left(\\prod p_k\\right)$.", "0-1"),
    "min_weight": ("Smallest edge weight along the path.", "integer"),
    "length": ("Number of hops (edges) in the path.", "integer"),
    "nt_type": ("Predicted neurotransmitter type (ACh, GABA, glutamate, ...).", "text"),
    "nt_types": ("Neurotransmitter types along the path.", "text"),
    "conn_layer": ("Layer transition label of the connection, e.g. '0->1'.", "text"),
    "node": ("Neuron body id the shortest-discovery label describes.", "integer or text"),
    "dist": ("BFS distance in hops to the row's target (0 = the target itself).", "integer"),
    "probability": ("Edge traversal probability.", "0-1"),
    # --- Enrollment / status flags --------------------------------------------
    "isInPath": ("Whether the source neuron participates in at least one found path.", "boolean"),
    # --- NeuronBridge find-lines expansion map --------------------------------
    "source_query": ("Original query chip whose expansion produced this row.", "text"),
    "expanded_name": ("Dataset-local equivalent name the query chip expanded into.", "text"),
    "nb_dataset": ("NeuronBridge hosted release the expanded name belongs to.", "text"),
    "mapping_status": ("Type-mapper resolution status for the expansion (mapped, valid_split_evidence, evidence_only, unmapped, conflict, mapper unavailable).", "text"),
    "mapping_kind": ("Resolution kind behind the status (e.g. renamed or split; empty when unmapped).", "text"),
    "Checked": ("Whether the target neuron was reached/checked during traversal.", "boolean"),
    "Layer": ("Traversal layer at which the target neuron was reached.", "integer"),
    "viz_layer": ("Layer index assigned to the neuron in the 3D visualization.", "integer"),
    "status": ("Resolution/match status label.", "text"),
    "source_status": ("Resolution status of the source neuron profile.", "text"),
    "target_status": ("Resolution status of the target neuron profile.", "text"),
    "weak_source": ("Source profile has fewer than the minimum partner types.", "boolean"),
    "weak_target": ("Target profile has fewer than the minimum partner types.", "boolean"),
    "source_partner_count": ("Number of partners in the source profile.", "integer"),
    "target_partner_count": ("Number of partners in the target profile.", "integer"),
    "visualized": (
        "Whether the type row is included in the requested top-N visualization set.",
        "boolean",
    ),
    "visualization_rank": (
        "One-based rank in the sorted type-level visualization order.",
        "integer",
    ),
    "in_a": ("Partner is present in profile A (query).", "boolean"),
    "in_b": ("Partner is present in profile B (candidate).", "boolean"),
    "rank": ("Row rank (1 = best).", "integer"),
    "rank_a": ("Rank of the partner within profile A.", "integer"),
    "rank_b": ("Rank of the partner within profile B.", "integer"),
    "is_same_type": ("Source and target share the same neuron type.", "boolean"),
    "is_same_dataset": ("Source and target come from the same dataset.", "boolean"),
    "is_intra_type": ("The candidate is the same type as the query.", "boolean"),
    "conserved": ("Edge exists on both compared sides/conditions.", "boolean"),
    "present_L": ("Edge present on the left side.", "boolean"),
    "present_R": ("Edge present on the right side.", "boolean"),
    "note": ("Free-text annotation (e.g. why an edge is unconserved).", "text"),
    # --- Profile similarity metrics --------------------------------------------
    # Formulas are written inline so the exported run guide explains HOW each
    # score is computed, not only what it means.  A = one profile's partner set,
    # B = the other's; w_a/w_b are partner weights; ρ = Spearman correlation.
    "jaccard": ("Jaccard similarity of the two partner sets: "
                 "$\\lvert A \\cap B\\rvert / \\lvert A \\cup B\\rvert$.", "0-1"),
    "weighted_jaccard": (
        "Score-weighted (Ruzicka) Jaccard similarity: "
        "$\\sum \\min(w_a, w_b) / \\sum \\max(w_a, w_b)$ "
        "over the partner-type union.", "0-1"),
    "cosine": (
        "Cosine similarity of the partner-weight vectors: "
        "$(A \\cdot B) / (\\lVert A \\rVert \\cdot \\lVert B \\rVert)$ "
        "over the partner-type union (missing = 0).", "0-1"),
    "rank_corr": (
        "Raw Spearman rank correlation ($\\rho$) of partner weights on "
        "shared partners; NaN when fewer than 3 shared partners.", "-1 to 1"),
    "rank_union": (
        "Spearman rank correlation over the union of both partner sets "
        "(missing partners weighted 0), raw -1 to 1.", "-1 to 1"),
    "similarity": ("Overall similarity score used for sorting; see method/metric.", "0-1"),
    "profile_similarity": (
        "Connectivity-profile similarity evidence: "
        "$\\text{shared-count} / \\text{max-shared-count}$ (0-1).", "0-1"),
    "roi_similarity": (
        "ROI overlap similarity: cosine of input/output synapse distributions "
        "over primary ROIs, mirrored across the midline.", "0-1"),
    "intra_type_similarity": (
        "Mean pairwise similarity of the type's members — the intra-type "
        "reference other rows are ranked against.", "0-1"),
    "method": ("Similarity method used (vector / nblast).", "text"),
    "metric": ("Distance metric used (cosine / pearson).", "text"),
    "adjacency_score": (
        "Direct adjacency (shared-partner / synaptic-contact) score between "
        "the pair.", "number"),
    "shared_type_count": ("Number of partner types shared by both profiles: "
                         "$\\lvert A \\cap B\\rvert$.", "integer"),
    "union_type_count": ("Number of unique partner types across both profiles: "
                         "$\\lvert A \\cup B\\rvert$.", "integer"),
    "overlap_a_in_b": ("Fraction of A's partners present in B: "
                       "$\\lvert A \\cap B\\rvert / \\lvert A\\rvert$.", "0-1"),
    "overlap_b_in_a": ("Fraction of B's partners present in A: "
                       "$\\lvert A \\cap B\\rvert / \\lvert B\\rvert$.", "0-1"),
    "overlap_avg": ("Mean of overlap_a_in_b and overlap_b_in_a.", "0-1"),
    # --- Type-level aggregates --------------------------------------------------
    "avg_jaccard": ("Mean jaccard over all bodyId pairs of the type pair.", "0-1"),
    "avg_rank_union": ("Mean rank_union over all bodyId pairs of the type pair.", "-1 to 1"),
    "avg_cosine": ("Mean cosine over all bodyId pairs of the type pair.", "0-1"),
    "avg_adjacency_score": ("Mean adjacency score over all bodyId pairs.", "number"),
    "avg_shared_type_count": ("Mean shared partner-type count.", "number"),
    "avg_union_type_count": ("Mean union partner-type count.", "number"),
    "n_bodyid_comparisons": ("Number of bodyId-vs-bodyId comparisons aggregated.", "integer"),
    "n_bodyids": ("Number of bodyIds of the candidate type.", "integer"),
    "target_type_members": ("Member count of the candidate type in the target "
                            "dataset; very large counts indicate coarse or "
                            "hemilineage-scale annotations (e.g. Mi15).", "integer"),
    "n_complete_sources": ("Source bodyIds with complete profiles.", "integer"),
    "n_incomplete_sources": ("Source bodyIds with incomplete profiles.", "integer"),
    "source_dataset": ("Dataset of the query/source neurons.", "text"),
    "target_dataset": ("Dataset of the candidate/target neurons.", "text"),
    "source_bodyId": ("BodyId of the source neuron.", "integer"),
    "source_type": ("Type of the source neuron.", "text"),
    "source_instance": ("Instance name of the source neuron.", "text"),
    "target_bodyId": ("BodyId of the candidate neuron.", "integer"),
    "target_type": ("Type of the candidate neuron.", "text"),
    "target_instance": ("Instance name of the candidate neuron.", "text"),
    "anchor": ("Anchor type name the mapping row was resolved from.", "text"),
    "same name": ("The dataset uses the identical type name (no mapping needed).", "boolean"),
    # --- Cross-dataset comparison ------------------------------------------------
    "threshold": ("Minimum threshold applied (ASKED value — synapse count, "
                  "or the min connection ratio on weight-basis='connection_ratio' "
                  "runs; see applied_threshold for the real cutoff in "
                  "effect).", "number"),
    "requested_threshold": ("The user-entered threshold before any budget "
                            "effect (Min Synapse Count; float min connection "
                            "ratio on weight-basis='connection_ratio' runs).",
                            "number"),
    "applied_threshold": ("The CANONICAL (minimal) threshold that reproduces "
                          "this run's output: w2 + 1 for a budget-bitten synapse "
                          "run (w2 = strongest dropped path bottleneck — every "
                          "threshold in (w2, τ] yields the identical set; the "
                          "weakest distinct ratio tier above w2 on "
                          "connection-ratio runs), else the asked threshold for "
                          "complete runs (whose natural τ equals it). See "
                          "applied_threshold_source for which mechanism(s) set "
                          "it.", "number"),
    "applied_threshold_source": ("Which mechanism(s) determined "
                                 "applied_threshold: 'requested' (no budget "
                                 "bit), 'strongest_first_budget', "
                                 "'edge_budget', or "
                                 "'strongest_first_budget+edge_budget'.",
                                 "text"),
    "strongest_first_budget": ("Effective StrongestFirst path budget for the "
                               "run (the auto default 1,000,000 when the "
                               "user left Max Paths at 0).", "integer"),
    "strongest_first_budget_bitten": ("True when the StrongestFirst path "
                                      "budget was reached and the output was "
                                      "τ-bounded.", "boolean"),
    "strongest_first_tau": ("StrongestFirst budget LANDING τ (same value as "
                            "the τ column).", "number"),
    "tau_canonical": ("The minimal threshold that reproduces this run's "
                      "materialized path set: w2 + 1 when a budget bite left "
                      "a gap in (w2, τ] on synapse runs (the weakest distinct "
                      "ratio tier above w2 on connection-ratio runs); the "
                      "natural τ otherwise. Distinct from the landing τ, "
                      "which is only the collapse bound.", "number"),
    "strongest_dropped": ("w2 — the strongest path bottleneck NOT in the "
                          "output (budget-bitten runs): lowering the threshold "
                          "to w2 or below admits new paths; any value in "
                          "[w2+1, τ] changes nothing. Empty for complete "
                          "runs.", "number"),
    "strongest_dropped_bottleneck": ("w2 — the strongest path bottleneck NOT "
                                     "emitted after the StrongestFirst budget "
                                     "bit (same value as strongest_dropped). "
                                     "Empty for complete runs.", "number"),
    "strongest_retained_bottleneck": ("W* — the widest-path (best bottleneck) "
                                      "ceiling after the lossless pruning "
                                      "passes. Lossless pruning never changes "
                                      "it, so it states that the top paths "
                                      "are untouched by pruning.", "number"),
    "edge_budget": ("Configured Edge Budget cap in bodyId-level edges "
                    "(0/empty = off). 'all' path mode only — shortest mode is "
                    "never floored.", "integer"),
    "edge_budget_applied": ("True when the Edge Budget floor fired for this "
                            "run (the lossless-pruned cone exceeded the "
                            "cap).", "boolean"),
    "edge_budget_landing": ("w1 — the edge-weight tier that determined the "
                            "Edge Budget landing: the strongest tier whose "
                            "admission would exceed the budget. The floor "
                            "w0 lands just above it (w0 = w1 + 1 "
                            "conceptually).", "number"),
    "drop_untyped": ("Neuron-label filter: when True, edges touching untyped "
                     "neurons (empty / Unknown / NaN / bodyId-fallback type "
                     "labels — the shared predicate in utils.label_utils) "
                     "were removed. Single-dataset tabs apply it after label "
                     "enrichment and before graph construction; cross-dataset "
                     "comparison applies it after standardized labels and "
                     "before aggregation. Dropped rows are exported to "
                     "untyped_dropped_records.csv.",
                     "boolean"),
    "untyped_side": ("Which side of a dropped edge is untyped: 'pre', "
                     "'post', or 'pre+post'.", "text"),
    "untyped_dropped_rows": ("Connections removed by the Drop Untyped "
                             "Neurons filter for this run.", "integer"),
    "untyped_dropped_neurons": ("Distinct untyped neurons whose incident "
                                "connections were dropped for this run.",
                                "integer"),
    "pruned": ("True when the Edge-Budget floor (the only lossy stage) fired "
               "for this run: the cone exceeded the budget and was floored at "
               "w0 = (N-th strongest edge) + 1 — the run is exactly a "
               "complete run at edge_weight_floor. τ-collapsed rows are "
               "marked skipped, not pruned.", "boolean"),
    "edge_weight_floor": ("Fix D floor weight (w0) when the Edge Budget fired; "
                          "every path in the output has bottleneck >= this value "
                          "and the run is equivalent to a complete run at this "
                          "threshold. Empty when no floor was applied.", "number"),
    "applied_folder": ("Folder containing the materialized output for this "
                       "requested threshold. A skipped threshold aliases the "
                       "real applied folder instead of creating duplicate "
                       "output.", "integer"),
    "conservation": ("Number/fraction of datasets in which the edge is present.", "text"),
    "conserved_at_lowest": ("Edge present in every dataset at the lowest threshold.", "boolean"),
    "tau": ("StrongestFirst budget LANDING τ: the weakest kept path's "
           "bottleneck — the maximal threshold equivalent to this run (the "
           "collapse bound; every threshold up to this value yields the "
           "identical set). For COMPLETE runs this is the natural τ — the "
           "weakest emitted path's bottleneck, which is also the canonical "
           "minimal threshold.", "number"),
    "paths_complete": ("True when the path set is complete; False when a budget "
                       "cutoff applied.", "boolean"),
    "skipped": ("Feature G (duplicate-threshold skipping): True when this input "
                "threshold was NOT re-enumerated — its path set is identical to the "
                "earlier run it duplicates (every threshold up to that run's τ "
                "yields the same set). 'all' path mode only.", "boolean"),
    "duplicate_of": ("For skipped thresholds: the earlier input threshold (or "
                     "the τ folder, applied_folder) whose run this row "
                     "duplicates.", "integer"),
    "applied_folder": ("F5 τ-folder discipline: the applied-threshold value "
                       "holding this threshold's real output. Collapsed "
                       "thresholds have no folder of their own — their frames "
                       "alias that applied folder's materialization (fresh τ "
                       "denominators). On disk: a genuine requested run keeps "
                       "the bare minsyn_{requested}; the collapse floor is "
                       "minsyn_{applied}_applied_floor. A requested level "
                       "coinciding with the floor it aliases to is skipped "
                       "silently (no _skipped marker).", "integer"),
    "max_paths_bodyid": ("Path budget for StrongestFirst enumeration (0/empty "
                         "uses the internal 1,000,000 auto budget; legacy "
                         "unbounded enumerators require an explicit API "
                         "algorithm).", "integer"),
    "jaccard_similarity": ("Jaccard similarity of the two datasets' edge sets: "
                            "$\\lvert E_1 \\cap E_2\\rvert / "
                            "\\lvert E_1 \\cup E_2\\rvert$.", "0-1"),
    "ruzicka_similarity": ("Ruzicka (weighted Jaccard) of edge weights: "
                            "$\\sum \\min(W_1, W_2) / \\sum \\max(W_1, W_2)$.", "0-1"),
    "pearson_correlation": ("Pearson correlation of matched edge weights (sparse-data caution).", "-1 to 1"),
    "edges_in_d1": ("Number of edges in dataset 1.", "integer"),
    "edges_in_d2": ("Number of edges in dataset 2.", "integer"),
    "common_edges": ("Number of edges present in both datasets: "
                      "$\\lvert E_1 \\cap E_2\\rvert$.", "integer"),
    "union_edges": ("Number of unique edges across both datasets: "
                     "$\\lvert E_1 \\cup E_2\\rvert$.", "integer"),
    "unique_to_d1": ("Edges present only in dataset 1.", "integer"),
    "unique_to_d2": ("Edges present only in dataset 2.", "integer"),
    "edge_rank_correlation": ("Spearman correlation of edge-weight ranks over the union (missing = 0).", "-1 to 1"),
    "cosine_similarity": ("Cosine similarity of edge-weight vectors over the union: "
                            "$(E_1 \\cdot E_2) / "
                            "(\\lVert E_1 \\rVert \\cdot \\lVert E_2 \\rVert)$.", "0-1"),
    "path_rank_correlation": ("Spearman correlation of path-probability ranks.", "-1 to 1"),
    "spearman_rank_correlation": ("Spearman rank correlation of the compared values.", "-1 to 1"),
    "rv_coefficient": ("RV coefficient (multivariate correlation) of the edge "
                         "matrices: $\\langle A, B\\rangle_F^2 / "
                         "(\\lVert A \\rVert_F^2 \\cdot \\lVert B \\rVert_F^2)$.", "0-1"),
    "dataset_1": ("First dataset of the pairwise comparison.", "text"),
    "dataset_2": ("Second dataset of the pairwise comparison.", "text"),
    # --- Dataset metadata ---------------------------------------------------------
    "total_neurons": ("Total number of neurons in the dataset.", "integer"),
    "typed_neurons": ("Number of neurons with a type assignment.", "integer"),
    "untyped_neurons": ("Number of neurons without a type assignment.", "integer"),
    "type_coverage_pct": ("Fraction of neurons that are typed.", "percent"),
    "total_presynaptic": ("Total presynaptic site count.", "integer"),
    "total_postsynaptic": ("Total postsynaptic site count.", "integer"),
    "total_synapses": ("Total synapse count.", "integer"),
    "roi_count": ("Number of ROI regions covered.", "integer"),
    "coverage_notes": ("Free-text coverage/caveat notes.", "text"),
    "base_pre": ("Hemisphere-neutral (suffix-stripped) presynaptic type.", "text"),
    "base_post": ("Hemisphere-neutral (suffix-stripped) postsynaptic type.", "text"),
    # --- NeuronBridge ---------------------------------------------------------------
    "score": (
        "NeuronBridge match score; higher means a better EM↔LM match "
        "(typical range 0-50000).", "number"),
    "image_id": ("Identifier of the matched light-microscopy image.", "text"),
    "lm_sample": ("Light-microscopy sample identifier.", "text"),
    "match_type": ("Match algorithm that produced the score (cds / pppm).", "text"),
    "library": ("Driver-line library (e.g. Split-GAL4, GAL4/LexA).", "text"),
    "canonical_type": ("Cross-dataset standardized (canonical) type name.", "text"),
    "best_max_score": ("Highest max score across datasets for the canonical type.", "number"),
    "total_labeled_N": ("Total number of labeled neurons across datasets.", "integer"),
    "n_neurons": ("Number of matched neurons for the line.", "integer"),
    "n_types": ("Number of distinct neuron types matched.", "integer"),
    "mean_score": ("Mean NeuronBridge score of the matched neurons.", "number"),
    "max_score": ("Best NeuronBridge score among the matched neurons.", "number"),
    "n_neurons_HMS": ("Neurons with a high match score (above the high cutoff).", "integer"),
    "n_types_HMS": ("Types with a high match score.", "integer"),
    "n_neurons_MS": ("Neurons above the minimum score cutoff.", "integer"),
    "n_types_MS": ("Types above the minimum score cutoff.", "integer"),
    "Qf": ("Quality factor of the line labeling.", "number"),
    "colabel_sparsity": ("Sparsity of the line's co-labeling pattern.", "number"),
    "labeled_N": ("Number of labeled neurons of the type.", "integer"),
    "avg_score": ("Average NeuronBridge score of the matched neurons.", "number"),
    "median_score": ("Median NeuronBridge score of the matched neurons.", "number"),
    "Q1_score": ("First-quartile NeuronBridge score of the matched neurons.", "number"),
    "Q3_score": ("Third-quartile NeuronBridge score of the matched neurons.", "number"),
    "typed_N_in_dataset": ("Number of typed neurons in the dataset.", "integer"),
    "_passes_min_score": ("Whether the match passes the run's min_score cutoff.", "boolean"),
    # --- Auto threshold-density alignment (plan §8c.2) -----------------------
    "density": ("Edge density at the threshold: E(t)/N with a t-independent "
                "N (annotated nodes in the searched graph), i.e. bodyId edges "
                "per searched neuron at or above the Min Synapse Count.", "number"),
    "normalizer": ("Density normalizer in use (per_node = E(t)/N primary; "
                   "raw/per_source/cone are diagnostics only).", "text"),
    "basis": ("Graph object the density counts: bodyId_edges for auto mode "
              "(type_pairs for the whole-dataset prober diagnostic).", "text"),
    "w_start": ("Low end of a dataset's threshold window: max(3, applied "
                "threshold).", "integer"),
    "w_star_measured": ("Measured retained ceiling: max bottleneck over the "
                        "actually enumerated paths (= high end of the "
                        "window).", "integer"),
    "w_star_stored": ("Provenance-stored strongest_retained_bottleneck; "
                      "untrusted when it disagrees with w_star_measured.",
                      "integer"),
    "w_star_mismatch": ("True when the stored strongest_retained_bottleneck "
                        "differs from the measured retained ceiling.",
                        "boolean"),
    "path_complete_from": ("Threshold from which the path curve is complete "
                           "(max(w_start, tau_canonical)); below it the path "
                           "curve is unknown.", "integer"),
    "is_materialized": ("Whether this threshold was actually materialized "
                        "(vs an interpolated curve point).", "boolean"),
    "edge_count": ("Number of bodyId edges with weight >= the threshold in the "
                   "query-scoped searched graph.", "integer"),
    "path_count": ("Number of enumerated bodyId paths with bottleneck >= the "
                   "threshold (diagnostic; hub-inflated).", "integer"),
    "applied": ("Applied (materialized) threshold actually compared for the "
                "dataset.", "integer"),
    "budget_bitten": ("Whether the StrongestFirst path budget cut the "
                      "enumeration at the requested threshold.", "boolean"),
    "n_paths": ("Number of enumerated bodyId paths.", "integer"),
    "n_edges": ("Number of bodyId edges in the query-scoped searched graph "
                "at the applied threshold.", "integer"),
    "n_nodes": ("Number of bodyId nodes enrolled in the searched graph.",
                "integer"),
    "mode": ("Row origin for aligned density rows: vertical (same threshold), "
             "horizontal (same density), or both.", "text"),
    "level_continuous": ("Vertical row's shared integer threshold, or the "
                         "density level for a horizontal row.", "number"),
    "level_normalized": ("Normalized density level a horizontal row is "
                         "aligned at.", "number"),
    "degenerate": ("True when a mode collapsed to a single (or empty) "
                   "aligned row.", "boolean"),
    "clamped": ("True when an inverted threshold was clamped to its dataset's "
                "window boundary.", "boolean"),
    "max_abs_deviation": ("Largest distance between the target density level "
                          "and the density actually achieved at each dataset's "
                          "chosen integer threshold (quantization gap).",
                          "number"),
    "n_nodes_typed": ("Searched-graph nodes whose type label is a real name "
                      "(Unknown/empty/digit labels are untyped) — the primary "
                      "density denominator N.", "integer"),
    "n_nodes_untyped": ("Searched-graph nodes with an untyped label (present "
                        "in the curated table but Unknown/empty/digit).",
                        "integer"),
    "n_nodes_debris": ("Searched-graph ids absent from the curated neuron "
                       "table (segmentation debris); always excluded from "
                       "the density universe.", "integer"),
    "n_edges_active_basis": ("Cone edges at w_start counted under the "
                             "active basis (typed with Drop Untyped on, "
                             "all-but-debris when off) — the same number "
                             "density_curves.csv reports at w_start.",
                             "integer"),
    # v2.2 similarity-matrix columns (plan-similarity-matrix-schema-v2)
    "path_jaccard_similarity": (
        "v2.2 path-level representative: Jaccard similarity of the two "
        "datasets' path sets (union of paths as node sequences). NaN when "
        "no path data was provided.", "0-1"),
    "path_top20_overlap": (
        "v2.2 detail metric: overlap of the two datasets' top-20 paths "
        "(ranked by traversal probability). NaN without path data.",
        "0-1"),
    "hop_profile_w1": (
        "v2.2 detail metric: W1 (earth-mover) distance between the two "
        "datasets' hop-count profiles of paths, mapped to a similarity "
        "(1/(1+W1)). NaN without path data.", "0-1"),
    "netsimile_similarity": (
        "v2.2 graph-level representative: NetSimile-lite similarity of the "
        "two weighted edge lists (degree/strength signature vectors, "
        "1/(1+normalized Canberra distance); the range floors at 0.5 "
        "because every Canberra term is <= 1).", "0.5-1"),
    "coverage_d1": (
        "v2.2 detail metric: overlap coefficient restricted to dataset 1's "
        "edges (|shared| / |d1 edges|).", "0-1"),
    "coverage_d2": (
        "v2.2 detail metric: overlap coefficient restricted to dataset 2's "
        "edges (|shared| / |d2 edges|).", "0-1"),
    "coverage_min": (
        "v2.2 detail metric: conservative overlap coefficient "
        "(|shared| / min(|d1|, |d2|) edges) — the smaller dataset's "
        "coverage.", "0-1"),
    "top20_overlap": (
        "v2.2 detail metric: overlap of the two datasets' top-20 edges by "
        "weight.", "0-1"),
    "strength_w1_out": (
        "v2.2 detail metric: W1 distance between outgoing-strength "
        "distributions, mapped to a similarity (1/(1+W1)).", "0-1"),
    "strength_w1_in": (
        "v2.2 detail metric: W1 distance between incoming-strength "
        "distributions, mapped to a similarity (1/(1+W1)).", "0-1"),
    "type_coverage": (
        "Morphological similarity: coverage of the target type's members "
        "found for the query (types with a coverage snapshot only — "
        "written whenever coverage exists).", "0-1"),
    "edges_retained_from_prev": (
        "Threshold sensitivity: unique edges retained from the previous "
        "(lower) threshold; blank on the first threshold of a schedule and "
        "in combination mode.", "integer"),
    "retention_rate": (
        "Threshold sensitivity: edges_retained_from_prev / previous "
        "edge_count; blank where the retention columns are blank.",
        "0-1"),
    "edges_lost": (
        "Threshold sensitivity: unique edges present at the previous "
        "threshold but not this one; blank on the first threshold of a "
        "schedule and in combination mode.", "integer"),
    "total_edges": (
        "Unified summary: unique (source, target) type pairs at this "
        "threshold (per-layer occurrences of the same pair are not "
        "double-counted).", "integer"),
    "total_layer_rows": (
        "Unified summary: raw connection rows including per-layer "
        "occurrences of the same pair.", "integer"),
    "total_weight": (
        "Unified summary: sum of connection weights at this threshold.",
        "float"),
    "mean_weight": (
        "Unified summary: mean connection weight at this threshold.",
        "float"),
    "unique_sources": (
        "Unified summary: distinct source types with at least one edge at "
        "this threshold.", "integer"),
    "unique_targets": (
        "Unified summary: distinct target types with at least one edge at "
        "this threshold.", "integer"),
}


def glossary_entry(column: str) -> tuple:
    """Return (description, range) for a column, with a safe fallback."""
    entry = COLUMN_GLOSSARY.get(column)
    if entry is None and str(column).startswith("threshold_"):
        # Comparison exports suffix the shared field name with the dataset
        # (for example ``threshold_banc_v888``). Keep one glossary rule for
        # the dynamic family while still documenting the actual column name.
        entry = COLUMN_GLOSSARY.get("threshold_")
    return entry or ("(see docs/OUTPUT_FILES.md)", "")


# Inline-math markers: descriptions may wrap a formula fragment in ``$...$``
# (LaTeX). Each renderer presents those fragments as formatted math:
#   - markdown: keeps ``$...$`` so the viewer's MathJax/KaTeX renders it
#   - html:     rewrites to MathJax ``\(...\)`` and loads MathJax in the head
#   - txt:      strips the markers for plain text

def _math_to_md(description: str) -> str:
    """Return the description unchanged (already uses $...$ inline math)."""
    return description


def _math_to_html(description: str) -> str:
    """Render $...$ fragments as self-contained styled HTML math.

    No MathJax / external script is required: each fragment becomes a Unicode
    math string with <sub>/<sup> markup wrapped in a <span class="math">, so
    the guide renders the formula correctly even when opened offline.
    """
    return re.sub(
        r"\$([^$]+)\$",
        lambda m: '<span class="math">' + _latex_to_html(m.group(1)) + '</span>',
        description,
    )


# Shared LaTeX -> glyph substitution map (order-sensitive: long commands first).
# Note: escaped underscores (\_) are intentionally NOT mapped here so the
# subscript parser can tell a real subscript from a literal underscore.
_MATH_SYMBOL_SUBS = [
    (r"\lVert", "‖"), (r"\rVert", "‖"),
    (r"\lvert", "|"), (r"\rvert", "|"),
    (r"\left(", "("), (r"\right)", ")"),
    (r"\langle", "⟨"), (r"\rangle", "⟩"),
    (r"\min", "min"), (r"\max", "max"),
    (r"\cdot", "·"), (r"\sum", "Σ"), (r"\prod", "Π"),
    (r"\rho", "ρ"), (r"\cap", "∩"), (r"\cup", "∪"),
    (r"\,", " "),
]


def _latex_to_uniform(math_text: str) -> str:
    """Convert the small LaTeX subset used by the glossary to a uniform string
    with ``_{<..>}`` / ``^{<..>}`` and single-token sub/superscript markers.
    """
    stash: dict = {}

    def _store(m):
        key = f"\x00T{len(stash)}\x00"
        stash[key] = m.group(1).replace(r"\_", "_")
        return key

    math_text = re.sub(r"\\text\{([^}]*)\}", _store, math_text)
    for old, new in _MATH_SYMBOL_SUBS:
        math_text = math_text.replace(old, new)
    # Subscripts / superscripts: braced forms first, then single-token forms.
    math_text = re.sub(r"(?<!\\)_\{([^}]*)\}", r"_{<\1>}", math_text)
    math_text = re.sub(r"(?<!\\)\^\{([^}]*)\}", r"^{<\1>}", math_text)
    math_text = re.sub(r"(?<!\\)_([A-Za-z0-9]+)", r"_{<\1>}", math_text)
    math_text = re.sub(r"(?<!\\)\^([A-Za-z0-9]+)", r"^{<\1>}", math_text)
    math_text = math_text.replace(r"\_", "_").replace("\\", "")
    # LaTeX puts a space after an opening delimiter (| A -> |A) and before a
    # closing one (A | -> A|); drop it so set/norm formulas read tightly.
    math_text = re.sub(r"(?<=[|‖⟨])\s+", "", math_text)
    math_text = re.sub(r"\s+(?=[|‖⟩])", "", math_text)
    math_text = re.sub(r"\s{2,}", " ", math_text)
    for key, inner in stash.items():
        math_text = math_text.replace(key, inner)
    return math_text


def _latex_to_html(math_text: str) -> str:
    """Render a math fragment as HTML with </sub>/<sup> markup."""
    uniform = _latex_to_uniform(math_text)
    uniform = re.sub(r"_\{<([^>]*)>\}", r"<sub>\1</sub>", uniform)
    uniform = re.sub(r"\^\{<([^>]*)>\}", r"<sup>\1</sup>", uniform)
    return uniform


def _math_to_txt(description: str) -> str:
    """Strip $...$ markers and render readable plain text for the txt output."""
    return re.sub(
        r"\$([^$]+)\$",
        lambda m: _latex_to_plain(m.group(1)),
        description,
    )


def _latex_to_plain(text: str) -> str:
    """Convert the small LaTeX set the glossary uses back to plain text.

    Keep it minimal and order-sensitive (long commands first), so the plain
    txt guide stays readable without embedding raw TeX. Function names
    (min/max) are preserved as words; sub/superscripts are flattened to a
    single line (w_a -> wa) and set/norm delimiters lose the inner spacing.
    """
    uniform = _latex_to_uniform(text)
    uniform = re.sub(r"[_^]\{<([^>]*)>\}", r"\1", uniform)
    return re.sub(r"\\[a-zA-Z]+\b", "", uniform)  # drop any leftover commands


# =============================================================================
# Per-tool file specifications. Each entry describes one file (or a glob of
# files) in the run folder. ``columns`` reference COLUMN_GLOSSARY keys;
# ``matrix`` marks matrix-style tables whose axes, not columns, carry meaning.
# =============================================================================

_NEURON_TABLE_NOTE = (
    "plus the full NeuPrint neuron-table columns (size, status, soma side, "
    "cross-dataset type names, ...)"
)

_PATH_COLUMNS = [
    "path", "weights", "probabilities", "ratios", "min_weight",
    "path_prob", "min_ratio", "length", "nt_types",
]

_CONNECTION_TYPE_COLUMNS = [
    "type_pre", "type_post", "weight", "connection_ratio",
    "traversal_probability", "block_probability", "nt_type",
]

_HOMOLOG_RESULT_COLUMNS = [
    "source_bodyId", "source_type", "target_bodyId", "target_type",
    "target_dataset", "adjacency_score", "shared_type_count",
    "union_type_count", "rank_union",
    "jaccard", "weighted_jaccard", "cosine", "is_same_type",
    "is_same_dataset", "source_status", "target_status", "weak_source",
    "weak_target", "source_partner_count", "target_partner_count",
]

# Profiles carry one similarity matrix per metric/direction.  These are the
# score columns whose values fill the matrix cells; the glossary explains the
# formula behind each so the profiling run guide documents the score logic.
_PROFILING_METRIC_COLUMNS = [
    "jaccard", "weighted_jaccard", "cosine", "rank_corr", "rank_union",
]

_FIND_NETWORK = {
    "title": "Find Network",
    "summary": (
        "Direct connections among the neurons matching one query set "
        "(patterns supported)."
    ),
    "files": [
        {"pattern": "data_details/connection_type.csv",
         "description": "Direct connections aggregated by type pair.",
         "preview": True,
         "preview_title": "Connections by type pair",
         "columns": _CONNECTION_TYPE_COLUMNS},
        {"pattern": "data_details/neurons.csv",
         "description": "The resolved neuron set. Key columns: bodyId, "
                        "instance, type, pre, post — " + _NEURON_TABLE_NOTE,
         "preview": True,
         "preview_title": "Resolved neuron set"},
        {"pattern": "data_details/parameters.csv",
         "description": "Run parameters as a table."},
        {"pattern": "parameters.txt",
         "description": "Human-readable record of all run parameters."},
        {"pattern": "all_attributes.json",
         "description": "Serialized run attributes (machine-readable)."},
        {"pattern": WARNING_FILENAME,
         "description": "Warnings/notes collected during the run "
                        "(rendered in the Warnings section above)."},
        {"pattern": "visualization/Network_*.html",
         "description": "Interactive network graph of the found connections."},
        {"pattern": "visualization/Heatmap_*.html",
         "description": "Connection weight heatmap with interactive Ward "
                        "dendrograms."},
        {"pattern": "visualization/visualization_data/*_data_connections.csv",
         "description": "Edge list backing the HTML visualizations.",
         "columns": ["source", "target", "weight", "ratio", "probability",
                     "nt_type"]},
        {"pattern": "visualization/visualization_data/*_data_original_paths.csv",
         "description": "Original path records backing the HTML files."},
        {"pattern": "visualization/visualization_data/network_edges_input.csv",
         "description": "Input edge list used to build the network.",
         "columns": ["source", "target", "weight"]},
    ],
}

_PATHFINDING_FILES = [
    {"pattern": "*_allpaths_type.csv",
     "description": "Primary path table (type-level, UI default).",
     "preview": True,
     "preview_title": "Path table (type-level)",
     "columns": _PATH_COLUMNS},
    {"pattern": "*_allpaths_bodyId_paths.csv",
     "description": "BodyId-level path table (written when Skip BodyId is "
                    "off). Same score columns plus bodyId-level endpoints.",
     "preview": True,
     "preview_title": "Path table (bodyId-level)",
     "columns": _PATH_COLUMNS},
    {"pattern": "source_neurons.csv",
     "description": "Resolved source neurons. Key columns: isInPath, bodyId, "
                    "instance, type, pre, post — " + _NEURON_TABLE_NOTE,
     "preview": True,
     "preview_title": "Source neurons",
     "columns": ["isInPath"]},
    {"pattern": "target_neurons.csv",
     "description": "Resolved target neurons. Key columns: Checked, Layer, "
                    "bodyId, instance, type — " + _NEURON_TABLE_NOTE,
     "preview": True,
     "preview_title": "Target neurons",
     "columns": ["Checked", "Layer"]},
    {"pattern": "all_attributes.json",
     "description": "Serialized run attributes (machine-readable). Includes "
                    "the applied-threshold provenance block: "
                    "requested_threshold, applied_threshold, "
                    "applied_threshold_source, strongest_first_tau, "
                    "tau_canonical, strongest_dropped_bottleneck (w2), "
                    "edge_budget, edge_budget_landing (w1), "
                    "edge_weight_floor (w0), strongest_retained_bottleneck "
                    "(W*), paths_complete."},
    {"pattern": "parameters.txt",
     "description": "Human-readable record of all run parameters, including "
                    "the same applied-threshold provenance keys."},
    {"pattern": "data_details/untyped_dropped_records.csv",
     "description": "Connections removed by the Drop Untyped Neurons filter "
                    "(written only when the filter actually dropped rows) "
                    "with dataset/threshold/layer context and an untyped_side "
                    "flag. An untyped source/target can remain enrolled in "
                    "source_neurons.csv/target_neurons.csv while its incident "
                    "edges were removed.",
     "columns": ["drop_untyped", "untyped_side"]},
    {"pattern": "visualization/visualization_data/type_paths_visualized.csv",
     "description": "When the Visualization Edge Limit trimmed the drawn "
                    "graph: the exact complete type-level path rows the "
                    "rendered network represents."},
    {"pattern": "bodyId_visualization/Network_*.html",
     "description": "BodyId-level interactive network graph (written when "
                    "Skip BodyId is off). Shares the drawing cap with the "
                    "type-level visualization."},
    {"pattern": "bodyId_visualization/Heatmap_*.html",
     "description": "BodyId-level connection weight heatmap (Skip BodyId "
                    "off) with interactive Ward dendrograms. Same drawing cap "
                    "as the type-level heatmap."},
    {"pattern": "bodyId_visualization/Sankey_*.html",
     "description": "BodyId-level Sankey flow diagram (Skip BodyId off). "
                    "Same drawing cap as the type-level Sankey."},
    {"pattern": "bodyId_visualization/visualization_data/*_data_connections.csv",
     "description": "Edge list backing the bodyId-level HTML "
                    "visualizations.",
     "columns": ["source", "target", "weight", "ratio", "probability",
                 "nt_type"]},
    {"pattern": "bodyId_visualization/visualization_data/*_data_original_paths.csv",
     "description": "Original bodyId-level path records backing the HTML "
                    "files."},
    {"pattern": "bodyId_visualization/visualization_data/bodyId_paths_visualized.csv",
     "description": "When the Visualization Edge Limit trimmed the bodyId "
                    "graph: the exact bodyId path rows the rendered network "
                    "represents (parallel to type_paths_visualized.csv)."},
    {"pattern": "network_early/Network_*.html",
     "description": "Early type-level network preview drawn from the "
                    "discovered graph before path reconstruction (only with "
                    "visualize_before_reconstruct=True)."},
    {"pattern": "network_early_bodyId/Network_*.html",
     "description": "Early bodyId-level network preview (only with "
                    "visualize_before_reconstruct=True and Skip BodyId "
                    "off)."},
    {"pattern": WARNING_FILENAME,
     "description": "Warnings/notes collected during the run (rendered in "
                    "the Warnings section above)."},
    {"pattern": "data_details/connection_type.csv",
     "description": "Edges aggregated by type pair.",
     "columns": _CONNECTION_TYPE_COLUMNS},
    {"pattern": "data_details/conn_mat_type_weight.csv",
     "description": "Type-level edge-weight matrix.",
     "matrix": "rows/columns = neuron types, values = synapse counts"},
    {"pattern": "data_details/conn_mat_type_ratio.csv",
     "description": "Type-level connection-ratio matrix.",
     "matrix": "rows/columns = neuron types, values = connection_ratio"},
    {"pattern": "data_details/conn_mat_type_prob.csv",
     "description": "Type-level traversal-probability matrix.",
     "matrix": "rows/columns = neuron types, values = traversal_probability"},
    {"pattern": "data_details/conn_mat_type_nt.csv",
     "description": "Type-level neurotransmitter matrix.",
     "matrix": "rows/columns = neuron types, values = nt_type"},
    {"pattern": "data_details/neurons_included.csv",
     "description": "All neurons participating in the found connections.",
     "columns": ["group", "bodyId", "type", "instance", "nt_type"]},
    {"pattern": "data_details/total_weight_layer.csv",
     "description": "Total weight per intermediate layer.",
     "columns": ["conn_layer", "weight"]},
    {"pattern": "data_details/parameters.csv",
     "description": "Run parameters as a table."},
    {"pattern": "data_details/connection_info_bodyId.csv",
     "description": "BodyId-level edge table (written when Skip BodyId is "
                    "off)."},
    {"pattern": "visualization/Network_*.html",
     "description": "Interactive network graph of the found paths."},
    {"pattern": "visualization/Heatmap_*.html",
     "description": "Connection weight heatmap."},
    {"pattern": "visualization/Sankey_*.html",
     "description": "Sankey flow diagram of the pathways."},
    {"pattern": "visualization/visualization_data/*_data_connections.csv",
     "description": "Edge list backing the HTML visualizations.",
     "columns": ["source", "target", "weight", "ratio", "probability",
                 "nt_type"]},
    {"pattern": "visualization/visualization_data/*_data_original_paths.csv",
     "description": "Original path records backing the HTML files."},
    {"pattern": "hemisphere_symmetry/symmetry_summary.json",
     "description": "Ipsilateral/contralateral Jaccard and conserved/union "
                    "counts (when Symmetry Analysis is on)."},
    {"pattern": "hemisphere_symmetry/symmetry_ipsi.csv",
     "description": "L-L vs R-R edge comparisons.",
     "columns": ["base_pre", "base_post", "weight_L", "weight_R",
                 "present_L", "present_R", "conserved", "ratio"]},
    {"pattern": "hemisphere_symmetry/symmetry_contra.csv",
     "description": "L-R vs R-L edge comparisons.",
     "columns": ["base_pre", "base_post", "weight_LR", "weight_RL",
                 "present_LR", "present_RL", "conserved", "ratio"]},
    {"pattern": "hemisphere_symmetry/conserved_edges.csv",
     "description": "Edges conserved across hemispheres.",
     "columns": ["base_pre", "base_post", "type", "note"]},
    {"pattern": "hemisphere_symmetry/unconserved_edges.csv",
     "description": "Edges not conserved across hemispheres.",
     "columns": ["base_pre", "base_post", "type", "note"]},
    {"pattern": "hemisphere_symmetry/pairwise_strength.csv",
     "description": "Weight comparisons for matched hemisphere edge pairs.",
     "columns": ["base_pre", "base_post", "type", "weight_L", "weight_R",
                 "diff", "ratio", "weight_LR", "weight_RL"]},
    {"pattern": "hemisphere_symmetry/type_counts_by_role.csv",
     "description": "Per-type source/intermediate/target counts."},
    {"pattern": "find_reciprocal/reciprocal_connection_type.csv",
     "description": "Type-level reciprocal connections (when Find Reciprocal "
                    "is on).",
     "columns": _CONNECTION_TYPE_COLUMNS},
    {"pattern": "find_reciprocal/reciprocal_type_network.html",
     "description": "Reciprocal network visualization."},
    {"pattern": "find_reciprocal/reciprocal_type_heatmap.html",
     "description": "Reciprocal heatmap visualization."},
    {"pattern": "find_reciprocal/parameters.csv",
     "description": "Reciprocal-analysis parameters."},
    # Post-hoc pair-report additions (scripts/PathsPairReport.py writes
    # these INTO the run folder after the fact; the entries surface in the
    # run guide only when the files exist).
    {"pattern": "path_report.html",
     "description": "Per-source-target-pair HTML report (written "
                    "automatically after pathfinding runs; also via "
                    "scripts/PathsPairReport.py): Overview / Global / "
                    "Pair Explorer / Data tabs, interactive networks; "
                    "capped views link to the lossless breakdown CSVs "
                    "below."},
    {"pattern": "dataset_data/*/path_report.html",
     "description": "Per-delegate single-unit pair report inside each "
                    "cross-dataset dataset_data/<dataset>/<delegate>/ "
                    "folder; the run-root report's Data tab links them."},
    {"pattern": "paths_pair_breakdown/pair_breakdown_paths.csv",
     "description": "Pair-report breakdown, one row per path (uncapped): "
                    "rank within (pair, length), bottleneck-first ordering.",
     "preview": True,
     "preview_title": "Pair breakdown (per path)",
     "columns": ["unit", "source", "target", "rank_in_pair_length", "path",
                 "min_weight", "length", "source_bodyid_coverage",
                 "target_bodyid_coverage", "paths_in_pair"]},
    {"pattern": "paths_pair_breakdown/pair_breakdown_intermediates.csv",
     "description": "Pair-report breakdown, one row per (pair, intermediate): "
                    "shared (>=2 paths of the pair) vs unique (exactly 1), "
                    "min hop position.",
     "preview": True,
     "preview_title": "Pair breakdown (intermediates)",
     "columns": ["unit", "source", "target", "intermediate", "n_paths_using",
                 "classification", "min_hop_position"]},
    # Shortest-mode batched discovery store (plan-shortest-batched-discovery):
    # written when discovery batching is active (default on for Shortest
    # Paths); like the pair-report entries above, these surface in the run
    # guide only when the files exist.
    {"pattern": "shortest_discovery_store/connections/*.parquet",
     "description": "Discovery store: the finalized per-reverse-layer "
                    "union edge frames exactly as fetched (one file per "
                    "conn_layer; with Discovery Store = compact these are "
                    "merged into one connections_all.parquet holding the "
                    "same rows and conn_layer multiplicity in four "
                    "columns). The per-layer granularity is what the "
                    "graph's cross-layer weight sums are built from — "
                    "keep the files if you want the run to stay "
                    "auditable.",
     "columns": ["bodyId_pre", "bodyId_post", "weight", "conn_layer"]},
    {"pattern": "shortest_discovery_store/node_distances/*.parquet",
     "description": "Discovery store: per-target BFS distance labels "
                    "streamed during discovery (one row per target x "
                    "node). Enumeration rehydrates these per batch.",
     "columns": ["target", "node", "dist"]},
    {"pattern": "shortest_discovery_store/dag_edges/*.parquet",
     "description": "Discovery store: per-target shortest-DAG candidate "
                    "edges recorded at scan time "
                    "(dist[pre] == dist[post] + 1).",
     "columns": ["target", "bodyId_pre", "bodyId_post"]},
    {"pattern": "shortest_discovery_store/pairs.parquet",
     "description": "Discovery store: every reachable (source, target) "
                    "bodyId pair with its own shortest distance.",
     "columns": ["source", "target", "dist"]},
    {"pattern": "shortest_discovery_store/meta.json",
     "description": "Discovery store manifest: query tokens, hop bound, "
                    "completeness census, per-target distance-state "
                    "counts, edge-filter configuration, the realized "
                    "batch composition, and (when Discovery Store = "
                    "compact/prune) the retention record — mode, sizes, "
                    "removals, and the dag-edges derivation recipe."},
    # Type-level refill (post-hoc, scripts/TypeLevelRefill.py --in-run):
    # standalone refill of the type-level connection strength for runs
    # whose applied threshold exceeded the asked threshold. Like the
    # pair-report entries above, these surface in the run guide only when
    # the files exist.
    {"pattern": "data_details/ratio_synapse_map.csv",
     "description": "Ratio-basis runs only: the per-neuron synapse cutoffs "
                    "the min-connection-ratio threshold implies — "
                    "`implied_syn_cutoff = max(1, ceil(t_r * "
                    "total_incoming))` per post, plus kept-edge synapse "
                    "ranges. A sub-synapse cutoff would admit every edge "
                    "into that neuron; those no-op posts are counted in "
                    "ratio_provenance.json.",
     "columns": ["bodyId_post", "total_incoming", "implied_syn_cutoff",
                 "kept_in_edges", "kept_syn_min", "kept_syn_max"]},
    {"pattern": "data_details/type_level_refill/refill_type_pairs.csv",
     "description": "Refill summary, one row per EMITTED type pair (zeros "
                    "included): emitted vs refill weight, the refilled "
                    "total/ratio/probability, and how the split was "
                    "reconstructed. Never capped.",
     "preview": True,
     "preview_title": "Type-level refill (per type pair)",
     "columns": ["type_pre", "type_post", "emitted_weight", "refill_weight",
                 "refilled_total", "emitted_pair_count",
                 "refill_pair_count", "refilled_connection_ratio",
                 "refilled_traversal_probability", "split_status"]},
    {"pattern": "data_details/type_level_refill/refill_bodyId_pairs.csv",
     "description": "The refilled bodyId pairs (capped, bottleneck-first "
                    "order): the pruned mass behind refill_weight, with "
                    "re-exploration traversal counts and hop positions.",
     "preview": True,
     "preview_title": "Type-level refill (per bodyId pair)",
     "columns": ["bodyId_pre", "bodyId_post", "type_pre", "type_post",
                 "weight", "traversal_count", "min_hop", "max_hop"]},
    {"pattern": "data_details/type_level_refill/refill_provenance.json",
     "description": "Refill provenance: asked/applied thresholds, budgets, "
                    "induced-graph sizes, table_reproduced (the re-derived "
                    "cut reproduced connection_type.csv exactly — the "
                    "consistency anchor), and the disclosure counters "
                    "(skipped non-emitted type pairs, boundary edges at or "
                    "above the floor, refill truncation)."},
    {"pattern": "data_details/type_level_refill/README.md",
     "description": "Semantics, invariants, and the join recipe for the "
                    "refill records."},
]

# Reusable pathfinding explanation, rendered by all three run-guide formats
# (HTML / Markdown / plain text) for find_path, find_shortest, and
# inter_dataset. Sections: heading, paragraphs, optional table
# (first row = header).
_PATHFINDING_EXPLANATION = [
    {
        "heading": "How the path set was produced",
        "paragraphs": [
            "Every pathfinding run passes through the same stages, in this "
            "order — each stage can only narrow the previous one:",
        ],
        "table": None,
        "pipeline": [
            "requested threshold (Min Synapse Count, or the Min "
            "Connection Ratio on weight-basis='connection_ratio' runs — "
            "the stage is identical, the units change)",
            "lossless hop/dead-end pruning (never changes which paths exist)",
            "optional Edge Budget floor w0 ('all' mode only — a graph "
            "budget, exactly equivalent to raising the threshold)",
            "StrongestFirst path-budget ordering and τ (a path-output "
            "budget; applies in both Complete and Shortest Paths)",
            "applied threshold / retained path set",
            "visualization-only edge limit (drawing only — never affects "
            "the analysis outputs)",
        ],
    },
    {
        "heading": "Threshold & bottleneck vocabulary",
        "paragraphs": [
            "The bottleneck of a path is its MINIMUM edge weight — the "
            "weakest link (synapse count by default; the F9 connection "
            "ratio on weight_basis='connection_ratio' runs, where the "
            "allpaths min_ratio column IS the bottleneck). All budgeted "
            "outputs are strength-bounded path "
            "sets: the StrongestFirst enumerator emits intact paths in "
            "descending bottleneck order, so a budgeted result is exactly "
            "'all intact paths with bottleneck >= τ', never an arbitrary "
            "first-N truncation. The ‘This run’ column shows the value this "
            "run actually produced (— = the mechanism did not apply).",
        ],
        "table": [
            ["Name", "Key", "Meaning"],
            ["W*", "strongest_retained_bottleneck",
             "Widest-path ceiling after lossless pruning; pruning must not "
             "change it."],
            ["τ", "strongest_first_tau",
             "StrongestFirst landing: all intact paths with bottleneck at "
             "least τ are retained when the path budget bites. For a "
             "complete run it is the natural weakest emitted-path "
             "bottleneck."],
            ["w2", "strongest_dropped_bottleneck",
             "Strongest path not emitted when the StrongestFirst budget "
             "bites."],
            ["w0", "edge_weight_floor",
             "Edge Budget floor; the effective graph is equivalent to "
             "raising the threshold to this floor."],
            ["w1", "edge_budget_landing",
             "Edge-weight tier used to determine the Edge Budget "
             "landing/floor."],
            ["applied threshold", "applied_threshold",
             "Canonical threshold describing the materialized output; "
             "applied_threshold_source names the contributing mechanism(s) "
             "(requested / strongest_first_budget / edge_budget / both)."],
            ["paths_complete", "paths_complete",
             "True when every intact path within the search bound was "
             "emitted — no budget bit."],
            ["pruned", "pruned",
             "True when the Edge Budget floor fired (the only lossy graph "
             "stage)."],
        ],
        "values_column": True,
        "pipeline": None,
    },
    {
        "heading": "How the thresholds relate",
        "paragraphs": [
            "Once the run is known the pruning levels form a strict chain: "
            "w0 <= w2 < tau_canonical <= τ <= W*. The Edge Budget acts on "
            "the GRAPH — it raises the effective threshold to w0, one tier "
            "above the landing tier w1 (the strongest tier whose admission "
            "would exceed the budget) — while the StrongestFirst budget "
            "acts on the OUTPUT — it bounds the emitted paths at the "
            "landing τ and excludes everything weaker than w2. "
            "applied_threshold collapses both mechanisms into the single "
            "number that matters for interpretation.",
            "Read applied_threshold as the EQUIVALENT threshold — the Min "
            "Synapse Count, or the min connection ratio on "
            "weight-basis='connection_ratio' runs: a "
            "complete run at that threshold produces exactly this path "
            "set. τ alone is only the landing/collapse bound — when the "
            "budget bite leaves a gap in the bottleneck distribution, "
            "every threshold in (w2, τ] yields the identical set and the minimal "
            "one is tau_canonical (w2+1 on synapse runs; the weakest "
            "distinct ratio tier above w2 on connection-ratio runs).",
        ],
        "table": [
            ["Run state", "applied_threshold", "τ", "Reading"],
            ["Complete (no budget bit)", "requested_threshold",
             "natural weakest emitted bottleneck (= tau_canonical)",
             "Every Min Synapse Count up to τ yields this identical "
             "set."],
            ["StrongestFirst budget bit", "w2 + 1 (= tau_canonical)",
             "landing bound of the drained tie group",
             "The strongest excluded path is w2; every threshold in "
             "[w2+1, τ] gives the same set."],
            ["Edge Budget floor only ('all' mode)", "natural τ (>= w0)",
             "natural weakest emitted bottleneck",
             "The graph was floored at w0 first; flooring cannot inflate "
             "hop distances — it only removes weak routes."],
            ["Both budgets", "w2 + 1 (= tau_canonical)", "landing bound",
             "The floor raised the graph threshold first; the bite then "
             "bounded the output."],
        ],
        "values_column": False,
        "pipeline": None,
    },
    {
        "heading": "Reading τ vs applied_threshold",
        "paragraphs": [
            "τ is a landing/collapse bound: the maximal threshold "
            "equivalent to this run. It is not always the minimal one — "
            "when the budget bite leaves a gap in the bottleneck "
            "distribution, every threshold in [w2+1, τ] yields the "
            "identical set, and w2+1 (tau_canonical) is the minimal "
            "equivalent threshold. applied_threshold reports that canonical "
            "minimal value with its source; for an unbounded/complete run "
            "it stays at the requested threshold and the natural τ is "
            "reported separately.",
            "Shortest Paths can take the StrongestFirst path budget (and "
            "reports the same τ metadata) but is NEVER floored: the Edge "
            "Budget does not apply in shortest mode because trimming edges "
            "could remove the only shortest route.",
            "bodyId-level and type-level visualizations share the same "
            "drawing cap (Visualization Edge Limit). The drawing cap is not "
            "a path or graph budget: when it trims the rendered graph, the "
            "exact path rows still represented by the drawing are exported "
            "as *_paths_visualized.csv companion files, and a single "
            "complete path may exceed the cap to stay intact.",
        ],
        "table": None,
        "pipeline": None,
    },
    {
        "heading": "Filters and what they affect",
        "paragraphs": [],
        "table": [
            ["Control", "Level", "Effect"],
            ["Min Synapse Count", "threshold",
             "Edges below this weight are never fetched into the graph."],
            ["Edge Budget", "graph",
             "'all' mode only: lossy floor w0 on the discovery cone; "
             "exactly equivalent to raising the threshold. Shortest mode "
             "is never floored."],
            ["Max Paths (BodyId)", "path output",
             "StrongestFirst budget on the emitted paths; the graph is not "
             "trimmed."],
            ["Drop Untyped Neurons", "neuron labels",
             "Removes edges touching untyped neurons (empty / Unknown / "
             "NaN / bodyId-fallback labels — the shared predicate). In the "
             "single-dataset tabs this happens before graph construction; in "
             "Cross-Dataset Comparison it happens after standardized labels "
             "and before comparison aggregation. Dropped rows are exported "
             "to untyped_dropped_records.csv and counted in "
             "user_warning_notes.txt."],
            ["Visualization Edge Limit", "drawing only",
             "Caps unique edges drawn per HTML view; fetched connections, "
             "the graph, and the path tables are unaffected."],
        ],
        "pipeline": None,
    },
    {
        "heading": "Cross-dataset threshold queries",
        "paragraphs": [
            "Cross-Dataset Comparison has two threshold modes. Standard "
            "N-chip thresholds create one query per scalar N and apply the "
            "same requested threshold to every dataset. Custom combination "
            "mode creates one complete query row at a time: each row "
            "contains one requested threshold for every dataset column. "
            "Those rows are comparison identities; the union of their cell "
            "values is only the deduplicated raw-run schedule.",
            "For each query, use threshold_combinations.csv or the queries "
            "block in effective_thresholds.json to join the requested "
            "threshold with the dataset-specific applied threshold and its "
            "bottleneck/budget provenance. Never compare a dataset column "
            "from one query with a different query's column just because "
            "their scalar threshold values match.",
        ],
        "table": [
            ["Term", "Meaning in a cross-dataset query"],
            ["requested threshold",
             "The cell entered for this dataset in this query row."],
            ["applied threshold",
             "The canonical equivalent Min Synapse Count for that dataset's "
             "materialized run; it may differ from the requested cell when a "
             "budget changes the output."],
            ["Edge Budget / w0 / w1",
             "Graph-level cap, floor, and landing tier for this dataset's "
             "query cell; the floor is never applied in Shortest Paths."],
            ["StrongestFirst budget / tau / w2",
             "Path-output budget, landing/collapse bound, and strongest "
             "dropped bottleneck for this dataset's query cell."],
            ["bottleneck / W*",
             "A path's weakest edge; W* is the strongest retained path "
             "bottleneck after lossless pruning."],
        ],
        "values_column": False,
        "pipeline": None,
    },
]

# Symmetry columns also appear as base_pre/base_post pairs (defined in the
# glossary above).

_HOMOLOG_FILES = [
    {"pattern": "results/homolog_results.csv",
     "description": "Full type-level results with all similarity columns, "
                    "sorted by the chosen metric.",
     "columns": _HOMOLOG_RESULT_COLUMNS},
    {"pattern": "results/bodyid_results.csv",
     "description": "BodyId-level results (sorted by source, then metric).",
     "preview": True,
     "preview_title": "BodyId-level homologs",
     "columns": [
         "source_bodyId", "source_type", "source_instance",
         "target_bodyId", "target_type", "target_instance",
         "rank_union", "jaccard",
         "cosine", "adjacency_score", "shared_type_count",
         "union_type_count", "is_same_type", "is_same_dataset",
         "source_status", "target_status", "weak_source", "weak_target",
         "source_partner_count", "target_partner_count"]},
    {"pattern": "results/type_summary.csv",
     "description": "Aggregated results at the neuron type level "
                    "(sorted by avg_jaccard, descending).",
     "preview": True,
     "preview_title": "Type-mean (from bodyId level)",
     "columns": [
         "query", "source_dataset", "target_dataset", "source_type",
         "target_type", "n_bodyid_comparisons",
         "avg_jaccard", "avg_rank_union", "avg_cosine",
         "avg_adjacency_score", "avg_shared_type_count",
         "avg_union_type_count", "n_complete_sources",
         "n_incomplete_sources", "visualized", "visualization_rank"]},
    {"pattern": "results/type_level_results.csv",
     "description": "True type-level homolog ranking — pooled all-adjacency "
                    "type profiles scored against every typed target type "
                    "(top N per source type under the run's metric).",
     "preview": True,
     "preview_title": "Type-level homologs (pooled profiles)",
     "columns": [
         "source_type", "target_type", "target_type_members", "is_same_type",
         "target_dataset", "jaccard", "weighted_jaccard", "cosine",
         "rank_union", "rank"]},
    {"pattern": "results/shuffle_test.json",
     "description": "Null-model shuffle test for the type-level ranking "
                    "(metric values of true pairings vs shuffled-label "
                    "nulls): written when the shuffle test runs."},
    {"pattern": "by_type/**",
     "description": "Per-query-type subfolders for multi-type runs: each "
                    "by_type/<query type>/ holds the same results/ tables "
                    "restricted to that query type."},
    {"pattern": "auto_type_mapping.json",
     "description": "Auto-type-mapping provenance for this search: which "
                    "mapper was requested vs active, source table and version, "
                    "any load error, per-status candidate-expansion resolution "
                    "counts (basis unique_type_resolutions), and whether "
                    "raw-name fallback occurred."},
    {"pattern": "results/source_status_summary.json",
     "description": "Per-source-neuron status (resolved bodyIds, candidate "
                    "counts)."},
    {"pattern": "results/intra_type_results.csv",
     "description": "Intra-type comparison table (same-dataset similarity "
                    "searches only).",
     "columns": _HOMOLOG_RESULT_COLUMNS},
    {"pattern": "profiles/query/*.csv",
     "description": "Connectivity profile of the query neuron(s).",
     "columns": ["neuron_type", "dataset", "direction", "partner_type",
                 "weight", "rank"]},
    {"pattern": "profiles/query/source_bodyids.csv",
     "description": "BodyIds enrolled for the query neuron(s)."},
    {"pattern": "profiles/matches/*.csv",
     "description": "Connectivity profiles of the top candidate matches.",
     "columns": ["neuron_type", "dataset", "direction", "partner_type",
                 "weight", "rank"]},
    {"pattern": "profiles/matches/top_target_bodyids.csv",
     "description": "BodyIds enrolled for the top candidate matches."},
    {"pattern": "overlaps/*.csv",
     "description": "Partner overlap details for top candidates (which "
                    "partners are shared vs unique).",
     "columns": ["partner_type", "in_a", "in_b", "weight_a", "weight_b",
                 "rank_a", "rank_b", "status", "direction"]},
    {"pattern": "results/morph_qualification.json",
     "description": "Morph-qualification provenance (only with the morph "
                    "option on): mode (null bar / mapping-referenced "
                    "floor), null level percentile (default 95), bar "
                    "offset, per-source bars and scored pairs with their "
                    "qualified flags."},
    {"pattern": "README.txt",
     "description": "Analysis parameters, summary, and column descriptions."},
]

TOOL_GUIDE_SPECS = {
    "find_path": {
        "title": "Complete Paths",
        "summary": "Multi-hop pathways between source and target neuron "
                   "groups in a single dataset.",
        "files": _PATHFINDING_FILES,
        "explanation": _PATHFINDING_EXPLANATION,
    },
    "find_shortest": {
        "title": "Shortest Paths",
        "summary": "Minimum-hop paths between source and target neuron "
                   "groups (all ties kept).",
        "files": _PATHFINDING_FILES,
        "explanation": _PATHFINDING_EXPLANATION,
    },
    "find_network": _FIND_NETWORK,
    "plot3d_skeleton": {
        "title": "3D Skeleton Visualization",
        "summary": "3D visualization of neuron skeletons, synapses, and ROI "
                   "meshes.",
        "files": [
            {"pattern": "*.html",
             "description": "The interactive 3D scene. Open in a web browser "
                            "to view neurons, synapses, and ROIs."},
            {"pattern": "*_neuron_info.csv",
             "description": "Merged neuron metadata table for all layers. "
                            "Key columns: viz_layer, bodyId, instance, type, "
                            "pre, post — " + _NEURON_TABLE_NOTE,
             "preview": True,
             "preview_title": "Neuron info",
             "columns": ["viz_layer"]},
            {"pattern": "viz_layer_info.csv",
             "description": "Reusable layer-map CSV with one-based layers, "
                            "resolved neuron identifiers, and effective neuron, "
                            "synapse, pre-site, and post-site colors.",
             "preview": True,
             "preview_title": "Layer map",
             "columns": ["layer", "neuron", "color"]},
            {"pattern": "*_synapses.*",
             "description": "Merged synapse data. In paired (connector) mode "
                            "the viz_layer column reads e.g. 0->1; in pre/post "
                            "site mode (synapse_mode=pre_post) it reads e.g. "
                            "0:pre / 0:post and holds the rendered input/output "
                            "sites of the queried neurons.",
             "columns": ["viz_layer", "bodyId_pre", "bodyId_post"]},
            {"pattern": "parameters.txt",
             "description": "Visualization parameters (colors, alphas, "
                            "modes, backend, ...)."},
            {"pattern": WARNING_FILENAME,
             "description": "Notes/warnings collected during rendering "
                            "(rendered in the Warnings section above)."},
            {"pattern": "*.png",
             "description": "Exported screenshots (when Export Views is on)."},
            {"pattern": "*.gif",
             "description": "Exported rotating animation (when Export Video "
                            "is on)."},
            {"pattern": "*.mp4",
             "description": "Exported rotating video (when Export Video is "
                            "on)."},
            {"pattern": "*_simplified.html",
             "description": "Degraded copy of the interactive scene, written "
                            "only when the full page was too large for the "
                            "browser export. Re-export from the main page."},
            {"pattern": "visualization_manifest.json",
             "description": "Machine-readable record of the run: dataset, "
                            "canonical viewer page, legend mode, freeze "
                            "state, render settings, profile levels, and "
                            "per-view cameras. The re-exporter reads its "
                            "cameras from here and re-checks the recorded "
                            "trace roles against the page."},
            {"pattern": "individual_profiles/*",
             "description": "One PNG per profile group (per legend entry, "
                            "layer, type, or bodyId leaf, per Profile "
                            "Granularity), plus the PDF/PPTX summary."},
            {"pattern": "exported_views/*",
             "description": "Fixed-camera PNGs of the whole scene (when Export "
                            "Views is on), and the temporary pages the browser "
                            "export drives."},
            {"pattern": "pics_*fps_*/*",
             "description": "The individual frames behind a rotating video, "
                            "reused when the same settings are exported again."},
        ],
    },
    "plot3d_reexport": {
        "title": "Skeleton Re-export",
        "summary": "Re-render individual profiles and the rotating video from a "
                   "stored 3D skeleton HTML page, without querying the dataset.",
        "files": [
            {"pattern": "individual_profiles/*",
             "description": "One PNG per profile group at the chosen "
                            "granularity, grouped from the page's own traces "
                            "(its drocatTrace tags; a pre-manifest page falls "
                            "back to its legend groups). Capped at 300 renders "
                            "per run (groups x views) - the groups that fit "
                            "still render and the rest are named in the log, so "
                            "separate the plots into smaller runs if a page "
                            "asks for more."},
            {"pattern": "*_reexport/*",
             "description": "The whole output folder, written beside the source "
                            "page: profiles, frames, videos and GIFs. The run's "
                            "own files are never modified."},
            {"pattern": "pics_*fps_*/*",
             "description": "Frames of the rotating video, captured through one "
                            "Chrome session (webdriver method) or rendered per "
                            "frame (kaleido)."},
            {"pattern": "*.mp4",
             "description": "Forward and backward rotating videos."},
            {"pattern": "*.gif",
             "description": "Small GIF conversions of those videos."},
        ],
    },
    "plot_path": {
        "title": "Net-Viz (Path Network Visualization)",
        "summary": "Pathway graphs rendered from Complete Paths outputs or a "
                   "custom edge-list CSV.",
        "files": [
            {"pattern": "*_network.html",
             "description": "Interactive pathway graph."},
            {"pattern": "*_heatmap.html",
             "description": "Edge weight heatmap."},
            {"pattern": "*_Sankey.html",
             "description": "Sankey flow diagram."},
            {"pattern": "*_data.xlsx",
             "description": "Data workbook with sheets: connections, "
                            "original_paths, connMatrix_weight, "
                            "connMatrix_ratio, connMatrix_prob (plus "
                            "connMatrix_nt_type when the input carries "
                            "neurotransmitter data)."},
        ],
    },
    "find_homologs": {
        "title": "Connectivity · Find Homolog",
        "summary": "Similar neurons found by connectivity-profile similarity, "
                   "across datasets (homolog search) or within one dataset "
                   "when Target = Source.",
        "files": _HOMOLOG_FILES,
    },
    "find_similar_morphology": {
        "title": "Morphological Similarity",
        "summary": "Morphologically similar neurons found by skeleton "
                   "comparison.",
        "files": [
            {"pattern": "results.csv",
             "description": "BodyId-level similarity results. The leading "
                            "columns are always present; the source_* and "
                            "pool_stage columns appear only when applicable "
                            "(cross-dataset runs), and type_coverage is "
                            "written whenever a coverage snapshot exists.",
             "preview": True,
             "preview_title": "Similarity results",
             "columns": [
                 "rank", "target_bodyId",
                 "target_type", "target_instance", "profile_similarity",
                 "roi_similarity", "similarity", "is_same_type",
                 "intra_type_similarity", "method", "metric",
                 "type_coverage", "source_bodyId", "source_type"]},
            {"pattern": "type_summary.csv",
             "description": "Type-level summary.",
             "preview": True,
             "preview_title": "Type-level summary",
             "columns": [
                 "rank", "target_type", "similarity", "n_bodyids",
                 "profile_similarity", "roi_similarity", "is_intra_type",
                 "intra_type_similarity", "method", "metric"]},
            {"pattern": "README.txt",
             "description": "Run summary and parameter record."},
        ],
    },
    "connectivity_profiling": {
        "title": "Connectivity · Comparison",
        "summary": "Connectivity profiles and their pairwise similarity "
                   "within and across datasets.",
        "files": [
            {"pattern": "report.html",
             "description": "Overall HTML report linking every metric and "
                            "heatmap.",
             "columns": _PROFILING_METRIC_COLUMNS},
            {"pattern": "parameters.json",
             "description": "All analysis parameters (query, datasets, "
                            "top_k/top_m, thresholds, metrics)."},
            {"pattern": "README.txt",
             "description": "Human-readable summary listing only the folders "
                            "this run wrote."},
            {"pattern": "type_level/results/type_similarity_*.csv",
             "description": "ONE-DATASET runs at the type level: pooled "
                            "profiles compared, one file per direction × "
                            "metric. A bodyId-level run writes no such folder "
                            "(nothing was pooled), and a custom group run "
                            "writes group_level/ instead, because its axes "
                            "are groups.",
             "matrix": "rows/columns = neuron types, values = similarity for "
                       "the metric/direction in the "
                       "file name",
             "columns": _PROFILING_METRIC_COLUMNS},
            {"pattern": "group_level/results/group_similarity_*.csv",
             "description": "ONE-DATASET runs at the custom group level: the "
                            "same pooled matrix, named for its group axes.",
             "matrix": "rows/columns = custom group labels, values = "
                       "similarity for the metric/direction in the file name",
             "columns": _PROFILING_METRIC_COLUMNS},
            {"pattern": "type_level/visualization/heatmap_type_*.html",
             "description": "ONE-DATASET type-level interactive heatmaps "
                            "(same folder as its matrices) with interactive "
                            "Ward dendrograms; at the custom group level the "
                            "twin is group_level/visualization/"
                            "heatmap_group_*.html."},
            {"pattern": "bodyid_level/results/bodyid_similarity_*.csv",
             "description": "ONE-DATASET runs: member-to-member similarity. "
                            "The rows are individual neurons at the type and "
                            "bodyId levels; at the custom group level they are "
                            "the grouping board's members, so a member named "
                            "as a type contributes that type's pooled profile.",
             "matrix": "rows/columns = '{bodyId}_{instance}' (or "
                       "'{bodyId}_{type}_{L|R}' on FAFB/BANC) labels, "
                       "values = similarity",
             "columns": _PROFILING_METRIC_COLUMNS},
            {"pattern": "bodyid_level/results/type_avg_bodyid_similarity_*.csv",
             "description": "ONE-DATASET runs: per-type averages of the bodyId "
                            "pair scores (diagonal = intra-type cohesion). At "
                            "the bodyId level this is folded from the matrices "
                            "beside it, so a bodyId run still yields a "
                            "per-type view.",
             "matrix": "rows/columns = neuron types, values = averaged "
                       "similarity",
             "columns": _PROFILING_METRIC_COLUMNS},
            {"pattern": "bodyid_level/visualization/heatmap_bodyid_*.html",
             "description": "ONE-DATASET bodyId-level interactive heatmaps "
                            "with interactive Ward dendrograms."},
            {"pattern": "bodyid_level/visualization/heatmap_type_avg_*.html",
             "description": "ONE-DATASET type-average-of-bodyId heatmaps."},
            {"pattern": "profiles/individual/*_profile.json",
             "description": "ONE-DATASET runs: the connectivity profile of "
                            "each individual neuron compared. Read the keys "
                            "against the run's level: 'type' holds the MATRIX "
                            "ROW the neuron fed (its own display label at the "
                            "bodyId level, the group label at the custom "
                            "group level), while 'neuron_type' is always the "
                            "neuron's real resolved type."},
            {"pattern": "profiles/aggregated/*_profile.json",
             "description": "ONE-DATASET runs at the type or custom group "
                            "level: the pooled profile behind each matrix row "
                            "('type' is that row label, so a group's file "
                            "carries the group name). Absent for a "
                            "bodyId-level run."},
            {"pattern": "intra_dataset/*/results/similarity_*.csv",
             "description": "Type-level N×N similarity matrices, one file "
                            "per direction × metric.",
             "matrix": "rows/columns = neuron types, values = similarity "
                       "for the metric/direction in the file name",
             "columns": _PROFILING_METRIC_COLUMNS},
            {"pattern": "intra_dataset/*/results/bodyid_similarity_*.csv",
             "description": "BodyId-to-bodyId similarity matrices.",
             "matrix": "rows/columns = '{bodyId}_{instance}' (or "
                       "'{bodyId}_{type}_{L|R}' on FAFB/BANC) labels, "
                       "values = similarity",
             "columns": _PROFILING_METRIC_COLUMNS},
            {"pattern": "intra_dataset/*/results/type_avg_bodyid_similarity_*.csv",
             "description": "Type similarities averaged from bodyId pairs.",
             "matrix": "rows/columns = neuron types, values = averaged "
                       "similarity",
             "columns": _PROFILING_METRIC_COLUMNS},
            {"pattern": "intra_dataset/*/visualization/heatmap_*.html",
             "description": "Interactive intra-dataset heatmaps with Ward "
                            "dendrograms beside the clustered matrices."},
            {"pattern": "cross_dataset/mapping_summary.csv",
             "description": "Resolved type names per dataset with same-name "
                            "flags.",
             "preview": True,
             "preview_title": "Cross-dataset type mapping",
             "columns": ["anchor", "same name"]},
            {"pattern": "cross_dataset/all_types/results/similarity_*.csv",
             "description": "N×M similarity matrices comparing the queried "
                            "types across datasets.",
             "matrix": "rows = types in one dataset, columns = types in the "
                       "other, values = similarity",
             "columns": _PROFILING_METRIC_COLUMNS},
            {"pattern": "cross_dataset/all_types/visualization/heatmap_*.html",
             "description": "Interactive cross-dataset heatmaps with Ward "
                            "dendrograms beside the clustered matrices."},
            {"pattern": "profiles/*/aggregated/*_profile.json",
             "description": "Type-aggregated connectivity profiles."},
            {"pattern": "profiles/*/individual/*_profile.json",
             "description": "Individual bodyId connectivity profiles "
                            "(bodyId, type, instance, dataset)."},
        ],
    },
    "morph_cross_dataset": {
        "title": "Morphology · Cross-Dataset Comparison",
        "summary": "Compares the queried neurons' morphology across FAFB / "
                   "male-cns / BANC: pairwise vector_v2 per dataset pair "
                   "with a seeded null baseline and overlay scenes.",
        "files": [
            {"pattern": "report.html",
             "description": "Tabbed report (same generator as the "
                            "connectivity-profiling report): scrollable "
                            "overview table with frame-asymmetry "
                            "disclosure, per-pair tabs with Type/BodyId "
                            "level heatmaps (square cells via explicit-width "
                            "sizing), CSV and VisPath editor links, null "
                            "baselines, and overlay-scene links."},
            {"pattern": "parameters.json",
             "description": "All analysis parameters (queries, datasets, "
                            "member caps, null sample size, reference "
                            "template) plus run warnings."},
            {"pattern": "README.txt",
             "description": "Human-readable summary with the output "
                            "structure."},
            {"pattern": "overview.csv",
             "description": "Queried type x dataset-pair best target-type "
                            "match score, target name, and baseline p95. "
                            "Each column block is scored in the target's "
                            "render space — A\u2192B and B\u2192A are not "
                            "symmetric.",
             "preview": True,
             "preview_title": "Overview",
             "columns": ["queried_type"]},
            {"pattern": "members_summary.csv",
             "description": "Number of compared members per queried type "
                            "and dataset."},
            {"pattern": "*_to_*/results/morph_bodyid_scores.csv",
             "description": "Member-level scores for one dataset pair: "
                            "vector_v2 per source x target bodyId with the "
                            "null p95 and the above_baseline flag.",
             "preview": True,
             "preview_title": "BodyId scores",
             "columns": ["source_bodyId", "target_bodyId", "morph_v2"]},
            {"pattern": "*_to_*/results/morph_bodyid_matrix.csv",
             "description": "BodyId x bodyId vector_v2 matrix for one "
                            "dataset pair (tree-legend axis labels)."},
            {"pattern": "*_to_*/results/morph_type_matrix.csv",
             "description": "Type x type mean vector_v2 matrix for one "
                            "dataset pair (mean over cross-member pairs)."},
            {"pattern": "*_to_*/results/null_baseline.json",
             "description": "Seeded random-target null reference for one "
                            "dataset pair (p95/median/std/n per query)."},
            {"pattern": "*_to_*/visualization/heatmap_morph_*.html",
             "description": "Interactive VisPath heatmaps (type and "
                            "bodyId level) for one dataset pair, with "
                            "interactive Ward dendrograms."},
            {"pattern": "plot-3d_*/**",
             "description": "3D overlay scenes: each dataset's members "
                            "bridged into the reference template."},
        ],
    },
    "morphology_comparison": {
        "title": "Morphology · Comparison",
        "summary": "Intra-dataset N×N morphology comparison of the queried "
                   "neurons. Aggregation Level picks the row: type (default), "
                   "bodyId (one neuron per row) or custom group.",
        "files": [
            {"pattern": "report.html",
             "description": "Summary report: parameters, compared neurons, "
                            "and the similarity matrices with links to the "
                            "interactive heatmaps. Only the levels this run "
                            "computed get a tab, so a bodyId-level report "
                            "shows the BodyId matrix alone."},
            {"pattern": "parameters.json",
             "description": "All analysis parameters, including "
                            "aggregation_level (member/total caps, method)."},
            {"pattern": "README.txt",
             "description": "Human-readable summary with the output "
                            "structure."},
            {"pattern": "members.csv",
             "description": "Resolved comparison population: one row per "
                            "compared neuron, with the matrix row it feeds "
                            "and its availability status.",
             "preview": True,
             "preview_title": "Compared neurons",
             "columns": ["row", "type", "bodyId", "instance", "status"]},
            {"pattern": "user_warning_notes.txt",
             "description": "How the query resolved, written only when "
                            "something needs disclosing: taxonomy-label "
                            "expansions (e.g. a cell_type value expanding "
                            "into member types), instance-name matches, "
                            "member/total caps, and tokens nothing matched."},
            {"pattern": "type_level/type_similarity_*.csv",
             "description": "Type×type similarity matrix: each entry is the "
                            "mean over the cross-member bodyId pairs, the "
                            "diagonal the type's intra-type cohesion. Written "
                            "at the type level only — at the bodyId level the "
                            "bodyId matrix IS the comparison and no aggregate "
                            "is computed; at the custom group level the same "
                            "aggregate lands in group_level/ instead.",
             "matrix": "rows/columns = neuron types, values = mean "
                       "morphological similarity"},
            {"pattern": "group_level/group_similarity_*.csv",
             "description": "Group×group similarity matrix at the custom "
                            "group level: rows are the grouping board's "
                            "source-side groups, each entry the mean over "
                            "their members' cross pairs.",
             "matrix": "rows/columns = custom group labels, values = mean "
                       "morphological similarity"},
            {"pattern": "bodyid_level/bodyid_similarity_*.csv",
             "description": "BodyId-to-bodyId similarity matrix (every "
                            "individual pair). Always written, and the "
                            "run's only matrix at the bodyId level.",
             "matrix": "rows/columns = '{bodyId}_{instance}' (or "
                       "'{bodyId}_{type}_{L|R}' on FAFB/BANC) labels, "
                       "values = morphological similarity"},
            {"pattern": "visualization/heatmap_*.html",
             "description": "Interactive heatmaps, one per computed level — "
                            "heatmap_type_* / heatmap_group_* / "
                            "heatmap_bodyid_*. Matrix cells render square "
                            "(1:1) in both the VisPath renderer and the "
                            "offline fallback, and the clustered pages draw "
                            "Ward dendrograms beside the matrix."},
            {"pattern": "plot-3d_*/**",
             "description": "Optional 3D skeleton scene (3D Skeleton "
                            "Visualization checkbox): one layer per "
                            "compared matrix row, linked from report.html."},
        ],
    },
    "inter_dataset": {
        "title": "Cross-Dataset Comparison",
        "summary": "Connectivity pathways compared across multiple datasets.",
        "files": [
            {"pattern": "path_report.html",
             "description": "Per-source-target-pair HTML report, written "
                            "automatically after the run: Overview / Global "
                            "/ Pair Explorer / Data pages with capped "
                            "top-paths tables (bodyId coverage per path) "
                            "and offline interactive networks."},
            {"pattern": "paths_pair_breakdown/pair_breakdown_paths.csv",
             "description": "Lossless per-path breakdown written with "
                            "path_report.html: rank within (pair, length), "
                            "bottleneck-first ordering.",
             "preview": True,
             "preview_title": "Pair breakdown (per path)",
             "columns": ["unit", "source", "target", "rank_in_pair_length",
                         "path", "min_weight", "length",
                         "source_bodyid_coverage", "target_bodyid_coverage",
                         "paths_in_pair"]},
            {"pattern": "paths_pair_breakdown/pair_breakdown_intermediates.csv",
             "description": "Per (pair, intermediate) breakdown: shared "
                            "(>=2 paths of the pair) vs unique (exactly 1), "
                            "min hop position.",
             "preview": True,
             "preview_title": "Pair breakdown (intermediates)",
             "columns": ["unit", "source", "target", "intermediate",
                         "n_paths_using", "classification",
                         "min_hop_position"]},
            {"pattern": "comparison_report.html",
             "description": "Comprehensive interactive HTML report. Standard "
                            "and Custom combination runs use the same report "
                            "sections; Custom repeats every section for every "
                            "query row with query-keyed networks, matrices, "
                            "provenance, conservation, overlap and statistics."},
            {"pattern": "comparison_report.txt",
             "description": "Plain-text summary of the report."},
            {"pattern": "parameters.json",
             "description": "JSON dump of all comparison parameters, plus "
                             "the pathfinding provenance field list and the "
                            "definitions of tau and applied_threshold. It "
                            "also includes normalized threshold_queries with "
                            "requested/applied thresholds and provenance in "
                            "Custom combination mode."},
            {"pattern": "effective_thresholds.json",
             "description": "Threshold notice used by the UI and run guide. "
                            "Always written for pathfinding comparisons; its "
                            "runs rows contain requested and applied "
                            "thresholds, source, tau, w0, w1, w2, W*, and "
                            "paths_complete. In combination mode, queries "
                            "preserve the row-wise dataset threshold map."},
            {"pattern": "label_map.json",
             "description": "Label mappings for source/target neurons across "
                            "datasets (incl. auto type mapping)."},
            {"pattern": "run_manifest.json",
             "description": "Self-describing manifest of the whole run: the "
                            "parameters dump, dataset nicknames, applied "
                            "thresholds and query views per dataset, "
                            "threshold comparability, per-dataset coverage, "
                            "untyped-drop stats, alignment suggestions, and — "
                            "on auto threshold runs — auto_mode_status, the "
                            "machine-readable record of how the density-aligned "
                            "schedule resolved (outcome installed / "
                            "verticals_only / degraded, the bootstrap floor, "
                            "the installed row counts, the datasets that "
                            "produced no density capture, their failure "
                            "reasons, and any schedule-level reason). A "
                            "partially resolved auto run is also described in "
                            "user_warning_notes.txt and bannered in the "
                            "report's density section."},
            {"pattern": "dataset_metadata_comparison.csv",
             "description": "Per-dataset metadata comparison.",
             "preview": True,
             "preview_title": "Dataset metadata comparison",
             "columns": [
                 "dataset", "total_neurons", "typed_neurons",
                 "untyped_neurons", "type_coverage_pct", "total_presynaptic",
                 "total_postsynaptic", "total_synapses", "roi_count",
                 "coverage_notes"]},
            {"pattern": "auto_type_mapping.csv",
             "description": "Cross-dataset type mapping table. The "
                            "mapping_origin column states how each row "
                            "was derived: crosswalk, same name, the "
                            "curated label lanes ('cross-dataset cell "
                            "type'), or the annotation bridge "
                            "(additional_type(s) / Alternative Cell "
                            "Type(s)) with its tokens. "
                            "The trailing mapping_support column carries "
                            "the row-based bodyId-level bridge evidence: "
                            "label votes per source type with curated vs "
                            "auto provenance (e.g. "
                            "'cross-dataset cell type: X=2 (auto 2)'), "
                            "'same name (all bodyIds pooled)' for direct "
                            "same-name pairs, or empty when the pair "
                            "carries none."},
            {"pattern": "auto_type_mapping_conflicts.csv",
             "description": "Conflicting cross-dataset type mappings "
                            "(N-to-1 / 1-to-N, never guessed). The origin "
                            "column distinguishes crosswalk conflicts from "
                            "annotation-bridge ones; multivalue_source / "
                            "source_parts flag comma-joined multi-value type "
                            "cells, and same_name_candidate / same_name_path / "
                            "same_name_disposition record the same-name-first "
                            "verdict for the row."},
            {"pattern": "auto_type_mapping_suspects.csv",
             "description": "Same-name-first SUSPECT relations: one row per "
                            "rival candidate of a fan-out whose candidate set "
                            "contained the source type's own name. "
                            "rival_has_own_clean_pair + rival_pair_status "
                            "(own_1to1_pair / no_own_1to1_pair) tell whether "
                            "the rival's own name also pairs 1-to-1 in this "
                            "direction — an observation, never a verdict; "
                            "reverse_target/backs_source says whether it "
                            "points back at the source. Includes per-candidate "
                            "votes, populations, the pair-level "
                            "selection_disposition, and a ready-made "
                            "custom-label-mapper entry "
                            "(custom_mapper_dataset/from/to) — the user's "
                            "inclusion path. Rivals are never merged "
                            "automatically."},
            {"pattern": "auto_type_mapping_per_bridge.csv",
             "description": "Auto-type-mapping decisions per BRIDGE (one "
                            "row per dataset pair the mapper walked, not "
                            "per source type): the lane that carried the "
                            "mapping and its support counts, with the "
                            "unified mapping_origin vocabulary — "
                            "annotation lanes read 'annotation bridge via "
                            "<tokens>', and a glue chain rescued by the "
                            "backward-reciprocity check reads "
                            "'annotation bridge via <tokens> "
                            "(reciprocal)'. Companion of "
                            "auto_type_mapping.csv for auditing multi-hop "
                            "chains."},
            {"pattern": "auto_type_mapping.json",
             "description": "Auto-type-mapping provenance for this run: "
                            "which mapper was requested vs active, its source "
                            "table and version, any load error, per-status "
                            "resolution counts (basis unique_type_resolutions) "
                            "plus the partner-occurrence metric, and whether "
                            "raw-name fallback occurred."},
            {"pattern": "comparison_report_used_data/query_resolution.csv",
             "description": "Per-token, per-dataset query resolution with "
                            "method, status, confidence and evidence "
                            "(same_name_identity, taxonomy expansions, "
                            "taxonomy member mapping, fallbacks, conflicts).",
             "columns": ["token", "dataset", "role", "status", "method",
                         "target_types", "evidence", "confidence"]},
            {"pattern": "comparison_report_used_data/*.csv",
             "description": "Aggregated metrics backing the report. Standard "
                            "files use threshold keys; Custom combination files "
                            "use query_id/query_label and query-keyed tables. "
                            "Ratio/probability backing files are not emitted "
                            "for pathfinding comparisons."},
            {"pattern": "comparison_results/edge_presence_matrix*.csv",
             "description": "Edge presence across datasets (one file per "
                            "standard threshold or advanced query). Advanced "
                            "filenames use edge_presence_matrix_query_<query_id> "
                            "with a filesystem-safe query-ID slug "
                            "(the manifest retains the original ID); rows "
                            "include per-dataset requested thresholds. Presence flags, "
                            "weights, counts and conservation per edge. "
                            "source_status_<dataset>/target_status_<dataset> "
                            "columns carry the union type-resolution verdict "
                            "for each endpoint type in that dataset (present, "
                            "below_threshold, not_in_dataset, unmapped, ...).",
             "columns": ["edge_key", "source", "target", "conserved_at_lowest"]},
            {"pattern": "comparison_results/edge_weight_comparison.csv",
             "description": "Edge weights compared across all datasets. "
                            "Combination rows carry query_id/query_label and "
                            "requested/applied threshold columns per dataset."},
            {"pattern": "comparison_results/path_presence_matrix*.csv",
             "description": "Path presence across datasets. Advanced filenames "
                            "use path_presence_matrix_query_<query_id> with a "
                            "filesystem-safe query-ID slug rather than "
                            "a scalar threshold union; the manifest preserves "
                            "the original query ID."},
            {"pattern": "comparison_results/unified_edge_comparison.csv",
             "description": "Combined edge data: per-dataset weight/presence "
                            "columns for every edge; combination rows also "
                            "carry query_id and per-dataset threshold columns "
                            "plus the per-endpoint source_status/target_status "
                            "union-resolution columns (the "
                            "query_id/query_label/threshold_mode columns "
                            "below exist only in combination mode).",
             "columns": ["query_id", "query_label", "edge_key", "source",
                         "target", "threshold_mode", "conservation"]},
            {"pattern": "comparison_results/type_resolution_union.csv",
             "description": "Union type-resolution coverage: the union of "
                            "types that appeared in ANY dataset for a query, "
                            "resolved into EVERY dataset via the type mapper. "
                            "One row per query x type x dataset with the "
                            "resolved native name and, for absent types, why "
                            "(below_threshold with the max connected edge "
                            "weight, not_recruited, no_edges, not_in_dataset, "
                            "unmapped/conflict, resolved_absent when the "
                            "diagnosis inputs were unavailable). This is the "
                            "full-union counterpart of query_resolution.csv, "
                            "which covers only the queried source/target "
                            "tokens.",
             "columns": ["query_id", "query_label", "type", "dataset",
                         "present", "resolved_type", "resolution_status",
                         "detail"]},
            {"pattern": "comparison_results/unified_summary.csv",
             "description": "Run summary of the unified comparison.",
             "preview": True,
             "preview_title": "Unified comparison summary",
             "columns": ["query_id", "query_label", "dataset", "threshold",
                         "requested_threshold",
                         "applied_threshold", "applied_threshold_source",
                         "strongest_first_budget", "strongest_first_tau",
                         "tau_canonical", "strongest_dropped_bottleneck",
                         "edge_budget", "edge_budget_applied",
                         "edge_budget_landing", "edge_weight_floor",
                         "strongest_retained_bottleneck", "paths_complete",
                         "total_edges", "total_layer_rows", "total_weight",
                         "mean_weight", "unique_sources", "unique_targets"]},
            {"pattern": "comparison_results/unique_to_*.csv",
             "description": "Edges unique to one dataset. Standard mode "
                            "only: combination runs omit these files instead "
                            "of inferring a union threshold."},
            {"pattern": "comparison_results/top_edges_comparison.csv",
             "description": "Top conserved/divergent edges. Standard mode "
                            "only (middle scalar threshold)."},
            {"pattern": "comparison_results/top_edges_overlap.csv",
             "description": "Overlap of the top edges across datasets. "
                            "Standard mode only."},
            {"pattern": "comparison_results/degree_*.csv",
             "description": "Degree analysis by type (in/out/statistics). "
                            "Standard mode only; combination runs omit "
                            "these files."},
            {"pattern": "comparison_results/neuron_counts_*.csv",
             "description": "Neuron counts per type and overall."},
            {"pattern": "comparison_results/dataset_metadata_comparison.csv",
             "description": "Per-dataset metadata overview: one row per "
                            "dataset with total/typed/untyped neuron counts, "
                            "type coverage percentage, pre/post-synaptic and "
                            "total synapse counts, ROI count and coverage "
                            "notes."},
            {"pattern": "comparison_results/motif_analysis.csv",
             "description": "Network motif analysis."},
            {"pattern": "comparison_results/threshold_sensitivity.csv",
             "description": "Per-dataset edge counts per threshold with "
                            "retention vs the previous threshold (unique "
                            "source-target pairs). Includes the complete "
                            "requested/applied threshold provenance: the "
                            "StrongestFirst budget and bite, tau, tau_canonical, "
                            "w2, the Edge Budget/w0/w1 state, W*, "
                            "paths_complete, and skipped/duplicate_of. In "
                            "combination mode rows are query-keyed with "
                            "query_id/query_label; adjacent-threshold "
                            "retention is intentionally blank because the "
                            "query rows are not a monotone schedule.",
             "columns": ["dataset", "threshold", "requested_threshold",
                         "applied_threshold", "applied_threshold_source",
                         "strongest_first_budget",
                         "strongest_first_budget_bitten",
                         "strongest_first_tau", "tau_canonical",
                         "strongest_dropped_bottleneck", "edge_budget",
                         "edge_budget_applied", "edge_budget_landing",
                         "edge_weight_floor",
                         "strongest_retained_bottleneck", "paths_complete",
                         "edge_count", "edges_retained_from_prev",
                         "retention_rate", "edges_lost",
                         "skipped", "duplicate_of", "drop_untyped",
                         "untyped_dropped_rows", "untyped_dropped_neurons"]},
            {"pattern": "comparison_results/pathfinding_provenance.csv",
             "description": "One complete provenance row per dataset and "
                            "requested threshold. Use applied_threshold for "
                            "the canonical equivalent Min Synapse Count; tau "
                            "is the StrongestFirst landing/collapse bound; "
                            "w0/w1 are the Edge Budget floor/landing; w2 is "
                            "the strongest dropped bottleneck and W* the "
                            "strongest retained bottleneck.",
             "preview": True,
             "preview_title": "Pathfinding threshold provenance",
             "columns": ["dataset", "threshold", "threshold_scope",
                         "requested_threshold",
                         "applied_threshold", "applied_threshold_source",
                         "strongest_first_budget",
                         "strongest_first_budget_bitten",
                         "strongest_first_tau", "tau", "tau_canonical",
                         "strongest_dropped_bottleneck",
                         "strongest_retained_bottleneck", "edge_budget",
                         "edge_budget_applied", "edge_budget_landing",
                         "edge_weight_floor", "paths_complete", "pruned",
                         "skipped", "duplicate_of", "applied_folder",
                         "path_mode", "comparison_mode", "drop_untyped"]},
            {"pattern": "comparison_results/threshold_combinations.csv",
             "description": "Canonical query manifest. One row per query ID "
                            "and dataset, joining the requested threshold to "
                            "the raw run's applied threshold, StrongestFirst "
                            "budget/tau, Edge Budget/w0/w1, bottlenecks and "
                            "paths_complete. In standard mode the rows are "
                            "same-threshold queries; in advanced mode each "
                            "row is an explicit threshold combination.",
             "preview": True,
             "preview_title": "Threshold query manifest",
             "columns": ["query_id", "query_label", "threshold_mode",
                         "threshold_scope",
                         "dataset", "requested_threshold",
                         "applied_threshold", "applied_threshold_source",
                         "strongest_first_budget", "strongest_first_tau",
                         "edge_budget", "edge_weight_floor",
                         "strongest_dropped_bottleneck",
                         "strongest_retained_bottleneck", "paths_complete"]},
            {"pattern": "comparison_results/untyped_dropped_records.csv",
             "description": "Rows removed by the comparison-level Drop "
                            "Untyped Neurons filter after standardized labels "
                            "were resolved. Delegated per-dataset pathfinding "
                            "folders keep their raw rows; this is the single "
             "comparison-level records file.",
             "columns": ["dataset", "threshold", "untyped_side",
                         "query_id", "query_label"]},
            {"pattern": "comparison_results/threshold_alignment_best_matches.csv",
             "description": "Feature C: per (dataset pair, anchor threshold) the "
                             "best-matching threshold in the other dataset, found by a "
                             "bisection prober over the whole-dataset edge-density curve "
                             "(edge-count distance is primary; Jaccard/rank similarity "
                             "at the matched point). Includes a global-best row. "
                             "In combination mode this is a raw-run schedule "
                             "diagnostic; use threshold_combinations.csv for "
                             "query comparisons."},
            {"pattern": "comparison_results/threshold_alignment_matrix.csv",
             "description": "Feature C: pairwise alignment metrics over the TYPED "
                             "threshold grid points only (edge-count distance, Jaccard, "
                             "rank similarity per dataset-pair/threshold-pair). "
                             "Combination-mode rows are explicitly diagnostic, "
                             "not scalar comparison identities."},
            {"pattern": "comparison_results/edge_density_per_threshold.csv",
             "description": "Feature D data: per dataset, distinct connection-pair "
                             "counts (absolute and per-neuron) over the extended "
                             "threshold grid used by the prober; typed thresholds are "
                             "flagged. In combination mode, threshold_scope marks "
                             "this as a raw-run schedule diagnostic. Rendered as "
                             "edge_density_threshold_curves.png."},
            {"pattern": "comparison_results/density_curves.csv",
             "description": "Per-dataset query-scoped density curves for "
                            "every pathfinding mode, from each dataset's "
                            "minimal available threshold to its measured "
                            "ceiling. y is a DENSITY (E(t)/N, fixed N); x is "
                            "the per-connection Min Synapse Count. The edge "
                            "basis follows the run's Drop Untyped setting "
                            "(bodyId_edges_typed when on, "
                            "bodyId_edges_all_but_debris when off; "
                            "segmentation debris is always excluded).",
             "columns": ["dataset", "threshold", "edge_count", "path_count",
                         "density", "normalizer", "basis", "w_start",
                         "w_star_measured", "path_complete_from",
                         "is_materialized"]},
            {"pattern": "comparison_results/density_windows.csv",
             "description": "Per-dataset completeness window [w_start, "
                            "w_star_measured] and the stored-vs-measured "
                            "ceiling check, with the searched-graph node "
                            "classes: typed neurons, untyped (Unknown/empty "
                            "label; counted only when Drop Untyped is off) "
                            "and segmentation debris (never counted). "
                            "n_edges is the all-class cone count at the "
                            "applied threshold; n_edges_active_basis "
                            "re-counts it under the active basis so it "
                            "matches density_curves.csv edge_count at "
                            "w_start.",
             "columns": ["dataset", "applied", "w_start", "w_star_stored",
                         "w_star_measured", "w_star_mismatch", "budget_bitten",
                         "paths_complete", "path_complete_from", "n_paths",
                         "n_edges", "n_edges_active_basis", "n_nodes",
                         "n_nodes_typed", "n_nodes_untyped",
                         "n_nodes_debris"]},
            {"pattern": "comparison_results/density_alignment_best_matches.csv",
             "description": "Auto threshold mode: vertical (same-threshold) and "
                            "horizontal (same-density) aligned threshold rows, "
                            "one integer per dataset. The rows this run "
                            "installed are its queries (see "
                            "threshold_combinations.csv); when auto mode "
                            "resolved only partially, the remaining rows are "
                            "advisory — the report names them and the density "
                            "section carries a banner. When both modes coexist, "
                            "read vertical rows as the like-for-like spine and "
                            "horizontal rows as the density-matched envelope. "
                            "Horizontal rows also carry "
                            "density_at_<dataset> columns (the achieved density "
                            "at each chosen integer threshold) and "
                            "max_abs_deviation (worst distance from the target "
                            "level after integer quantization).",
             "columns": ["mode", "level_continuous", "level_normalized",
                         "degenerate", "clamped", "max_abs_deviation"]},
            {"pattern": "dataset_data/*/_density/density_meta.json",
             "description": "The query-scoped density capture persisted by "
                            "every delegated pathfinding run (one copy per "
                            "dataset, referenced by each minsyn_* folder's "
                            "density_meta.json)."},
            {"pattern": "comparison_results/path_count_comparison.csv",
             "description": "Path-count comparison across datasets. "
                            "Combination rows carry query_id/query_label and "
                            "one requested threshold per dataset."},
            {"pattern": "comparison_visualizations/*.png",
             "description": "Static heatmaps and path-count plots per "
                            "threshold. Custom combination plots use "
                            "query_<query_id> stems for every query."},
            {"pattern": "comparison_visualizations/by_ratio/**",
             "description": "Connection-ratio heatmaps (PNG + backing CSV) "
                            "when enabled; not emitted for pathfinding "
                            "comparisons."},
            {"pattern": "comparison_visualizations/by_probability/**",
                            "description": "Traversal-probability heatmaps (PNG + backing "
                            "CSV), when enabled; not emitted for pathfinding "
                            "comparisons."},
            {"pattern": "comparison_visualizations/visualization_data/*.csv",
             "description": "CSVs backing the HTML report (edge overlap, key "
                            "findings, overlap matrices, path counts). Custom "
                            "rows carry query_id/query_label and per-dataset "
                            "requested/applied threshold columns; the scalar "
                            "threshold field is blank rather than overloaded "
                            "with a query ID."},
            {"pattern": "similarity_matrices/similarity_threshold_*.csv",
             "description": "Cross-dataset similarity rows per threshold.",
             "preview": True,
             "preview_title": "Cross-dataset similarity per threshold",
             "columns": [
                 "dataset_1", "dataset_2", "jaccard_similarity",
                 "ruzicka_similarity", "pearson_correlation", "edges_in_d1",
                 "edges_in_d2", "common_edges", "union_edges", "unique_to_d1",
                 "unique_to_d2", "edge_rank_correlation", "cosine_similarity",
                 "path_rank_correlation", "spearman_rank_correlation",
                 "rv_coefficient", "path_jaccard_similarity",
                 "path_top20_overlap", "hop_profile_w1",
                 "netsimile_similarity", "coverage_d1", "coverage_d2",
                 "coverage_min", "top20_overlap", "strength_w1_out",
                 "strength_w1_in", "threshold"]},
            {"pattern": "similarity_matrices/similarity_query_*.csv",
             "description": "Combination-mode similarity rows for one query. "
                            "The filename uses a filesystem-safe query-ID slug, "
                            "while the original query ID and per-dataset threshold "
                            "columns are the comparison join key; the raw "
                            "threshold union is not used here.",
             "preview": True,
             "preview_title": "Similarity per threshold query",
             "columns": ["query_id", "query_label", "dataset_1", "dataset_2",
                         "threshold_", "jaccard_similarity",
                         "ruzicka_similarity", "pearson_correlation",
                         "common_edges", "path_jaccard_similarity",
                         "path_top20_overlap", "hop_profile_w1",
                         "netsimile_similarity", "coverage_d1", "coverage_d2",
                         "coverage_min", "top20_overlap", "strength_w1_out",
                         "strength_w1_in"]},
            {"pattern": "similarity_matrices/similarity_by_query.csv",
             "description": "Combined combination-mode similarity export; "
                            "filter by query_id before comparing rows."},
            {"pattern": "conserved_reciprocal_graph/*.html",
             "description": "Network graph of hemisphere-conserved "
                            "reciprocal connections (when both options are "
                            "on)."},
            {"pattern": "conserved_paths/*.html",
             "description": "Network graphs of hemisphere-conserved "
                            "connections per threshold (Conserved Hemisphere "
                            "analysis): same renderer as the reciprocal "
                            "graphs, one page per threshold."},
            {"pattern": "type_resolution_topology.json",
             "description": "Machine-readable record of the type-resolution "
                            "topology used to union types across datasets "
                            "(merge-policy decisions: anchors, residuals, "
                            "fan-ins)."},
            {"pattern": "comparison_report_used_data/type_appearance_order.csv",
             "description": "Order in which types first appear in the "
                            "comparison tables — the canonical ordering the "
                            "report's tables and heatmaps follow."},
            {"pattern": "dataset_data/**",
             "description": "Raw per-dataset FindNeuronConnection runs "
                            "(one subfolder per dataset/threshold, same "
                            "layout as Complete Paths — see the pathfinding "
                            "model sections above for the threshold/budget "
                            "provenance and file naming — plus "
                            "connections_edge.csv and its "
                            "connections_edge.fingerprint.json query-"
                            "fingerprint sidecar, whose match/mismatch "
                            "governs cache reuse)."},
        ],
        "explanation": _PATHFINDING_EXPLANATION,
    },
    "nb_find_lines": {
        "title": "NeuronBridge — Find Driver Lines",
        "summary": "EM→LM mapping: driver lines matching the queried EM "
                   "neurons.",
        "files": [
            {"pattern": "*_lines.csv",
             "description": "All matched driver lines with scores "
                            "(bodyId-level source data; Compact output "
                            "detail removes it after the summary is "
                            "written).",
             "preview": True,
             "preview_title": "Matched driver lines",
             "columns": ["line", "score", "match_type", "library"]},
            {"pattern": "line_summary.csv",
             "description": "Summary statistics per line.",
             "preview": True,
             "preview_title": "Line summary",
             "columns": ["line", "n_neurons", "n_types", "mean_score",
                         "max_score"]},
            {"pattern": "gal4_lexa_summary.csv",
             "description": "GAL4/LexA library summary (when Separate "
                            "Split-GAL4 is on)."},
            {"pattern": "split_gal4_summary.csv",
             "description": "Split-GAL4 library summary (when Separate "
                            "Split-GAL4 is on)."},
            {"pattern": "images/**",
             "description": "Downloaded CDM/FlyLight images (only when "
                            "image download is enabled; Compact output "
                            "detail removes them after the PDF/PPTX is "
                            "generated)."},
            {"pattern": "parameters.json",
             "description": "Analysis parameters."},
            {"pattern": WARNING_FILENAME,
             "description": "Notes collected during the run (rendered in "
                            "the Warnings section above)."},
        ],
    },
    "nb_find_lines_expanded": {
        "title": "NeuronBridge — Find Driver Lines (cross-dataset expansion)",
        "summary": "Coverage-routed, name-expanded EM→LM mapping: every "
                   "query chip expands into all NeuronBridge-hosted "
                   "releases before searching.",
        "files": [
            {"pattern": "expansion_map.csv",
             "description": "Per chip: expanded name → hosted release → "
                            "mapping status/kind.",
             "preview": True,
             "preview_title": "Expansion map",
             "columns": ["source_query", "expanded_name", "nb_dataset",
                         "mapping_status", "mapping_kind"]},
            {"pattern": "expansion_summary.json",
             "description": "Original selection, coverage routing "
                            "(selected vs covered vs unavailable "
                            "datasets), output detail, and cleanup audit."},
            {"pattern": "chip_*/**",
             "description": "Per-query-chip Find Lines outputs (one folder "
                            "per chip; single-chip runs are flat in the "
                            "run root)."},
            {"pattern": "images/**",
             "description": "Downloaded CDM/FlyLight images (only when "
                            "image download is enabled; Compact output "
                            "detail removes them after the PDF/PPTX is "
                            "generated)."},
            {"pattern": "images_summary.pdf",
             "description": "PDF contact sheet of the top ranked lines "
                            "(when a PDF summary is requested)."},
            {"pattern": "line_summary.csv",
             "description": "Summary statistics per line (single-chip "
                            "flat runs)."},
            {"pattern": "*_summary.csv",
             "description": "GAL4/LexA and Split-GAL4 library summaries "
                            "(single-chip flat runs)."},
            {"pattern": "*_lines.csv",
             "description": "BodyId-level match tables (source data; "
                            "Compact output detail removes them)."},
            {"pattern": "cleanup_audit.json",
             "description": "Output-detail cleanup audit (Compact runs): "
                            "removed paths and reclaimed bytes."},
            {"pattern": WARNING_FILENAME,
             "description": "Run notes; coverage warnings carry a "
                            "`coverage:` prefix."},
            {"pattern": "parameters.json",
             "description": "Analysis parameters (per chip)."},
        ],
    },
    "nb_find_neuron": {
        "title": "NeuronBridge — Find EM Neurons",
        "summary": "LM→EM mapping: EM neurons matching a driver line.",
        "files": [
            {"pattern": "all_neurons.csv",
             "description": "Combined matched neurons across all datasets.",
             "preview": True,
             "preview_title": "Matched EM neurons",
             "columns": ["bodyId", "dataset", "instance", "type", "status",
                         "score", "image_id", "lm_sample", "match_type",
                         "library", "source_line"]},
            {"pattern": "*_neurons.csv",
             "description": "Matched neurons for the line (combined and "
                            "per-dataset; bodyId-level source data — "
                            "Compact output detail removes them after the "
                            "type summaries are written).",
             "columns": ["bodyId", "dataset", "instance", "type", "status",
                         "score"]},
            {"pattern": "by_dataset/*_neurons.csv",
             "description": "Per-dataset matched neurons (bodyId-level "
                            "source data; Compact output detail removes "
                            "them after the type summaries are written)."},
            {"pattern": "*_types.csv",
             "description": "Per-dataset type aggregates of the matches.",
             "preview": True,
             "preview_title": "Type aggregates",
             "columns": ["type", "labeled_N", "max_score", "median_score",
                         "Q3_score", "Q1_score", "avg_score",
                         "typed_N_in_dataset"]},
            {"pattern": "*_type_mapped.csv",
             "description": "Cross-dataset type mapping summary.",
             "columns": ["canonical_type", "best_max_score",
                         "total_labeled_N"]},
            {"pattern": "labeling_distribution.html",
             "description": "Score distribution visualization."},
            {"pattern": "parameters.json",
             "description": "Analysis parameters."},
            {"pattern": WARNING_FILENAME,
             "description": "Notes when score-cutoff behavior affects "
                            "interpretation (rendered in the Warnings "
                            "section above)."},
            {"pattern": "plot-3d_*/**",
             "description": "Per-dataset 3D skeleton visualizations (only "
                            "when Visualize Top N > 0)."},
        ],
    },
    "nb_colabel": {
        "title": "NeuronBridge — Co-Labeling Analysis",
        "summary": "Multi-line co-labeling analysis of driver lines.",
        "files": [
            {"pattern": "expression_matrix.csv",
             "description": "Type × line score matrix (types prefixed with "
                            "dataset abbreviations).",
             "preview": True,
             "preview_title": "Expression matrix",
             "matrix": "rows = neuron types, columns = driver lines, "
                       "values = max NeuronBridge score"},
            {"pattern": "expression_matrix_merged.csv",
             "description": "Same matrix with types merged across datasets "
                            "(max score aggregation).",
             "preview": True,
             "preview_title": "Merged expression matrix",
             "matrix": "rows = merged types, columns = driver lines, "
                       "values = max score"},
            {"pattern": "expression_matrix*.html",
             "description": "Interactive expression-matrix heatmaps."},
            {"pattern": "expression_matrix_viz.csv",
             "description": "Reduced matrix for visualization."},
            {"pattern": "expression_matrix_merged_viz.csv",
             "description": "Reduced merged matrix for visualization."},
            {"pattern": "colabeling_matrix_jaccard.csv",
             "description": "Binary Jaccard similarity between lines.",
             "matrix": "rows/columns = driver lines, values = Jaccard "
                       "similarity"},
            {"pattern": "colabeling_matrix_weighted_jaccard.csv",
             "description": "Weighted Jaccard similarity between lines.",
             "matrix": "rows/columns = driver lines, values = weighted "
                       "Jaccard similarity"},
            {"pattern": "colabeling_matrix_*.html",
             "description": "Interactive co-labeling similarity heatmaps."},
            {"pattern": "labeling_distribution_*.html",
             "description": "Labeling distribution visualizations (by type, "
                            "by neuron, stacked)."},
            {"pattern": "distribution_data_by_type.csv",
             "description": "Raw distribution data per type.",
             "columns": ["type", "score", "source_line", "dataset"]},
            {"pattern": "distribution_data_by_neuron.csv",
             "description": "Raw distribution data per neuron (row-level "
                            "source data; Compact output detail removes "
                            "it).",
             "columns": ["bodyId", "dataset", "instance", "type", "status",
                         "score", "image_id", "lm_sample", "match_type",
                         "library", "_passes_min_score", "source_line"]},
            {"pattern": "labeling_info.csv",
             "description": "Case-sensitive type × line matrix with dataset "
                            "column.",
             "columns": ["type", "dataset"]},
            {"pattern": "line_summary.csv",
             "description": "Summary statistics per line.",
             "preview": True,
             "preview_title": "Line summary",
             "columns": ["line", "n_neurons", "n_types", "mean_score",
                         "max_score", "n_neurons_HMS", "n_types_HMS",
                         "n_neurons_MS", "n_types_MS", "Qf",
                         "colabel_sparsity"]},
            {"pattern": "line_labeled_neurons/**",
             "description": "Per-line neuron details (neurons, per-dataset "
                            "neurons/types, type mapping; row-level source "
                            "data — Compact output detail removes the "
                            "folder after the matrices/report are "
                            "written)."},
            {"pattern": "parameters.json",
             "description": "Analysis parameters."},
            {"pattern": WARNING_FILENAME,
             "description": "Notes describing score-cutoff filtering and "
                            "retained top-N records (rendered in the "
                            "Warnings section above)."},
            {"pattern": "colabeling_report.html",
             "description": "Comprehensive HTML report."},
            {"pattern": "plot-3d_*/**",
             "description": "Per-dataset 3D visualizations (only when "
                            "Visualize Top N > 0)."},
        ],
    },
    "flylight_download": {
        "title": "FlyLight Image Download",
        "summary": "Confocal image downloads for driver lines.",
        "files": [
            {"pattern": "*_mip.png",
             "description": "Downloaded maximum-intensity-projection images, "
                            "organized by collection and line (or directly "
                            "in the output root in flat-structure mode). "
                            "File names "
                            "encode slide, sex, zoom, region, and driver "
                            "type."},
            {"pattern": "*.jpg",
             "description": "Downloaded JPG images (when jpg format is "
                            "requested), nested by collection or flat "
                            "depending on the structure mode."},
            {"pattern": "*_summary.pdf",
             "description": "Summary document with the downloaded images "
                            "(when a PDF summary is requested)."},
            {"pattern": "*_summary.pptx",
             "description": "Summary slides with the downloaded images "
                            "(when a PDF summary is requested)."},
        ],
    },
    "type_mapping_validation": {
        "title": "Type-Mapping Validation (TM VEV)",
        "summary": "BodyId-level cross-dataset type validation "
                   "(validate · expand · visualize): branch pools, "
                   "expansion bins, fill proposals, out-map expansion, "
                   "morph qualification, and 3D scenes. Proposals only — "
                   "the mapping is never rewritten.",
        "files": [
            {"pattern": "report.html",
             "description": "The per-run report's 14 tabs: Coverage "
                            "(headline + the three coverage levels L1 claim "
                            "/ L2 provenance / L3 validation, plus the "
                            "disclosure-evidence card), Branches "
                            "(marks same-name-first selections), Targets, "
                            "Fill, Reciprocal (stage-5d reverse evidence, "
                            "one row per scanned neuron), Out-map, Backward "
                            "source status, Homolog · forward and "
                            "Homolog · backward (the offline morph "
                            "top-match tables), Suspects verification "
                            "(opt-in runs), Morph, Pooling, Scenes "
                            "(which also prints the palette the run actually "
                            "wore, one chip per category, each recolored bin "
                            "naming the default it replaced) "
                            "and Log. Hover any dotted "
                            "term — or any table header, which explains its "
                            "own column — for its definition. Regenerable "
                            "for any past run: python -m "
                            "comparison.mapping_validation_report <run_dir>"},
            {"pattern": "README.txt",
             "description": "Slim directions (what file is what, where to "
                            "start) + the raw run log. The analysis content "
                            "lives in report.html."},
            {"pattern": "user_warning_notes.txt",
             "description": "Bracketed-tag warning lines: scene self-check "
                            "status, the null-sample run-sensitivity "
                            "advisory, the [reciprocal] share of scanned "
                            "gap-fill members whose own branch source type "
                            "reaches a reverse top-3, and mapper-gap "
                            "evidence (types with "
                            "no backward mapping)."},
            {"pattern": "set_coverage.json",
             "description": "Set-level coverage (the deliverable), in two "
                            "role-named blocks — `source` (assigned / "
                            "fill-proposed / unpaired) and `target` (in-pool "
                            "by tier, holes with bodyIds, family_material) — "
                            "labelled by source_dataset / target_dataset, "
                            "plus mapper_gap."},
            {"pattern": "validation/pair_summary.csv", "preview": True,
             "description": "Per branch: pools, best (the mutual-best 1:1 "
                            "pair count), gap = smaller pool − best, "
                            "gap_triggered, verdict/noise counters, "
                            "hemisphere symmetry. The report's Branches tab "
                            "shows Mapped (verified_strong+verified+"
                            "borderline) and measures its own gap against "
                            "that, because a source can carry a verdict "
                            "without being paired — both numbers hover."},
            {"pattern": "mapping/mapping_export.csv",
             "description": "Branch-level mapping: chains, linker values, "
                            "refined bodyId pools (the mapper-facing "
                            "export). Additive route_basis/via_mid columns "
                            "label full-map COMPOSED pairs (route_basis "
                            "= direct|composed; curated runs leave them "
                            "at their defaults) — the same additive "
                            "columns ride validation_results.csv and "
                            "pair_summary.csv."},
            {"pattern": "validation/validation_results.csv",
             "description": "Source×branch verdict rows: verdict tier, "
                            "ranks + scores, connectivity flags, source "
                            "size. The published target is the ordering-"
                            "chain best (jaccard first, rank_union "
                            "breaking a jaccard tie); "
                            "ru_top_target_bodyId names the rank_union "
                            "claimant when the two differ."},
            {"pattern": "validation/pool_categories.csv",
             "description": "Per in-map target: tier (matched / verified / "
                            "borderline / unmatched) with the chain "
                            "claimant's best evidence."},
            {"pattern": "validation/examinees.csv",
             "description": "Per-pair pool-edge expansion rows (the "
                            "`examinee rows=` of the pair log; "
                            "mode-invariant set) with the Revision 3.12 "
                            "category partition (tier / sibling / "
                            "candidates / family / relative / examinees "
                            "- renamed from suspicious_candidates.csv). "
                            "The `examinees` BIN lives in "
                            "deep_candidates.csv, not here. "
                            "Leaf tokens, bars, and out-of-scope flags."},
            {"pattern": "mapping/same_name_excluded.csv",
             "description": "Queried types whose same-name fan-out was "
                            "held/excluded by the mapper, or multi-value "
                            "type cells (kept atomic) - advisory "
                            "accounting, never a gate."},
            {"pattern": "mapping/disclosure_evidence.csv",
             "description": "Evidence-tier disclosure (three-tier readout): "
                            "the ends the mapper decision DECLINED but the "
                            "derivation evidence reaches, with the decline "
                            "reason and - when --verify-suspects ran - the "
                            "verification verdicts. Advisory bin, never the "
                            "headline counts."},
            {"pattern": "mapping/suspects_verification.csv",
             "description": "Opt-in (--verify-suspects): advisory "
                            "connectivity verification of the mapper's "
                            "rival suspects - never merged into the "
                            "validation counts."},
            {"pattern": "validation/noise_filtered_candidates.csv",
             "description": "Gate-dropped expansion rows with "
                            "noise_reason."},
            {"pattern": "validation/deep_candidates.csv",
             "description": "Candidate-window rows below the pool best: the "
                            "top-rank_top_k band in family/aggressive "
                            "modes, the wider band in aggressive only."},
            {"pattern": "validation/forward_matches.csv",
             "description": "The Homolog · forward tab's data (user "
                            "2026-09-26): one row per appeared source "
                            "bodyId — assigned, fill-proposed, out-of-map "
                            "or unpaired alike — with the chain-best "
                            "target (primary_*), the serialized "
                            "forward_topN union neighbourhood (top-3 "
                            "rank_union ∪ top-3 jaccard, chain order), "
                            "n_scanned / scanned_at, and the primary "
                            "pair's morph display joins. Captured during "
                            "stage 2; the report re-derives ✓/✗ from the "
                            "branch bars — nothing is re-scored. Type "
                            "columns read `untyped` for unannotated "
                            "neurons, never the raw profiler 'nan'."},
            {"pattern": "gap_fill/gap_fill_proposals.csv",
             "description": "Fill proposals (in_pool / out_of_pool) for "
                            "every unpaired neuron — proposals only."},
            {"pattern": "gap_fill/gap_fill_levels.csv",
             "description": "Branch-level fill level: high / medium / low / "
                            "type_gated / advice."},
            {"pattern": "gap_fill/gap_fill_dedup.csv", "preview": True,
             "description": "The bodyId-unique fill (one row per target "
                            "bodyId, dedup precedence + dup flag) — the "
                            "real gap-fill list. On default runs "
                            "runs it also carries each neuron's reciprocal "
                            "ledger from its strongest branch: the grade, "
                            "the top-1 triple, and the branch-type hit the "
                            "grade rests on (with backward_own_type_via). "
                            "A rollup, not a scan record — the scores, "
                            "ranks and top-N live in "
                            "expansion/backward_matches.csv."},
            {"pattern": "expansion/family_candidates.csv",
             "description": "The whole family bin (out-map bodyIds of each "
                            "branch's target type). NOT the same number as "
                            "the Type Mapping panel's 'Out-map (in-map "
                            "types)' column: the panel subtracts the bridge "
                            "claim from the received types' populations and "
                            "has no morphology, so a candidate that closes a "
                            "hole here cannot close one there — on "
                            "circadian_clock → male-cns the panel reads 15 "
                            "(219 − 204) where this file holds 11 rows."},
            {"pattern": "expansion/relatives.csv",
             "description": "The whole relative bin (type-mates of "
                            "candidate types outside the map)."},
            {"pattern": "expansion/backward_matches.csv",
             "description": "Default ON (--no-backward-evidence opts "
                            "out): the reverse target -> source homolog "
                            "evidence per scanned "
                            "neuron - the candidates / family / relative "
                            "members first, then the UNMATCHED validated "
                            "pool members "
                            "(scan_role=pool_target); matched / verified / "
                            "borderline are not scanned - the symmetric "
                            "forward score is their evidence "
                            "(member_bodyId, "
                            "member_type, "
                            "member_category, scan_role): backward_evidence "
                            "with the backward_top1_source_bodyId / "
                            "backward_top1_source_type and the "
                            "backward_top1_in_branch flag, the "
                            "backward_own_type_* columns naming the "
                            "branch-type hit the grade actually rests on "
                            "(its bodyId, type, scores, ranks, thin flag "
                            "and backward_own_type_via = which ranking "
                            "placed it there), plus the serialized "
                            "backward_topN list in jaccard order. Each row "
                            "also states the "
                            "evidence base the score was computed over "
                            "(backward_shared_type_count / "
                            "backward_union_type_count) and "
                            "backward_thin_evidence (a top-1 resting on "
                            "<= 3 shared partner types - a reader's note "
                            "that never gates a verdict or a fill count). "
                            "Connectivity "
                            "only — morphology is never re-scored here — "
                            "and advisory: it labels the bins, it never "
                            "changes a fill count."},
            {"pattern": "expansion/target_matches.csv",
             "description": "The Homolog · backward tab's data (stage 5e, "
                            "user 2026-09-26): one row per appeared TARGET "
                            "bodyId — pool members including matched / "
                            "verified / borderline, plus expansion-bin, "
                            "out-map and proposal targets — scanned back "
                            "against the WHOLE source dataset. Carries "
                            "pool_category + pool_branches, the chain-best "
                            "primary_source_*, the backward_topN_union "
                            "payload (top-3 rank_union ∪ top-3 jaccard), "
                            "n_scanned / scanned_at and the same morph "
                            "display joins. No caps; advisory display "
                            "data only. Type columns read `untyped` for "
                            "unannotated neurons, never the raw "
                            "profiler 'nan'."},
            {"pattern": "expansion/out_map_expansion.csv",
             "description": "Top-k typed non-in-map candidates per "
                            "UNCLAIMED source, morph-checked against the "
                            "run null bar — mapper-gap evidence, not "
                            "fills."},
            {"pattern": "expansion/source_status.csv",
             "description": "Advisory backward (`source-`) status per "
                            "in-branch source: the column view of the same "
                            "pair scores. Never gates, never rewrites the "
                            "mapping."},
            {"pattern": "expansion/source_candidates.csv",
             "description": "Out-of-map sources whose best-ranked hits "
                            "reach a branch pool (null-bar "
                            "morph-qualified) — the scenes' "
                            "source-candidates roots. Advisory."},
            {"pattern": "pooling/pooling_candidates.csv",
             "description": "--mode pooling only: every (source, target) "
                            "pair the BAR admitted — each queried source keeps "
                            "the top-N rows of the configured metric "
                            "(`either` = the union of both metrics' own top-N) "
                            "— with the morph verdict, `morph_bar_kind` names "
                            "the rule that graded the pair, `morph_bar` its "
                            "binding value, `morph_similarity` the Track-A "
                            "score and `morph_pool_ref` the native pool "
                            "reference a native floor is applied to, so every "
                            "verdict recomputes from its own row — plus the "
                            "per-row tier (matched / verified / nominated), "
                            "the shared leaf token and the post-hoc "
                            "mapper_cell (confirmed / "
                            "type_miss / type_new). The Jaccard / rank_union "
                            "floors and the window ride along as ADVISORY "
                            "flags (`below_jaccard_floor`, "
                            "`below_rank_union_floor`, `outside_window`): "
                            "flagged rows are kept, never removed. The mapper "
                            "decides none of it — it is joined afterwards. "
                            "Includes the targets the morphology gate refused, "
                            "so the refusals stay auditable."},
            {"pattern": "pooling/pooling_sources.csv", "preview": True,
             "description": "--mode pooling only: one row per QUERIED source "
                            "(the mode's own unit and denominator) — its "
                            "chain-best finding, that row's tier (matched / "
                            "verified / nominated), how many targets it "
                            "admitted and how many survived the morphology "
                            "bar, source_claimed (mapper-derived, advisory), "
                            "and no_finding for a source the bar admitted "
                            "nothing for. A source that found nothing is "
                            "named, never absent."},
            {"pattern": "pooling/pooling_pool.csv",
             "description": "--mode pooling only: the same pool deduplicated "
                            "to one row per candidate target neuron on the "
                            "ordering chain, AFTER the morphology gate — with "
                            "`in_pool` saying what that gate decided. A target "
                            "stays while ANY admitting row survives the bar, so "
                            "the row standing for it is the chain-best row whose "
                            "verdict survived; `n_rows_refused` counts the rows "
                            "that did not, and a target every row was refused "
                            "for is still here with `in_pool=False` (it left the "
                            "scene, not the file). Targets only the supervised "
                            "path claims ride along as `mapper_cell="
                            "verified_only`. Carries `tiers` (the tier set its "
                            "admitting rows carry), `best_bar_rank`, `n_sources` "
                            "/ `dup`, and the same four morph columns as the "
                            "per-pair file."},
            {"pattern": "pooling/pooling_cross_validation.json",
             "description": "--mode pooling only: the unsupervised pool "
                            "compared with the mapper's claim sets, the bar "
                            "that admitted it (metric, top-N, the 2N tie-guard "
                            "row cap and `rows_cut`) beside `gate` (the floors "
                            "it only flags, with a `role` line saying so) and "
                            "`floor_flags` (how many admitted rows each one "
                            "flagged, and the row total they were counted "
                            "over), `tiers` counting ROWS per tier, "
                            "`pool_per_source` with its `pool_size_warning` "
                            "band, the morph record "
                            "(units / attempted / capped / scored / qualified "
                            "/ no-score / shared, `budget` naming the rule that "
                            "priced them — 3 units per queried source unless a "
                            "cap was configured — plus dropped_targets, how many "
                            "targets the bar refused, and the target-vector "
                            "store's ledger: loaded / "
                            "stale_dropped / saved / targets), `body_ids` "
                            "listing the pool_miss and verified_only targets, "
                            "`pool_miss_by_type` and `map_tags` (the harvest by "
                            "target type), the input "
                            "fingerprint the scores came from (git rev, the "
                            "scanned universes, both profile caches, the mapper "
                            "snapshot, and `morph_stores` — the vector cache, "
                            "whitener and skeleton count a native verdict was "
                            "read out of, keyed by pass (supervised / "
                            "pooling)), and the "
                            "reading notes that say which cells are NOT "
                            "recall measures."},
            {"pattern": "morphology_calibration.json",
             "description": "Per-branch qualification bars (floors v3), "
                            "the run null bar, AUC gate record, and score "
                            "frames."},
            {"pattern": "parameters.json",
             "description": "Every knob incl. validation_mode / mode_rank (a "
                            "pooling run has mode_rank null: the mode is "
                            "parallel to the nested chain, not above it), and "
                            "the scene look (neuron_alpha / scene_viz / "
                            "scene_category_colors — with scenes on these are "
                            "the EFFECTIVE values the pages wore, not an echo "
                            "of what was sent: resolved kwargs plus the full "
                            "legend palette, defaults and aliases included)."},
            {"pattern": "pipeline_progress.jsonl",
             "description": "Stage timeline events (pre-flight, scans, "
                            "out-map expansion, the stage-5d reciprocal "
                            "scan, run_done)."},
            {"pattern": "visualization/*/branches_*.html",
             "description": "One 3D review scene per parent type with "
                            "expansion content (source coordinates; a PNG "
                            "preview sits next to it). The legend tree is "
                            "one root per expansion category with ordered "
                            "leaf tokens on every bodyId; each category's "
                            "color is its own default and is adjustable per "
                            "category, a recolor moving the whole bin across "
                            "every branch; on "
                            "--backward-evidence runs a scanned gap-fill "
                            "leaf carries its reciprocal grade as a "
                            "trailing · high / · medium / · low tag "
                            "(an unscanned member keeps a bare leaf)."},
            {"pattern": "visualization/*/SCENE_FAILED.txt",
             "description": "This scene did NOT render: the page, PNG and "
                            "manifest are absent. Two causes. (1) A "
                            "mid-render failure: the scene folder is "
                            "created before the figure, so it would "
                            "otherwise look like a finished scene on disk, "
                            "and everything else inside is a partial "
                            "artifact. (2) The empty-query-layer gate: the "
                            "parent's queried source neurons all failed to "
                            "load skeletons, so the folder is synthetic "
                            "and holds nothing but this marker. Names the "
                            "parent type and the error (plus the traceback "
                            "for render failures). NOTE: this marker is "
                            "written by the TM VEV scene builder only — a "
                            "plain 3D-Skeleton viewer run that dies "
                            "mid-render leaves an unmarked partial folder."},
        ],
    },
}


def preview_views(tool_name: str) -> list:
    """Result-table previews for *tool_name*, in spec order.

    Returns the ``TOOL_GUIDE_SPECS`` file entries flagged ``preview: True``
    as dicts with ``pattern``, ``title`` and ``description`` keys. The UI
    Output panel renders a top-N table preview for each after a successful
    run; tools without flagged entries (e.g. image downloads) get none.
    """
    spec = TOOL_GUIDE_SPECS.get(tool_name) or {}
    views = []
    for file_spec in spec.get("files", []):
        if not file_spec.get("preview"):
            continue
        views.append({
            "pattern": file_spec["pattern"],
            "title": file_spec.get("preview_title")
                     or Path(file_spec["pattern"]).name,
            "description": file_spec.get("description", ""),
        })
    return views


# =============================================================================
# Content assembly
# =============================================================================

def _match_files(run_folder: Path, pattern: str) -> list:
    """Return sorted relative POSIX paths in the run folder matching pattern."""
    matches = []
    for path in run_folder.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(run_folder).as_posix()
        if fnmatch.fnmatch(rel, pattern):
            matches.append(rel)
    return sorted(matches)


def _describe_unmatched(path_rel: str) -> str:
    """Generic fallback description for files no spec entry covered."""
    suffix = Path(path_rel).suffix.lower()
    if suffix == ".csv":
        return "Tabular data file."
    if suffix in (".xlsx", ".xls"):
        return "Excel workbook."
    if suffix == ".json":
        return "JSON data file."
    if suffix == ".html":
        return "HTML visualization."
    if suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        return "Image file."
    if suffix in (".gif", ".mp4", ".avi", ".mov"):
        return "Video/animation file."
    if suffix in (".pdf", ".pptx"):
        return "Summary document."
    if suffix in (".txt", ".md", ".log"):
        return "Text file."
    return "Output file."


def _ordered_unique(values) -> list:
    """Return values in first-seen order without duplicates."""
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _resolve_entry_columns(run_folder: Path, file_spec: dict,
                           matched: list) -> list:
    """Expand dynamic column families against matched CSV headers.

    The static spec uses ``threshold_`` as a family marker because the actual
    comparison export names the column per dataset.  Replace that marker with
    the real headers when a matched CSV is available; an unmatched/empty
    preview retains the marker and still gets its prefix-aware glossary entry.
    """
    columns = list(file_spec.get("columns", []))
    if "threshold_" not in columns:
        return columns

    dynamic = []
    for relative in matched:
        if Path(relative).suffix.lower() != ".csv":
            continue
        try:
            with (run_folder / relative).open(
                    "r", encoding="utf-8-sig", newline="") as handle:
                header = next(csv.reader(handle), [])
        except (OSError, UnicodeDecodeError, csv.Error):
            continue
        dynamic.extend(
            column for column in header
            if str(column).startswith("threshold_")
        )
    dynamic = _ordered_unique(dynamic)
    if not dynamic:
        return columns

    resolved = []
    for column in columns:
        if column == "threshold_":
            resolved.extend(dynamic)
        else:
            resolved.append(column)
    return resolved


def _metric_anchor(name) -> str:
    """HTML-safe anchor fragment for a metric/column name."""
    return re.sub(r"[^A-Za-z0-9_-]+", "-", str(name)).strip("-")


def _tmvev_glossary() -> list:
    """TM VEV terms explained one by one (user 2026-09-18): the canonical
    definitions from the run report's glossary, so the exported UserGuide
    and the report can never drift apart."""
    try:
        from comparison.mapping_validation_report import TERM_DEFS
        return [{"term": k, "description": TERM_DEFS[k]}
                for k in sorted(TERM_DEFS)]
    except Exception:
        return []


def assemble_run_content(run_folder: Path, tool_name: str,
                         params: Optional[dict]) -> dict:
    """Build the format-independent content model for one run folder."""
    spec = TOOL_GUIDE_SPECS.get(tool_name, {
        "title": tool_name.replace("_", " ").title(),
        "summary": "",
        "files": [],
    })
    params = params or {}

    # Match spec entries against the actual files present in the folder.
    # Each real file is claimed by the FIRST matching entry so overlapping
    # patterns (e.g. *_neurons.csv vs all_neurons.csv) never list it twice.
    entries = []
    matched_all = set()
    for file_spec in spec["files"]:
        matched = [
            rel for rel in _match_files(run_folder, file_spec["pattern"])
            if rel not in matched_all
        ]
        matched_all.update(matched)
        entries.append({
            "pattern": file_spec["pattern"],
            "description": file_spec["description"],
            "columns": _resolve_entry_columns(run_folder, file_spec, matched),
            "matrix": file_spec.get("matrix"),
            "matched": matched,
        })

    # Everything the spec did not cover (including unexpected outputs).
    leftovers = []
    for path in sorted(run_folder.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(run_folder).as_posix()
        if rel.startswith(GUIDE_BASENAME) or rel in matched_all:
            continue
        leftovers.append({"path": rel,
                          "description": _describe_unmatched(rel)})

    # Metrics & parameters: the ordered union of every column/metric the
    # matched files carry, with its description.  The renderers present these
    # once in a dedicated reference panel; the Output Files section only lists
    # the metric names and links back here, instead of repeating each
    # description inline.
    metrics = []
    for column in _ordered_unique(
            c for e in entries if e["matched"] for c in e["columns"]):
        description, value_range = glossary_entry(column)
        metrics.append({"name": column, "anchor": _metric_anchor(column),
                        "description": description, "range": value_range})

    warnings = _read_warnings(run_folder)

    explanation = list(spec.get("explanation") or [])
    output_detail = _output_detail_section(tool_name, params)
    if output_detail:
        explanation.append(output_detail)

    return {
        "tool_name": tool_name,
        "title": spec["title"],
        "summary": spec["summary"],
        "folder": run_folder.name,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "params": params,
        "explanation": explanation,
        "applied": _read_applied_state(run_folder),
        "applied_by_dataset": _read_applied_thresholds_by_dataset(run_folder),
        "threshold_queries": _read_threshold_query_manifest(run_folder),
        "entries": entries,
        "metrics": metrics,
        "leftovers": leftovers,
        "warnings": warnings,
        # TM VEV term glossary (type-mapping validation only; empty for
        # the other tools).
        "glossary": (_tmvev_glossary()
                     if tool_name == "type_mapping_validation" else []),
    }


def _output_detail_section(tool_name: str, params: Optional[dict]) -> Optional[dict]:
    """Tool-aware "Output detail" explanation section for the NeuronBridge
    tools (plan `_plan/plan-nb-find-lines-output-modes.md` D5b).

    Full runs learn which files are source-data-only and that the app's
    Output detail control (Compact) drops them; Compact runs learn what was
    pruned and how to regenerate it.  Non-NB tools return None.
    """
    source_data = {
        "nb_find_lines": (
            "the per-query match tables (`{query}_lines.csv`) and the "
            "downloaded images (`images/`, once the PDF/PPTX contact sheet "
            "is generated)"
        ),
        "nb_find_lines_expanded": (
            "the per-chip match tables (`*_lines.csv`) and the downloaded "
            "images (`images/`, once the PDF/PPTX contact sheet is "
            "generated)"
        ),
        "nb_find_neuron": (
            "`all_neurons.csv`, the per-line `{line}_neurons.csv` tables, "
            "and `by_dataset/*_neurons.csv` (type summaries, the "
            "type-mapped table, and 3D renders always stay)"
        ),
        "nb_colabel": (
            "`line_labeled_neurons/` and `distribution_data_by_neuron.csv` "
            "(matrices, expression data, by-type distributions, and the "
            "HTML report always stay)"
        ),
    }.get(tool_name)
    if not source_data:
        return None

    compact = params.get("keep_per_match_csv") is False
    if compact:
        paragraphs = [
            "This run used COMPACT output detail: the source-data-only "
            f"files ({source_data}) were removed after the summaries/report "
            "were written. Every removal is listed in the run's "
            "`cleanup_audit.json`.",
            "To regenerate them, re-run the same query — the NeuronBridge "
            "match cache is off by default, so this refetches from the "
            "NeuronBridge API. Enable Settings → NeuronBridge Match Cache "
            "to keep match tables across runs.",
        ]
    else:
        paragraphs = [
            "This run used FULL output detail: every file is kept, "
            f"including the source-data-only files ({source_data}).",
            "These files are intermediate inputs for the summaries and "
            "reports above — if you do not need them, switch the **Output "
            "detail** control to **Compact** on the tool tab and Compact "
            "will remove them automatically after summarization (audited "
            "in `cleanup_audit.json`).",
        ]
    return {"heading": "Output detail", "paragraphs": paragraphs}


def _read_warnings(run_folder: Path) -> Optional[str]:
    """Return the run's user_warning_notes.txt content, or None."""
    note_path = run_folder / WARNING_FILENAME
    if note_path.exists():
        try:
            text = note_path.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                return text.strip()
        except OSError:
            pass
    return None


# Applied-threshold / bottleneck provenance keys the pathfinding backend
# writes (all_attributes.json + parameters.txt). The run guide surfaces
# them so the reader sees the ACTUAL pruning level without opening the
# raw files.
_PROVENANCE_KEYS = (
    "requested_threshold",
    "applied_threshold",
    "applied_threshold_source",
    "strongest_first_budget",
    "strongest_first_budget_bitten",
    "strongest_first_tau",
    "tau_canonical",
    "strongest_dropped_bottleneck",
    "edge_budget",
    "edge_budget_applied",
    "edge_budget_landing",
    "edge_weight_floor",
    "strongest_retained_bottleneck",
    "paths_complete",
)

# parameters.txt alias lines (pre-provenance readers) -> canonical keys
_PARAMETERS_ALIASES = {
    "applied_tau (min path bottleneck)": "strongest_first_tau",
    "applied_tau": "strongest_first_tau",
}


def _coerce_provenance_value(text: str):
    """Coerce a parameters.txt value string to int/float/bool when possible."""
    s = text.strip()
    if s.lower() in ("not applied", "not reached", "n/a"):
        return None
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def _read_applied_state(run_folder: Path) -> Optional[dict]:
    """Read this run's applied-threshold provenance.

    Primary source: ``all_attributes.json`` (machine-readable, written by
    the post-enumeration re-stamp). Fallback: the ``key: value`` lines of
    ``parameters.txt`` (including the legacy ``applied_tau`` alias) so
    guides regenerated for older post-fix runs still show the values.
    Returns None for runs without provenance (other tools, pre-fix runs).
    """
    attrs_path = run_folder / "all_attributes.json"
    if attrs_path.exists():
        try:
            attrs = json.loads(attrs_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            attrs = None
        if isinstance(attrs, dict) and "applied_threshold" in attrs:
            return {key: attrs[key] for key in _PROVENANCE_KEYS
                    if key in attrs}

    params_path = run_folder / "parameters.txt"
    if params_path.exists():
        try:
            state = {}
            for line in params_path.read_text(
                    encoding="utf-8", errors="replace").splitlines():
                if ":" not in line:
                    continue
                key, _, value = line.partition(":")
                key = key.strip()
                value = _coerce_provenance_value(value)
                if key in _PROVENANCE_KEYS:
                    state[key] = value
                elif key in _PARAMETERS_ALIASES:
                    state.setdefault(_PARAMETERS_ALIASES[key], value)
            if "applied_threshold" in state:
                return state
        except OSError:
            pass
    return None


def _read_applied_thresholds_by_dataset(run_folder: Path) -> Optional[dict]:
    """Read per-dataset threshold provenance for comparison run guides.

    ``effective_thresholds.json`` is the primary UI notice.  The CSV fallback
    keeps regenerated guides informative when the notice was removed or when
    a comparison was exported by an older build that only wrote the durable
    provenance table.
    """
    banner_path = run_folder / "effective_thresholds.json"
    if banner_path.exists():
        try:
            payload = json.loads(banner_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = None
        datasets = payload.get("datasets") if isinstance(payload, dict) else None
        if datasets:
            return datasets

    provenance_path = (run_folder / "comparison_results"
                       / "pathfinding_provenance.csv")
    if not provenance_path.exists():
        return None
    try:
        grouped = {}
        with provenance_path.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                dataset = raw.get("dataset")
                if not dataset:
                    continue
                row = {
                    key: _coerce_provenance_value(value)
                    for key, value in raw.items()
                    if key and key != "dataset"
                }
                threshold = row.get("threshold")
                info = grouped.setdefault(dataset, {
                    "input": [], "effective": [], "skipped": [],
                    "applied_folder": {}, "runs": [],
                })
                if threshold is not None:
                    info["input"].append(threshold)
                applied = row.get("applied_threshold")
                if applied is not None:
                    info["effective"].append(applied)
                if row.get("skipped"):
                    info["skipped"].append(threshold)
                    if row.get("applied_folder") is not None:
                        info["applied_folder"][str(threshold)] = \
                            row["applied_folder"]
                tau = row.get("tau", row.get("strongest_first_tau"))
                if tau is not None:
                    info["tau"] = max(info.get("tau", tau), tau)
                info["runs"].append(row)
        for info in grouped.values():
            info["input"] = sorted(set(info["input"]))
            info["effective"] = sorted(set(info["effective"]))
        return grouped or None
    except (OSError, csv.Error, TypeError, ValueError):
        return None


def _read_threshold_query_manifest(run_folder: Path) -> Optional[dict]:
    """Read the row-wise threshold-query identity for a comparison run.

    The CSV is the durable source because it contains one dataset-specific
    provenance row per query.  The JSON notice is the fallback for runs where
    only the UI notice was retained.
    """
    manifest_path = (run_folder / "comparison_results"
                     / "threshold_combinations.csv")
    grouped = {}
    dataset_order = []
    mode = None
    if manifest_path.exists():
        try:
            with manifest_path.open(newline="", encoding="utf-8") as handle:
                for raw in csv.DictReader(handle):
                    query_id = raw.get("query_id")
                    dataset = raw.get("dataset")
                    if not query_id or not dataset:
                        continue
                    mode = raw.get("threshold_mode") or mode
                    if dataset not in dataset_order:
                        dataset_order.append(dataset)
                    query = grouped.setdefault(query_id, {
                        "id": query_id,
                        "label": raw.get("query_label") or query_id,
                        "thresholds": {},
                        "runs": [],
                    })
                    requested = _coerce_provenance_value(
                        raw.get("requested_threshold", ""))
                    query["thresholds"][dataset] = requested
                    query["runs"].append({
                        key: _coerce_provenance_value(value)
                        for key, value in raw.items()
                        if key not in ("query_id", "query_label")
                    })
            if grouped:
                return {
                    "mode": mode or "standard",
                    "dataset_order": dataset_order,
                    "queries": list(grouped.values()),
                }
        except (OSError, csv.Error, TypeError, ValueError):
            pass

    banner_path = run_folder / "effective_thresholds.json"
    if not banner_path.exists():
        return None
    try:
        payload = json.loads(banner_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    raw_queries = payload.get("queries") or payload.get("combinations")
    if not raw_queries:
        return None
    queries = []
    for raw_query in raw_queries:
        if not isinstance(raw_query, dict):
            continue
        query_id = raw_query.get("id") or raw_query.get("query_id")
        if not query_id:
            continue
        thresholds = (raw_query.get("requested_thresholds")
                      or raw_query.get("thresholds")
                      or raw_query.get("thresholds_by_dataset")
                      or {})
        queries.append({
            "id": str(query_id),
            "label": raw_query.get("label") or str(query_id),
            "thresholds": thresholds,
            "runs": raw_query.get("runs") or [],
        })
        for dataset in thresholds:
            if dataset not in dataset_order:
                dataset_order.append(dataset)
    if not queries:
        return None
    return {
        "mode": payload.get("threshold_mode", "standard"),
        "dataset_order": (payload.get("threshold_dataset_order")
                          or dataset_order),
        "queries": queries,
    }


def _format_applied_value(value) -> str:
    """Compact display for a provenance value (— when it did not apply)."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _this_run_value(applied: Optional[dict], key: str) -> Optional[object]:
    """Map a vocabulary-table key onto this run's provenance value."""
    if applied is None:
        return None
    if key == "pruned":
        # the run-state name for the vocabulary row 'pruned'
        return applied.get("edge_budget_applied")
    return applied.get(key)


# Applied-threshold block: the value rows shown under Run parameters,
# ordered so the pruning story reads top-down (what was asked -> what the
# budgets did -> what to believe).
_APPLIED_ROWS = (
    ("requested threshold (Min Synapse Count as entered)",
     "requested_threshold"),
    ("applied threshold (equivalent Min Synapse Count)", "applied_threshold"),
    ("source", "applied_threshold_source"),
    ("τ (StrongestFirst landing / natural bottleneck)",
     "strongest_first_tau"),
    ("tau_canonical (minimal equivalent threshold)", "tau_canonical"),
    ("w2 (strongest dropped path bottleneck)",
     "strongest_dropped_bottleneck"),
    ("w0 (Edge Budget floor)", "edge_weight_floor"),
    ("w1 (Edge Budget landing tier)", "edge_budget_landing"),
    ("W* (strongest retained path bottleneck)",
     "strongest_retained_bottleneck"),
    ("paths_complete", "paths_complete"),
)


def _applied_headline(applied: dict) -> str:
    """One-sentence reading of this run's pruning level."""
    source = applied.get("applied_threshold_source")
    applied_threshold = applied.get("applied_threshold")
    if source in (None, "requested"):
        tau = applied.get("strongest_first_tau")
        head = ("No budget bit: the output is the complete path set at "
                f"Min Synapse Count = "
                f"{_format_applied_value(applied.get('requested_threshold'))}.")
        if tau is not None:
            head += (f" The natural τ (weakest emitted-path bottleneck) "
                     f"is {_format_applied_value(tau)} — every Min Synapse "
                     "Count up to it yields this identical set.")
        return head
    return (f"Applied threshold = {_format_applied_value(applied_threshold)} "
            f"(source: {source}) — the materialized output is EXACTLY a "
            "complete run at Min Synapse Count = "
            f"{_format_applied_value(applied_threshold)}. Raise Max Paths "
            "(BodyId) / the Edge Budget, or lower Min Synapse Count, to "
            "enumerate weaker paths.")


def _format_dataset_threshold_banner(by_dataset: dict) -> list:
    """Per-dataset asked -> applied lines for comparison runs."""
    lines = []
    for ds, info in sorted(by_dataset.items()):
        if not isinstance(info, dict):
            continue
        asked = ", ".join(str(t) for t in info.get("input") or [])
        effective = ", ".join(str(t) for t in info.get("effective") or [])
        collapsed = info.get("applied_folder") or {}
        collapse_txt = ""
        if collapsed:
            collapse_txt = " (collapsed: " + ", ".join(
                f"{t}->{applied}" for t, applied in sorted(
                    collapsed.items(), key=lambda kv: str(kv[0]))) + ")"
        tau = info.get("tau")
        tau_txt = f", τ {_format_applied_value(tau)}" \
            if tau is not None else ""
        lines.append(f"  {ds}: asked [{asked}] -> applied [{effective}]"
                     f"{collapse_txt}{tau_txt}")
        for row in info.get("runs") or []:
            if not isinstance(row, dict):
                continue
            lines.append(
                "    threshold "
                f"{_format_applied_value(row.get('threshold'))}: "
                f"requested {_format_applied_value(row.get('requested_threshold'))} "
                f"-> applied {_format_applied_value(row.get('applied_threshold'))} "
                f"(source {_format_applied_value(row.get('applied_threshold_source'))}; "
                f"SF budget {_format_applied_value(row.get('strongest_first_budget'))}, "
                f"SF bite {_format_applied_value(row.get('strongest_first_budget_bitten'))}; "
                f"tau {_format_applied_value(row.get('tau', row.get('strongest_first_tau')))}; "
                f"edge budget {_format_applied_value(row.get('edge_budget'))} "
                f"(applied {_format_applied_value(row.get('edge_budget_applied'))}); "
                f"w0 {_format_applied_value(row.get('edge_weight_floor'))}; "
                f"w1 {_format_applied_value(row.get('edge_budget_landing'))}; "
                f"w2 {_format_applied_value(row.get('strongest_dropped_bottleneck'))}; "
                f"W* {_format_applied_value(row.get('strongest_retained_bottleneck'))}; "
                f"paths_complete {_format_applied_value(row.get('paths_complete'))})")
    return lines


def _format_threshold_query_banner(manifest: Optional[dict]) -> list:
    """Format the comparison query rows for text/Markdown renderers."""
    if not manifest:
        return []
    lines = [
        f"  Threshold mode: {manifest.get('mode', 'standard')}",
        "  Query rows (requested threshold by dataset):",
    ]
    dataset_order = manifest.get("dataset_order") or []
    for query in manifest.get("queries") or []:
        if not isinstance(query, dict):
            continue
        query_id = query.get("id") or query.get("query_id")
        label = query.get("label") or query_id
        thresholds = query.get("thresholds") or {}
        requested = ", ".join(
            f"{dataset}={thresholds.get(dataset, '—')}"
            for dataset in dataset_order
        ) or ", ".join(f"{dataset}={value}"
                       for dataset, value in thresholds.items())
        lines.append(f"    {query_id} ({label}): {requested}")
        for run in query.get("runs") or []:
            if not isinstance(run, dict):
                continue
            dataset = run.get("dataset")
            if not dataset:
                continue
            tau = run.get("strongest_first_tau", run.get("tau"))
            lines.append(
                f"      {dataset}: requested "
                f"{_format_applied_value(run.get('requested_threshold'))}"
                f" -> applied "
                f"{_format_applied_value(run.get('applied_threshold'))}"
                f" (source "
                f"{_format_applied_value(run.get('applied_threshold_source'))}; "
                f"SF budget {_format_applied_value(run.get('strongest_first_budget'))}, "
                f"SF bite {_format_applied_value(run.get('strongest_first_budget_bitten'))}; "
                f"tau {_format_applied_value(tau)}; w0 "
                f"{_format_applied_value(run.get('edge_weight_floor'))}; "
                f"w1 {_format_applied_value(run.get('edge_budget_landing'))}; "
                f"edge budget {_format_applied_value(run.get('edge_budget'))} "
                f"(applied {_format_applied_value(run.get('edge_budget_applied'))}); "
                f"w2 {_format_applied_value(run.get('strongest_dropped_bottleneck'))}; "
                f"W* {_format_applied_value(run.get('strongest_retained_bottleneck'))}; "
                f"paths_complete {_format_applied_value(run.get('paths_complete'))})")
    return lines


def _key_params(params: dict) -> list:
    """Pick the most informative run parameters for the summary table."""
    priority = (
        "dataset", "datasets", "source_dataset", "target_dataset",
        "sourceNeurons", "targetNeurons", "source_neurons", "target_neurons",
        "source", "query", "line_names", "lines", "line_name",
        "threshold_mode", "threshold_dataset_order", "threshold_combinations",
        "weight_basis", "min_synapse_num", "min_synapse_threshold", "min_ratio",
        "min_traversal_probability", "max_interlayer", "thresholds",
        "graph_edge_limit_bodyid", "max_paths_bodyid", "drop_untyped",
        "edgeN_limit",
        "output_format", "skip_bodyId", "similarity_metric", "top_n",
        "top_k", "top_m", "match_type",
    )
    rows = []
    for key in priority:
        if key in params and params[key] not in (None, "", []):
            rows.append((key, params[key]))
    return rows[:12]


# =============================================================================
# Renderers
# =============================================================================

def _render_explanation_txt(sections, applied=None) -> list:
    """Plain-text rendering of the pathfinding explanation block."""
    lines = []
    for section in sections:
        lines.append(section.get("heading", "") + ":")
        for para in section.get("paragraphs") or []:
            lines.append(f"  {_math_to_txt(para)}")
        for step in section.get("pipeline") or []:
            lines.append(f"  -> {step}")
        table = section.get("table")
        if table:
            rows = [list(row) for row in table]
            if section.get("values_column") and applied:
                rows[0].append("This run")
                for row in rows[1:]:
                    row.append(_format_applied_value(
                        _this_run_value(applied, str(row[1]))))
            width = max(
                sum(len(str(cell)) for cell in row) + 3 * (len(row) - 1)
                for row in rows)
            for idx, row in enumerate(rows):
                lines.append("  " + " | ".join(str(cell) for cell in row))
                if idx == 0:
                    lines.append("  " + "-" * width)
        lines.append("")
    return lines


def _render_applied_txt(applied, by_dataset, threshold_queries=None) -> list:
    """Plain-text rendering of the applied-threshold block."""
    lines = []
    combinations = bool(
        threshold_queries and threshold_queries.get("mode") == "combinations")
    if applied:
        lines.append(_applied_headline(applied))
        lines.append("")
        for label, key in _APPLIED_ROWS:
            lines.append(
                f"  {label}: {_format_applied_value(applied.get(key))}")
    if by_dataset and not combinations:
        lines.append("")
        lines.append("  Per-dataset asked -> applied thresholds:")
        lines.extend(_format_dataset_threshold_banner(by_dataset))
    if combinations:
        lines.append("")
        lines.append("  Cross-dataset threshold query rows:")
        lines.extend(_format_threshold_query_banner(threshold_queries))
    return lines


def render_txt(content: dict) -> str:
    lines = []
    bar = "=" * 72
    lines.append(bar)
    lines.append(f"DROCAT RUN GUIDE — {content['title']}")
    lines.append(bar)
    lines.append("")
    if content["summary"]:
        lines.append(content["summary"])
        lines.append("")
    lines.append(f"Run folder : {content['folder']}")
    lines.append(f"Generated  : {content['generated']}")
    lines.append(f"Storage    : this folder appears in Settings -> Storage, "
                 f"where it can be pruned (source data) or deleted.")
    lines.append("")

    key_params = _key_params(content["params"])
    if key_params:
        lines.append("RUN PARAMETERS")
        lines.append("-" * 72)
        for key, value in key_params:
            lines.append(f"  {key}: {value}")
        lines.append("")

    if (content.get("applied") or content.get("applied_by_dataset")
            or content.get("threshold_queries")):
        lines.append("APPLIED THRESHOLD (THIS RUN)")
        lines.append("-" * 72)
        lines.extend(_render_applied_txt(
            content.get("applied"), content.get("applied_by_dataset"),
            content.get("threshold_queries")))
        lines.append("")

    if content.get("explanation"):
        lines.append("PATHFINDING MODEL")
        lines.append("-" * 72)
        lines.extend(_render_explanation_txt(
            content["explanation"], applied=content.get("applied")))
        lines.append("")

    lines.append("WARNINGS & NOTES")
    lines.append("-" * 72)
    lines.append(content["warnings"] or NO_WARNINGS_TEXT)
    lines.append("")

    lines.append("METRICS & PARAMETERS")
    lines.append("-" * 72)
    if content["metrics"]:
        for metric in content["metrics"]:
            suffix = f" [{metric['range']}]" if metric["range"] else ""
            lines.append(f"  {metric['name']}{suffix}: "
                         f"{_math_to_txt(metric['description'])}")
    else:
        lines.append("  (none)")
    lines.append("")

    if content.get("glossary"):
        lines.append("GLOSSARY - TM VEV TERMS")
        lines.append("-" * 72)
        for g in content["glossary"]:
            lines.append(f"  {g['term']}: "
                         f"{_math_to_txt(g['description'])}")
        lines.append("")

    lines.append("OUTPUT FILES")
    lines.append("-" * 72)
    for entry in content["entries"]:
        if not entry["matched"]:
            continue
        names = ", ".join(entry["matched"][:4])
        if len(entry["matched"]) > 4:
            names += f", ... ({len(entry['matched'])} files)"
        lines.append(f"* {names}")
        lines.append(f"    {entry['description']}")
        if entry["matrix"]:
            lines.append(f"    Layout: {entry['matrix']}")
        if entry["columns"]:
            lines.append(f"    Metrics: {', '.join(entry['columns'])} — "
                         "definitions in Metrics & Parameters above.")
        lines.append("")

    if content["leftovers"]:
        lines.append("OTHER FILES IN THIS RUN")
        lines.append("-" * 72)
        for item in content["leftovers"]:
            lines.append(f"* {item['path']}")
            lines.append(f"    {item['description']}")
        lines.append("")

    lines.append(bar)
    lines.append("Full output reference: docs/OUTPUT_FILES.md in the "
                 "DROCAT installation.")
    lines.append(bar)
    return "\n".join(lines) + "\n"


def _render_explanation_markdown(sections, applied=None) -> list:
    """Markdown rendering of the pathfinding explanation block."""
    md = []
    for section in sections:
        md.append(f"### {section.get('heading', '')}")
        md.append("")
        for para in section.get("paragraphs") or []:
            md.append(_math_to_md(para))
            md.append("")
        pipeline = section.get("pipeline")
        if pipeline:
            md.append("```")
            md.extend(pipeline)
            md.append("```")
            md.append("")
        table = section.get("table")
        if table:
            rows = [list(row) for row in table]
            if section.get("values_column") and applied:
                rows[0].append("This run")
                for row in rows[1:]:
                    row.append(_format_applied_value(
                        _this_run_value(applied, str(row[1]))))
            header = rows[0]
            md.append("| " + " | ".join(str(c) for c in header) + " |")
            md.append("|" + "|".join([" --- "] * len(header)) + "|")
            for row in rows[1:]:
                cells = [
                    str(c).replace("|", "\\|") for c in row]
                md.append("| " + " | ".join(cells) + " |")
            md.append("")
    return md


def _render_applied_markdown(applied, by_dataset, threshold_queries=None) -> list:
    """Markdown rendering of the applied-threshold block."""
    md = []
    combinations = bool(
        threshold_queries and threshold_queries.get("mode") == "combinations")
    if applied:
        md.append(_applied_headline(applied))
        md.append("")
        md.append("| Parameter | Value |")
        md.append("| --- | --- |")
        for label, key in _APPLIED_ROWS:
            value = _format_applied_value(applied.get(key)).replace(
                "|", "\\|")
            md.append(f"| {label} | {value} |")
        md.append("")
    if by_dataset and not combinations:
        md.append("Per-dataset asked -> applied thresholds:")
        md.append("")
        md.extend(f"    {line.strip()}"
                  for line in _format_dataset_threshold_banner(by_dataset))
        md.append("")
    if combinations:
        md.append("Cross-dataset threshold query rows:")
        md.append("")
        md.extend(f"    {line.strip()}"
                  for line in _format_threshold_query_banner(threshold_queries))
        md.append("")
    return md


def render_markdown(content: dict) -> str:
    md = []
    md.append(f"# DROCAT Run Guide — {content['title']}")
    md.append("")
    if content["summary"]:
        md.append(content["summary"])
        md.append("")
    md.append(f"- **Run folder:** `{content['folder']}`")
    md.append(f"- **Generated:** {content['generated']}")
    md.append("- **Storage:** this folder appears in the app's Settings → "
              "Storage card, where it can be pruned (source data) or "
              "deleted.")
    md.append("")

    key_params = _key_params(content["params"])
    if key_params:
        md.append("## Run parameters")
        md.append("")
        md.append("| Parameter | Value |")
        md.append("| --- | --- |")
        for key, value in key_params:
            md.append(f"| `{key}` | `{value}` |")
        md.append("")

    if (content.get("applied") or content.get("applied_by_dataset")
            or content.get("threshold_queries")):
        md.append("## Applied threshold (this run)")
        md.append("")
        md.extend(_render_applied_markdown(
            content.get("applied"), content.get("applied_by_dataset"),
            content.get("threshold_queries")))

    if content.get("explanation"):
        md.append("## Pathfinding model")
        md.append("")
        md.extend(_render_explanation_markdown(
            content["explanation"], applied=content.get("applied")))

    md.append("## Warnings & notes")
    md.append("")
    if content["warnings"]:
        md.append("```")
        md.append(content["warnings"])
        md.append("```")
    else:
        md.append(NO_WARNINGS_TEXT)
    md.append("")

    md.append("## Metrics & parameters")
    md.append("")
    if content["metrics"]:
        md.append("| Metric | Description | Range |")
        md.append("| --- | --- | --- |")
        for metric in content["metrics"]:
            cell = _math_to_md(metric["description"]).replace("|", "\\|")
            md.append(f"| `{metric['name']}` | {cell} | {metric['range']} |")
    else:
        md.append("*No metrics or parameters are documented for this run.*")
    md.append("")

    if content.get("glossary"):
        md.append("## Glossary — TM VEV terms")
        md.append("")
        for g in content["glossary"]:
            md.append(f"- **{g['term']}** — {g['description']}")
        md.append("")

    md.append("## Output files")
    md.append("")
    for entry in content["entries"]:
        if not entry["matched"]:
            continue
        names = ", ".join(f"`{m}`" for m in entry["matched"][:4])
        if len(entry["matched"]) > 4:
            names += f", … ({len(entry['matched'])} files)"
        md.append(f"### {names}")
        md.append("")
        md.append(entry["description"])
        if entry["matrix"]:
            md.append("")
            md.append(f"*Layout: {entry['matrix']}*")
        if entry["columns"]:
            md.append("")
            cols = ", ".join(f"`{c}`" for c in entry["columns"])
            md.append(f"*Metrics: {cols} — definitions in the Metrics & "
                      "parameters section above.*")
        md.append("")

    if content["leftovers"]:
        md.append("## Other files in this run")
        md.append("")
        for item in content["leftovers"]:
            md.append(f"- `{item['path']}` — {item['description']}")
        md.append("")

    md.append("---")
    md.append("")
    md.append("Full output reference: `docs/OUTPUT_FILES.md` in the DROCAT "
              "installation.")
    return "\n".join(md) + "\n"


_HTML_STYLE = """
body { font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial,
       sans-serif; margin: 0; background: #f4f6fb; color: #1f2733; }
main { max-width: 960px; margin: 24px auto; padding: 0 20px 48px; }
.head { background: #12305e; color: #fff; border-radius: 12px;
        padding: 20px 24px; margin-bottom: 20px; }
.head h1 { margin: 0 0 6px; font-size: 1.5em; }
.head .meta { color: #b9c6dd; font-size: 0.9em; }
h2 { font-size: 1.15em; margin: 28px 0 10px; color: #12305e; }
h3 { font-size: 1em; margin: 12px 0 6px; color: #12305e; }
.pipeline { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas,
            monospace; font-size: 0.85em; margin: 6px 0; white-space: pre;
            color: #33415c; }
.card { background: #fff; border: 1px solid #dde4ef; border-radius: 10px;
        padding: 14px 18px; margin-bottom: 12px; }
.warn { background: #fff8e6; border: 1px solid #ecd9a0; border-radius: 10px;
        padding: 14px 18px; white-space: pre-wrap; font-size: 0.92em; }
.ok { background: #eefaf0; border: 1px solid #bfe5c8; border-radius: 10px;
      padding: 12px 18px; color: #1f6b34; }
table { border-collapse: collapse; width: 100%; margin-top: 8px;
        font-size: 0.9em; }
th, td { border: 1px solid #dde4ef; padding: 5px 9px; text-align: left;
         vertical-align: top; }
th { background: #f0f4fb; }
code { background: #eef1f7; border-radius: 4px; padding: 1px 5px;
       font-size: 0.9em; }
.fname { font-weight: 600; color: #12305e; }
.desc { margin: 6px 0; }
.small { color: #5b6b84; font-size: 0.85em; }
.math { font-family: 'Cambria Math', 'STIX Two Math', 'Latin Modern Math',
        Georgia, 'Times New Roman', serif; white-space: nowrap; }
.math sub, .math sup { font-size: 0.7em; line-height: 0; }
"""


def _html_escape(text) -> str:
    return html.escape(str(text))


def render_html(content: dict) -> str:
    parts = []
    parts.append("<!doctype html>")
    parts.append('<html lang="en">')
    parts.append("<head>")
    parts.append('<meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width, '
                 'initial-scale=1">')
    parts.append(f"<title>DROCAT Run Guide — "
                 f"{_html_escape(content['title'])}</title>")
    parts.append(f"<style>{_HTML_STYLE}</style>")
    parts.append("</head>")
    parts.append("<body>")
    parts.append("<main>")

    parts.append('<div class="head">')
    parts.append(f"<h1>DROCAT Run Guide — {_html_escape(content['title'])}"
                 "</h1>")
    if content["summary"]:
        parts.append(f"<p>{_html_escape(content['summary'])}</p>")
    parts.append(f'<div class="meta">Run folder: '
                 f"<code>{_html_escape(content['folder'])}</code> · "
                 f"Generated: {_html_escape(content['generated'])} · "
                 f"Manage this folder in Settings → Storage</div>")
    parts.append("</div>")

    key_params = _key_params(content["params"])
    if key_params:
        parts.append("<h2>Run parameters</h2>")
        parts.append('<div class="card">')
        parts.append("<table><tr><th>Parameter</th><th>Value</th></tr>")
        for key, value in key_params:
            parts.append(f"<tr><td><code>{_html_escape(key)}</code></td>"
                         f"<td><code>{_html_escape(value)}</code></td></tr>")
        parts.append("</table></div>")

    applied = content.get("applied")
    by_dataset = content.get("applied_by_dataset")
    threshold_queries = content.get("threshold_queries")
    combinations = bool(
        threshold_queries and threshold_queries.get("mode") == "combinations")
    if applied or by_dataset or combinations:
        parts.append("<h2>Applied threshold (this run)</h2>")
        parts.append('<div class="card">')
        if applied:
            parts.append(
                f"<p><strong>{_html_escape(_applied_headline(applied))}"
                "</strong></p>")
            parts.append("<table><tr><th>Parameter</th><th>Value</th></tr>")
            for label, key in _APPLIED_ROWS:
                parts.append(
                    f"<tr><td>{_html_escape(label)}</td><td>"
                    f"<code>{_html_escape(_format_applied_value(applied.get(key)))}"
                    "</code></td></tr>")
            parts.append("</table>")
        if by_dataset and not combinations:
            parts.append("<p class=\"small\">Per-dataset asked &rarr; "
                         "applied thresholds:</p>")
            has_run_rows = any(
                isinstance(info, dict) and info.get("runs")
                for info in by_dataset.values())
            if has_run_rows:
                parts.append(
                    "<table><tr><th>Dataset</th><th>Threshold</th>"
                    "<th>Requested</th><th>Applied</th><th>Source</th>"
                    "<th>SF budget</th><th>SF bite</th><th>tau</th>"
                    "<th>Edge budget</th><th>Edge floor applied</th>"
                    "<th>w0</th><th>w1</th><th>w2</th>"
                    "<th>W*</th><th>paths_complete</th></tr>")
            else:
                parts.append("<table><tr><th>Dataset</th>"
                             "<th>Asked &rarr; applied</th></tr>")
            for ds, info in sorted(by_dataset.items()):
                if not isinstance(info, dict):
                    continue
                if has_run_rows and info.get("runs"):
                    for row in info.get("runs") or []:
                        if not isinstance(row, dict):
                            continue
                        cells = (
                            ds,
                            row.get("threshold"),
                            row.get("requested_threshold"),
                            row.get("applied_threshold"),
                            row.get("applied_threshold_source"),
                            row.get("strongest_first_budget"),
                            row.get("strongest_first_budget_bitten"),
                            row.get("tau", row.get("strongest_first_tau")),
                            row.get("edge_budget"),
                            row.get("edge_budget_applied"),
                            row.get("edge_weight_floor"),
                            row.get("edge_budget_landing"),
                            row.get("strongest_dropped_bottleneck"),
                            row.get("strongest_retained_bottleneck"),
                            row.get("paths_complete"),
                        )
                        parts.append(
                            "<tr>" + "".join(
                                f"<td><code>{_html_escape(_format_applied_value(cell))}"
                                "</code></td>" for cell in cells)
                            + "</tr>")
                    continue
                asked = ", ".join(str(t) for t in info.get("input") or [])
                effective = ", ".join(
                    str(t) for t in info.get("effective") or [])
                collapsed = info.get("applied_folder") or {}
                collapse_txt = ""
                if collapsed:
                    collapse_txt = " (collapsed: " + ", ".join(
                        f"{t}\u2192{applied_t}" for t, applied_t in sorted(
                            collapsed.items(), key=lambda kv: str(kv[0]))) \
                        + ")"
                tau = info.get("tau")
                tau_txt = f", τ {_format_applied_value(tau)}" \
                    if tau is not None else ""
                parts.append(
                    f"<tr><td><code>{_html_escape(ds)}</code></td><td>"
                    f"[{_html_escape(asked)}] &rarr; "
                    f"[{_html_escape(effective)}]{_html_escape(collapse_txt)}"
                    f"{_html_escape(tau_txt)}</td></tr>")
            parts.append("</table>")
        if combinations:
            parts.append("<p class=\"small\">Cross-dataset threshold query "
                         "rows (each row is one comparison identity; "
                         "applied values are per dataset):</p>")
            parts.append(
                "<table><tr><th>Query</th><th>Label</th>"
                "<th>Requested thresholds</th><th>Dataset</th>"
                "<th>Applied</th><th>Source</th><th>SF budget</th>"
                "<th>SF bite</th><th>tau</th><th>Edge budget</th>"
                "<th>Edge budget applied</th>"
                "<th>w0</th><th>w1</th><th>w2</th><th>W*</th>"
                "<th>paths_complete</th></tr>")
            dataset_order = threshold_queries.get("dataset_order") or []
            for query in threshold_queries.get("queries") or []:
                if not isinstance(query, dict):
                    continue
                query_id = query.get("id") or query.get("query_id")
                label = query.get("label") or query_id
                thresholds = query.get("thresholds") or {}
                requested = ", ".join(
                    f"{dataset}={thresholds.get(dataset, '—')}"
                    for dataset in dataset_order)
                if not requested:
                    requested = ", ".join(
                        f"{dataset}={value}"
                        for dataset, value in thresholds.items())
                runs = query.get("runs") or [{}]
                for run in runs:
                    if not isinstance(run, dict):
                        continue
                    tau = run.get("strongest_first_tau", run.get("tau"))
                    cells = (
                        query_id, label, requested, run.get("dataset"),
                        run.get("applied_threshold"),
                        run.get("applied_threshold_source"),
                        run.get("strongest_first_budget"),
                        run.get("strongest_first_budget_bitten"), tau,
                        run.get("edge_budget"),
                        run.get("edge_budget_applied"),
                        run.get("edge_weight_floor"),
                        run.get("edge_budget_landing"),
                        run.get("strongest_dropped_bottleneck"),
                        run.get("strongest_retained_bottleneck"),
                        run.get("paths_complete"),
                    )
                    parts.append(
                        "<tr>" + "".join(
                            f"<td><code>{_html_escape(_format_applied_value(cell))}"
                            "</code></td>" for cell in cells)
                        + "</tr>")
            parts.append("</table>")
        parts.append("</div>")

    if content.get("explanation"):
        applied_state = content.get("applied")
        parts.append("<h2>Pathfinding model</h2>")
        for section in content["explanation"]:
            parts.append('<div class="card">')
            parts.append(f"<h3>{_html_escape(section.get('heading', ''))}"
                         "</h3>")
            for para in section.get("paragraphs") or []:
                parts.append(f"<p>{_math_to_html(_html_escape(para))}</p>")
            pipeline = section.get("pipeline")
            if pipeline:
                parts.append('<div class="pipeline">'
                             + "<br>".join(
                                 "&rarr; " + _html_escape(step)
                                 for step in pipeline)
                             + "</div>")
            table = section.get("table")
            if table:
                rows = [list(row) for row in table]
                if section.get("values_column") and applied_state:
                    rows[0].append("This run")
                    for row in rows[1:]:
                        row.append(_format_applied_value(
                            _this_run_value(applied_state, str(row[1]))))
                parts.append("<table><tr>"
                             + "".join(f"<th>{_html_escape(c)}</th>"
                                       for c in rows[0])
                             + "</tr>")
                for row in rows[1:]:
                    parts.append("<tr>"
                                 + "".join(f"<td>{_html_escape(c)}</td>"
                                           for c in row)
                                 + "</tr>")
                parts.append("</table>")
            parts.append("</div>")

    parts.append("<h2>Warnings &amp; notes</h2>")
    if content["warnings"]:
        parts.append(f'<div class="warn">{_html_escape(content["warnings"])}'
                     "</div>")
    else:
        parts.append(f'<div class="ok">{_html_escape(NO_WARNINGS_TEXT)}'
                     "</div>")

    if content.get("glossary"):
        parts.append('<h2 id="glossary">Glossary — TM VEV terms</h2>')
        parts.append("<p class=\"small\">Every pipeline term, explained. "
                     "The report's dotted terms link to the same "
                     "definitions.</p>")
        parts.append("<table>")
        parts.append("<thead><tr><th>Term</th><th>Meaning</th></tr>"
                     "</thead><tbody>")
        for g in content["glossary"]:
            parts.append(f"<tr><td><code>"
                         f"{_html_escape(g['term'])}</code></td>"
                         f"<td>{_html_escape(g['description'])}</td></tr>")
        parts.append("</tbody></table>")

    parts.append('<h2 id="metrics">Metrics &amp; parameters</h2>')
    if content["metrics"]:
        parts.append('<div class="card"><table>'
                     '<tr><th>Metric</th><th>Description</th><th>Range</th></tr>')
        for metric in content["metrics"]:
            parts.append(
                f'<tr id="metric-{_html_escape(metric["anchor"])}">'
                f'<td><code>{_html_escape(metric["name"])}</code></td>'
                f'<td>{_math_to_html(_html_escape(metric["description"]))}</td>'
                f'<td>{_html_escape(metric["range"])}</td></tr>')
        parts.append("</table></div>")
    else:
        parts.append('<div class="card"><span class="small">No metrics or '
                     'parameters are documented for this run.</span></div>')

    parts.append("<h2>Output files</h2>")
    for entry in content["entries"]:
        if not entry["matched"]:
            continue
        parts.append('<div class="card">')
        names = []
        for rel in entry["matched"][:6]:
            if rel.endswith(".html") and "/" not in rel:
                names.append(f'<a href="{_html_escape(rel)}">'
                             f"<code>{_html_escape(rel)}</code></a>")
            else:
                names.append(f"<code>{_html_escape(rel)}</code>")
        shown = ", ".join(names)
        if len(entry["matched"]) > 6:
            shown += f", … ({len(entry['matched'])} files total)"
        parts.append(f'<div class="fname">{shown}</div>')
        parts.append(f'<div class="desc">{_html_escape(entry["description"])}'
                     "</div>")
        if entry["matrix"]:
            parts.append(f'<div class="small">Layout: '
                         f'{_html_escape(entry["matrix"])}</div>')
        if entry["columns"]:
            links = ", ".join(
                f'<a href="#metric-{_html_escape(_metric_anchor(c))}">'
                f"<code>{_html_escape(c)}</code></a>"
                for c in entry["columns"])
            parts.append(
                f'<div class="small">Metrics: {links} — '
                '<a href="#metrics">definitions in Metrics &amp; '
                'parameters</a>.</div>')
        parts.append("</div>")

    if content["leftovers"]:
        parts.append("<h2>Other files in this run</h2>")
        parts.append('<div class="card"><table>'
                     "<tr><th>File</th><th>Description</th></tr>")
        for item in content["leftovers"]:
            parts.append(f"<tr><td><code>{_html_escape(item['path'])}</code>"
                         f"</td><td>{_html_escape(item['description'])}</td>"
                         "</tr>")
        parts.append("</table></div>")

    parts.append('<p class="small">Full output reference: '
                 "<code>docs/OUTPUT_FILES.md</code> in the DROCAT "
                 "installation.</p>")
    parts.append("</main>")
    parts.append("</body>")
    parts.append("</html>")
    return "\n".join(parts) + "\n"


_RENDERERS = {
    "html": render_html,
    "txt": render_txt,
    "markdown": render_markdown,
}


# =============================================================================
# Public API
# =============================================================================

def resolve_guide_format(preferred: Optional[str] = None) -> str:
    """Resolve the effective guide format.

    Order: explicit argument > DROCAT_RUN_GUIDE_FORMAT env var > the saved
    user default (Settings) > built-in default 'html'. Invalid values fall
    back to the built-in default.
    """
    if preferred in GUIDE_FORMATS:
        return preferred
    env_value = os.environ.get(GUIDE_FORMAT_ENV, "").strip().lower()
    if env_value in GUIDE_FORMATS:
        return env_value
    try:
        from .config import get_user_default
        saved = str(get_user_default("run_guide_format")).strip().lower()
        if saved in GUIDE_FORMATS:
            return saved
    except Exception:
        pass
    return "html"


def write_run_guide(run_folder, tool_name: str,
                    params: Optional[dict] = None,
                    fmt: Optional[str] = None) -> Optional[Path]:
    """Write the exported run guide into *run_folder*.

    Returns the written path, or None when the guide is disabled or could
    not be written. Never raises: a failed guide write must not break a run.
    """
    try:
        resolved = resolve_guide_format(fmt)
        if resolved == "disabled":
            return None
        folder = Path(run_folder)
        if not folder.is_dir():
            return None
        content = assemble_run_content(folder, tool_name, params)
        text = _RENDERERS[resolved](content)
        guide_path = folder / (GUIDE_BASENAME + GUIDE_EXTENSIONS[resolved])
        guide_path.write_text(text, encoding="utf-8")
        return guide_path
    except Exception:
        return None

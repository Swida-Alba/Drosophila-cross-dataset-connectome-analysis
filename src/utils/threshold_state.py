"""
Canonical applied-threshold / bottleneck provenance for DROCAT
pathfinding runs.

Single source of truth for the applied-threshold contract, shared by
``FindNeuronConnection`` (all modes + replay folders), the exported run
guides, and Cross-Dataset Comparison's summary exports
(``ComparisonAnalyzer._applied_state_for``). Pure scalars in, dict out —
no heavy imports, so any subsystem can compute the same numbers.
"""

def applied_threshold_provenance(
    requested_threshold,
    strongest_first_tau=None,
    strongest_first_budget_bitten=False,
    strongest_dropped_bottleneck=None,
    tau_canonical=None,
    edge_weight_floor=None,
    edge_budget_landing=None,
    edge_budget=None,
    strongest_retained_bottleneck=None,
):
    """Canonical threshold/bottleneck provenance for one pathfinding run
    (SYNAPSE basis — int units; ratio-basis runs use
    ``connection_ratio_paths.ratio_threshold_provenance``, the float
    mirror with identical key names).

    Single source for the applied-threshold contract shared by
    parameters.txt, all_attributes.json, data_details/parameters.csv,
    the replay folders, and the run guides:

    - ``requested_threshold``: the user-entered Min Synapse Count before
      any budget effect.
    - ``strongest_first_tau``: the StrongestFirst LANDING tau — all intact
      paths with bottleneck >= tau are retained when the path budget
      bites. For a complete run it is the natural weakest emitted-path
      bottleneck.
    - ``tau_canonical`` / applied threshold: the MINIMAL threshold that
      reproduces the materialized output set (``w2 + 1`` when the budget
      bite leaves a gap; the landing tau otherwise).
    - ``strongest_dropped_bottleneck`` (w2): the strongest path NOT
      emitted after the StrongestFirst budget bites.
    - ``edge_weight_floor`` (w0) / ``edge_budget_landing`` (w1): the Edge
      Budget floor and the tier that determined it; a floored run is
      exactly a complete run at ``max(requested, w0)``.
    - ``strongest_retained_bottleneck`` (W*): the widest-path ceiling
      after lossless pruning; lossless pruning must not change it.

    ``applied_threshold`` is the requested threshold for an
    unbounded/complete run (the natural tau is reported separately);
    when a lossy budget affects the output it is the **canonical
    equivalent threshold** that reproduces the materialized set
    (``tau_canonical``: ``w2 + 1`` when the StrongestFirst bite leaves a
    gap, else the landing/natural tau), and ``applied_threshold_source``
    names the contributing mechanism(s): 'requested',
    'strongest_first_budget', 'edge_budget', or
    'strongest_first_budget+edge_budget'.

    There is deliberately **no** ``max(canonical, requested)`` clamp: the
    pipeline order (filter by requested -> edge budget -> graph build ->
    StrongestFirst) makes a per-slice run's canonical at or above its own
    requested threshold — a bitten slice drops only paths with bottleneck
    >= requested (so ``w2 + 1 > requested``), a binding floor has
    ``w0 > requested``, and a complete slice's natural tau is
    ``>= requested``. A canonical below the request therefore indicates the
    slice inherited a lower threshold's provenance, which is fixed at the
    source (the multi-threshold replay), not by clamping the label.

    The Edge Budget contributes to the source only when the floor is
    **binding for this threshold** (``w0 > requested``): when
    ``w0 <= requested`` the floored graph reproduces the unfiltered
    graph's output at that threshold, so the run is untouched
    (``edge_budget_applied`` False, ``paths_complete`` True).
    """
    requested_threshold = int(requested_threshold)
    bitten = bool(strongest_first_budget_bitten)
    floor = None if edge_weight_floor is None else int(edge_weight_floor)
    # The floor only changes THIS threshold's output when it removes edges
    # at or above the requested cutoff, i.e. when w0 > requested.
    floor_binding = floor is not None and floor > requested_threshold
    sources = []
    if bitten:
        sources.append('strongest_first_budget')
    if floor_binding:
        sources.append('edge_budget')

    if not sources:
        applied_threshold = requested_threshold
        applied_source = 'requested'
        paths_complete = True
        # Complete runs: the natural tau IS the canonical (minimal)
        # threshold for this set.
        if strongest_first_tau is not None:
            canonical_tau = int(strongest_first_tau)
        elif tau_canonical is not None:
            canonical_tau = int(tau_canonical)
        else:
            canonical_tau = None
    else:
        if tau_canonical is not None:
            canonical_tau = int(tau_canonical)
        elif strongest_dropped_bottleneck is not None:
            canonical_tau = int(strongest_dropped_bottleneck) + 1
        elif strongest_first_tau is not None:
            canonical_tau = int(strongest_first_tau)
        else:
            canonical_tau = None
        if canonical_tau is None and floor_binding:
            canonical_tau = floor
        if canonical_tau is not None and floor_binding:
            # Invariant: the floored graph cannot emit a path weaker than
            # the floor, so canonical >= w0. Keep the max as a guard for
            # degenerate inputs rather than emitting a contradiction.
            canonical_tau = max(canonical_tau, floor)
        # The applied threshold IS the canonical equivalent threshold. With
        # the pipeline order (filter by requested -> edge budget -> graph ->
        # StrongestFirst) a per-slice run cannot produce a canonical below
        # its own requested value: a bitten slice drops only paths with
        # bottleneck >= requested (so w2+1 > requested), and a floored slice
        # has w0 > requested; a complete slice has natural tau >= requested.
        # Any observed canonical < requested means the slice inherited a
        # LOWER threshold's provenance — fix that at the source (replay),
        # never by clamping the label here.
        applied_threshold = (canonical_tau if canonical_tau is not None
                             else requested_threshold)
        applied_source = '+'.join(sources)
        paths_complete = False

    return {
        'requested_threshold': requested_threshold,
        'applied_threshold': applied_threshold,
        'applied_threshold_source': applied_source,
        'strongest_first_budget_bitten': bitten,
        'strongest_first_tau': strongest_first_tau,
        'tau_canonical': canonical_tau,
        'strongest_dropped_bottleneck': strongest_dropped_bottleneck,
        'edge_budget': (int(edge_budget) if edge_budget else None),
        # Per-threshold binding, not mere presence of a graph floor.
        'edge_budget_applied': floor_binding,
        # Explicit alias for consumers that want the unambiguous name.
        'edge_floor_binding': floor_binding,
        'edge_budget_landing': edge_budget_landing,
        'edge_weight_floor': (int(edge_weight_floor)
                              if edge_weight_floor is not None else None),
        'strongest_retained_bottleneck': strongest_retained_bottleneck,
        'paths_complete': paths_complete,
    }

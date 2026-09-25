# DROCAT UI Display & Interaction Refinement Plan

Status: In progress — reliability foundation implemented
Date: 2026-09-11
Scope: DROCAT v4.5.0 NiceGUI display, interaction, execution-state continuity, refresh/reconnect behavior, and UI quality assurance.

Implementation checkpoint: the durable run-state layer, page-teardown protection, same-session refresh rehydration, explicit backend termination on UI-server shutdown, startup cleanup of old terminal run-cache records, active-tab persistence, Activity view, tab activity indicators, output metadata/recovery messaging, session-scoped restoration, and focused regression tests are implemented. The remaining phased work is the broader form-validation standardization, visual regression/accessibility gate, and incremental migration of additional shared interaction patterns.

## 1. Desired outcome

Make every DROCAT tool feel like one dependable application rather than a collection of independently assembled tabs. A user should be able to start a comparison, navigate away, refresh the page, or reconnect and still understand:

- what ran, what is running, and what finished;
- whether the run succeeded, failed, was cancelled, or was interrupted;
- how far the run progressed and what it is currently doing;
- where its output files are and which result belongs to which run;
- what action is available next.

The refinement should preserve the current scientific workflows and backend result semantics while making the interface safer, clearer, more responsive, and easier to test.

### Non-goals

- Changing connectome algorithms, thresholds, result schemas, or analysis semantics.
- Automatically rerunning work after a refresh.
- Cancelling a subprocess merely because a browser page was refreshed or disconnected.
- Replacing the existing worktree changes unrelated to this plan.

## 2. Baseline and evidence

The plan is based on the repository structure as of 2026-09-11 (line
anchors below are from that baseline) and the reported v4.5.0 runtime
trace.

| Area | Current evidence | Planning implication |
|---|---|---|
| Page construction | `ui/app.py:2492-2677` builds the page and tab panels per NiceGUI client. | Page-local objects are recreated on refresh. |
| Shared layout | `ui/components/common.py:271-322` provides the common two-column tool workspace. | Keep this as the layout foundation and formalize its variants. |
| Inputs | `ui/components/common.py:619-875` and `:2064-2510` cover datasets, neuron chips, numeric/select controls, and directories. | Establish consistent validation, dependency, keyboard, and empty-state rules. |
| Output display | `ui/components/output_panel.py:166-710` owns status, progress, logs, files, previews, and cancellation. | Make it a lifecycle-safe view subscribed to durable run state. |
| Progress | `ui/components/page_progress.py:345-574` updates a NiceGUI progress element from async work. | Guard all updates against deleted clients and make progress recoverable. |
| Execution | `ui/runner.py:303-511` and `:786-930` manage subprocesses, streamed output, and result metadata. | Separate job ownership from browser/page presentation. |
| Reusable styling | `ui/app.py` contains the global CSS and design tokens. | Consolidate tokens and component states before broad visual changes. |
| Existing UI coverage | `tests/ui/test_ui_e2e.py` has substantial server-side component/layout coverage. | Add browser-level refresh, reconnect, accessibility, and visual checks. |
| Existing design work | `docs/archive/UI_REDESIGN_AND_BACKEND_REPORT.md` documents the current workspace/output-panel direction. | Extend the established visual language instead of introducing a second design system. |

## 3. Prioritized findings

| ID | Priority | Finding | User impact | Evidence/confidence | Status |
|---|---|---|---|---|---|
| UI-01 | P0 | A background comparison can finish after its page client is deleted; final UI updates then raise NiceGUI client/slot teardown errors. | Noisy traceback and missing final status even when the backend returns successfully. | `inter_dataset.py:703-740`, `output_panel.py:408-417`, `page_progress.py:353-360`; high confidence. | ✅ Closed — teardown-safe updates (`_ui_alive` guard), regression-tested `test_output_panel_is_safe_after_client_delete`. |
| UI-02 | P0 | Run state is primarily held by page-local `ScriptRunner`/`OutputPanel` instances. | Refreshing the page empties status, progress, logs, and result presentation. | Tab factories plus `runner.py:303+`; high confidence. | ✅ Closed — durable `ui/run_state.py` layer + same-session refresh rehydration; regression-tested in `tests/ui/test_run_state.py`. |
| UI-03 | P1 | Execution ownership and UI lifecycle are coupled, with repeated run/clear/status code across tabs. | Fixes are inconsistent and future tabs can reintroduce lifecycle bugs. | Repeated tab patterns; high confidence. | ✅ Closed (with the run-state layer). |
| UI-04 | P1 | There is no application-wide activity view or durable run history. | Users must inspect tabs individually and can miss a completed or failed run. | Current page/tab structure; high confidence. | ✅ Closed (Activity view + tab activity indicators; the interaction-contract half is UI-05). |
| UI-05 | P1 | The form and navigation surfaces expose many controls without one explicit interaction contract for required, advanced, dependent, invalid, and recovering states. | Higher cognitive load and inconsistent feedback between tools. | Shared inputs plus per-tab configuration; medium confidence. |
| UI-06 | P2 | Output panels can become long log/result surfaces without a uniform hierarchy for summary, details, files, and previews. | Important outcome information is harder to scan. | `output_panel.py:166+`; medium confidence. |
| UI-07 | P2 | CSS and component state styling are centralized but broad; visual variants and state rules are not fully explicit. | Small changes can cause cross-tab drift or regressions. | `ui/app.py` global stylesheet; medium confidence. |
| UI-08 | P2 | Existing UI tests are mainly server-side; browser refresh/disconnect and visual regression paths are not a defined gate. | The reported failure mode can escape automated testing. | `tests/ui` inventory; high confidence. |
| UI-09 | P2 | Accessibility and responsive behavior need a deliberate verification matrix for dense navigation, chips, dialogs, logs, and result cards. | Keyboard, mobile-width, focus, and screen-reader users may receive uneven support. | Current breakpoint/style inventory; medium confidence. |

## 4. Target interaction contract

The following behavior should be consistent across every analysis tab.

1. **Page load:** restore the last selected tab and show the latest known run for each tab when one exists in the current DROCAT server session. A new server session starts with explicit empty states instead of silently restoring old terminal logs.
2. **Form editing:** preserve user-entered values during ordinary UI updates; distinguish required, optional, advanced, and disabled-by-dependency fields; validate before launching work.
3. **Run launch:** create a stable `run_id`, show a concise preflight summary, disable only conflicting actions, and expose an explicit cancel action.
4. **Progress:** show status, phase, elapsed time, and an honest progress value. If exact progress is unavailable, use an indeterminate state rather than a misleading percentage.
5. **Navigation:** allow users to move between tabs while work continues; show a small activity indicator on tabs with queued/running/unread results.
6. **Refresh/reconnect:** detach the old page subscriber, keep the job alive, create a new subscriber, replay the latest snapshot/log, and never cancel solely because the client disappeared.
7. **Completion:** show success/failure/cancelled/interrupted state, return code when useful, elapsed time, output folder, and grouped output files. Preserve the log and result preview.
8. **Failure/recovery:** show a human-readable summary first and technical details second. If a process disappears or state is stale, label it `Interrupted` rather than leaving it at `Running`.
9. **Repeat execution:** keep prior runs inspectable as history while clearly marking the current/latest run. Never silently replace a completed result with an empty panel.

## 5. Workstreams

### A. Execution lifecycle and refresh continuity — highest priority

Introduce an application-scoped run manager that owns execution independently of a browser client.

Deliverables:

- A stable `tab_key` and `run_id` for every execution.
- A server-session identifier so browser refresh/reconnect can restore state without making old terminal sessions appear current.
- A job record with status, phase, progress, timestamps, return code, output folder, log reference, and process identity where applicable.
- Durable per-run state, using an atomic manifest and append-only execution log/JSONL. Do not persist secrets or unnecessary raw parameters.
- A subscription/replay API: a newly built page can obtain a snapshot, replay recent log lines, and subscribe to future events.
- Lifecycle-safe page binding: page deletion detaches subscriptions and stops page-owned timers; it does not cancel the underlying job.
- Explicit cancellation owned by the run manager and initiated only by the user’s cancel action.
- Explicit server-shutdown cleanup that terminates active backends when the DROCAT UI process exits; browser refresh remains non-cancelling.
- Startup/reconnect reconciliation that marks missing or stale processes `Interrupted` with an explanatory message.
- Startup cache cleanup that reconciles orphaned active records, removes stale terminal manifests/logs from older sessions, and preserves current-session active records plus all output folders/results.
- Retention limits and cleanup for old manifests/logs/output references.

Recommended state vocabulary: `Idle`, `Queued`, `Running`, `Completed`, `Failed`, `Cancelled`, and `Interrupted`.

### B. Shared run and output components

Turn `OutputPanel` into a view of run state instead of the owner of run state.

Deliverables:

- One shared run action bar with Run, Cancel, Retry, Clear view, and View history actions.
- One status model used by progress, badges, tab indicators, notifications, and output summaries.
- Safe no-op behavior for every UI update after client deletion, including status, progress, logs, files, previews, and timers.
- Explicit states for empty, loading, restoring, active, completed, failed, cancelled, and interrupted.
- Consistent result summary at the top, collapsible logs in the middle, and grouped files/previews below.
- Stable test IDs and semantic labels on run controls, status, log, output folder, and file actions.

After the shared contract is stable, reduce duplicated tab-level lifecycle code incrementally rather than performing a broad rewrite.

### C. Navigation and application-level display

Make work discoverable even when the user is not looking at the originating tab.

Deliverables:

- Persistent active-tab selection across refresh, using a URL/hash or another explicit browser-safe mechanism.
- Per-tab activity indicators for queued/running, completed/unread, failed, and interrupted runs.
- A compact Activity/Recent Runs surface showing tool, start time, status, duration, and an open-results action.
- A clear distinction between current run, latest run, and historical run.
- Responsive navigation that remains usable at desktop, tablet, and narrow mobile widths.
- Keyboard-accessible tab order and visible focus indication.

### D. Form interaction and input quality

Apply one interaction model to dataset selection, neuron chips, directories, thresholds, options, uploads, and advanced controls.

Deliverables:

- A field specification for label, description/help, requiredness, default, validation, dependency, and persistence.
- Inline validation with actionable messages before Run; avoid relying on a late toast or traceback.
- Clear dependency behavior: explain why a control is disabled and preserve or intentionally reset its value according to a documented rule.
- A consistent Advanced section with disclosure state retained per tool.
- Preflight summary of resolved inputs, selected datasets, output location, and potentially expensive options.
- Keyboard support for chip creation/removal, suggestions, upload actions, dialogs, and directory controls.
- Consistent handling of loading, no suggestions, invalid input, duplicate input, upload failure, and restored form state.

### E. Results, logs, and output presentation

Optimize the results area for scanning first and investigation second.

Deliverables:

- A result header containing status, duration, phase/final message, run time, and output location.
- A short success/failure summary with an expandable technical detail section.
- Log controls for expand/collapse, clear visual separation, copy, and bounded rendering for very long output.
- Grouped output files by type or purpose, with predictable labels and actions.
- Result preview cards that retain the current visual language and show loading/error/unsupported states.
- A “restored from previous execution” indicator that is informative but not alarming.
- Preservation of log scroll/section expansion where practical during incremental updates.

### F. Design system and visual consistency

Formalize the existing visual language before adding more local styling.

Deliverables:

- Named tokens for color, typography, spacing, radius, elevation, focus ring, disabled state, and status colors.
- Component variants for primary/secondary/destructive actions, status badges, cards, empty states, and alerts.
- A documented density scale for forms, navigation, logs, and result cards.
- Light/dark theme checks for contrast, status meaning, and disabled controls.
- A gradual organization/extraction of global CSS from `ui/app.py` after token and variant rules are agreed; avoid a risky one-shot stylesheet rewrite.

### G. Accessibility, responsiveness, and performance

Treat these as acceptance criteria for each workstream, not a final polish pass.

Deliverables:

- Semantic labels and descriptions for all inputs and actions.
- Predictable focus order, focus restoration after dialogs, and keyboard operation for navigation and run controls.
- Visible non-color status cues and contrast-checked status colors.
- Responsive checks at approximately 1440, 1024, 768, and 390 CSS pixels.
- Bounded log rendering and throttled progress/log updates to avoid excessive client traffic.
- No active timers or event subscriptions left attached to deleted pages.
- Clear behavior when multiple tools run concurrently.

### H. QA, documentation, and release discipline

Deliverables:

- A fake/deterministic runner fixture for lifecycle and refresh tests.
- Unit tests for state transitions, manifest writes, stale-run reconciliation, and subscriber replay.
- Browser smoke tests for launch, navigation, refresh, reconnect, completion, failure, cancellation, and output reopening.
- Visual snapshots for representative tools and all major UI states.
- Accessibility checks for keyboard flow, labels, focus, and contrast.
- Documentation for the run-state contract and user-visible recovery behavior.
- A source-of-truth process for UI guides and generated/manual HTML companions.

## 6. Phased delivery plan

### Phase 0 — Baseline and interaction contract

Activities:

- Inventory representative tabs: one simple form, one dataset-heavy form, one visualization/output-heavy form, and `inter_dataset`.
- Capture screenshots and interaction notes for fresh, running, completed, failed, and empty states.
- Agree on state vocabulary, status wording, test IDs, responsive widths, and the refresh contract.
- Add no behavior changes in this phase; use it to establish comparable before/after evidence.

Exit criteria: signed-off interaction contract, representative baseline captures, and a test matrix mapped to the findings above.

### Phase 1 — Reliability and durable execution state

Activities:

- Implement the application-scoped run manager and durable state/log records.
- Add lifecycle-safe subscriber attach/detach and safe UI update boundaries.
- Rehydrate status, progress, logs, output files, and active-tab selection after refresh.
- Reconcile stale or missing processes as `Interrupted`.

Exit criteria: no client/slot teardown traceback during refresh or disconnect; a running job survives refresh; completed and failed jobs reappear with their evidence; explicit Cancel remains functional.

### Phase 2 — Shared component and navigation contract

Activities:

- Refactor `OutputPanel` around the shared run-state contract.
- Introduce the shared run action bar, status presentation, result summary, and activity indicators.
- Migrate representative tabs first, then migrate remaining tabs in small batches.

Exit criteria: representative tabs behave identically for all run states; duplicated lifecycle paths are reduced; inactive tabs expose meaningful activity state.

### Phase 3 — Form and interaction refinement

Activities:

- Apply the field specification and validation rules.
- Standardize advanced sections, dependencies, preflight summaries, chip controls, suggestions, uploads, and directory persistence.
- Verify that asynchronous updates never erase in-progress form edits.

Exit criteria: invalid runs are blocked with actionable inline feedback; keyboard-only operation works for core workflows; dependent controls are understandable and deterministic.

### Phase 4 — Display, responsive layout, and visual system

Activities:

- Improve result hierarchy, file grouping, log controls, previews, and recovery messaging.
- Formalize tokens and variants, then organize global CSS with minimal visual regression.
- Tune navigation and two-column workspaces across target widths and themes.

Exit criteria: users can identify status and next action within a quick scan; no critical overflow or contrast/focus defects remain at target widths; representative snapshots are stable.

### Phase 5 — QA, documentation, and rollout

Activities:

- Run the full lifecycle/browser/accessibility/visual matrix.
- Update the run-state and UI interaction documentation.
- Add release gates for teardown safety, refresh recovery, and output discoverability.
- Roll out by representative tool group, monitoring regressions before migrating all tabs.

Exit criteria: all P0/P1 findings are closed or explicitly accepted, the test matrix is green, and the recovery behavior is documented for users and maintainers.

## 7. Refresh and reconnect behavior in detail

The intended event sequence is:

1. The user starts a run and receives a `run_id`.
2. The run manager persists `Queued`/`Running` state and appends execution events independently of the page.
3. The page subscribes to the run and renders a snapshot plus live events.
4. The browser refreshes or temporarily disconnects. The page subscriber and page-owned timers are detached; the run continues.
5. The new page loads the latest run for each tab from the same server session, restores the selected tab, replays the latest snapshot/log tail, and subscribes to new events. A newly started server does not rehydrate terminal records from an older session.
6. The run completes, fails, is cancelled explicitly, or is reconciled as interrupted. The terminal state and output references remain visible after another refresh.

Minimum persisted record:

```text
run_id, tab_key, tool_name
status, phase, progress, message
created_at, started_at, finished_at, duration
return_code, process_id, host_identity
output_folder, output_files, log_reference
safe_input_summary, error_summary
```

Writes should be atomic, records should be scoped to the DROCAT instance/user as appropriate, and sensitive values should be excluded or redacted.

## 8. Validation matrix

| Scenario | Expected result | Test layer |
|---|---|---|
| Fresh page, no prior run | Consistent empty state and enabled form. | Component/browser |
| Run queued/running | Status, phase, progress/log updates, and disabled conflicting actions are visible. | Unit/integration/browser |
| Refresh during run | New page restores the same `run_id`; job is not duplicated or cancelled. | Browser/integration |
| Refresh after success | Completed status, log, files, preview, and output folder return. | Browser |
| Refresh after failure | Failure summary and technical details return without traceback. | Browser |
| Explicit cancel | Process stops, terminal state is `Cancelled`, and output is clearly labeled. | Integration/browser |
| Process/server restart | Stale active work becomes `Interrupted` with recovery guidance. | Integration |
| Tab navigation during run | Work continues; originating tab shows activity when revisited. | Browser |
| Two concurrent runs | Events and output files remain associated with the correct run. | Integration/browser |
| Long log/output list | UI remains responsive and bounded. | Performance/browser |
| Keyboard-only flow | User can select inputs, launch/cancel, navigate, and inspect results. | Accessibility/browser |
| Narrow/mobile width | No clipped primary controls or inaccessible result content. | Visual/browser |
| Light/dark themes | Status remains understandable through text/icon plus color. | Visual/accessibility |

## 9. Open decisions to resolve in Phase 0

- Should active jobs survive a full DROCAT server restart, or only page refresh/reconnect? Current policy: survive page refresh/reconnect; mark jobs interrupted after server restart unless a worker-backed persistence mechanism is introduced. The current `ScriptRunner` subprocess is not a detached, reconnectable worker.
- How many run records and how much log history should be retained? Current policy removes terminal run-cache manifests/logs from older sessions at UI startup, retains the current session for refresh recovery, and preserves output folders/results. Longer-term age/count limits can still be added if a multi-session history view is needed.
- Is state app-wide for the local single-user application, or should it be scoped per browser/session? Recommended initial policy: app-wide job state with a per-client view of selected tab and expansion state.
- Should closing the browser behave like refresh? Recommended initial policy: yes for job ownership; only explicit Cancel stops work.
- Which output files are safe to persist as references, and which should be redacted from logs/manifests?
- Which browser/OS combinations are release targets for keyboard, layout, and visual checks?

## 10. Definition of done

- Refreshing during any supported execution state restores the correct run rather than an empty panel.
- A deleted NiceGUI client cannot produce the reported teardown traceback from normal run completion, cancellation, or log/progress updates.
- Page teardown detaches presentation resources without cancelling the underlying job.
- Every tab uses the same user-visible status vocabulary and result hierarchy.
- Users can find active, completed, failed, cancelled, and interrupted work without manually reopening every tab.
- Core forms provide consistent validation, dependency feedback, keyboard operation, and preserved input values.
- Desktop, tablet, narrow layouts, light mode, dark mode, and keyboard-only workflows pass the agreed checks.
- Browser lifecycle, visual regression, accessibility, and state-rehydration tests are part of the release gate.
- Existing algorithm behavior and output semantics remain unchanged.

## 11. Implementation guardrails

- Add this plan as the only new artifact for the planning task; do not modify existing source, tests, or guides until an implementation phase is explicitly approved.
- Preserve unrelated modifications already present in the worktree.
- Prefer small, reversible migrations by representative tab group.
- Keep the run manager and persistence contract independent of NiceGUI so the execution state can be tested without a browser.
- Treat the page as a disposable view and the run record as the source of truth for execution status.
- Validate each phase against the baseline captures and the matrix above before broadening the rollout.

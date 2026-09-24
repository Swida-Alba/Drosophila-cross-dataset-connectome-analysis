#!/bin/bash
# Run a TM VEV matrix — targets x modes over one query — and record it.
#
# This is the durable form of the ad-hoc queue drivers that used to sit in
# local_data/: runs STRICTLY SEQUENTIALLY in the live repo (its datasets/ and
# cache/ are the point), waits for the machine to be free so no run's stage
# timings are measured under a competitor, and appends one manifest row per run
# (folder, exit code, wall seconds). A failing run is recorded and the queue
# carries on: a failure is a finding, not a reason to stop collecting data.
#
#   scripts/maintenance/run_tmvev_matrix.sh \
#       --targets "male-cns:v1.0,banc_v888" \
#       --modes "restrictive,family,aggressive,pooling" \
#       --types circadian_clock [--out local_data/tmvev-matrix-$(date +%Y%m%d)] \
#       [--source flywire_FAFB_v783] [--no-scenes] [--now]
#
# Read the result with scripts/verify_tmvev_run_exports.py <out> and, for a
# before/after pair, scripts/verify_tmvev_run_parity.py compare <pre> <post>.
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${PY:-python3}
SRC=flywire_FAFB_v783
TYPES=circadian_clock
TARGETS="male-cns:v1.0,banc_v888"
MODES="restrictive,family,aggressive,pooling"
OUT=""
SCENES=1
WAIT=1

while [ $# -gt 0 ]; do
    case $1 in
        --targets) TARGETS=$2; shift 2 ;;
        --modes) MODES=$2; shift 2 ;;
        --types) TYPES=$2; shift 2 ;;
        --source) SRC=$2; shift 2 ;;
        --out) OUT=$2; shift 2 ;;
        --no-scenes) SCENES=0; shift ;;
        --now) WAIT=0; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[ -n "$OUT" ] || OUT=$REPO/local_data/tmvev-matrix-$(date +%Y%m%d_%H%M)
mkdir -p "$OUT"
MAN=$OUT/manifest.tsv
LOG=$OUT/queue.log
[ -f "$MAN" ] || printf 'started\ttarget\tmode\tlabel\trun_dir\texit\twall_s\n' \
    > "$MAN"
SHORT_OF() { case $1 in *male-cns*) echo mcns ;; *banc*) echo banc ;;
                 *hemibrain*) echo hemi ;; *) echo tgt ;; esac; }

if [ "$WAIT" = 1 ]; then
    echo "== matrix $(date '+%F %T'): waiting for the machine to be free" >> "$LOG"
    # the rule this honours: one test suite / one benchmark at a time, and a
    # stage timing measured against a competitor is not a measurement. The
    # patterns name a process's actual invocation, never a word that appears in
    # this script's own command line, so the wait cannot match itself.
    while pgrep -f "python.* -m pytest|python.* -m pytest tests|RunMappingValidation.py|verify_tmvev_frozen_gate.sh" \
            > /dev/null; do sleep 60; done
fi
echo "== matrix starts $(date '+%T')" >> "$LOG"

IFS=, read -ra TGT_ARR <<< "$TARGETS"
IFS=, read -ra MODE_ARR <<< "$MODES"
for tgt in "${TGT_ARR[@]}"; do
    short=$(SHORT_OF "$tgt")
    for mode in "${MODE_ARR[@]}"; do
        label="matrix_${short}_${mode}"
        t0=$(date +%s)
        echo "---- $label start $(date '+%T')" >> "$LOG"
        args=(--source "$SRC" --target "$tgt" --types "$TYPES"
              --mode "$mode" --label "$label" --output-dir "$OUT")
        [ "$SCENES" = 1 ] && args+=(--scene-selfcheck) || args+=(--no-visualize)
        ( cd "$REPO" && "$PY" -u scripts/RunMappingValidation.py \
            "${args[@]}" ) > "$OUT/$label.log" 2>&1
        rc=$?
        run=$(grep -m1 -E "^(Results:|.*Output will be saved to:)" \
                "$OUT/$label.log" | sed -E 's/.*(Results: |saved to: )//')
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date '+%T')" "$tgt" "$mode" \
            "$label" "$run" "$rc" "$(( $(date +%s) - t0 ))" >> "$MAN"
        echo "---- $label exit=$rc wall=$(( $(date +%s) - t0 ))s $run" >> "$LOG"
        [ -n "$run" ] && grep -E "^\[pooling\]|^\[stage 5\] !|^! " \
            "$run/README.txt" 2>/dev/null | cut -c1-220 | head -12 >> "$LOG"
    done
done
echo "== matrix DONE $(date '+%F %T')" >> "$LOG"
touch "$OUT/DONE"

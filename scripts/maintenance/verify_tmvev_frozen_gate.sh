#!/bin/bash
# Cold/warm parity gate for anything that CACHES morphometry.
#
# A cache on this route is only allowed to make the next run faster. If it
# changes a score it is a different instrument, and no consumer would see it:
# the float32 sidecar this gate was written for moved a score in its 8th
# decimal (0.5295437067915649 vs 0.5295436978544568) while every verdict
# stayed put. So this runs the SAME query twice against a private clone of the
# stores — once with the target-vector store deleted, once with what the first
# run wrote — and demands byte-identical exports, then reports the stage delta
# the speed-up is supposed to buy.
#
#   scripts/maintenance/verify_tmvev_frozen_gate.sh \
#       [--harness local_data/tmvev-harness] \
#       [--source flywire_FAFB_v783] [--target male-cns:v1.0] \
#       [--types l-LNv,APDN3,s-CPDN3A] [--py python3]
#
# The harness is an APFS clone of cache/ + neuron_indexes/ with the working
# tree's src/scripts/ui/datasets rsynced into it: a run mutates its own clone,
# never the shared cache, which is what makes the two runs comparable at all.
# Exit 0 only when every compared file is identical.
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${PY:-python3}
H=$REPO/local_data/tmvev-harness
SRC=flywire_FAFB_v783
TGT=male-cns:v1.0
TYPES=l-LNv,APDN3,s-CPDN3A

while [ $# -gt 0 ]; do
    case $1 in
        --harness) H=$2; shift 2 ;;
        --source) SRC=$2; shift 2 ;;
        --target) TGT=$2; shift 2 ;;
        --types) TYPES=$2; shift 2 ;;
        --py) PY=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[ -d "$H/cache" ] || { echo "no frozen store at $H/cache" >&2; exit 2; }

sync_code() {
    for d in src scripts ui datasets; do
        rsync -a --delete --exclude '__pycache__' "$REPO/$d/" "$H/$d/" || return 1
    done
    cp "$REPO/scripts/RunMappingValidation.py" "$H/RunMappingValidation.py"
}

run_one() {
    local label=$1
    echo "== $label start ($(date '+%T'))"
    local t0=$(date +%s)
    ( cd "$H" && $PY -u scripts/RunMappingValidation.py \
        --source "$SRC" --target "$TGT" --types "$TYPES" --mode pooling \
        --label "$label" --output-dir "$H/out" --scene-selfcheck \
        > "$H/$label.log" 2>&1 )
    local rc=$?
    echo "== $label exit=$rc elapsed=$(( $(date +%s) - t0 ))s"
    local RUN
    RUN=$(grep -m1 -E "^(Results:|.*Output will be saved to:)" "$H/$label.log" \
          | sed -E 's/.*(Results: |saved to: )//')
    echo "$RUN" > "$H/$label.rundir"
    echo "== $label folder: $RUN"
    grep -E "^\[pooling\]" "$RUN/README.txt" 2>/dev/null | cut -c1-330
    $PY - "$RUN" "$label" <<'PY'
import json, sys, datetime as dt
from pathlib import Path
ev = [json.loads(l) for l in
      Path(sys.argv[1], 'pipeline_progress.jsonl').read_text().splitlines()
      if l.strip()]
ts = {}
for r in ev:
    if r.get('event') in ('stage_start', 'stage_done'):
        ts.setdefault(str(r.get('stage')), []).append(r['ts'])
for k, v in ts.items():
    if len(v) >= 2:
        a = dt.datetime.fromisoformat(v[0])
        b = dt.datetime.fromisoformat(v[-1])
        print(f'   {sys.argv[2]} stage {k}: {int((b - a).total_seconds())} s')
PY
}

sync_code || { echo "sync failed" >&2; exit 1; }
echo "== cold run: no target-vector store"
find "$H/cache" -name 'cross_dataset_targetvec_*.npz' -print -delete | head -3
run_one cold
STORE=$(find "$H/cache" -name 'cross_dataset_targetvec_*.npz' | head -1)
echo "== store after the cold run: ${STORE:-NONE}"
[ -n "$STORE" ] || { echo "the cold run wrote no store — gate cannot proceed" >&2; exit 1; }
ls -l "$STORE"
run_one warm

echo "== compare"
A=$(cat "$H/cold.rundir"); B=$(cat "$H/warm.rundir")
$PY "$REPO/scripts/verify_tmvev_run_parity.py" compare "$A" "$B" \
    --expect-identical
rc=$?
echo "== gate $([ $rc = 0 ] && echo PASSED || echo FAILED) ($(date '+%T'))"
exit $rc

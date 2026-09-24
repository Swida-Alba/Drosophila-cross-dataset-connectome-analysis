#!/bin/bash
# Announce the full-suite result: a macOS notification for the user plus stdout
# (so the agent session running this in the background gets its completion
# event). Exits non-zero if the summary never appeared inside the window.
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
STAMP=${1:?usage: scripts/maintenance/announce_suite.sh <run stamp, e.g. 20260922_190520>}
SUM=local_data/full_suite_$STAMP.summary.json
LOG=local_data/full_suite_$STAMP.runner.log
for _ in $(seq 1 480); do
  [ -f "$SUM" ] && break
  sleep 30
done
if [ ! -f "$SUM" ]; then
  osascript -e 'display notification "no summary after 4h — check the runner log" with title "DROCAT full suite" sound name "Basso"'
  echo "TIMEOUT waiting for $SUM"
  tail -4 "$LOG"
  exit 1
fi
LINE=$(/usr/bin/python3 -c "
import json
d = json.load(open('$SUM'))
parts = [r['summary'].split(' in ')[0] for r in d['stages']]
bad = [r['stage'] for r in d['stages'] if r['rc'] != 0]
print(('FAILED in ' + ', '.join(bad) + ' | ') if bad else 'all stages clean | '
      + ' ; '.join(parts))
")
osascript -e "display notification \"$(echo "$LINE" | sed 's/"/\\"/g')\" with title \"DROCAT full suite\" sound name \"Glass\"" 2>/dev/null
echo "$LINE"
cat "$SUM"

#!/bin/bash
# The code of the Gemma 4 back-translation, from this checkout (committed, no local changes), to <dir>/scripts/ of each
# region's directory, where keeper.sh copies it onto the VMs of that region (a VM reads only its own region's bucket):
#   bash stage.sh gs://<bucket-a>/<dir> gs://<bucket-b>/<dir> ...
# With PILOT=<file>: also that file as scripts/probe-ids.json (bt_gemma4.py probeplan --pilot).
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: stage.sh gs://<bucket>/<dir> ..."; exit 1; }
HERE=$(cd "$(dirname "$0")" && pwd)
TOP=$(dirname "$HERE")
(cd "$TOP" && git diff --quiet && git diff --cached --quiet) || { echo "$TOP has uncommitted changes"; exit 1; }
COMMIT=$(cd "$TOP" && git rev-parse --short HEAD)
FILES=("$HERE/bt_gemma4.py" "$HERE/gemma4_engine.py" "$HERE/models.yml" "$HERE/run_bt_gemma4.sh" "$HERE/setup.sh"
       "$TOP/pairs.py" "$TOP/structure.py" "$TOP/publish.py" "$TOP/select_passages.py" "$TOP/metricx_qe.py")
for D in "$@"; do
  [[ $D == gs://* ]] || { echo "not a gs:// directory: $D"; exit 1; }
  T=${D%/}/scripts/
  timeout 300 gsutil -q -m cp "${FILES[@]}" "$T"
  if [ -n "${PILOT:-}" ]; then timeout 60 gsutil -q cp "$PILOT" "${T}probe-ids.json"; fi
  echo "$COMMIT $(date -u +%FT%TZ)" | timeout 60 gsutil -q cp - "${T}STAGED"
  echo "staged $COMMIT to $T"
done

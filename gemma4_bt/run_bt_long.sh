#!/bin/bash
# On a worker of a TPU v4-32 (each worker one process of its own 4 chips): the TPU steps of the long passages
# (bt_long.py), each run by a control machine on every worker of a VM. The code is this checkout, staged as it is.
#   O=gs://<bucket>/<dir> CKPT=gs://<bucket>/<dir>/gemma-4-31b-it-maxtext bash run_bt_long.sh setup
#   ... bash run_bt_long.sh smoke <smoke-nl|smoke-rt> <class> ...     (ROOT=gs://... of this VM's smoke test)
#   ... bash run_bt_long.sh judge                                     (this worker's share of the judge's input)
#   ... bash run_bt_long.sh work <nl-en|rt> <class> ...               (every unit of the classes, with the others)
# setup: the venv of MaxText at FORK_COMMIT in $SHM (one already there with that commit and every package is taken;
#   else setup.sh builds it), Gemma 4's tokenizer, the judge's environment (gemma_judge/setup.sh: MaxText 0.2.5, which
#   scores without a decode cache); a check that the 4 chips are one process.
# smoke: per class the units of the smoke plan $O/smoke-plan/<plan> under ROOT; a class that runs out of memory runs
#   again with its fallback_slots of models.yml; the slots that ran go to ROOT/slots/<class>. The logs go to ROOT/logs/.
# judge: judge.py score (task source_topic_origin, mesh fsdp 1 x tensor 4, as the existing passages) on
#   $O/judge-in/w<k> into $O/scores/w<k>; out of memory: again with half the token budget.
# work: the plan $O/<plan>/plan, its classes in the order given, claims and shards under $O/<plan>/, the slots of
#   $O/slots.json; until the plan is complete (bt_gemma4.py status), every worker of every VM taking free units.
# CPU steps, on one worker (the control machine runs each once, under a lock in the bucket):
#   measure: smoke.json (bt_long.py smoke-measures) and slots.json from ROOT, the first VM's smoke test
#   plan: choose at the target, budget, choose within it, the nl->en plan (bt_gemma4.py plan) to $O/nl-en/plan
#   rtplan: the English of the nl->en shards (bt_long.py english) and the round trip's plan to $O/rt/plan
#   build: the -gemma4 datasets at the heads the candidates read, the new rows (bt_long.py build) to $O/build
# Every step ends with a line "STEP <name> OK" or exits non-zero. Needs ~/.cache/huggingface/token.
set -euo pipefail
O=${O:?set O to the gs:// directory of the run}
CKPT=${CKPT:?set CKPT to the gs:// directory of the MaxText checkpoint of Gemma 4 31B-it}
SHM=${SHM:-/dev/shm/bt_long}
FORK_COMMIT=${FORK_COMMIT:-f4d8615f478b6c18de9df8f2dff1f57f775a2521}
PROMPT=${PROMPT:-natural-v1-fmt}
RUN=${RUN:-$(date +%s)}
C=$(cd "$(dirname "$0")" && pwd)
TOP=$(dirname "$C")
K=$(hostname | sed 's/.*-w-//')
TOK=$SHM/tokenizer-gemma4
JUDGE=${JUDGE:-/dev/shm/judge}  # gemma_judge/setup.sh's environment
PARTS=fineweb,handelingen,raad,dpc,rechtspraak
WORKER_HOURS=${WORKER_HOURS:-96}
TARGET=${TARGET:-5000}
region() { local zone; zone=$(curl -fs -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}'); echo "${zone%-*}"; }
export XLA_FLAGS=--xla_dump_hlo_as_text=false TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
# The host's 4 chips as one process: without these JAX takes the slice from the metadata and waits for the other hosts
export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476
export TPU_PROCESS_ADDRESSES=localhost:8476 CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
mkdir -p ~/logs "$SHM"
# Out of memory, as XLA and the runtime say it
OOM='RESOURCE_EXHAUSTED|CompileTimeHbmOom|Ran out of memory|Error allocating device buffer'

py() { "$(cat "$SHM/venv.txt")/bin/python" "$@"; }

fork_venv() {  # an existing venv with MaxText at FORK_COMMIT and every package bt_gemma4.py needs, or nothing
  local cand got
  for cand in "$SHM/venv" /dev/shm/translate-generic/venv-fork-${FORK_COMMIT:0:9} /dev/shm/flan_nl/venv-g4-${FORK_COMMIT:0:7}; do
    [ -x "$cand/bin/python" ] || continue
    got=$("$cand/bin/python" -c 'import importlib.metadata as m, json; d = m.distribution("maxtext"); print(json.loads(d.read_text("direct_url.json") or "{}").get("vcs_info", {}).get("commit_id"))' 2> /dev/null || true)
    if [ "$got" = "$FORK_COMMIT" ] && "$cand/bin/python" -c 'import yaml, langid, pyarrow, transformers, jax, sacrebleu, tokenizers, huggingface_hub, numpy, google.cloud.storage' 2> /dev/null; then
      echo "$cand"; return 0
    fi
  done
  return 0
}

slots_of() {  # model class: the slots of models.yml (or its fallback_slots with "fallback")
  py -c "import sys, yaml; m = yaml.safe_load(open('$C/models.yml'))['models'][sys.argv[1]]; print(m['fallback_slots' if sys.argv[3:] else 'slots'][sys.argv[2]])" "$@"
}

work_class() {  # plan root class slots log: bt_gemma4.py work on one class; the exit code
  local plan=$1 root=$2 class=$3 slots=$4 log=$5
  py -u "$C/bt_gemma4.py" work --plan "$plan" --root "$root" --length-class "$class" --prompt "$PROMPT" \
    --direction forward --owner "$(hostname)/h0" --run "$RUN" --region "$(region)" --tokenizer "$TOK" \
    --ckpt "$CKPT/0/items" --work-dir "$SHM/work" --slots "$slots" > "$log" 2>&1
}

step=${1:-}
shift || true
case $step in
setup)
  [ -s ~/.cache/huggingface/token ] || { echo "$(hostname): no ~/.cache/huggingface/token"; exit 1; }
  V=$(fork_venv)
  if [ -z "$V" ]; then
    SHM=$SHM FORK_COMMIT=$FORK_COMMIT HF_HUB_OFFLINE=0 bash "$C/setup.sh"
    V=$SHM/venv
  fi
  echo "$V" > "$SHM/venv.txt"
  if [ ! -s "$TOK/tokenizer.json" ]; then
    HF_HUB_OFFLINE=0 py -c "from huggingface_hub import snapshot_download as s; s('google/gemma-4-31B-it', revision='842da3794eaa0b77d5f08bae87a17459d91ff475', local_dir='$TOK', allow_patterns=['tokenizer*', 'chat_template.jinja', 'config.json', 'generation_config.json'])"
  fi
  CKPT=$CKPT HF_HUB_OFFLINE=0 bash "$TOP/gemma_judge/setup.sh"
  timeout 600 "$(cat "$SHM/venv.txt")/bin/python" -c "import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count()); print('devices', jax.devices())"
  echo "venv $V: $(py -c 'import importlib.metadata as m, json; print(json.loads(m.distribution("maxtext").read_text("direct_url.json"))["vcs_info"]["commit_id"])')"
  echo "STEP setup OK w$K $(df -h /dev/shm | tail -1 | awk '{print $4}') free in /dev/shm"
  ;;
smoke)
  plan=$1; shift
  ROOT=${ROOT:?set ROOT to the gs:// directory of the smoke test of this VM}
  mkdir -p "$SHM/smoke-plan/$plan"
  gsutil -m -q rsync -r "$O/smoke-plan/$plan" "$SHM/smoke-plan/$plan"
  model=$(py -c "import json, sys; print(json.load(open(sys.argv[1]))['model'])" "$SHM/smoke-plan/$plan/units.json")
  for class in "$@"; do
    slots=$(slots_of "$model" "$class")
    log=~/logs/btl_smoke_${plan}_${class}_$RUN.log
    if ! work_class "$SHM/smoke-plan/$plan" "$ROOT/$plan" "$class" "$slots" "$log"; then
      if grep -aqE "$OOM" "$log" && [ "$(slots_of "$model" "$class" fallback)" -lt "$slots" ]; then
        gsutil -q cp "$log" "$ROOT/logs/$(basename "$log" .log)-oom-w$K.log"
        slots=$(slots_of "$model" "$class" fallback)
        echo "$class: out of memory with $(slots_of "$model" "$class") slots; again with $slots"
        log=~/logs/btl_smoke_${plan}_${class}_${RUN}_s$slots.log
        work_class "$SHM/smoke-plan/$plan" "$ROOT/$plan" "$class" "$slots" "$log" || { tail -n 30 "$log"; exit 1; }
      else
        tail -n 30 "$log"; exit 1
      fi
    fi
    gsutil -q cp "$log" "$ROOT/logs/$(basename "$log" .log)-w$K.log"
    echo "$slots" | gsutil -q cp - "$ROOT/slots/$class"
    grep -aE '^(SHARD|MEM|engine ready)' "$log" | cut -c1-300 || true
  done
  echo "STEP smoke OK w$K $*"
  ;;
judge)
  IN=$SHM/judge-in
  mkdir -p "$IN"
  gsutil -m -q rsync "$O/judge-in/w$K" "$IN"
  budget=${TOKEN_BUDGET:-16384}
  run_judge() {
    JUDGE_JAX_CACHE=$SHM/judge-jax-cache TPU_LOG_DIR=$SHM/tpu_logs "$JUDGE/venv/bin/python" -u \
      "$TOP/gemma_judge/judge.py" score --ckpt "$CKPT/0/items" --tokenizer "$JUDGE/tokenizer" --fsdp 1 \
      --tensor 4 --token-budget "$1" --task source_topic_origin --pairs "$IN" --out "$O/scores/w$K" > "$2" 2>&1
  }
  log=~/logs/btl_judgerun_$RUN.log
  if ! run_judge "$budget" "$log"; then
    grep -aqE "$OOM" "$log" || { tail -n 30 "$log"; exit 1; }
    gsutil -q cp "$log" "$O/logs/btl_judgerun_${RUN}_oom-w$K.log"
    log=~/logs/btl_judgerun_${RUN}_b$((budget / 2)).log
    run_judge $((budget / 2)) "$log" || { tail -n 30 "$log"; exit 1; }
  fi
  gsutil -q cp "$log" "$O/logs/btl_judgerun_$RUN-w$K.log"
  grep -aE '^(DONE|model ready)' "$log" || true
  n_in=$(ls "$IN" | grep -c '^shard-.*\.parquet$' || true)
  n_out=$(gsutil ls "$O/scores/w$K/" | grep -c '/shard-.*\.parquet$' || true)
  [ "$n_in" = "$n_out" ] || { echo "w$K: $n_out score shards for $n_in input shards"; exit 1; }
  echo "STEP judge OK w$K $n_out shards"
  ;;
work)
  plan=$1; shift
  P=$SHM/$plan
  mkdir -p "$P"
  gsutil -m -q rsync -r "$O/$plan/plan" "$P"
  gsutil -q cp "$O/slots.json" "$SHM/slots.json"
  declare -A fails
  while true; do
    for class in "$@"; do
      slots=$(py -c "import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$SHM/slots.json" "$class")
      log=~/logs/btl_work_${plan}_${class}_$RUN.log
      [ -e "$log" ] && log=~/logs/btl_work_${plan}_${class}_${RUN}_$(date +%s).log
      if work_class "$P" "$O/$plan" "$class" "$slots" "$log"; then
        fails[$class]=0
      else
        # the claim stays this worker's: the next pass takes the same unit again; a class failing twice in a row stops
        fails[$class]=$(( ${fails[$class]:-0} + 1 ))
        tail -n 30 "$log"
        gsutil -q cp "$log" "$O/logs/$(basename "$log" .log)-failed-w$K.log" || true
        [ "${fails[$class]}" -ge 2 ] && { echo "w$K: class $class failed in two passes in a row"; exit 1; }
      fi
      # the informational lines; set -e and pipefail would stop at a log without one
      grep -aE '^(CLASS|SHARD)' "$log" | cut -c1-240 | tail -n 3 || true
    done
    st=$(py -u "$C/bt_gemma4.py" status --plan "$P" --root "$O/$plan" | tail -n 1)
    echo "STATUS $(date -u +%T) $st"
    [ "$st" = COMPLETE ] && break
    sleep 60
  done
  echo "STEP work OK w$K $plan"
  ;;
measure)
  ROOT=${ROOT:?set ROOT to the gs:// directory of the smoke test of this VM}
  W=$SHM/measure-$RUN
  mkdir -p "$W/logs" "$W/shards" "$W/candidates"
  gsutil -m -q rsync -r "$O/candidates" "$W/candidates"
  gsutil -m -q rsync -r "$ROOT/logs" "$W/logs"
  gsutil -m -q rsync -r "$ROOT/smoke-nl/shards" "$W/shards"
  py -u "$C/bt_long.py" smoke-measures --logs "$W"/logs/*.log --shards "$W/shards" --candidates "$W/candidates" \
    --out "$W/smoke.json"
  mkdir -p "$W/slots"
  gsutil -m -q rsync -r "$ROOT/slots" "$W/slots"
  py - "$W/slots" "$C/models.yml" > "$W/slots.json" <<'PYEOF'
import json, os, sys
import yaml
models = yaml.safe_load(open(sys.argv[2]))["models"]
slots = {k: int(open(os.path.join(sys.argv[1], k)).read()) for k in sorted(os.listdir(sys.argv[1]))}
wanted = [c for m in ("gemma4-31b-bt-long", "gemma4-31b-rt-long") for c in models[m]["length_classes"]]
missing = [c for c in wanted if c not in slots]
if missing:
    sys.exit(f"no slots measured for {missing}")
print(json.dumps(slots))
PYEOF
  # once: the first VM's measures stand; a later smoke test only checks its own VM
  gsutil -q -h x-goog-if-generation-match:0 cp "$W/smoke.json" "$O/smoke.json" || echo "smoke.json was there"
  gsutil -q -h x-goog-if-generation-match:0 cp "$W/slots.json" "$O/slots.json" || echo "slots.json was there"
  cat "$W/slots.json"
  echo "STEP measure OK"
  ;;
plan)
  W=$SHM/plan-$RUN
  mkdir -p "$W/candidates" "$W/scores"
  gsutil -m -q rsync -r "$O/candidates" "$W/candidates"
  gsutil -m -q rsync -r "$O/scores" "$W/scores"
  gsutil -q cp "$O/smoke.json" "$W/smoke.json"
  py -u "$C/bt_long.py" choose --candidates "$W/candidates" --scores "$W/scores" --per-source "$TARGET" --out "$W/at-target"
  py -u "$C/bt_long.py" budget --smoke "$W/smoke.json" --report "$W/candidates/report.json" --selection "$W/at-target" \
    --worker-hours "$WORKER_HOURS" --out "$W/budget.json"
  py -u "$C/bt_long.py" choose --candidates "$W/candidates" --scores "$W/scores" --budget "$W/budget.json" \
    --out "$W/selection"
  py -u "$C/bt_gemma4.py" plan --data "$W/selection" --model gemma4-31b-bt-long --parts "$PARTS" --expect 0 \
    --check-prompt "$PROMPT" --tokenizer "$TOK" --out "$W/plan"
  gsutil -m -q rsync -r "$W/selection" "$O/selection"
  gsutil -q cp "$W/budget.json" "$O/budget.json"
  gsutil -m -q rsync -r "$W/plan" "$O/nl-en/plan"
  echo "STEP plan OK $(py -c "import json, sys; d = json.load(open(sys.argv[1])); print(len(d['units']), 'units,', d['passages'], 'passages, hash', d['hash'])" "$W/plan/units.json")"
  ;;
rtplan)
  W=$SHM/rtplan-$RUN
  mkdir -p "$W/selection" "$W/shards"
  gsutil -m -q rsync -r "$O/selection" "$W/selection"
  gsutil -m -q rsync -r "$O/nl-en/shards" "$W/shards"
  py -u "$C/bt_long.py" english --selection "$W/selection" --shards "$W/shards" --out "$W/english"
  py -u "$C/bt_gemma4.py" plan --data "$W/english" --text en --model gemma4-31b-rt-long --parts "$PARTS" --expect 0 \
    --check-prompt "$PROMPT" --tokenizer "$TOK" --out "$W/plan"
  gsutil -m -q rsync -r "$W/plan" "$O/rt/plan"
  echo "STEP rtplan OK $(py -c "import json, sys; d = json.load(open(sys.argv[1])); print(len(d['units']), 'units,', d['passages'], 'passages, hash', d['hash'])" "$W/plan/units.json")"
  ;;
build)
  W=$SHM/build-$RUN
  mkdir -p "$W/candidates" "$W/selection" "$W/en" "$W/rt"
  gsutil -q cp "$O/candidates/report.json" "$W/candidates/report.json"
  py -c "import json, sys; json.dump(json.load(open(sys.argv[1]))['heads'], open(sys.argv[2], 'w'))" \
    "$W/candidates/report.json" "$W/heads.json"
  HF_HUB_OFFLINE=0 py -u "$C/bt_long.py" fetch --heads "$W/heads.json" --out "$W/existing"
  gsutil -m -q rsync -r "$O/selection" "$W/selection"
  gsutil -m -q rsync -r "$O/nl-en/shards" "$W/en"
  gsutil -m -q rsync -r "$O/rt/shards" "$W/rt"
  py -u "$C/bt_long.py" build --selection "$W/selection" --existing "$W/existing" --en-shards "$W/en" \
    --rt-shards "$W/rt" --out "$W/build"
  gsutil -m -q rsync -r "$W/build" "$O/build"
  echo "STEP build OK"
  ;;
*)
  echo "usage: run_bt_long.sh setup | smoke <plan> <class>... | judge | work <plan> <class>... | measure | plan | rtplan | build"
  exit 1 ;;
esac

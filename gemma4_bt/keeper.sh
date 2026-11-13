#!/bin/bash
# On a control machine with gcloud and gsutil: keeps the spot v6e-8 VMs of the Gemma 4 back-translation going
# (run_bt_gemma4.sh on each) until every unit is on Hugging Face. All settings are environment variables:
#   PROJECT    the Google Cloud project of the TPU VMs
#   VMS        "name:zone ...", the VMs to keep (each in a zone of a region of REGIONS)
#   REGIONS    "region=dir,direction,ckpt ...": per region the gs:// directory of the run in a bucket of that region
#              (stage.sh put scripts/ there), forward or backward (one region each), and Gemma 4's MaxText checkpoint
#              in that region (<checkpoint>/0/items)
#   REPO       the private Hugging Face dataset of the shards
#   TPU_UP     a command (split on spaces) that requests a VM: called with --accelerator v6e-8 --zone <zone> --name <name>, and with
#              --replace for a VM that was preempted or stopped (the old request is deleted first)
#   TPU_DOWN   a command (split on spaces) that deletes a VM and its request: called with --name <name>
#   STATE      a directory for the keeper's state (default ~/logs/bt4_keeper_state)
#   REPORTS    a directory for copies of the selection and the final report (default $STATE/report)
#   POLL       seconds between cycles (default 60); RETRIES  restarts of a failed run per VM (default 2)
#   STEP       the step of run_bt_gemma4.sh each VM runs: all (default) or roundtrip
# TPU_UP and TPU_DOWN are where a quota guard belongs: the keeper never creates or deletes a VM in another way.
# The HF token goes from ~/.cache/huggingface/token here to each VM through stdin; it is never an argument.
# Each cycle:
#   - per VM at most one action:
#       region complete and the VM's run ended or the VM not READY  TPU_DOWN
#       no request                     TPU_UP
#       VM preempted/stopped/failed    TPU_UP --replace
#       VM READY, no run log           HF token through stdin, scripts from the region's bucket, run_bt_gemma4.sh STEP
#                                      detached
#       run log without EXIT, no run   (rebooted VM) the log moved aside, the next cycle starts again
#       run log ends EXIT 0            TPU_DOWN (its region is complete)
#       run log ends EXIT n > 0        started again at most RETRIES times, then left as it is (FAILED in the log)
#   - other-done.txt: per region the units with a shard in another region's bucket, written into the region's bucket
#     when it changed (from listings only; no VM reads another region's bucket)
#   - the probe's selection and the final report copied to REPORTS once they are there (small files)
# A region is complete once its bucket has full/region-complete-<region>.json, or once any such file counts every
# unit of the plan on Hugging Face. Ends (exit 0) once every region is complete and no VM of VMS is left.
set -uo pipefail
PROJECT=${PROJECT:?set PROJECT}
VMS=${VMS:?set VMS to "name:zone ..."}
REGIONS=${REGIONS:?set REGIONS to "region=gs://dir,forward|backward,gs://ckpt/0/items ..."}
REPO=${REPO:?set REPO to the private Hugging Face dataset}
TPU_UP=${TPU_UP:?set TPU_UP to the command that requests a TPU VM}
TPU_DOWN=${TPU_DOWN:?set TPU_DOWN to the command that deletes a TPU VM}
STATE=${STATE:-$HOME/logs/bt4_keeper_state}
REPORTS=${REPORTS:-$STATE/report}
POLL=${POLL:-60}
RETRIES=${RETRIES:-2}
STEP=${STEP:-all}
case $STEP in all|roundtrip) ;; *) echo "STEP is all or roundtrip, not '$STEP'"; exit 1 ;; esac
SHM=/dev/shm/bt_gemma4
mkdir -p "$STATE" "$REPORTS"
log() { echo "$(date -u +%FT%TZ) $*"; }
declare -A DIR DIRECTION CKPT
for spec in $REGIONS; do
  r=${spec%%=*}; IFS=, read -r d dir ck <<< "${spec#*=}"
  case $dir in forward|backward) ;; *) echo "REGIONS: $r has direction '$dir', not forward or backward"; exit 1 ;; esac
  [[ $d == gs://* && $ck == gs://* ]] || { echo "REGIONS: $r needs a gs:// directory and a gs:// checkpoint"; exit 1; }
  DIR[$r]=${d%/}; DIRECTION[$r]=$dir; CKPT[$r]=$ck
done
region_of() { local z=$1; echo "${z%-*}"; }
for spec in $VMS; do
  r=$(region_of "${spec##*:}")
  [ -n "${DIR[$r]:-}" ] || { echo "VMS: ${spec%%:*} is in $r, which REGIONS lacks"; exit 1; }
done
ZONES=$(for spec in $VMS; do echo "${spec##*:}"; done | sort -u)
[ -s ~/.cache/huggingface/token ] || { echo "no ~/.cache/huggingface/token here"; exit 1; }

ssh_vm() {  # name zone command
  timeout 180 gcloud compute tpus tpu-vm ssh "$1" --zone "$2" --project "$PROJECT" --quiet \
    --ssh-flag="-o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=4" --command "$3"
}

read_states() {  # "name qr vm" of every queued resource and VM in the zones of VMS; fails when a zone cannot be read
  local z qr vm
  for z in $ZONES; do
    qr=$(timeout 120 gcloud compute tpus queued-resources list --zone "$z" --project "$PROJECT" \
      --format="value(name.basename(),state.state)") || return 1
    vm=$(timeout 120 gcloud compute tpus tpu-vm list --zone "$z" --project "$PROJECT" \
      --format="value(name.basename(),state)") || return 1
    LC_ALL=C join -a 1 -a 2 -e none -o 0,1.2,2.2 <(echo "$qr" | grep . | LC_ALL=C sort -k1,1) \
      <(echo "$vm" | grep . | LC_ALL=C sort -k1,1)
  done
}

read_complete() {  # once a cycle: which regions are complete
  local r f all=no
  for r in "${!DIR[@]}"; do
    f=${DIR[$r]}/full/region-complete-$r.json
    timeout 60 gsutil cat "$f" 2> /dev/null | python3 -c "import json, sys; d = json.load(sys.stdin); sys.exit(0 if d['units_on_hf'] == d['units'] else 1)" 2> /dev/null && all=yes
  done
  declare -gA COMPLETE=()
  for r in "${!DIR[@]}"; do
    COMPLETE[$r]=no
    { [ $all = yes ] || timeout 60 gsutil -q stat "${DIR[$r]}/full/region-complete-$r.json"; } && COMPLETE[$r]=yes
  done
}

shards() {  # dir: the units with a shard there, sorted; fails on a listing error (none yet is no error)
  local out
  if ! out=$(timeout 120 gsutil ls "$1/full/shards/**" 2>&1); then
    grep -q "matched no objects" <<< "$out" || { log "listing $1 failed: $(echo "$out" | tail -n 1)" >&2; return 1; }
    out=""
  fi
  echo "$out" | sed -n "s|^$1/full/shards/\(.*\)\.parquet$|\1|p" | sort
}

sync_other_done() {
  local r o
  for r in "${!DIR[@]}"; do shards "${DIR[$r]}" > "$STATE/shards-$r.txt" || return; done
  for r in "${!DIR[@]}"; do
    for o in "${!DIR[@]}"; do [ "$o" = "$r" ] || cat "$STATE/shards-$o.txt"; done | sort -u \
      | comm -23 - "$STATE/shards-$r.txt" | grep . > "$STATE/other-$r.new"
    if ! cmp -s "$STATE/other-$r.new" "$STATE/other-$r.txt"; then
      timeout 60 gsutil -q cp "$STATE/other-$r.new" "${DIR[$r]}/full/other-done.txt" \
        && mv "$STATE/other-$r.new" "$STATE/other-$r.txt" \
        && log "other-done $r: $(wc -l < "$STATE/other-$r.txt") units of other regions"
    fi
  done
}

copy_reports() {
  local r f
  for r in "${!DIR[@]}"; do
    for f in probe/report/selection.md probe/report/selection.json full/report/bt-gemma4.md full/report/bt-gemma4.json; do
      [ -s "$REPORTS/$(basename "$f")" ] && [[ $f == full/* ]] && continue
      if timeout 60 gsutil -q stat "${DIR[$r]}/$f"; then
        timeout 60 gsutil -q cp "${DIR[$r]}/$f" "$REPORTS/$(basename "$f").tmp" && mv "$REPORTS/$(basename "$f").tmp" "$REPORTS/$(basename "$f")"
      fi
    done
  done
}

declare -A last
while true; do
  if ! states=$(read_states); then log "state read failed"; sleep "$POLL"; continue; fi
  read_complete
  left=0
  for spec in $VMS; do
    name=${spec%%:*}; zone=${spec##*:}; region=$(region_of "$zone")
    read -r qr vm < <(echo "$states" | awk -v n="$name" '$1 == n {print $2, $3}')
    qr=${qr:-none}; vm=${vm:-none}
    [ -f "$STATE/$name.gone" ] && [ "$qr $vm" = "none none" ] && continue
    [ "$qr $vm" != "none none" ] && left=$((left + 1))
    now="qr=$qr vm=$vm"
    [ "$now" != "${last[$name]:-}" ] && log "$name: $now"
    last[$name]=$now
    if [ "${COMPLETE[$region]}" = yes ] && [ "$vm" != READY ] && [ "$qr $vm" != "none none" ]; then
      log "$name: region complete, VM not READY: down"; $TPU_DOWN --name "$name"; touch "$STATE/$name.gone"; continue
    fi
    case "$qr $vm" in
      "none none")
        if [ "${COMPLETE[$region]}" = yes ]; then touch "$STATE/$name.gone"; continue; fi
        log "$name: requesting"; $TPU_UP --accelerator v6e-8 --zone "$zone" --name "$name" ;;
      *" PREEMPTED"|*" TERMINATED"|*" STOPPED"|"FAILED "*|"SUSPENDED "*)
        log "$name: re-requesting"; $TPU_UP --accelerator v6e-8 --zone "$zone" --name "$name" --replace ;;
      *" READY")
        out=$(ssh_vm "$name" "$zone" 'L=~/logs/bt4_run.log; P=~/logs/bt4_run.pid
          alive=no; [ -f $P ] && kill -0 "$(cat $P)" 2> /dev/null && alive=yes
          if [ -f $L ] && [ $alive = no ] && ! tail -n 1 $L | grep -q "^EXIT"; then mv $L ~/logs/bt4_run.$(date +%s).log; fi
          if [ -f $L ]; then echo "RUNLOG alive=$alive"; tail -n 2 $L | cut -c1-250; else echo NORUN; fi' < /dev/null 2> /dev/null)
        if grep -q '^NORUN' <<< "$out"; then
          o=${DIR[$region]}
          log "$name: new run"
          ssh_vm "$name" "$zone" 'mkdir -p ~/.cache/huggingface && umask 077 && cat > ~/.cache/huggingface/token' < ~/.cache/huggingface/token > /dev/null 2>&1 \
            && ssh_vm "$name" "$zone" "mkdir -p $SHM/code ~/logs && gsutil -m -q cp '$o/scripts/*' $SHM/code/ && (setsid nohup bash -c 'echo \$\$ > ~/logs/bt4_run.pid; O=$o DIRECTION=${DIRECTION[$region]} CKPT=${CKPT[$region]} REPO=$REPO bash $SHM/code/run_bt_gemma4.sh $STEP > ~/logs/bt4_run.log 2>&1; echo EXIT \$? >> ~/logs/bt4_run.log' > /dev/null 2>&1 < /dev/null &) && echo started" < /dev/null 2>&1 | tail -n 1
        elif grep -q '^RUNLOG' <<< "$out"; then
          tail_line=$(echo "$out" | grep -v '^RUNLOG' | tail -n 1)
          [ "$tail_line" != "$(cat "$STATE/$name.tail" 2> /dev/null)" ] && log "$name run: $tail_line"
          echo "$tail_line" > "$STATE/$name.tail"
          if grep -q '^EXIT 0$' <<< "$out"; then
            log "$name: run ended in EXIT 0: down"; $TPU_DOWN --name "$name"; touch "$STATE/$name.gone"
          elif grep -q '^EXIT [1-9]' <<< "$out"; then
            n=$(cat "$STATE/$name.retries" 2> /dev/null || echo 0)
            if [ "$n" -lt "$RETRIES" ]; then
              echo $((n + 1)) > "$STATE/$name.retries"
              log "$name: run FAILED ($(echo "$out" | tr '\n' ' ' | cut -c1-300)); started again ($((n + 1))/$RETRIES)"
              ssh_vm "$name" "$zone" 'mv ~/logs/bt4_run.log ~/logs/bt4_run.failed.$(date +%s).log' < /dev/null > /dev/null 2>&1
            else
              [ -f "$STATE/$name.failed" ] || log "$name: run FAILED, no retries left; VM left as it is"
              touch "$STATE/$name.failed"
            fi
          fi
        fi
        ;;
    esac
  done
  sync_other_done
  copy_reports
  done_all=yes
  for r in "${!DIR[@]}"; do [ "${COMPLETE[$r]}" = yes ] || done_all=no; done
  if [ $done_all = yes ] && [ "$left" = 0 ]; then
    copy_reports; log "every region complete, no VM left: keeper ends"; exit 0
  fi
  sleep "$POLL"
done

#!/bin/bash
# On a v6e-8 TPU VM: the environment of the Gemma 4 back-translation in /dev/shm/bt_gemma4. Each step is skipped when
# its result is there.
#   bash setup.sh
# venv: MaxText from git at a fixed commit (FORK_COMMIT of github.com/yhavinga/maxtext, branch
# fix-nnx-decode-cache-abstract-init: MaxText's main of 9 Nov 2026, d1c0d0afe, with the NNX decode cache made from
# shapes inside the init jit), installed with its tpu-requirements.txt; plus what the measures (sacrebleu, langid) and
# MetricX-24 (CPU torch, to read its pytorch_model*.bin) need. Without JetStream: gemma4_engine.py registers MaxText's
# stub ResultTokens as a pytree where it is not one yet. tokenizer-gemma4: Gemma 4 31B-it's tokenizer and chat template
# at the revision of models.yml. Needs ~/.cache/huggingface/token; the token is never printed.
set -euo pipefail
SHM=/dev/shm/bt_gemma4
FORK=https://github.com/yhavinga/maxtext.git
FORK_COMMIT=2249474b93b971d8248520e3a09b94b4ce626ce4
GEMMA4=google/gemma-4-31B-it
GEMMA4_REV=842da3794eaa0b77d5f08bae87a17459d91ff475
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=$SHM/uv-cache UV_PYTHON_INSTALL_DIR=$SHM/python HF_HUB_DISABLE_PROGRESS_BARS=1
[ -s ~/.cache/huggingface/token ] || { echo "$(hostname): no ~/.cache/huggingface/token"; exit 1; }
mkdir -p "$SHM" ~/logs
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
command -v uv > /dev/null || { echo "$(hostname): uv is not on PATH after installing it"; exit 1; }

V=$SHM/venv
if [ ! -s "$V/done.txt" ]; then
  [ -x "$V/bin/python" ] || uv venv -q --python 3.12 "$V"
  SRC=$SHM/maxtext-src
  [ -d "$SRC/.git" ] || git clone -q "$FORK" "$SRC"
  git -C "$SRC" fetch -q origin "$FORK_COMMIT"
  git -C "$SRC" checkout -q "$FORK_COMMIT"
  VIRTUAL_ENV=$V uv pip install -q -r "$SRC/src/dependencies/requirements/generated_requirements/tpu-requirements.txt"
  VIRTUAL_ENV=$V uv pip install -q --no-deps "maxtext @ git+$FORK@$FORK_COMMIT"
  VIRTUAL_ENV=$V uv pip install -q sacrebleu==2.5.1 langid==1.1.6 pytest
  VIRTUAL_ENV=$V UV_TORCH_BACKEND=cpu uv pip install -q torch==2.9.1
  "$V/bin/python" -c "import importlib.metadata as m, json; d = m.distribution('maxtext'); u = json.loads(d.read_text('direct_url.json') or '{}'); print({p: m.version(p) for p in ('maxtext', 'jax', 'jaxlib', 'libtpu', 'flax', 'transformers', 'tokenizers', 'pyarrow', 'torch')}, u.get('vcs_info', {}).get('commit_id'))" \
    | tee "$V/done.txt"
fi
cat "$V/done.txt"
T=$SHM/tokenizer-gemma4
if [ ! -s "$T/tokenizer.json" ]; then
  "$V/bin/python" -c "from huggingface_hub import snapshot_download as s; s('$GEMMA4', revision='$GEMMA4_REV', local_dir='$T', allow_patterns=['tokenizer*', 'chat_template.jinja', 'config.json', 'generation_config.json'])"
fi
echo "$(hostname): ready in $SHM, $(df -h /dev/shm | tail -1 | awk '{print $4}') free in /dev/shm"

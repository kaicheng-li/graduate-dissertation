#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
data_root="$(dirname -- "$PWD")"
export XDG_CACHE_HOME="$data_root/.cache"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
printf 'ROLE_SWAP_CACHE HF_HOME=%s VLLM_CACHE_ROOT=%s\n' "$HF_HOME" "$VLLM_CACHE_ROOT"
python -u -m scripts.run_role_swap_test \
  --manager-model "${MANAGER_MODEL:-}" --answer-model "${ANSWER_MODEL:-}" \
  --gpus "${TEST_GPUS:-0,1,2,3}" \
  --output-dir "${TEST_OUTPUT_DIR:-evaluation/results/role_swap_test}" "$@"

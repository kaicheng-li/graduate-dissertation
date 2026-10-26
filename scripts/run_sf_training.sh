#!/usr/bin/env bash
# One backend. Manager: sa/sf. Fixed-Manager data: answer-data. Answer: answer.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
kind="${1:-}"
case "$kind" in
  sa|sf|answer|answer-data|smoke) shift ;;
  *) echo 'Usage: bash scripts/run_sf_training.sh {sa|sf|answer-data|answer|smoke} [arguments]' >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export MEMORY_EMBEDDING_DEVICE=cpu
extra=()
[[ -z "${RESUME_FROM:-}" ]] || extra+=(--resume-from "$RESUME_FROM")
if [[ "$kind" == answer ]]; then
  : "${ANSWER_MODEL:?Set ANSWER_MODEL to the local base model directory}"
  [[ -z "${ANSWER_ADAPTER:-}" ]] || extra+=(--answer-adapter "$ANSWER_ADAPTER")
  python -u -m verl_memory.train --task answer --answer-model "$ANSWER_MODEL" \
    --output-dir "${TRAIN_OUTPUT:-output/answer_verl_seed42}" "${extra[@]}" "$@"
elif [[ "$kind" == answer-data ]]; then
  : "${MANAGER_MODEL:?Set MANAGER_MODEL}"
  : "${MANAGER_ADAPTER:?Set MANAGER_ADAPTER to the selected lora_adapter directory}"
  python -u scripts/build_answer_training_data.py --manager-model "$MANAGER_MODEL" \
    --manager-adapter "$MANAGER_ADAPTER" --output-dir "${ANSWER_DATA_OUTPUT:-data/answer_verl}" "$@"
else
  : "${MANAGER_MODEL:?Set MANAGER_MODEL}"
  : "${ANSWER_ADAPTER:?Set ANSWER_ADAPTER to the frozen trained Answer adapter}"
  [[ -z "${MANAGER_ADAPTER:-}" ]] || extra+=(--manager-adapter "$MANAGER_ADAPTER")
  task_kind="$kind"
  if [[ "$kind" == smoke ]]; then task_kind=sa; extra+=(--smoke-turns 4 --epochs 1); fi
  python -u -m verl_memory.train --task manager --kind "$task_kind" \
    --manager-model "$MANAGER_MODEL" --answer-model "${ANSWER_MODEL:-$MANAGER_MODEL}" \
    --answer-adapter "$ANSWER_ADAPTER" --output-dir "${TRAIN_OUTPUT:-output/${kind}_verl_seed42}" \
    "${extra[@]}" "$@"
fi

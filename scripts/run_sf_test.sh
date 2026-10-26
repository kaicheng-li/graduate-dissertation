#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
: "${MANAGER_MODEL:?Set MANAGER_MODEL}"
: "${MANAGER_ADAPTER:?Set MANAGER_ADAPTER to the selected Manager adapter directory}"
: "${ANSWER_ADAPTER:?Set ANSWER_ADAPTER to the selected Answer adapter directory}"
TEST_OUTPUT="${TEST_OUTPUT:-evaluation/results/test_$(date +%Y%m%d_%H%M%S).jsonl}"
python -u evaluation/evaluate_locomo.py \
  --timeline "${TEST_TIMELINE:-data/stage0/dialogue_timelines_test.json}" \
  --manager-model "$MANAGER_MODEL" --manager-adapter "$MANAGER_ADAPTER" \
  --answer-model "${ANSWER_MODEL:-$MANAGER_MODEL}" --answer-adapter "$ANSWER_ADAPTER" \
  --gpus "${TEST_GPUS:-0,1,2,3}" --output "$TEST_OUTPUT" "$@"
if [[ "${RUN_JUDGE:-0}" == 1 ]]; then
  : "${JUDGE_MODEL_PATH:?Set JUDGE_MODEL_PATH for the optional local judge}"
  python -u evaluation/score_judge.py "$TEST_OUTPUT" --backend local \
    --local-model-path "$JUDGE_MODEL_PATH" --gpus "${TEST_GPUS:-0,1,2,3}" \
    --max-tokens "${JUDGE_MAX_TOKENS:-2048}"
fi

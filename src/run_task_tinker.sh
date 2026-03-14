#!/bin/bash
#
# Run a PostTrainBench task using Tinker API instead of a local GPU.
# No container, no GPU required. Runs the agent directly on the host.
#
# Usage:
#   bash src/run_task_tinker.sh <benchmark> <agent> <model_to_train> <num_hours> <agent_config>
#
# Example:
#   bash src/run_task_tinker.sh gsm8k claude_tinker Qwen/Qwen3.5-4B 10 claude-sonnet-4-5
#

set -euo pipefail

EVALUATION_TASK="$1"
AGENT="$2"
MODEL_TO_TRAIN="$3"
NUM_HOURS="$4"
AGENT_CONFIG="$5"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Results directory
RESULTS_DIR="${POST_TRAIN_BENCH_RESULTS_DIR:-results}"
AGENT_CONFIG_SAFE=$(echo "$AGENT_CONFIG" | tr '/:' '_')
RESULT_PREFIX_SAFE=$(echo "$MODEL_TO_TRAIN" | tr '/:' '_')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

EVAL_DIR="${RESULTS_DIR}/${AGENT}_${AGENT_CONFIG_SAFE}_${NUM_HOURS}h_tinker/${EVALUATION_TASK}_${RESULT_PREFIX_SAFE}_${TIMESTAMP}"
mkdir -p "${EVAL_DIR}"

echo "=== PostTrainBench Tinker Runner ==="
echo "Task: ${EVALUATION_TASK}"
echo "Agent: ${AGENT} (${AGENT_CONFIG})"
echo "Model: ${MODEL_TO_TRAIN}"
echo "Hours: ${NUM_HOURS}"
echo "Results: ${EVAL_DIR}"
echo "====================================="

# Create task working directory
TASK_DIR="${EVAL_DIR}/task"
mkdir -p "${TASK_DIR}"

# Copy Tinker-based evaluate.py as evaluate.py
cp "${REPO_ROOT}/src/eval/tasks/${EVALUATION_TASK}/evaluate_tinker.py" "${TASK_DIR}/evaluate.py"

# Symlink tinker-cookbook repo for the agent to explore
TINKER_COOKBOOK_PATH="${TINKER_COOKBOOK_PATH:-$HOME/Repos/tinker-cookbook}"
if [ -d "$TINKER_COOKBOOK_PATH" ]; then
    ln -s "$TINKER_COOKBOOK_PATH" "${TASK_DIR}/tinker-cookbook"
else
    echo "WARNING: tinker-cookbook not found at $TINKER_COOKBOOK_PATH"
    echo "Set TINKER_COOKBOOK_PATH to the correct location."
    exit 1
fi

# Generate system prompt
export POST_TRAIN_BENCH_PROMPT="prompt_tinker"
PROMPT=$(python "${REPO_ROOT}/src/eval/general/get_prompt.py" \
    --model-to-train "$MODEL_TO_TRAIN" \
    --benchmark-id "$EVALUATION_TASK" \
    --num-hours "$NUM_HOURS" \
    --agent "${AGENT}")
echo "$PROMPT" > "${EVAL_DIR}/prompt.txt"

# Create timer
bash "${REPO_ROOT}/src/utils/create_timer.sh" "$NUM_HOURS" "${TASK_DIR}/timer.sh"

# Verify Tinker API key is set
if [ -z "${TINKER_API_KEY:-}" ]; then
    echo "ERROR: TINKER_API_KEY is not set."
    exit 1
fi

# Export environment for the agent
export AGENT_CONFIG="$AGENT_CONFIG"
export PROMPT="$PROMPT"

# Copy agent solve script
cp "${REPO_ROOT}/agents/${AGENT}/solve.sh" "${EVAL_DIR}/agent_solve.sh"

echo "=== Starting agent ==="
echo "Working directory: ${TASK_DIR}"

# Run the agent from the task directory
cd "${TASK_DIR}"

SOLVE_OUT="${EVAL_DIR}/solve_out.txt"
START_TIME=$(date +%s)

timeout --signal=TERM --kill-after=30s "$((NUM_HOURS * 60 + 5))m" \
    bash "${EVAL_DIR}/agent_solve.sh" > "${SOLVE_OUT}" 2>&1
SOLVE_EXIT=$?

END_TIME=$(date +%s)
TIME_TAKEN=$((END_TIME - START_TIME))
printf '%02d:%02d:%02d\n' \
    $((TIME_TAKEN / 3600)) \
    $(((TIME_TAKEN % 3600) / 60)) \
    $((TIME_TAKEN % 60)) > "${EVAL_DIR}/time_taken.txt"

echo "--- SOLVE DIAGNOSTICS ---"
echo "exit_code: $SOLVE_EXIT"
if [ $SOLVE_EXIT -eq 0 ]; then
    echo "status: exited normally"
elif [ $SOLVE_EXIT -eq 124 ]; then
    echo "status: killed by timeout (reached ${NUM_HOURS}h limit)"
else
    echo "status: exited with error code $SOLVE_EXIT"
fi
echo "time_taken: $(cat ${EVAL_DIR}/time_taken.txt)"

# Parse agent trace
cd "${REPO_ROOT}"
TRACE_PARSER="agents/${AGENT}/human_readable_trace.py"
if [ -f "$TRACE_PARSER" ]; then
    python "$TRACE_PARSER" "${SOLVE_OUT}" -o "${EVAL_DIR}/solve_parsed.txt" || true
fi

# Check if agent produced a final model path
if [ -f "${TASK_DIR}/final_model_path.txt" ]; then
    FINAL_MODEL_PATH=$(cat "${TASK_DIR}/final_model_path.txt")
    echo "=== Final model path: ${FINAL_MODEL_PATH} ==="
    cp "${TASK_DIR}/final_model_path.txt" "${EVAL_DIR}/final_model_path.txt"
else
    echo "WARNING: No final_model_path.txt found."
fi

echo "=== DONE ==="
echo "Results saved to: ${EVAL_DIR}"

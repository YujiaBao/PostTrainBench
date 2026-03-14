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
# Required environment variables:
#   TINKER_API_KEY      - API key for the Tinker service
#   ANTHROPIC_API_KEY   - API key for the Claude agent
#
# Optional environment variables:
#   TINKER_COOKBOOK_PATH         - Path to tinker-cookbook repo (default: ~/Repos/tinker-cookbook)
#   POST_TRAIN_BENCH_RESULTS_DIR - Results directory (default: results)
#   CODEX_API_KEY               - Required for contamination checking (optional but recommended)
#   OPENAI_API_KEY              - Required for arenahardwriting/healthbench evaluation

set -euo pipefail

EVALUATION_TASK="$1"
AGENT="$2"
MODEL_TO_TRAIN="$3"
NUM_HOURS="$4"
AGENT_CONFIG="$5"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Read benchmark display name
BENCHMARK=""
if [ -f "${REPO_ROOT}/src/eval/tasks/${EVALUATION_TASK}/benchmark.txt" ]; then
    BENCHMARK=$(cat "${REPO_ROOT}/src/eval/tasks/${EVALUATION_TASK}/benchmark.txt")
else
    BENCHMARK="${EVALUATION_TASK}"
fi

# Results directory
RESULTS_DIR="${POST_TRAIN_BENCH_RESULTS_DIR:-results}"
AGENT_CONFIG_SAFE=$(echo "$AGENT_CONFIG" | tr '/:' '_')
RESULT_PREFIX_SAFE=$(echo "$MODEL_TO_TRAIN" | tr '/:' '_')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

EVAL_DIR="${RESULTS_DIR}/${AGENT}_${AGENT_CONFIG_SAFE}_${NUM_HOURS}h_tinker/${EVALUATION_TASK}_${RESULT_PREFIX_SAFE}_${TIMESTAMP}"
mkdir -p "${EVAL_DIR}"

echo "=== PostTrainBench Tinker Runner ==="
echo "Task: ${EVALUATION_TASK} (${BENCHMARK})"
echo "Agent: ${AGENT} (${AGENT_CONFIG})"
echo "Model: ${MODEL_TO_TRAIN}"
echo "Hours: ${NUM_HOURS}"
echo "Results: ${EVAL_DIR}"
echo "====================================="

# ---------------------------------------------------------------------------
# Phase 1: Task preparation
# ---------------------------------------------------------------------------

# Create task working directory
TASK_DIR="${EVAL_DIR}/task"
mkdir -p "${TASK_DIR}"

# Copy Tinker-based evaluate script as evaluate.py (what the agent sees)
cp "${REPO_ROOT}/src/eval/tasks/${EVALUATION_TASK}/evaluate_tinker.py" "${TASK_DIR}/evaluate.py"

# Copy task_context files if they exist (e.g. bfcl evaluation code)
if [ -d "${REPO_ROOT}/src/eval/tasks/${EVALUATION_TASK}/task_context" ]; then
    cp -r "${REPO_ROOT}/src/eval/tasks/${EVALUATION_TASK}/task_context/"* "${TASK_DIR}/"
    echo "[prep] Copied task_context into task directory."
fi

# Copy evaluation_code if it exists (e.g. healthbench, arenahardwriting)
if [ -d "${REPO_ROOT}/src/eval/tasks/${EVALUATION_TASK}/evaluation_code" ]; then
    cp -r "${REPO_ROOT}/src/eval/tasks/${EVALUATION_TASK}/evaluation_code" "${TASK_DIR}/"
    echo "[prep] Copied evaluation_code into task directory."
fi

# Symlink tinker-cookbook repo for the agent to explore
TINKER_COOKBOOK_PATH="${TINKER_COOKBOOK_PATH:-$HOME/Repos/tinker-cookbook}"
if [ -d "$TINKER_COOKBOOK_PATH" ]; then
    ln -sf "$TINKER_COOKBOOK_PATH" "${TASK_DIR}/tinker-cookbook"
else
    echo "ERROR: tinker-cookbook not found at $TINKER_COOKBOOK_PATH"
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

# ---------------------------------------------------------------------------
# Phase 2: Agent execution
# ---------------------------------------------------------------------------

echo "=== Starting agent ==="
echo "Working directory: ${TASK_DIR}"

cd "${TASK_DIR}"

SOLVE_OUT="${EVAL_DIR}/solve_out.txt"
START_TIME=$(date +%s)

set +e
timeout --signal=TERM --kill-after=30s "$((NUM_HOURS * 60 + 5))m" \
    bash "${EVAL_DIR}/agent_solve.sh" > "${SOLVE_OUT}" 2>&1
SOLVE_EXIT=$?
set -e

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
echo "time_taken: $(cat "${EVAL_DIR}/time_taken.txt")"

# Parse agent trace
cd "${REPO_ROOT}"
TRACE_PARSER="agents/${AGENT}/human_readable_trace.py"
if [ -f "$TRACE_PARSER" ]; then
    python "$TRACE_PARSER" "${SOLVE_OUT}" -o "${EVAL_DIR}/solve_parsed.txt" || true
fi

# Check if agent produced a final model path
FINAL_MODEL_PATH=""
if [ -f "${TASK_DIR}/final_model_path.txt" ]; then
    FINAL_MODEL_PATH=$(cat "${TASK_DIR}/final_model_path.txt")
    echo "=== Final model path: ${FINAL_MODEL_PATH} ==="
    cp "${TASK_DIR}/final_model_path.txt" "${EVAL_DIR}/final_model_path.txt"
else
    echo "WARNING: No final_model_path.txt found. Skipping evaluation and contamination check."
fi

# ---------------------------------------------------------------------------
# Phase 3: Contamination check
# ---------------------------------------------------------------------------

if [ -n "${CODEX_API_KEY:-}" ] && [ -n "$FINAL_MODEL_PATH" ]; then
    echo "=== Running contamination check ==="

    JUDGE_TASK=$(python "${REPO_ROOT}/src/disallowed_usage_judge/get_judge_prompt.py" \
        --benchmark "${BENCHMARK}" --model "${MODEL_TO_TRAIN}")

    cd "${TASK_DIR}"
    set +e
    codex --search -a never exec --json \
        -c model_reasoning_summary=detailed \
        --skip-git-repo-check --yolo \
        --model "gpt-5.1-codex" \
        "$JUDGE_TASK" 2>&1 | tee "${EVAL_DIR}/judge_output.json"
    JUDGE_EXIT=$?
    set -e
    cd "${REPO_ROOT}"

    if [ $JUDGE_EXIT -eq 0 ]; then
        # Copy judgement files if the judge created them
        if [ -f "${TASK_DIR}/contamination_judgement.txt" ]; then
            cp "${TASK_DIR}/contamination_judgement.txt" "${EVAL_DIR}/contamination_judgement.txt"
            echo "Contamination judgement: $(cat "${EVAL_DIR}/contamination_judgement.txt")"
        fi
        if [ -f "${TASK_DIR}/disallowed_model_judgement.txt" ]; then
            cp "${TASK_DIR}/disallowed_model_judgement.txt" "${EVAL_DIR}/disallowed_model_judgement.txt"
            echo "Model judgement: $(cat "${EVAL_DIR}/disallowed_model_judgement.txt")"
        fi
    else
        echo "WARNING: Contamination judge exited with code $JUDGE_EXIT"
    fi
elif [ -z "${CODEX_API_KEY:-}" ] && [ -n "$FINAL_MODEL_PATH" ]; then
    echo "WARNING: CODEX_API_KEY not set, skipping contamination check."
    echo "Set CODEX_API_KEY to enable data contamination and model substitution detection."
fi

# ---------------------------------------------------------------------------
# Phase 4: Post-agent evaluation (full benchmark)
# ---------------------------------------------------------------------------

if [ -n "$FINAL_MODEL_PATH" ] && [ ! -f "${EVAL_DIR}/metrics.json" ]; then
    echo "=== Running final evaluation ==="

    EVAL_COUNTER=0

    run_tinker_evaluation() {
        local max_tokens_arg="$1"
        local eval_num="$2"

        echo "Evaluation attempt ${eval_num} ${max_tokens_arg:+(with $max_tokens_arg)}"

        cd "${REPO_ROOT}/src/eval/tasks/${EVALUATION_TASK}"
        set +e
        timeout --signal=TERM --kill-after=60s 28800s \
            python evaluate_tinker.py \
                --model-path "$FINAL_MODEL_PATH" \
                --base-model "$MODEL_TO_TRAIN" \
                --limit -1 \
                ${max_tokens_arg} \
                --json-output-file "${EVAL_DIR}/metrics.json" \
                > "${EVAL_DIR}/final_eval_${eval_num}.txt" 2>&1
        local eval_exit=$?
        set -e
        cd "${REPO_ROOT}"

        if [ $eval_exit -ne 0 ]; then
            echo "Evaluation attempt ${eval_num} failed with exit code ${eval_exit}"
        fi
        return $eval_exit
    }

    run_evaluation_with_retry() {
        local max_retries="$1"
        local max_tokens_arg="$2"

        for ((attempt=1; attempt<=max_retries; attempt++)); do
            if [ -f "${EVAL_DIR}/metrics.json" ]; then
                return 0
            fi

            EVAL_COUNTER=$((EVAL_COUNTER + 1))
            run_tinker_evaluation "$max_tokens_arg" "$EVAL_COUNTER" || true

            if [ -f "${EVAL_DIR}/metrics.json" ]; then
                return 0
            fi
            sleep 5
        done
        return 1
    }

    # Phase 1: default parameters (3 attempts)
    run_evaluation_with_retry 3 "" || true

    # Phase 2: reduced max-tokens (2 attempts)
    if [ ! -f "${EVAL_DIR}/metrics.json" ]; then
        case "$EVALUATION_TASK" in
            aime2025)       MAX_TOKENS_ARG="--max-tokens 12000" ;;
            arenahardwriting) MAX_TOKENS_ARG="--max-new-tokens 12288" ;;
            bfcl)           MAX_TOKENS_ARG="--max-tokens 12000" ;;
            gpqamain)       MAX_TOKENS_ARG="--max-tokens 12000" ;;
            gsm8k)          MAX_TOKENS_ARG="--max-tokens 3000" ;;
            healthbench)    MAX_TOKENS_ARG="--max-new-tokens 12288" ;;
            humaneval)      MAX_TOKENS_ARG="--max-tokens 3000" ;;
            *)              MAX_TOKENS_ARG="" ;;
        esac
        run_evaluation_with_retry 2 "$MAX_TOKENS_ARG" || true
    fi

    # Phase 3: further reduced max-tokens (2 attempts)
    if [ ! -f "${EVAL_DIR}/metrics.json" ]; then
        case "$EVALUATION_TASK" in
            aime2025)       MAX_TOKENS_ARG="--max-tokens 8000" ;;
            arenahardwriting) MAX_TOKENS_ARG="--max-new-tokens 8192" ;;
            bfcl)           MAX_TOKENS_ARG="--max-tokens 8000" ;;
            gpqamain)       MAX_TOKENS_ARG="--max-tokens 8000" ;;
            gsm8k)          MAX_TOKENS_ARG="--max-tokens 2000" ;;
            healthbench)    MAX_TOKENS_ARG="--max-new-tokens 8192" ;;
            humaneval)      MAX_TOKENS_ARG="--max-tokens 2000" ;;
            *)              MAX_TOKENS_ARG="" ;;
        esac
        run_evaluation_with_retry 2 "$MAX_TOKENS_ARG" || true
    fi

    if [ -f "${EVAL_DIR}/metrics.json" ]; then
        echo "=== Evaluation complete ==="
        cat "${EVAL_DIR}/metrics.json"
    else
        echo "WARNING: All evaluation attempts failed. Check final_eval_*.txt logs."
    fi
else
    if [ -f "${EVAL_DIR}/metrics.json" ]; then
        echo "=== metrics.json already exists (agent produced it), skipping final eval ==="
        cat "${EVAL_DIR}/metrics.json"
    fi
fi

echo "=== DONE ==="
echo "Results saved to: ${EVAL_DIR}"

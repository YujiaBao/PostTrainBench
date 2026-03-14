# Tinker Mode Implementation Summary

This document summarizes the changes made to bring the Tinker API execution mode to full parity with the container-based (H100) baseline.

## Context

PostTrainBench evaluates LLM agents' ability to post-train base models. The original pipeline runs agents inside Apptainer containers with local H100 GPU access and vLLM for inference. The Tinker mode replaces local GPU access with the Tinker API for remote training and inference — no GPU or container required.

## Changes

### 1. Tinker evaluation scripts for all 7 benchmarks

Previously only `gsm8k` had an `evaluate_tinker.py`. Now all 7 benchmarks are covered:

| Benchmark | File | Approach |
|-----------|------|----------|
| aime2025 | `src/eval/tasks/aime2025/evaluate_tinker.py` | Inspect-AI + Tinker sampling |
| arenahardwriting | `src/eval/tasks/arenahardwriting/evaluate_tinker.py` | Tinker sampling for generation, OpenAI judging preserved |
| bfcl | `src/eval/tasks/bfcl/evaluate_tinker.py` | Inspect-AI + Tinker sampling; tool calling via inspect-ai framework |
| gpqamain | `src/eval/tasks/gpqamain/evaluate_tinker.py` | Inspect-AI + Tinker sampling; custom `@task` with `record_to_sample` |
| gsm8k | `src/eval/tasks/gsm8k/evaluate_tinker.py` | *(already existed)* |
| healthbench | `src/eval/tasks/healthbench/evaluate_tinker.py` | Tinker sampling for generation, OpenAI grading preserved |
| humaneval | `src/eval/tasks/humaneval/evaluate_tinker.py` | Inspect-AI + Tinker sampling; `sandbox="local"` preserved |

**Inspect-AI based tasks** (aime2025, bfcl, gpqamain, humaneval, gsm8k) follow a uniform pattern: create a `tinker.ServiceClient`, wrap it in `InspectAPIFromTinkerSampling` from tinker-cookbook, then run `eval_async()`.

**Custom tasks** (arenahardwriting, healthbench) replace the `VLLMServer` class with `InspectAPIFromTinkerSampling` + `InspectAIModel` for generation. The OpenAI-based judging/grading pipelines are unchanged. These scripts were converted to async to support the Tinker API.

### 2. Hardened `run_task_tinker.sh`

The runner script was missing several steps that the container-mode `run_task.sh` performs. These have been added:

- **Copy `task_context/`** into the task directory (needed by bfcl).
- **Copy `evaluation_code/`** into the task directory (needed by arenahardwriting, healthbench).
- **Read `benchmark.txt`** for proper display names in the contamination judge prompt.
- **Contamination check** via codex CLI, run directly on the host when `CODEX_API_KEY` is set. Produces `contamination_judgement.txt` and `disallowed_model_judgement.txt`.
- **Post-agent evaluation** with 3-phase retry logic matching container mode (up to 7 attempts, decreasing `--max-tokens` per phase).
- **Fixed `set +e`** around agent execution so the script doesn't abort on non-zero agent exit codes.

### 3. Fairness with H100 baseline

The Tinker mode maintains fairness:

- Same 10-hour time limit and timer utility.
- Agent has no local GPU — all training and inference goes through the Tinker API.
- Same contamination and model-substitution detection.
- Same post-agent evaluation with `--limit -1` (full benchmark).
- Same evaluation metrics and scoring logic.

## Prerequisites for running

```bash
# Required
export TINKER_API_KEY="..."
export ANTHROPIC_API_KEY="..."

# Optional but recommended
export CODEX_API_KEY="..."       # contamination check
export OPENAI_API_KEY="..."      # arenahardwriting/healthbench judging

# tinker-cookbook location
export TINKER_COOKBOOK_PATH="$HOME/Repos/tinker-cookbook"

# Install dependencies
uv pip install tinker
uv pip install -e "$TINKER_COOKBOOK_PATH"
```

## Running

```bash
bash src/run_task_tinker.sh <benchmark> <agent> <model> <hours> <agent_config>

# Example
bash src/run_task_tinker.sh gsm8k claude_tinker Qwen/Qwen3.5-4B 10 claude-sonnet-4-5
```

Recommended to run in a tmux session for visibility into agent actions.

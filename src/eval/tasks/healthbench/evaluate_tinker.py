#!/usr/bin/env python3
# IMPORTANT: You are NOT allowed to use the OpenAI API for anything but this evaluation script.
"""
Tinker-based evaluation script for HealthBench.

Drop-in replacement for the vLLM-based evaluate.py, using Tinker's sampling API
instead of a local GPU. The OpenAI-based grading pipeline is preserved as-is.
"""
import os
import argparse
import asyncio
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

import tinker
from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling
from tinker_cookbook.model_info import get_recommended_renderer_name

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig as InspectAIGenerateConfig,
    Model as InspectAIModel,
)

from evaluation_code.data_loader import load_healthbench, HealthBenchExample
from evaluation_code.grader import grade_examples_parallel, ExampleResult
from evaluation_code.scoring import aggregate_scores, BenchmarkResult
from evaluation_code.text_utils import limit_repetitions


DEFAULT_JUDGE_WORKERS = 64
JUDGE_MODEL = "gpt-5-mini"


def _model_alias(model_path: str) -> str:
    if os.path.isdir(model_path):
        return Path(model_path).name
    return model_path.split("/")[-1]


def _convert_messages(conversation: List[Dict]):
    """Convert dict messages to inspect-ai ChatMessage objects."""
    result = []
    for msg in conversation:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            result.append(ChatMessageUser(content=content))
        elif role == "assistant":
            result.append(ChatMessageAssistant(content=content))
        elif role == "system":
            result.append(ChatMessageSystem(content=content))
    return result


async def generate_answers(
    args,
    examples: List[HealthBenchExample],
    model: InspectAIModel,
) -> List[str]:
    """Generate model responses for all examples using Tinker sampling API."""
    print(f"[generate] Generating answers for {len(examples)} examples via Tinker API.")

    responses = []
    for example in tqdm(examples, desc="Generating answers"):
        messages = _convert_messages(example.conversation)

        output = await model.generate(messages)
        answer_text = output.completion.strip()

        # Strip thinking tags if present (for reasoning models)
        if answer_text.startswith("<think>"):
            answer_text = answer_text.split("</think>", maxsplit=1)[-1].strip()

        answer_text = limit_repetitions(answer_text)
        responses.append(answer_text)

    return responses


def _compute_metrics(results: List[ExampleResult], examples: List[HealthBenchExample]) -> Dict:
    benchmark_result = aggregate_scores(results, examples)
    return {
        "accuracy": benchmark_result.accuracy,
        "stderr": benchmark_result.stderr,
        "n_examples": benchmark_result.n_examples,
        "total_grader_calls": benchmark_result.total_grader_calls,
        "by_theme": benchmark_result.by_theme,
        "by_axis": benchmark_result.by_axis,
    }


async def async_main(args) -> None:
    """Async entry point: set up Tinker model, generate answers, then grade."""
    service_client = tinker.ServiceClient()

    sampling_client = service_client.create_sampling_client(
        model_path=args.model_path,
        base_model=args.base_model,
    )

    renderer_name = get_recommended_renderer_name(args.base_model)
    print(f"[setup] Base model: {args.base_model}, renderer: {renderer_name}")
    if args.model_path:
        print(f"[setup] Fine-tuned checkpoint: {args.model_path}")

    api = InspectAPIFromTinkerSampling(
        renderer_name=renderer_name,
        model_name=args.base_model,
        sampling_client=sampling_client,
        verbose=False,
    )

    model = InspectAIModel(
        api=api,
        config=InspectAIGenerateConfig(
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
            top_p=1.0,
        ),
    )

    model_alias = _model_alias(args.model_path or args.base_model)

    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Please export your OpenAI API key before running."
        )

    # Load data
    print("[data] Loading HealthBench dataset...")
    examples = load_healthbench()
    random.Random(42).shuffle(examples)
    if args.limit != -1:
        examples = examples[: args.limit]

    # Generate answers
    responses = await generate_answers(args, examples, model)
    print(f"[generate] Generated {len(responses)} responses")

    # Save model outputs if requested
    if args.store_outputs:
        output_dir = Path(__file__).parent / "evaluation_code" / "data" / "model_answer"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{model_alias}.jsonl"
        print(f"[generate] Saving model outputs to {output_path}")
        with open(output_path, "w", encoding="utf-8") as fout:
            for example, response in zip(examples, responses):
                record = {
                    "example_id": example.example_id,
                    "model": model_alias,
                    "conversation": example.conversation,
                    "response": response,
                    "tstamp": time.time(),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Grade responses
    print("[judge] Grading responses...")
    pbar = tqdm(total=len(examples), desc="Judging answers")

    def update_progress(completed, total):
        pbar.n = completed
        pbar.refresh()

    results = grade_examples_parallel(
        examples=examples,
        responses=responses,
        grader_model=JUDGE_MODEL,
        example_workers=min(4, len(examples)),
        criteria_workers=8,
        max_concurrent_requests=args.judge_workers,
        progress_callback=update_progress,
    )
    pbar.close()

    # Compute metrics
    metrics = _compute_metrics(results, examples)

    print(f"\n[done] Evaluation Complete")
    print(f"  Model: {model_alias}")
    print(f"  Examples: {metrics['n_examples']}")
    print(f"  Accuracy: {metrics['accuracy']:.4f} (+/-{metrics['stderr']:.4f})")
    print(f"  Grader calls: {metrics['total_grader_calls']}")

    if metrics['by_theme']:
        print(f"\n  By Theme:")
        for theme, score in sorted(metrics['by_theme'].items()):
            print(f"    {theme}: {score:.4f}")

    if metrics['by_axis']:
        print(f"\n  By Axis:")
        for axis, score in sorted(metrics['by_axis'].items()):
            print(f"    {axis}: {score:.4f}")

    if args.json_output_file is not None:
        with open(args.json_output_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n[done] Metrics saved to {args.json_output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run HealthBench evaluation via Tinker API.")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Tinker model path (e.g. tinker://...) for a fine-tuned checkpoint.")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3.5-4B",
                        help="Base model name on Tinker.")
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--limit", type=int, default=32,
                        help="Limit number of examples for quicker runs.")
    parser.add_argument("--judge-workers", type=int, default=DEFAULT_JUDGE_WORKERS,
                        help="Number of concurrent judge jobs.")
    parser.add_argument('--json-output-file', type=str, default=None,
                        help="Optional path to output metrics as JSON.")
    parser.add_argument('--store-outputs', action='store_true',
                        help="Store model answers to disk.")
    parser.add_argument('--temperature', type=float, default=0.6,
                        help="Sampling temperature.")
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()

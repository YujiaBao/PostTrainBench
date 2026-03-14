#!/usr/bin/env python3
"""
Tinker-based evaluation script for AIME 2025.

Drop-in replacement for the vLLM-based evaluate.py, using Tinker's sampling API
instead of a local GPU.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os

import tinker
from inspect_ai import eval_async
from inspect_ai.model import GenerateConfig as InspectAIGenerateConfig
from inspect_ai.model import Model as InspectAIModel
from inspect_ai.util._display import init_display_type
from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling
from tinker_cookbook.model_info import get_recommended_renderer_name

import inspect_evals.aime2025  # noqa: F401  (registers task definitions)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AIME 2025 eval via Tinker sampling API.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Tinker model path (e.g. tinker://...) for a fine-tuned checkpoint.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3.5-4B",
        help="Base model name on Tinker.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to evaluate. Use -1 for all.",
    )
    parser.add_argument(
        "--json-output-file",
        type=str,
        default=None,
        help="Optional path to output metrics as JSON.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16000,
        help="Max tokens for generation.",
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        default=512,
        help="Max concurrent sampling requests to Tinker.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature.",
    )
    return parser.parse_args()


async def run_eval(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO)
    init_display_type("plain")

    service_client = tinker.ServiceClient()

    model_path = args.model_path
    base_model = args.base_model

    sampling_client = service_client.create_sampling_client(
        model_path=model_path,
        base_model=base_model,
    )

    renderer_name = get_recommended_renderer_name(base_model)
    logger.info(f"Using base model: {base_model}, renderer: {renderer_name}")
    if model_path:
        logger.info(f"Using fine-tuned checkpoint: {model_path}")

    api = InspectAPIFromTinkerSampling(
        renderer_name=renderer_name,
        model_name=base_model,
        sampling_client=sampling_client,
        verbose=False,
    )

    model = InspectAIModel(
        api=api,
        config=InspectAIGenerateConfig(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=1.0,
        ),
    )

    other_kwargs = {}
    if args.limit is not None and args.limit != -1:
        other_kwargs["limit"] = args.limit

    eval_out = await eval_async(
        "inspect_evals/aime2025",
        model=[model],
        score_display=False,
        log_realtime=False,
        log_format="json",
        log_dir=os.path.expanduser("~/inspect-logs"),
        timeout=18000000,
        max_connections=args.max_connections,
        retry_on_error=0,
        fail_on_error=False,
        **other_kwargs,
    )

    assert len(eval_out) == 1, eval_out
    assert len(eval_out[0].results.scores) == 1, eval_out[0].results.scores
    metrics = {}
    for k, v in eval_out[0].results.scores[0].metrics.items():
        metrics[k] = v.value

    logger.info(f"Evaluation metrics: {metrics}")

    if args.json_output_file is not None:
        with open(args.json_output_file, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics written to {args.json_output_file}")


def main() -> None:
    args = parse_args()
    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()

# IMPORTANT: You are NOT allowed to use the OpenAI API for anything but this evaluation script.
"""
Tinker-based evaluation script for Arena-Hard-v2.0 (Writing).

Drop-in replacement for the vLLM-based evaluate.py, using Tinker's sampling API
instead of a local GPU. The OpenAI-based judging pipeline is preserved as-is.
"""
import os

import argparse
import asyncio
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from concurrent.futures import ThreadPoolExecutor, as_completed

import shortuuid
import tiktoken
from tqdm import tqdm

import tinker
from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling
from tinker_cookbook.model_info import get_recommended_renderer_name

from inspect_ai.model import (
    ChatMessageUser,
    GenerateConfig as InspectAIGenerateConfig,
    Model as InspectAIModel,
)

from evaluation_code.utils.add_markdown_info import count_markdown_elements, remove_pattern
from evaluation_code.utils.completion import (
    load_model_answers,
    load_questions,
    make_config,
)
from evaluation_code.utils.judge_utils import JUDGE_SETTINGS
from evaluation_code.show_result import load_judgments, print_leaderboard


API_MAX_RETRY = 3
API_RETRY_SLEEP = 5
DEFAULT_JUDGE_WORKERS = 64
MAX_REPETITIONS = 5

BENCHMARK = "arena-hard-v2.0"
JUDGE_MODEL = "gpt-5-mini"
REASONING_EFFORT = "medium"
JUDGE_CONFIG = "evaluation_code/config/arena-hard-v2.0.yaml"
JUDGE_MAX_COMPLETION = 49152
DATA_PATH = Path("evaluation_code/data/" + BENCHMARK)


def limit_repetitions(text: str, max_reps: int = MAX_REPETITIONS) -> str:
    """Limit repetitive patterns in generated text to at most max_reps repetitions."""

    def _limit_consecutive_lines(txt: str) -> tuple:
        lines = txt.split('\n')
        result = []
        i = 0
        modified = False
        while i < len(lines):
            line = lines[i]
            count = 1
            j = i + 1
            while j < len(lines) and lines[j] == line:
                count += 1
                j += 1
            if count > max_reps:
                result.extend([line] * max_reps)
                modified = True
            else:
                result.extend([line] * count)
            i = j
        return '\n'.join(result), modified

    def _limit_block_patterns(txt: str) -> tuple:
        lines = txt.split('\n')
        modified = False
        for block_size in range(2, min(30, len(lines) // 5 + 1)):
            i = 0
            new_lines = []
            last_end = 0
            while i <= len(lines) - block_size:
                block = lines[i:i + block_size]
                count = 1
                j = i + block_size
                while j + block_size <= len(lines):
                    next_block = lines[j:j + block_size]
                    if next_block == block:
                        count += 1
                        j += block_size
                    else:
                        break
                if count > max_reps:
                    new_lines.extend(lines[last_end:i])
                    for _ in range(max_reps):
                        new_lines.extend(block)
                    remaining_lines = lines[j:]
                    if remaining_lines:
                        is_partial_repeat = True
                        for k, rem_line in enumerate(remaining_lines):
                            if k >= len(block):
                                is_partial_repeat = False
                                break
                            block_line = block[k % len(block)]
                            if rem_line != block_line:
                                if not (rem_line.strip() and block_line.startswith(rem_line.strip())):
                                    if not (block_line.strip() and rem_line.startswith(block_line.strip())):
                                        is_partial_repeat = False
                                        break
                        if is_partial_repeat:
                            last_end = len(lines)
                        else:
                            last_end = j
                    else:
                        last_end = j
                    i = len(lines)
                    modified = True
                else:
                    i += 1
            if modified:
                new_lines.extend(lines[last_end:])
                return '\n'.join(new_lines), True
        return txt, False

    def _limit_regex_patterns(txt: str) -> tuple:
        modified = False
        while True:
            changed_this_round = False
            for min_len, max_len in [(3, 30), (30, 60), (60, 120), (120, 250), (250, 500)]:
                pattern = rf'(.{{{min_len},{max_len}}}?)\1{{{max_reps},}}'

                def replace_func(match):
                    nonlocal changed_this_round, modified
                    unit = match.group(1)
                    full_match = match.group(0)
                    count = len(full_match) // len(unit)
                    if count > max_reps:
                        modified = True
                        changed_this_round = True
                        return unit * max_reps
                    return full_match

                txt = re.sub(pattern, replace_func, txt, flags=re.DOTALL)
            if not changed_this_round:
                break
        return txt, modified

    for _ in range(10):
        m_any = False
        text, m1 = _limit_consecutive_lines(text)
        m_any = m_any or m1
        text, m2 = _limit_block_patterns(text)
        m_any = m_any or m2
        text, m3 = _limit_regex_patterns(text)
        m_any = m_any or m3
        if not m_any:
            break
    return text


def get_questions(args):
    data_dir = DATA_PATH
    questions = load_questions(str(data_dir / "question.jsonl"))
    if args.limit is not None and args.limit != -1:
        random.Random(42).shuffle(questions)
        questions = questions[: args.limit]
    if args.limit == -1:
        random.Random(42).shuffle(questions)
    return questions


def _model_alias(model_path: str) -> str:
    return model_path.split("/")[-1]


def _make_metadata(answer: str) -> Dict:
    encoding = tiktoken.encoding_for_model("gpt-4o")
    token_len = len(encoding.encode(answer, disallowed_special=()))
    metadata = {"token_len": token_len}
    markdown_info = count_markdown_elements(
        remove_pattern(answer, re.compile("```([^`]*)```")),
        suffix="",
    )
    metadata.update(markdown_info)
    return metadata


async def generate_answers(args, model: InspectAIModel) -> tuple:
    """Generate answers using Tinker sampling API via inspect-ai model."""
    data_dir = DATA_PATH
    output_dir = data_dir / "model_answer"
    output_path = output_dir / f"{args.model_alias}.jsonl"

    questions = get_questions(args)
    answers_dict: Dict[str, Dict] = {}

    print(f"[generate] Generating answers for {len(questions)} questions via Tinker API.")

    for question in tqdm(questions, desc="Generating answers"):
        messages = [ChatMessageUser(content=question["prompt"])]

        output = await model.generate(messages)
        answer_text = output.completion.strip()

        if answer_text.startswith("<think>") and ("</think>" in answer_text):
            answer_text = answer_text.split("</think>", maxsplit=1)[1]
            answer_text = answer_text.strip()

        answer_text = limit_repetitions(answer_text)

        msg_records = [
            {"role": "user", "content": question["prompt"]},
            {"role": "assistant", "content": {"answer": answer_text}},
        ]

        record = {
            "uid": question["uid"],
            "ans_id": shortuuid.uuid(),
            "model": args.model_alias,
            "messages": msg_records,
            "tstamp": time.time(),
            "metadata": _make_metadata(answer_text),
        }
        answers_dict[question["uid"]] = record

    if args.store_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[generate] Writing answers to {output_path}")
        with open(output_path, "w", encoding="utf-8") as fout:
            for record in answers_dict.values():
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        return output_path, answers_dict

    return None, answers_dict


def call_openai(messages: List[Dict]):
    import openai

    client = openai.OpenAI()
    request_kwargs = {
        "model": JUDGE_MODEL,
        "messages": messages,
        "max_completion_tokens": JUDGE_MAX_COMPLETION,
    }
    if REASONING_EFFORT is not None:
        request_kwargs["reasoning_effort"] = REASONING_EFFORT

    for attempt in range(API_MAX_RETRY):
        try:
            completion = client.chat.completions.create(**request_kwargs)
            return {
                "answer": completion.choices[0].message.content,
            }
        except openai.BadRequestError as err:
            if "reasoning" in str(err).lower() and "reasoning_effort" in request_kwargs:
                print("[judge] reasoning_effort not supported; retrying without it.")
                request_kwargs.pop("reasoning_effort", None)
                continue
            wait_time = API_RETRY_SLEEP * (2**attempt)
            print(f"[judge] OpenAI API error ({type(err).__name__}): {err}. Retry in {wait_time}s.")
            time.sleep(wait_time)
        except Exception as err:
            wait_time = API_RETRY_SLEEP * (2**attempt)
            print(f"[judge] OpenAI API error ({type(err).__name__}): {err}. Retry in {wait_time}s.")
            time.sleep(wait_time)
    print("[judge] Exhausted retries; returning None.")
    return None


def get_score(judgment: str, patterns: Iterable[str]) -> Optional[str]:
    for pattern in patterns:
        compiled = re.compile(pattern)
        matches = [
            m for m in compiled.findall(judgment.upper()) if isinstance(m, str) and m
        ]
        if matches:
            return matches[-1].strip("\n")
        if matches and isinstance(matches[-1], tuple):
            for item in matches[-1]:
                if item:
                    return item.strip("\n")
    return None


def judge_answers(args, candidate_answers: Optional[Dict[str, Dict]] = None) -> tuple:
    judge_config = make_config(JUDGE_CONFIG)
    prompt_template = judge_config["prompt_template"]
    regex_patterns = judge_config["regex_patterns"]

    data_dir = DATA_PATH
    answer_dir = data_dir / "model_answer"
    judgment_dir = data_dir / "model_judgment" / JUDGE_MODEL
    output_path = judgment_dir / f"{args.model_alias}.jsonl"

    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Please export your OpenAI API key before judging."
        )

    questions = get_questions(args)
    model_answers = load_model_answers(str(answer_dir))

    if candidate_answers is not None:
        model_answers[args.model_alias] = candidate_answers

    if args.model_alias not in model_answers:
        raise FileNotFoundError(
            f"Cannot find answers for model '{args.model_alias}' in {answer_dir}."
        )

    results: List[Optional[Dict]] = [None] * len(questions)

    with ThreadPoolExecutor(max_workers=args.judge_workers) as executor:
        futures = {
            executor.submit(
                _judge_single_question,
                question,
                args,
                model_answers,
                prompt_template,
                regex_patterns,
            ): idx
            for idx, question in enumerate(questions)
        }
        with tqdm(total=len(futures), desc="Judging answers") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    pbar.update(1)
                    raise
                pbar.update(1)

    if args.store_outputs:
        judgment_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fout:
            for record in results:
                if record is None:
                    continue
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        return output_path, results

    return None, results


def _judge_single_question(
    question: Dict,
    args,
    model_answers: Dict[str, Dict],
    prompt_template: str,
    regex_patterns: Iterable[str],
):
    uid = question["uid"]
    category = question["category"]

    baseline_model = JUDGE_SETTINGS[category]["baseline"]
    if baseline_model not in model_answers:
        raise FileNotFoundError(
            f"Baseline model '{baseline_model}' answers not found in data/model_answer"
        )

    candidate_answer = model_answers[args.model_alias].get(uid)
    baseline_answer = model_answers[baseline_model].get(uid)

    if candidate_answer is None:
        print(f"[judge] Candidate missing answer for UID {uid}. Skipping.")
        return None
    if baseline_answer is None:
        print(f"[judge] Baseline missing answer for UID {uid}. Skipping.")
        return None

    prompt_args = {
        "QUESTION": question["prompt"],
        "ANSWER_A": baseline_answer["messages"][-1]["content"]["answer"],
        "ANSWER_B": candidate_answer["messages"][-1]["content"]["answer"],
    }
    user_prompt = prompt_template.format(**prompt_args)
    messages = [
        {"role": "system", "content": JUDGE_SETTINGS[category]["system_prompt"]},
        {"role": "user", "content": user_prompt},
    ]

    result_ab = call_openai(messages=messages)
    score_ab = get_score(result_ab["answer"], regex_patterns) if result_ab else None

    prompt_args_swap = {
        "QUESTION": question["prompt"],
        "ANSWER_A": candidate_answer["messages"][-1]["content"]["answer"],
        "ANSWER_B": baseline_answer["messages"][-1]["content"]["answer"],
    }
    user_prompt_swap = prompt_template.format(**prompt_args_swap)
    messages_swap = [
        {"role": "system", "content": JUDGE_SETTINGS[category]["system_prompt"]},
        {"role": "user", "content": user_prompt_swap},
    ]

    result_ba = call_openai(messages=messages_swap)
    score_ba = get_score(result_ba["answer"], regex_patterns) if result_ba else None

    return {
        "uid": uid,
        "category": category,
        "judge": JUDGE_MODEL,
        "model": candidate_answer["model"],
        "baseline": baseline_answer["model"],
        "games": [
            {"score": score_ab, "judgment": result_ab, "prompt": messages},
            {"score": score_ba, "judgment": result_ba, "prompt": messages_swap},
        ],
    }


def _compute_metrics(battles) -> Dict[str, float]:
    scores = battles["scores"].astype(float)
    num_samples = len(scores)
    if num_samples == 0:
        return {"accuracy": 0.0, "stderr": 0.0}
    accuracy = float(scores.mean())
    if num_samples == 1:
        stderr = 0.0
    else:
        std_dev = float(scores.std(ddof=1))
        stderr = std_dev / math.sqrt(num_samples) if not math.isnan(std_dev) else 0.0
    return {"accuracy": accuracy, "stderr": stderr}


def _judgments_to_battles(judgments: List[Optional[Dict]], weight: int = 3):
    import pandas as pd

    score_map = {
        "A>B": [1],
        "A>>B": [1] * weight,
        "A=B": [0.5],
        "A<<B": [0] * weight,
        "A<B": [0],
        "B>A": [0],
        "B>>A": [0] * weight,
        "B=A": [0.5],
        "B<<A": [1] * weight,
        "B<A": [1],
    }

    battles_data = []
    for record in judgments:
        if record is None:
            continue
        games = record.get("games", [])
        if len(games) < 2:
            continue
        score_ab = games[0].get("score")
        score_ba = games[1].get("score")
        if score_ab is None or score_ba is None:
            continue
        scores_ab = score_map[score_ab.upper()]
        scores_ba = score_map[score_ba.upper()]
        scores = [1 - s for s in scores_ab] + scores_ba
        battles_data.append({
            "uid": record["uid"],
            "category": record["category"],
            "model": record["model"],
            "baseline": record["baseline"],
            "scores": scores,
        })

    battles = pd.DataFrame(battles_data)
    battles = battles.explode('scores').reset_index(drop=True)
    return battles


def summarize_results(model_alias: str, judgments: Optional[List[Optional[Dict]]] = None) -> Optional[Dict[str, float]]:
    if judgments is not None:
        battles = _judgments_to_battles(judgments)
    else:
        try:
            battles = load_judgments([JUDGE_MODEL], BENCHMARK)
        except FileNotFoundError:
            print("[summary] No judgments found for summary.")
            return {"accuracy": 0.0, "stderr": 0.0}

    battles = battles[battles.model == model_alias].reset_index(drop=True)
    if battles.empty:
        print(f"[summary] No battles recorded for model '{model_alias}'.")
        return {"accuracy": 0.0, "stderr": 0.0}

    categories = battles.category.unique().tolist()
    for category in categories:
        print_leaderboard(battles[battles.category == category].reset_index(drop=True), category)

    return _compute_metrics(battles)


async def async_main(args) -> None:
    """Async entry point: set up Tinker model, generate answers, then judge."""
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

    candidate_answers = None

    if args.skip_generation:
        ans_path = DATA_PATH / "model_answer" / f"{args.model_alias}.jsonl"
        if ans_path.exists():
            print(f"[skip] Loading existing answers from {ans_path}")
            candidate_answers = {}
            with open(ans_path, "r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    candidate_answers[record["uid"]] = record
        else:
            print(f"[skip] File {ans_path} not found, generating answers instead")
            _, candidate_answers = await generate_answers(args, model)
    else:
        ans_path, candidate_answers = await generate_answers(args, model)
        if ans_path:
            print(f"[done] Answers saved to {ans_path}")

    judge_path, judgments = judge_answers(args, candidate_answers)
    if judge_path:
        print(f"[done] Judgments saved to {judge_path}")

    metrics = summarize_results(args.model_alias, judgments if not args.store_outputs else None)

    if args.json_output_file is not None and metrics is not None:
        with open(args.json_output_file, "w", encoding="utf-8") as metrics_file:
            json.dump(metrics, metrics_file, indent=2)
        print(f"[done] Metrics saved to {args.json_output_file}")
    if metrics is None:
        print("Failed to compute metrics.")

    print("Score (winrate) is:", metrics['accuracy'])


def main():
    parser = argparse.ArgumentParser(description="Run Arena-Hard evaluation via Tinker API.")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Tinker model path (e.g. tinker://...) for a fine-tuned checkpoint.")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3.5-4B",
                        help="Base model name on Tinker.")
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--limit", type=int, default=32,
                        help="Limit number of questions for quicker runs.")
    parser.add_argument("--judge-workers", type=int, default=DEFAULT_JUDGE_WORKERS,
                        help="Number of concurrent judge jobs.")
    parser.add_argument('--json-output-file', type=str, default=None,
                        help="Optional path to output metrics as JSON.")
    parser.add_argument('--skip-generation', action='store_true',
                        help="Skip answer generation and use existing answers.")
    parser.add_argument('--store-outputs', action='store_true',
                        help="Store model answers and judgments to disk.")
    parser.add_argument('--temperature', type=float, default=0.6,
                        help="Sampling temperature.")
    args = parser.parse_args()

    args.model_alias = _model_alias(args.model_path or args.base_model)
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()

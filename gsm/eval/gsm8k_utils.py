"""Shared GSM8K prompt construction and scoring helpers."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def extract_boxed(text: str) -> str | None:
    """Extract the final \\boxed{...} span using brace matching."""
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = idx + len("\\boxed")
    while i < len(text) and text[i] != "{":
        i += 1
    if i >= len(text):
        return None
    depth = 0
    start = i
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:j]
    return None


def extract_last_number(text: str) -> str | None:
    """Extract the last number from text as a fallback answer."""
    matches = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return matches[-1].replace(",", "") if matches else None


def normalize_answer(s: str) -> str:
    """Normalize an answer string before exact or numeric comparison."""
    return s.strip().replace("$", "").replace("%", "").replace(",", "").rstrip(".")


def extract_answer(text: str) -> str | None:
    """Extract a model answer, preferring \\boxed{} and falling back to last number."""
    boxed = extract_boxed(text)
    if boxed is not None:
        return normalize_answer(boxed)
    num = extract_last_number(text)
    return normalize_answer(num) if num is not None else None


def extract_gt_answer(answer_text: str) -> str:
    """Extract the GSM8K ground-truth answer from the `#### <answer>` suffix."""
    return normalize_answer(str(answer_text).split("####")[-1].strip())


def answers_match(pred: str, gt: str) -> bool:
    """Compare two normalized answers numerically when possible."""
    try:
        return abs(float(pred) - float(gt)) < 1e-5
    except (TypeError, ValueError):
        return pred == gt


def build_prompt(question: str, tokenizer, system_prompt: str = SYSTEM_PROMPT) -> str:
    """Render a GSM8K question with the model tokenizer's chat template."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_gsm8k(path: str | Path, max_examples: int | None = None) -> tuple[list[str], list[str]]:
    """Load GSM8K questions and normalized ground-truth answers from JSONL."""
    questions, answers = [], []
    with Path(path).open() as f:
        for i, line in enumerate(f):
            if max_examples is not None and i >= max_examples:
                break
            ex = json.loads(line)
            questions.append(ex["question"])
            answers.append(extract_gt_answer(ex["answer"]))
    return questions, answers


def score_greedy(responses: list[list[str]], gt_answers: list[str]) -> tuple[float, list[dict]]:
    """Score one completion per example."""
    correct = 0
    results = []
    for i, (resps, gt) in enumerate(zip(responses, gt_answers)):
        pred_text = resps[0]
        pred_answer = extract_answer(pred_text)
        is_correct = pred_answer is not None and answers_match(pred_answer, gt)
        correct += is_correct
        results.append({
            "idx": i,
            "pred_answer": pred_answer,
            "gt_answer": gt,
            "correct": is_correct,
            "response": pred_text,
        })
    return correct / len(gt_answers), results


def score_majority(responses: list[list[str]], gt_answers: list[str], k: int) -> tuple[float, list[dict]]:
    """Score by majority vote over the first k completions."""
    correct = 0
    results = []
    for i, (resps, gt) in enumerate(zip(responses, gt_answers)):
        answers = [ans for resp in resps[:k] if (ans := extract_answer(resp)) is not None]
        if answers:
            pred_answer = Counter(answers).most_common(1)[0][0]
            is_correct = answers_match(pred_answer, gt)
        else:
            pred_answer = None
            is_correct = False
        correct += is_correct
        results.append({
            "idx": i,
            "majority_answer": pred_answer,
            "gt_answer": gt,
            "correct": is_correct,
            "n_valid": len(answers),
        })
    return correct / len(gt_answers), results


def score_pass_at_k(responses: list[list[str]], gt_answers: list[str], k: int) -> tuple[float, list[dict]]:
    """Score whether any of the first k completions has the correct answer."""
    correct = 0
    results = []
    for i, (resps, gt) in enumerate(zip(responses, gt_answers)):
        is_correct = any(
            (ans := extract_answer(resp)) is not None and answers_match(ans, gt)
            for resp in resps[:k]
        )
        correct += is_correct
        results.append({"idx": i, "gt_answer": gt, "correct": is_correct})
    return correct / len(gt_answers), results


def parse_mode(mode: str) -> tuple[int, int, str]:
    """Return `(n_samples, k, score_name)` for greedy/pass@k/maj@k."""
    if mode == "greedy":
        return 1, 1, "greedy"
    if mode.startswith("pass@"):
        k = int(mode.split("@", 1)[1])
        return k, k, "pass"
    if mode.startswith("maj@"):
        k = int(mode.split("@", 1)[1])
        return k, k, "maj"
    raise ValueError(f"Unknown mode: {mode}. Use greedy, pass@k, or maj@k.")


def score_responses(
    responses: list[list[str]],
    gt_answers: list[str],
    mode: str,
) -> tuple[float, list[dict]]:
    """Score generated responses using `greedy`, `pass@k`, or `maj@k`."""
    _, k, score_name = parse_mode(mode)
    if score_name == "greedy":
        return score_greedy(responses, gt_answers)
    if score_name == "pass":
        return score_pass_at_k(responses, gt_answers, k)
    return score_majority(responses, gt_answers, k)

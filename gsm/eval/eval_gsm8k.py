"""Evaluate a model on GSM8K using vLLM for fast inference.

Supports greedy decoding and pass@k / maj@k evaluation.
Uses the Qwen2.5-Math prompt format: system prompt asks for step-by-step
reasoning with final answer in \\boxed{}.

Answer extraction: tries \\boxed{} first, then falls back to last number.

Usage:
    # Greedy eval on test split
    python eval_gsm8k.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --split test --name qwen_greedy

    # Maj@8 on test split
    python eval_gsm8k.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --split test --mode maj@8 --name qwen_maj8

    # Eval on both splits
    python eval_gsm8k.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --split both --name qwen_both

    # Multi-GPU (tensor parallel)
    python eval_gsm8k.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --split test --name qwen_tp2 --tp 2
"""

import argparse
import json
import os
import re
from collections import Counter

from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> str | None:
    """Extract content from \\boxed{...} with brace matching."""
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
                return text[start + 1 : j]
    return None


def extract_last_number(text: str) -> str | None:
    """Extract the last number from text as fallback."""
    matches = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if matches:
        return matches[-1].replace(",", "")
    return None


def extract_answer(text: str) -> str | None:
    """Extract answer from model output. Tries \\boxed{} first, then last number."""
    boxed = extract_boxed(text)
    if boxed is not None:
        return normalize_answer(boxed)
    num = extract_last_number(text)
    if num is not None:
        return normalize_answer(num)
    return None


def extract_gt_answer(answer_text: str) -> str:
    """Extract ground truth answer from GSM8K format: '...#### <number>'."""
    return normalize_answer(answer_text.split("####")[-1].strip())


def normalize_answer(s: str) -> str:
    """Normalize an answer string for comparison."""
    s = s.strip()
    s = s.replace("$", "").replace("%", "").replace(",", "")
    s = s.rstrip(".")
    return s


def answers_match(pred: str, gt: str) -> bool:
    """Check if predicted and ground truth answers match numerically."""
    try:
        return abs(float(pred) - float(gt)) < 1e-5
    except (ValueError, TypeError):
        return pred == gt


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def build_prompt(question: str, tokenizer) -> str:
    """Build chat prompt using the tokenizer's chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_greedy_accuracy(responses, gt_answers):
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
            "response": pred_text[:500],
        })
    accuracy = correct / len(gt_answers)
    return accuracy, results


def compute_majority_accuracy(responses, gt_answers, k):
    correct = 0
    results = []
    for i, (resps, gt) in enumerate(zip(responses, gt_answers)):
        answers = []
        for resp in resps[:k]:
            ans = extract_answer(resp)
            if ans is not None:
                answers.append(ans)
        if answers:
            counter = Counter(answers)
            majority_answer = counter.most_common(1)[0][0]
            is_correct = answers_match(majority_answer, gt)
        else:
            majority_answer = None
            is_correct = False
        correct += is_correct
        results.append({
            "idx": i,
            "majority_answer": majority_answer,
            "gt_answer": gt,
            "correct": is_correct,
            "n_valid": len(answers),
        })
    accuracy = correct / len(gt_answers)
    return accuracy, results


def compute_pass_at_k(responses, gt_answers, k):
    correct = 0
    results = []
    for i, (resps, gt) in enumerate(zip(responses, gt_answers)):
        any_correct = False
        for resp in resps[:k]:
            ans = extract_answer(resp)
            if ans is not None and answers_match(ans, gt):
                any_correct = True
                break
        correct += any_correct
        results.append({
            "idx": i,
            "gt_answer": gt,
            "correct": any_correct,
        })
    accuracy = correct / len(gt_answers)
    return accuracy, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on GSM8K (vLLM)")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "both"])
    parser.add_argument("--mode", type=str, default="greedy",
                        help="greedy, pass@k, or maj@k (e.g., pass@8, maj@8)")
    parser.add_argument("--name", type=str, required=True, help="Experiment name for output directory")
    parser.add_argument("--data_dir", type=str, default="data/gsm8k")
    parser.add_argument("--output_dir", type=str, default="output/eval")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--total_seqlen", type=int, default=None,
                        help="If set, cap total (prompt + generation) tokens per example. "
                             "max_new_tokens becomes max(1, total_seqlen - len(prompt_tokens)) per example. "
                             "Overrides --max_new_tokens.")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override temperature (default: 0 for greedy, 0.7 for sampling)")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Override top_p (default: 1.0 for greedy, 0.8 for sampling)")
    parser.add_argument("--max_examples", type=int, default=None, help="Limit examples (for debugging)")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size (number of GPUs)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    # Parse mode
    if args.mode == "greedy":
        n_samples = 1
        k = 1
        score_fn = "greedy"
    elif args.mode.startswith("pass@"):
        k = int(args.mode.split("@")[1])
        n_samples = k
        score_fn = "pass"
    elif args.mode.startswith("maj@"):
        k = int(args.mode.split("@")[1])
        n_samples = k
        score_fn = "maj"
    else:
        raise ValueError(f"Unknown mode: {args.mode}. Use greedy, pass@k, or maj@k.")

    # Set temperature/top_p defaults
    if args.temperature is None:
        args.temperature = 0.0 if score_fn == "greedy" else 0.7
    if args.top_p is None:
        args.top_p = 1.0 if score_fn == "greedy" else 0.8

    # Output directory
    exp_dir = os.path.join(args.output_dir, args.name)
    os.makedirs(exp_dir, exist_ok=True)

    # Load model with vLLM
    print(f"Loading model with vLLM: {args.model} (tp={args.tp})")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
    )
    tokenizer = llm.get_tokenizer()
    print(f"Model loaded.")

    # Sampling params — if --total_seqlen is set, we build per-prompt params below.
    stop_tokens = ["<|im_end|>", "<|endoftext|>", "</s>"]
    def make_sampling_params(max_tokens):
        return SamplingParams(
            n=n_samples,
            temperature=args.temperature if args.temperature > 0 else 0,
            top_p=args.top_p if args.temperature > 0 else 1.0,
            max_tokens=max_tokens,
            stop=stop_tokens,
            seed=args.seed if args.temperature > 0 else None,
        )
    sampling_params = make_sampling_params(args.max_new_tokens)

    # Determine splits
    splits = ["train", "test"] if args.split == "both" else [args.split]

    all_results = {
        "config": {
            "model": args.model,
            "mode": args.mode,
            "n_samples": n_samples,
            "k": k,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "tp": args.tp,
            "seed": args.seed,
            "name": args.name,
        },
    }

    for split in splits:
        # Load data
        data_path = os.path.join(args.data_dir, f"{split}.jsonl")
        print(f"\nLoading {split} data from {data_path}")
        examples = []
        with open(data_path) as f:
            for line in f:
                examples.append(json.loads(line.strip()))

        if args.max_examples:
            examples = examples[:args.max_examples]
        print(f"  {len(examples)} examples")

        # Build prompts
        questions = [ex["question"] for ex in examples]
        gt_answers = [extract_gt_answer(ex["answer"]) for ex in examples]
        prompts = [build_prompt(q, tokenizer) for q in questions]

        # Show example prompt
        print(f"\n--- Example prompt ---")
        print(prompts[0][:500])
        print("...\n")

        # Generate with vLLM (all prompts at once, vLLM handles batching)
        if args.total_seqlen is not None:
            per_prompt_sp = []
            for p in prompts:
                n_prompt = len(tokenizer.encode(p, add_special_tokens=False))
                per_prompt_sp.append(make_sampling_params(max(1, args.total_seqlen - n_prompt)))
            print(f"Generating ({args.mode}, n_samples={n_samples}, total_seqlen={args.total_seqlen})...")
            outputs = llm.generate(prompts, per_prompt_sp)
        else:
            print(f"Generating ({args.mode}, n_samples={n_samples}, max_new_tokens={args.max_new_tokens})...")
            outputs = llm.generate(prompts, sampling_params)

        # Collect responses: each output has n_samples completions
        responses = []
        for output in outputs:
            resps = [completion.text.strip() for completion in output.outputs]
            responses.append(resps)

        # Score
        if score_fn == "greedy":
            accuracy, details = compute_greedy_accuracy(responses, gt_answers)
        elif score_fn == "maj":
            accuracy, details = compute_majority_accuracy(responses, gt_answers, k)
        elif score_fn == "pass":
            accuracy, details = compute_pass_at_k(responses, gt_answers, k)

        print(f"\n=== {split} Results ===")
        print(f"  Accuracy ({args.mode}): {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"  Total: {len(examples)}")

        all_results[split] = {
            "accuracy": accuracy,
            "n_examples": len(examples),
            "details": details,
        }

    # Save results
    output_path = os.path.join(exp_dir, "results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Summary (no per-example details)
    summary = {"config": all_results["config"]}
    for split_key in splits:
        if split_key in all_results:
            summary[split_key] = {
                "accuracy": all_results[split_key]["accuracy"],
                "n_examples": all_results[split_key]["n_examples"],
            }
    summary_path = os.path.join(exp_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

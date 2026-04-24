"""Generate rollouts from a model on a dataset using vLLM.

Given a dataset of prompts (e.g., GSM8K questions), generate completions
using a specified model and sampling configuration. Outputs are saved as
JSONL with prompt and completion fields.

This is the analog of generate_noisy_rollouts.py from the addition experiments,
but here the "noisy expert" is the model itself (e.g., Qwen2.5-Math) rather
than a clean expert with manual corruption.

Usage:
    # Greedy rollout on GSM8K train set
    python rollout.py --model Qwen/Qwen2.5-Math-1.5B-Instruct \
        --data real_math/data/gsm8k/train.jsonl \
        --output real_math/data/rollouts/qwen_1.5b_greedy_train.jsonl

    # Temperature sampling, 8 completions per prompt
    python rollout.py --model Qwen/Qwen2.5-Math-1.5B-Instruct \
        --data real_math/data/gsm8k/train.jsonl \
        --output real_math/data/rollouts/qwen_1.5b_t07_n8_train.jsonl \
        --temperature 0.7 --top_p 0.8 --n 8

    # Custom prompt field and system prompt
    python rollout.py --model Qwen/Qwen2.5-Math-1.5B-Instruct \
        --data my_data.jsonl --prompt_field question \
        --system_prompt "Solve step by step." \
        --output rollouts.jsonl
"""

import argparse
import json
import os
import re
import time

from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# Answer extraction (for scoring rollouts against ground truth)
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


def extract_gt_answer(answer_text) -> str:
    """Extract ground truth answer. GSM8K format: '...#### <number>'.
    If not a string (e.g. TinyGSM's numeric answer), just stringify."""
    s = str(answer_text)
    return normalize_answer(s.split("####")[-1].strip())


def normalize_answer(s: str) -> str:
    s = s.strip().replace("$", "").replace("%", "").replace(",", "").rstrip(".")
    return s


def answers_match(pred: str, gt: str) -> bool:
    try:
        return abs(float(pred) - float(gt)) < 1e-5
    except (ValueError, TypeError):
        return pred == gt


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def build_prompt(question: str, tokenizer, system_prompt: str) -> str:
    """Build chat prompt using the tokenizer's chat template."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_prompt_no_chat(question: str, system_prompt: str) -> str:
    """Build a simple prompt without chat template."""
    if system_prompt:
        return f"{system_prompt}\n\n{question}"
    return question


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate rollouts from a model using vLLM")

    # Model
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    # Data
    parser.add_argument("--data", type=str, required=True, help="Input JSONL file with prompts")
    parser.add_argument("--prompt_field", type=str, default="question", help="JSONL field for the prompt")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--max_examples", type=int, default=None, help="Limit number of examples")

    # Prompt format
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT,
                        help="System prompt (set to '' to disable)")
    parser.add_argument("--no_chat_template", action="store_true",
                        help="Don't use tokenizer's chat template, just concatenate system+user")

    # Sampling
    parser.add_argument("--n", type=int, default=1, help="Number of completions per prompt")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0=greedy)")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")

    # Scoring (optional: if GT answers available, compute accuracy)
    parser.add_argument("--gt_answer_field", type=str, default="answer",
                        help="JSONL field for ground truth answer (set to '' to disable scoring)")

    # Stop tokens
    parser.add_argument("--stop", type=str, nargs="*",
                        default=["<|im_end|>", "<|endoftext|>", "</s>", "<end_of_turn>", "<eos>"],
                        help="Stop token strings")

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data}")
    examples = []
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    if args.max_examples:
        examples = examples[:args.max_examples]
    print(f"  {len(examples)} examples")

    # Load model
    print(f"Loading model: {args.model} (tp={args.tp})")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
    )
    tokenizer = llm.get_tokenizer()
    print("Model loaded.")

    # Build prompts
    prompts = []
    for ex in examples:
        question = str(ex[args.prompt_field])
        if args.no_chat_template:
            prompt = build_prompt_no_chat(question, args.system_prompt)
        else:
            prompt = build_prompt(question, tokenizer, args.system_prompt)
        prompts.append(prompt)

    # Show example
    print(f"\n--- Example prompt ---")
    print(prompts[0][:500])
    print("...\n")

    # Sampling params
    sampling_params = SamplingParams(
        n=args.n,
        temperature=args.temperature if args.temperature > 0 else 0,
        top_p=args.top_p if args.temperature > 0 else 1.0,
        max_tokens=args.max_new_tokens,
        stop=args.stop if args.stop else None,
        seed=args.seed if args.temperature > 0 else None,
    )

    # Generate
    mode_str = "greedy" if args.temperature == 0 else f"temp={args.temperature}"
    print(f"Generating ({mode_str}, n={args.n}, max_tokens={args.max_new_tokens})...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    print(f"Generation done in {elapsed:.1f}s ({len(prompts) * args.n / elapsed:.1f} completions/s)")

    # Save results
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    n_completions = 0
    with open(args.output, "w") as f:
        for i, output in enumerate(outputs):
            ex = examples[i]
            for j, completion in enumerate(output.outputs):
                record = {
                    "prompt": ex[args.prompt_field],
                    "completion": completion.text.strip(),
                    "prompt_formatted": prompts[i],
                    "completion_idx": j,
                }
                # Carry over any extra fields from the input (e.g., "answer" for GT)
                for key in ex:
                    if key != args.prompt_field and key not in record:
                        record[key] = ex[key]
                f.write(json.dumps(record) + "\n")
                n_completions += 1

    print(f"\nSaved {n_completions} completions ({len(examples)} prompts x {args.n}) to {args.output}")

    # Compute completion token length stats
    completion_lengths = []
    with open(args.output) as f:
        for line in f:
            r = json.loads(line)
            n_tokens = len(tokenizer.encode(r["completion"], add_special_tokens=False))
            completion_lengths.append(n_tokens)
    completion_lengths.sort()
    n = len(completion_lengths)
    thresholds = [256, 512, 768, 1024, 2048]
    length_stats = {
        "mean": round(sum(completion_lengths) / n, 1),
        "median": completion_lengths[n // 2],
        "max": completion_lengths[-1],
        "min": completion_lengths[0],
    }
    for t in thresholds:
        pct = sum(1 for l in completion_lengths if l <= t) / n * 100
        length_stats[f"pct_le_{t}"] = round(pct, 1)
    print(f"Completion token lengths: mean={length_stats['mean']}, median={length_stats['median']}, max={length_stats['max']}")
    for t in thresholds:
        print(f"  <= {t}: {length_stats[f'pct_le_{t}']}%")

    # Score completions against ground truth if available
    accuracy = None
    n_correct = None
    if args.gt_answer_field and args.gt_answer_field in examples[0]:
        n_correct = 0
        n_scored = 0
        with open(args.output) as f:
            for line in f:
                r = json.loads(line)
                if args.gt_answer_field not in r:
                    continue
                pred = extract_answer(r["completion"])
                gt = extract_gt_answer(r[args.gt_answer_field])
                if pred is not None and answers_match(pred, gt):
                    n_correct += 1
                n_scored += 1
        accuracy = n_correct / n_scored if n_scored > 0 else 0.0
        print(f"Accuracy: {n_correct}/{n_scored} = {accuracy * 100:.1f}%")

    # Save config alongside the output
    config_path = args.output.replace(".jsonl", "_config.json")
    config = {
        "model": args.model,
        "data": args.data,
        "prompt_field": args.prompt_field,
        "system_prompt": args.system_prompt,
        "no_chat_template": args.no_chat_template,
        "n": args.n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "stop": args.stop,
        "n_examples": len(examples),
        "n_completions": n_completions,
        "elapsed_seconds": round(elapsed, 1),
    }
    config["length_stats"] = length_stats
    if accuracy is not None:
        config["accuracy"] = round(accuracy, 4)
        config["n_correct"] = n_correct
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")


if __name__ == "__main__":
    main()

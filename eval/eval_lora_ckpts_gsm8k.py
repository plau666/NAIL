"""Evaluate all LoRA checkpoints in a training run dir on GSM8K (greedy).

Loads the base model once with vLLM (enable_lora=True) and swaps LoRA adapters
per checkpoint via lora_request, avoiding repeated model loads.

Writes eval_results.json inside each checkpoint directory:
    {"eval_data": ..., "accuracy": ..., "n_correct": ..., "n_total": ...}

Usage:
    python real_math/eval/eval_lora_ckpts_gsm8k.py \
        --run_dir real_math/output/tinygsm/obc_lora_r128_gemma270m_it_on_gemma3_1b_it_t1p0_n1_pure_maxlen1024 \
        --eval_data real_math/data/gsm8k/test.jsonl \
        --max_new_tokens 1024
"""

import argparse
import glob
import json
import os
import re
import sys

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_gsm8k import (
    SYSTEM_PROMPT,
    build_prompt,
    extract_answer,
    extract_gt_answer,
    answers_match,
)


def list_checkpoints(run_dir):
    ckpts = glob.glob(os.path.join(run_dir, "checkpoint-*"))
    def step_of(p):
        m = re.search(r"checkpoint-(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1
    ckpts = sorted(ckpts, key=step_of)
    final = os.path.join(run_dir, "final")
    if os.path.isdir(final):
        ckpts.append(final)
    return ckpts


def load_gsm8k(path):
    qs, gts = [], []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            qs.append(ex["question"])
            gts.append(extract_gt_answer(ex["answer"]))
    return qs, gts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--eval_data", default="real_math/data/gsm8k/test.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--total_seqlen", type=int, default=None,
                    help="If set, cap total (prompt + generation) tokens per example. "
                         "Overrides --max_new_tokens.")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    ap.add_argument("--max_lora_rank", type=int, default=128)
    ap.add_argument("--skip_if_done", action="store_true", default=True)
    args = ap.parse_args()

    # --- Figure out base model from the run's config.json ---
    config_path = os.path.join(args.run_dir, "config.json")
    with open(config_path) as f:
        run_config = json.load(f)
    base_model = run_config["student_model"]
    print(f"Base model: {base_model}")

    ckpts = list_checkpoints(args.run_dir)
    print(f"Found {len(ckpts)} checkpoint(s) in {args.run_dir}")

    # --- Load data ---
    questions, gts = load_gsm8k(args.eval_data)
    print(f"Loaded {len(questions)} eval examples from {args.eval_data}")

    # --- Load base model once with LoRA support ---
    llm = LLM(
        model=base_model,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        max_loras=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
    )
    tokenizer = llm.get_tokenizer()

    prompts = [build_prompt(q, tokenizer) for q in questions]
    print("\n--- Example prompt ---")
    print(prompts[0][:500])
    print("...\n")

    STOP = ["<|im_end|>", "<|endoftext|>", "</s>", "<end_of_turn>", "<eos>"]
    def make_sp(max_tokens):
        return SamplingParams(n=1, temperature=0.0, top_p=1.0,
                              max_tokens=max_tokens, stop=STOP)

    if args.total_seqlen is not None:
        prompt_tok_lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
        sampling_params = [make_sp(max(1, args.total_seqlen - L)) for L in prompt_tok_lens]
        print(f"Using total_seqlen={args.total_seqlen} (per-prompt max_new_tokens)")
    else:
        sampling_params = make_sp(args.max_new_tokens)
        print(f"Using max_new_tokens={args.max_new_tokens}")

    # --- Eval loop ---
    for i, ckpt in enumerate(ckpts):
        out_path = os.path.join(ckpt, "eval_results.json")
        if args.skip_if_done and os.path.exists(out_path):
            with open(out_path) as f:
                prev = json.load(f)
            print(f"[{i+1}/{len(ckpts)}] SKIP {os.path.basename(ckpt)}: "
                  f"acc={prev.get('accuracy')} already present")
            continue

        lora_name = os.path.basename(ckpt)
        lora_request = LoRARequest(lora_name=lora_name, lora_int_id=i + 1, lora_path=ckpt)

        print(f"\n[{i+1}/{len(ckpts)}] Evaluating {lora_name}...")
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

        n_correct = 0
        for out, gt in zip(outputs, gts):
            pred = extract_answer(out.outputs[0].text)
            if pred is not None and answers_match(pred, gt):
                n_correct += 1
        accuracy = n_correct / len(gts)

        result = {
            "eval_data": args.eval_data,
            "accuracy": round(accuracy, 4),
            "n_correct": n_correct,
            "n_total": len(gts),
            "max_new_tokens": args.max_new_tokens,
            "mode": "greedy",
        }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  accuracy: {accuracy:.4f} ({n_correct}/{len(gts)}) -> {out_path}")

    # --- Summary ---
    print("\n=== Summary ===")
    summary = {}
    for ckpt in ckpts:
        name = os.path.basename(ckpt)
        p = os.path.join(ckpt, "eval_results.json")
        if os.path.exists(p):
            with open(p) as f:
                r = json.load(f)
            summary[name] = r["accuracy"]
            print(f"  {name}: {r['accuracy']:.4f}")

    summary_path = os.path.join(args.run_dir, "ckpt_eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()

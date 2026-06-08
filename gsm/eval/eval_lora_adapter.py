"""Eval one LoRA adapter on GSM8K test at multiple max_new_tokens budgets.

Mirrors the output format of the existing base eval_gsm8k.py runs:
each (name, mnt) writes a directory under --output_dir containing:
    summary.json   : {config, test: {accuracy, n_examples}}
    results.json   : per-example details

One vLLM model load per script invocation, then it runs each requested
max_new_tokens in sequence. Run one process per GPU per adapter.

Usage:
    python eval_lora_adapter.py \
        --base_model google/gemma-3-270m-it \
        --adapter /path/to/.../final \
        --name nailf_et1p0_seed43 \
        --mnts 512 1024 2048 4096 \
        --output_dir eval
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_gsm8k import (
    SYSTEM_PROMPT,
    extract_answer,
    extract_gt_answer,
    answers_match,
    compute_greedy_accuracy,
)
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="google/gemma-3-270m-it")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data_dir", default="data/gsm8k")
    ap.add_argument("--name", required=True,
                    help="Base name. Final dir = '<name>_greedy_mnt<MNT>'.")
    ap.add_argument("--mnts", type=int, nargs="+", required=True,
                    help="max_new_tokens budgets to evaluate, e.g. 512 1024 2048 4096")
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--output_dir", default="eval")
    ap.add_argument("--max_lora_rank", type=int, default=128)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    ap.add_argument("--max_model_len", type=int, default=None,
                    help="vLLM max_model_len. Default = max(mnts) + 1024 for prompt headroom.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    max_model_len = args.max_model_len or (max(args.mnts) + 1024)

    # --- Load test data ---
    test_path = os.path.join(args.data_dir, "test.jsonl")
    qs, gts = [], []
    with open(test_path) as f:
        for i, line in enumerate(f):
            if args.max_examples and i >= args.max_examples:
                break
            ex = json.loads(line)
            qs.append(ex["question"])
            gts.append(extract_gt_answer(ex["answer"]))
    n = len(qs)
    print(f"Loaded {n} GSM8K test examples")

    # --- Build chat prompts ---
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    rendered = []
    for q in qs:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q}]
        rendered.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True))

    # --- Load vLLM once ---
    print(f"Loading vLLM: {args.base_model} + adapter {args.adapter}")
    print(f"  max_model_len={max_model_len}")
    llm = LLM(
        model=args.base_model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        max_model_len=max_model_len,
        seed=args.seed,
    )
    lora_request = LoRARequest("ckpt", 1, args.adapter)

    # --- For each mnt, generate + score + dump ---
    for mnt in args.mnts:
        run_name = f"{args.name}_greedy_mnt{mnt}_seed{args.seed}"
        run_dir = os.path.join(args.output_dir, run_name)
        if os.path.isfile(os.path.join(run_dir, "summary.json")):
            print(f"[SKIP] {run_dir}/summary.json already exists")
            continue
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n--- {run_name} ---")
        t0 = time.time()
        params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=mnt, n=1)
        outs = llm.generate(rendered, params, lora_request=lora_request)
        gen_t = time.time() - t0
        print(f"  vLLM took {gen_t:.1f}s")

        responses = [[o.outputs[0].text] for o in outs]
        accuracy, results = compute_greedy_accuracy(responses, gts)
        print(f"  accuracy: {accuracy:.4f}  ({sum(r['correct'] for r in results)}/{n})")

        summary = {
            "config": {
                "model": args.base_model,
                "adapter": args.adapter,
                "mode": "greedy",
                "n_samples": 1,
                "k": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": mnt,
                "tp": 1,
                "seed": args.seed,
                "name": run_name,
            },
            "test": {
                "accuracy": accuracy,
                "n_examples": n,
            },
            "timing": {
                "generation_seconds": round(gen_t, 1),
            },
        }
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump({"config": summary["config"], "test": results}, f, indent=2)
        print(f"  saved {run_dir}/summary.json")


if __name__ == "__main__":
    main()

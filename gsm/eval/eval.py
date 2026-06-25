"""Unified GSM8K evaluation for base models and LoRA adapters.

Usage examples (from inside `gsm/`):
    # Evaluate a raw/full model.
    python eval/eval.py \
        --model google/gemma-3-1b-it \
        --name gemma3_1b_it_greedy \
        --mnts 512

    # Evaluate one LoRA checkpoint or final adapter.
    python eval/eval.py \
        --model google/gemma-3-270m-it \
        --adapter output/<run_name>/checkpoint-200 \
        --name <run_name>_step200 \
        --mnts 512

    # Evaluate every checkpoint plus final/ in a training run.
    python eval/eval.py \
        --run_dir output/<run_name> \
        --mnts 512 \
        --gpu_memory_utilization 0.5

Raw/full models support `--mode greedy`, `--mode pass@k`, and `--mode maj@k`.
LoRA adapter evaluation is greedy-only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from gsm8k_utils import (
    build_prompt,
    load_gsm8k,
    parse_mode,
    score_responses,
)


def checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else 10**18


def list_adapters(run_dir: Path) -> list[Path]:
    adapters = sorted(run_dir.glob("checkpoint-*"), key=checkpoint_step)
    final = run_dir / "final"
    if final.is_dir():
        adapters.append(final)
    return [p for p in adapters if p.is_dir()]


def load_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"Run config not found: {config_path}")
    with config_path.open() as f:
        return json.load(f)


def default_name(model: str | None, adapter: Path | None, run_dir: Path | None) -> str:
    if adapter is not None:
        return adapter.name
    if run_dir is not None:
        return run_dir.name
    assert model is not None
    return model.rstrip("/").split("/")[-1]


def make_sampling_params(
    tokenizer,
    prompts: list[str],
    mode: str,
    mnt: int,
    total_seqlen: int | None,
    temperature: float | None,
    top_p: float | None,
    seed: int,
):
    from vllm import SamplingParams

    n_samples, _, score_name = parse_mode(mode)
    temp = temperature if temperature is not None else (0.0 if score_name == "greedy" else 0.7)
    p = top_p if top_p is not None else (1.0 if score_name == "greedy" else 0.8)
    stop = ["<|im_end|>", "<|endoftext|>", "</s>", "<end_of_turn>", "<eos>"]

    def one(max_tokens: int) -> SamplingParams:
        return SamplingParams(
            n=n_samples,
            temperature=temp if temp > 0 else 0,
            top_p=p if temp > 0 else 1.0,
            max_tokens=max_tokens,
            stop=stop,
            seed=seed if temp > 0 else None,
        )

    if total_seqlen is None:
        return one(mnt), temp, p

    prompt_lengths = [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
    return [one(max(1, total_seqlen - length)) for length in prompt_lengths], temp, p


def write_eval(run_dir: Path, config: dict, split: str, accuracy: float, results: list[dict], elapsed: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": config,
        split: {
            "accuracy": accuracy,
            "n_examples": len(results),
        },
        "timing": {
            "generation_seconds": round(elapsed, 1),
        },
    }
    with (run_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (run_dir / "results.json").open("w") as f:
        json.dump({"config": config, split: results}, f, indent=2)


def evaluate_once(
    llm: LLM,
    prompts: list[str],
    gt_answers: list[str],
    output_dir: Path,
    config_base: dict,
    mode: str,
    mnt: int,
    total_seqlen: int | None,
    temperature: float | None,
    top_p: float | None,
    seed: int,
    split: str,
    lora_request: Any = None,
) -> float:
    tokenizer = llm.get_tokenizer()
    sampling_params, actual_temp, actual_top_p = make_sampling_params(
        tokenizer, prompts, mode, mnt, total_seqlen, temperature, top_p, seed
    )
    config = {
        **config_base,
        "mode": mode,
        "temperature": actual_temp,
        "top_p": actual_top_p,
        "max_new_tokens": mnt,
        "total_seqlen": total_seqlen,
        "seed": seed,
    }

    print(f"Evaluating {output_dir.name}: mode={mode}, max_new_tokens={mnt}")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    elapsed = time.time() - t0

    responses = [[completion.text.strip() for completion in output.outputs] for output in outputs]
    accuracy, results = score_responses(responses, gt_answers, mode)
    write_eval(output_dir, config, split, accuracy, results, elapsed)
    print(f"  accuracy={accuracy:.4f} ({sum(r['correct'] for r in results)}/{len(results)}) -> {output_dir}")
    return accuracy


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate GSM8K with vLLM")
    parser.add_argument("--model", help="HF model ID or full model path. Required unless --run_dir is set.")
    parser.add_argument("--adapter", help="Single LoRA checkpoint/final adapter path.")
    parser.add_argument("--run_dir", help="Training run dir containing config.json, checkpoint-*/, and final/.")
    parser.add_argument("--name", help="Output name for raw-model or single-adapter eval.")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--data_dir", default="data/gsm8k")
    parser.add_argument("--output_dir", default="eval")
    parser.add_argument("--mode", default="greedy", help="greedy, pass@k, or maj@k. LoRA eval is greedy-only.")
    parser.add_argument("--mnts", type=int, nargs="+", default=[1024],
                        help="One or more max_new_tokens budgets.")
    parser.add_argument("--total_seqlen", type=int, default=None,
                        help="Cap prompt + generation length per example. Overrides each --mnts value per prompt.")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_lora_rank", type=int, default=128)
    parser.add_argument("--max_model_len", type=int, default=None,
                        help="vLLM max_model_len. Default is max(mnts)+1024, or total_seqlen if set.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Recompute even when summary.json already exists.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else None
    adapter = Path(args.adapter) if args.adapter else None

    if run_dir and (args.model or adapter or args.name):
        raise SystemExit("Use either --run_dir, or --model with optional --adapter/--name.")
    if not run_dir and not args.model:
        raise SystemExit("Pass --model for raw/single-adapter eval, or --run_dir for a checkpoint sweep.")
    if adapter and args.mode != "greedy":
        raise SystemExit("LoRA adapter eval currently supports --mode greedy only.")
    if run_dir and args.mode != "greedy":
        raise SystemExit("LoRA checkpoint sweeps currently support --mode greedy only.")

    if run_dir:
        run_config = load_run_config(run_dir)
        model = run_config["student_model"]
        adapters = list_adapters(run_dir)
        if not adapters:
            raise SystemExit(f"No checkpoint-* or final adapters found in {run_dir}")
        base_output_dir = run_dir / "eval"
    else:
        model = args.model
        adapters = [adapter] if adapter else [None]
        base_output_dir = Path(args.output_dir)

    data_path = Path(args.data_dir) / f"{args.split}.jsonl"
    questions, gt_answers = load_gsm8k(data_path, args.max_examples)
    print(f"Loaded {len(questions)} GSM8K {args.split} examples from {data_path}")

    max_model_len = args.max_model_len or args.total_seqlen or (max(args.mnts) + 1024)
    enable_lora = any(p is not None for p in adapters)
    print(f"Loading vLLM: {model}")
    print(f"  enable_lora={enable_lora}, max_model_len={max_model_len}")
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    llm_kwargs = {
        "model": model,
        "tensor_parallel_size": args.tp,
        "dtype": "bfloat16",
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": True,
        "enable_lora": enable_lora,
        "max_model_len": max_model_len,
        "seed": args.seed,
    }
    if enable_lora:
        llm_kwargs.update({"max_lora_rank": args.max_lora_rank, "max_loras": 1})
    llm = LLM(**llm_kwargs)

    tokenizer = llm.get_tokenizer()
    prompts = [build_prompt(question, tokenizer) for question in questions]
    print("\n--- Example prompt ---")
    print(prompts[0][:500])
    print("...\n")

    summary = {}
    lora_id = 1
    for adapter_path in adapters:
        if adapter_path is None:
            eval_name = args.name or default_name(model, None, None)
            lora_request = None
            config_base = {"model": model, "adapter": None, "name": eval_name, "tp": args.tp}
        else:
            eval_name = args.name or default_name(model, adapter_path, run_dir)
            lora_request = LoRARequest(eval_name, lora_id, str(adapter_path))
            lora_id += 1
            config_base = {
                "model": model,
                "adapter": str(adapter_path),
                "name": eval_name,
                "tp": args.tp,
            }

        for mnt in args.mnts:
            if len(args.mnts) == 1:
                out_name = eval_name
            else:
                out_name = f"{eval_name}_greedy_mnt{mnt}_seed{args.seed}"
            out_dir = base_output_dir / out_name
            if not args.force and (out_dir / "summary.json").exists():
                print(f"[SKIP] {out_dir}/summary.json already exists")
                with (out_dir / "summary.json").open() as f:
                    summary[out_name] = json.load(f)[args.split]["accuracy"]
                continue

            acc = evaluate_once(
                llm=llm,
                prompts=prompts,
                gt_answers=gt_answers,
                output_dir=out_dir,
                config_base=config_base,
                mode=args.mode,
                mnt=mnt,
                total_seqlen=args.total_seqlen,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
                split=args.split,
                lora_request=lora_request,
            )
            summary[out_name] = round(acc, 4)

    if run_dir:
        summary_path = base_output_dir / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nRun summary written to {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

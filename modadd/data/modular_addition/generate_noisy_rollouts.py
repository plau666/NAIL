import argparse
import os
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.modular_addition.offline_render import (
    DTYPE_LOOKUP,
    ROLLOUT_MODE_CHOICES,
    TEACHER_LAW_CHOICES,
    render_offline_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render an offline modular-addition dataset from a fixed prompt bank "
            "using a clean teacher checkpoint. Use eta=0 for the clean offline BC sweep."
        )
    )
    parser.add_argument(
        "--teacher_checkpoint",
        "--teacher_out_dir",
        dest="teacher_checkpoint",
        type=str,
        required=True,
        help="Path to ckpt.pt or an out_dir containing ckpt.pt.",
    )
    parser.add_argument("--prompt_bank_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--subset_size", type=int, required=True)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument(
        "--teacher_law",
        choices=TEACHER_LAW_CHOICES,
        default="distributional_noise",
    )
    parser.add_argument(
        "--rollout_mode",
        choices=ROLLOUT_MODE_CHOICES,
        default="greedy_then_corrupt",
    )
    parser.add_argument("--gen_batch_size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=sorted(DTYPE_LOOKUP), default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--random_suffix_noise_key_positions", choices=("semantic_key",), default="semantic_key")
    parser.add_argument("--random_suffix_noise_trigger_eta", type=float, default=None)
    parser.add_argument("--random_suffix_noise_mode", choices=("valid_tokens",), default="valid_tokens")
    parser.add_argument("--random_suffix_noise_keep_format_tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--random_suffix_noise_seed", type=int, default=1337)
    parser.add_argument("--random_suffix_noise_apply_to", choices=("modadd",), default="modadd")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if "cuda" in args.device and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    render_offline_dataset(
        teacher_checkpoint=args.teacher_checkpoint,
        prompt_bank_dir=args.prompt_bank_dir,
        save_dir=args.save_dir,
        subset_size=args.subset_size,
        eta=args.eta,
        rollout_mode=args.rollout_mode,
        teacher_law=args.teacher_law,
        random_suffix_noise_config={
            "enabled": True,
            "key_positions": args.random_suffix_noise_key_positions,
            "trigger_eta": args.random_suffix_noise_trigger_eta,
            "random_suffix_mode": args.random_suffix_noise_mode,
            "keep_format_tokens": args.random_suffix_noise_keep_format_tokens,
            "seed": args.random_suffix_noise_seed,
            "apply_to": args.random_suffix_noise_apply_to,
        },
        gen_batch_size=args.gen_batch_size,
        device=args.device,
        dtype_name=args.dtype,
        seed=args.seed,
    )
    print(f"saved dataset to {args.save_dir}")


if __name__ == "__main__":
    main()

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.modular_addition.task import sample_cot_example_ids_from_rng


def lengths_from_m(m):
    return m + 1, m


def fill_bank_split(prompt_ids, cot_ids, *, rng, p, m, split_name, offset, total):
    report_every = 10_000

    for row in range(prompt_ids.size(0)):
        prompt_row, cot_row = sample_cot_example_ids_from_rng(rng, p=p, m=m)
        prompt_ids[row] = torch.tensor(prompt_row, dtype=torch.int32)
        cot_ids[row] = torch.tensor(cot_row, dtype=torch.int32)

        done = offset + row + 1
        if done % report_every == 0 or done == total:
            print(f"{split_name}: generated {done}/{total} prompt+cot pairs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--p", type=int, default=7)
    parser.add_argument("--m", type=int, default=21)
    parser.add_argument("--n_train", type=int, default=100000)
    parser.add_argument("--n_val", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    prompt_len, cot_len = lengths_from_m(args.m)
    rng = random.Random(args.seed)
    total = args.n_train + args.n_val

    clean_train_prompt_ids = torch.empty((args.n_train, prompt_len), dtype=torch.int32)
    clean_train_cot_ids = torch.empty((args.n_train, cot_len), dtype=torch.int32)
    clean_val_prompt_ids = torch.empty((args.n_val, prompt_len), dtype=torch.int32)
    clean_val_cot_ids = torch.empty((args.n_val, cot_len), dtype=torch.int32)

    fill_bank_split(
        clean_train_prompt_ids,
        clean_train_cot_ids,
        rng=rng,
        p=args.p,
        m=args.m,
        split_name="train",
        offset=0,
        total=total,
    )
    fill_bank_split(
        clean_val_prompt_ids,
        clean_val_cot_ids,
        rng=rng,
        p=args.p,
        m=args.m,
        split_name="val",
        offset=args.n_train,
        total=total,
    )

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_order = torch.randperm(args.n_train, generator=g)

    torch.save(clean_train_prompt_ids, os.path.join(args.save_dir, "clean_train_prompt_ids.pt"))
    torch.save(clean_train_cot_ids, os.path.join(args.save_dir, "clean_train_cot_ids.pt"))
    torch.save(clean_val_prompt_ids, os.path.join(args.save_dir, "clean_val_prompt_ids.pt"))
    torch.save(clean_val_cot_ids, os.path.join(args.save_dir, "clean_val_cot_ids.pt"))
    torch.save(train_order, os.path.join(args.save_dir, "train_order.pt"))

    meta = {
        "task": "modadd",
        "p": args.p,
        "m": args.m,
        "prompt_len": prompt_len,
        "cot_len": cot_len,
        "target_len": cot_len,
        "final_answer_len": 1,
        "answer_len": 1,
        "target_span": "cot_with_final_answer_suffix",
        "n_train": args.n_train,
        "n_val": args.n_val,
        "seed": args.seed,
        "nested_subset_order_saved": True,
        "duplicate_check_performed": False,
    }
    with open(os.path.join(args.save_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"saved clean prompt bank to {args.save_dir}")


if __name__ == "__main__":
    main()

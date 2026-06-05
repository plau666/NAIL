import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.s5_cot.task import sample_cot_example_ids_from_rng


def lengths_from_m(m):
    return 7 * m + 1, 7 * m


def fill_bank_split(prompt_ids, cot_ids, *, rng, m, split_name, offset, total):
    report_every = 10_000

    for row in range(prompt_ids.size(0)):
        prompt_row, cot_row = sample_cot_example_ids_from_rng(rng, m=m)
        prompt_ids[row] = torch.tensor(prompt_row, dtype=torch.uint8)
        cot_ids[row] = torch.tensor(cot_row, dtype=torch.uint8)

        done = offset + row + 1
        if done % report_every == 0 or done == total:
            print(f"{split_name}: generated {done}/{total} prompt+cot pairs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--m", type=int, default=21)
    parser.add_argument("--n_train", type=int, default=100000)
    parser.add_argument("--n_val", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    prompt_len, cot_len = lengths_from_m(args.m)
    rng = random.Random(args.seed)
    total = args.n_train + args.n_val

    # We intentionally skip duplicate filtering here. The prompt space is 120^m,
    # so at m=21 the probability of any collision in a 6M-example bank is
    # effectively zero while exact dedupe would dominate memory use.
    clean_train_prompt_ids = torch.empty((args.n_train, prompt_len), dtype=torch.uint8)
    clean_train_cot_ids = torch.empty((args.n_train, cot_len), dtype=torch.uint8)
    clean_val_prompt_ids = torch.empty((args.n_val, prompt_len), dtype=torch.uint8)
    clean_val_cot_ids = torch.empty((args.n_val, cot_len), dtype=torch.uint8)

    fill_bank_split(
        clean_train_prompt_ids,
        clean_train_cot_ids,
        rng=rng,
        m=args.m,
        split_name="train",
        offset=0,
        total=total,
    )
    fill_bank_split(
        clean_val_prompt_ids,
        clean_val_cot_ids,
        rng=rng,
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
        "m": args.m,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "seed": args.seed,
        "nested_subset_order_saved": True,
        "prompt_len": prompt_len,
        "cot_len": cot_len,
        "target_len": cot_len,
        "final_answer_len": 7,
        "answer_len": 7,
        "target_span": "cot_with_final_answer_suffix",
        "duplicate_check_performed": False,
        "duplicate_collision_probability_negligible": True,
    }
    with open(os.path.join(args.save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"saved clean prompt bank to {args.save_dir}")


if __name__ == "__main__":
    main()

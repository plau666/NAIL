from __future__ import annotations

import json
import os
import random
from dataclasses import asdict
from pathlib import Path

import torch

from data.modular_addition.offline_render import render_offline_dataset
from data.modular_addition.task import sample_cot_example_ids_from_rng
from nanogpt.config_schema import AppConfig
from nanogpt.utils.repo import write_launch_metadata


def _lengths_from_m(m: int) -> tuple[int, int]:
    return m + 1, m


def _fill_bank_split(
    prompt_ids: torch.Tensor,
    cot_ids: torch.Tensor,
    *,
    rng: random.Random,
    p: int,
    m: int,
    split_name: str,
    offset: int,
    total: int,
) -> None:
    report_every = 10_000
    for row in range(prompt_ids.size(0)):
        prompt_row, cot_row = sample_cot_example_ids_from_rng(rng, p=p, m=m)
        prompt_ids[row] = torch.tensor(prompt_row, dtype=torch.int32)
        cot_ids[row] = torch.tensor(cot_row, dtype=torch.int32)
        done = offset + row + 1
        if done % report_every == 0 or done == total:
            print(f"{split_name}: generated {done}/{total} prompt+cot pairs")


def run_modadd_prompt_bank(cfg: AppConfig, *, launcher_command: list[str]) -> None:
    save_dir = Path(cfg.task.prompt_bank_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    write_launch_metadata(save_dir, cfg=cfg, command=launcher_command)

    prompt_len, cot_len = _lengths_from_m(cfg.task.modadd_m)
    rng = random.Random(cfg.task.bank_seed)
    total = cfg.task.n_train + cfg.task.n_val

    clean_train_prompt_ids = torch.empty((cfg.task.n_train, prompt_len), dtype=torch.int32)
    clean_train_cot_ids = torch.empty((cfg.task.n_train, cot_len), dtype=torch.int32)
    clean_val_prompt_ids = torch.empty((cfg.task.n_val, prompt_len), dtype=torch.int32)
    clean_val_cot_ids = torch.empty((cfg.task.n_val, cot_len), dtype=torch.int32)

    _fill_bank_split(
        clean_train_prompt_ids,
        clean_train_cot_ids,
        rng=rng,
        p=cfg.task.modadd_p,
        m=cfg.task.modadd_m,
        split_name="train",
        offset=0,
        total=total,
    )
    _fill_bank_split(
        clean_val_prompt_ids,
        clean_val_cot_ids,
        rng=rng,
        p=cfg.task.modadd_p,
        m=cfg.task.modadd_m,
        split_name="val",
        offset=cfg.task.n_train,
        total=total,
    )

    g = torch.Generator()
    g.manual_seed(cfg.task.bank_seed)
    train_order = torch.randperm(cfg.task.n_train, generator=g)

    torch.save(clean_train_prompt_ids, save_dir / "clean_train_prompt_ids.pt")
    torch.save(clean_train_cot_ids, save_dir / "clean_train_cot_ids.pt")
    torch.save(clean_val_prompt_ids, save_dir / "clean_val_prompt_ids.pt")
    torch.save(clean_val_cot_ids, save_dir / "clean_val_cot_ids.pt")
    torch.save(train_order, save_dir / "train_order.pt")

    meta = {
        "task": "modadd",
        "p": cfg.task.modadd_p,
        "m": cfg.task.modadd_m,
        "prompt_len": prompt_len,
        "cot_len": cot_len,
        "target_len": cot_len,
        "final_answer_len": 1,
        "answer_len": 1,
        "target_span": "cot_with_final_answer_suffix",
        "n_train": cfg.task.n_train,
        "n_val": cfg.task.n_val,
        "seed": cfg.task.bank_seed,
        "nested_subset_order_saved": True,
        "duplicate_check_performed": False,
    }
    with open(save_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"saved clean prompt bank to {save_dir}")


def run_modadd_render(cfg: AppConfig, *, launcher_command: list[str]) -> None:
    save_dir = Path(cfg.task.data_root) / cfg.task.dataset
    save_dir.mkdir(parents=True, exist_ok=True)
    write_launch_metadata(save_dir, cfg=cfg, command=launcher_command)

    random.seed(cfg.task.render_seed)
    torch.manual_seed(cfg.task.render_seed)
    if "cuda" in cfg.runtime.device and torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.task.render_seed)

    render_offline_dataset(
        teacher_checkpoint=cfg.task.teacher_checkpoint,
        prompt_bank_dir=cfg.task.prompt_bank_dir,
        save_dir=save_dir,
        subset_size=cfg.task.subset_size,
        eta=cfg.task.eta,
        rollout_mode=cfg.task.rollout_mode,
        teacher_law=cfg.task.teacher_law,
        random_suffix_noise_config=asdict(cfg.task.random_suffix_noise),
        gen_batch_size=cfg.task.gen_batch_size,
        device=cfg.runtime.device,
        dtype_name=cfg.runtime.dtype,
        seed=cfg.task.render_seed,
    )
    print(f"saved dataset to {save_dir}")

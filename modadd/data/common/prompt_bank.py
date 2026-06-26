from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def label_dtype_for_token_dtype(token_dtype: torch.dtype) -> torch.dtype:
    if token_dtype in (torch.uint8, torch.int8, torch.int16):
        return torch.int16
    if token_dtype == torch.int32:
        return torch.int32
    return torch.int64


def _normalize_meta(
    meta: dict[str, Any],
    *,
    clean_train_prompt_ids: torch.Tensor,
    clean_train_cot_ids: torch.Tensor,
) -> dict[str, Any]:
    normalized = dict(meta)
    prompt_len = int(clean_train_prompt_ids.size(1))
    cot_len = int(clean_train_cot_ids.size(1))
    task = str(normalized.get("task", "modadd"))
    normalized["task"] = task
    normalized.setdefault("prompt_len", prompt_len)
    normalized.setdefault("cot_len", cot_len)
    normalized.setdefault("target_len", cot_len)

    if task == "modadd":
        normalized.setdefault("p", int(clean_train_prompt_ids.max().item()))
        normalized.setdefault("m", cot_len)
        normalized.setdefault("final_answer_len", 1)
    else:
        normalized.setdefault("m", cot_len)
        normalized.setdefault("final_answer_len", cot_len)

    normalized.setdefault("answer_len", normalized["final_answer_len"])
    normalized.setdefault("target_span", "cot_with_final_answer_suffix")
    if int(normalized["target_len"]) != cot_len:
        raise ValueError(
            f"PromptBank target_len={normalized['target_len']} does not match "
            f"clean_train_cot_ids width={cot_len}"
        )
    if int(normalized["answer_len"]) != int(normalized["final_answer_len"]):
        raise ValueError(
            f"PromptBank answer_len={normalized['answer_len']} does not match "
            f"final_answer_len={normalized['final_answer_len']}"
        )
    return normalized


@dataclass
class PromptBank:
    clean_train_prompt_ids: torch.Tensor
    clean_train_cot_ids: torch.Tensor
    clean_val_prompt_ids: torch.Tensor
    clean_val_cot_ids: torch.Tensor
    train_order: torch.Tensor
    meta: dict[str, Any]

    def __post_init__(self) -> None:
        self.meta = _normalize_meta(
            self.meta,
            clean_train_prompt_ids=self.clean_train_prompt_ids,
            clean_train_cot_ids=self.clean_train_cot_ids,
        )

    @property
    def prompt_len(self) -> int:
        return int(self.meta["prompt_len"])

    @property
    def cot_len(self) -> int:
        return int(self.meta["cot_len"])

    @property
    def target_len(self) -> int:
        return int(self.meta["target_len"])

    @property
    def xy_len(self) -> int:
        return self.prompt_len + self.target_len - 1

    @property
    def m(self) -> int:
        return int(self.meta["m"])

    @property
    def p(self) -> int:
        return int(self.meta["p"])

    @property
    def task(self) -> str:
        return str(self.meta["task"])

    @property
    def final_answer_len(self) -> int:
        return int(self.meta["final_answer_len"])

    @property
    def answer_len(self) -> int:
        return int(self.meta["answer_len"])

    @property
    def token_dtype(self) -> torch.dtype:
        return self.clean_train_prompt_ids.dtype

    @property
    def label_dtype(self) -> torch.dtype:
        return label_dtype_for_token_dtype(self.token_dtype)


def load_prompt_bank(prompt_bank_dir: str | Path) -> PromptBank:
    prompt_bank_dir = Path(prompt_bank_dir)
    meta_path = prompt_bank_dir / "meta.json"
    meta: dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    clean_train_prompt_ids = torch.load(
        prompt_bank_dir / "clean_train_prompt_ids.pt",
        map_location="cpu",
    )
    clean_train_cot_ids = torch.load(
        prompt_bank_dir / "clean_train_cot_ids.pt",
        map_location="cpu",
    )
    return PromptBank(
        clean_train_prompt_ids=clean_train_prompt_ids,
        clean_train_cot_ids=clean_train_cot_ids,
        clean_val_prompt_ids=torch.load(
            prompt_bank_dir / "clean_val_prompt_ids.pt",
            map_location="cpu",
        ),
        clean_val_cot_ids=torch.load(
            prompt_bank_dir / "clean_val_cot_ids.pt",
            map_location="cpu",
        ),
        train_order=torch.load(prompt_bank_dir / "train_order.pt", map_location="cpu"),
        meta=_normalize_meta(
            meta,
            clean_train_prompt_ids=clean_train_prompt_ids,
            clean_train_cot_ids=clean_train_cot_ids,
        ),
    )


def select_train_subset(prompt_bank: PromptBank, subset_size: int) -> torch.Tensor:
    if subset_size < 0:
        raise ValueError(f"subset_size={subset_size} must be non-negative")
    if subset_size > prompt_bank.clean_train_prompt_ids.size(0):
        raise ValueError(
            f"subset_size={subset_size} exceeds prompt bank size "
            f"{prompt_bank.clean_train_prompt_ids.size(0)}"
        )
    return prompt_bank.train_order[:subset_size]


def build_xy_from_prompt_and_target(
    prompt_ids: torch.Tensor,
    target_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_dtype = prompt_ids.dtype
    label_dtype = label_dtype_for_token_dtype(token_dtype)
    prompt_ids = prompt_ids.to(dtype=token_dtype)
    target_ids = target_ids.to(dtype=token_dtype)
    seq = torch.cat((prompt_ids, target_ids), dim=1)
    x = seq[:, :-1].clone().contiguous()
    y = seq[:, 1:].to(dtype=label_dtype).clone().contiguous()
    y[:, :prompt_ids.size(1) - 1] = -1
    return x, y

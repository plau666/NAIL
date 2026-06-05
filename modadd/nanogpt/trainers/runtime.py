from __future__ import annotations

import random
from contextlib import nullcontext

import numpy as np
import torch

from torch_dtypes import DTYPE_LOOKUP


def resolve_device(device_arg: str | None) -> str:
    if device_arg is not None:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(dtype_name: str | None, device: str) -> torch.dtype:
    if dtype_name is not None:
        return DTYPE_LOOKUP[dtype_name]
    if "cuda" in device:
        return torch.float16
    return torch.float32


def build_autocast_context(device: str, torch_dtype: torch.dtype):
    if "cuda" not in device:
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=torch_dtype)


def build_grad_scaler(*, device: str, torch_dtype: torch.dtype):
    enabled = ("cuda" in device and torch_dtype == torch.float16)
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def get_nanogpt_lr(
    step: int,
    *,
    learning_rate: float,
    warmup_iters: int,
    lr_decay_iters: int,
    min_lr: float,
) -> float:
    lr_start = 1e-6

    if warmup_iters > 0 and step < warmup_iters:
        return lr_start + (learning_rate - lr_start) * (step + 1) / (warmup_iters + 1)

    if lr_decay_iters <= warmup_iters:
        return learning_rate

    if step > lr_decay_iters:
        return min_lr

    decay_ratio = (step - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def capture_rng_state(device: str) -> dict[str, object]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if "cuda" in device and torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, object], device: str) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "cuda" in device and torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])

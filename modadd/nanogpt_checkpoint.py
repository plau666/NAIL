from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from model import GPT, GPTConfig


UNWANTED_PREFIX = "_orig_mod."


def resolve_checkpoint_path(checkpoint_or_dir: str | Path) -> Path:
    path = Path(checkpoint_or_dir)
    if path.is_dir():
        path = path / "ckpt.pt"
    if not path.exists():
        raise FileNotFoundError(f"Could not find checkpoint at {path}")
    return path


def load_nanogpt_checkpoint(
    checkpoint_or_dir: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    ckpt_path = resolve_checkpoint_path(checkpoint_or_dir)
    return torch.load(ckpt_path, map_location=map_location, weights_only=False)


def normalize_nanogpt_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith(UNWANTED_PREFIX):
            key = key[len(UNWANTED_PREFIX):]
        cleaned[key] = value
    return cleaned


def build_nanogpt_model(model_args: Mapping[str, Any]) -> GPT:
    return GPT(GPTConfig(**dict(model_args)))


def load_nanogpt_model(
    checkpoint_or_dir: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    device: str | torch.device | None = None,
    eval_mode: bool = True,
    strict: bool = True,
    return_checkpoint: bool = False,
):
    checkpoint = load_nanogpt_checkpoint(checkpoint_or_dir, map_location=map_location)
    model = build_nanogpt_model(checkpoint["model_args"])
    state_dict = normalize_nanogpt_state_dict(checkpoint["model"])
    model.load_state_dict(state_dict, strict=strict)

    if device is not None:
        model.to(device)
    if eval_mode:
        model.eval()

    if return_checkpoint:
        return model, checkpoint
    return model

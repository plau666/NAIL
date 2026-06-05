from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch

from data.s5_cot.task import CORRUPTIBLE_IDS, stoi


S5_BLOCK_LEN = 7
S5_NUM_COORDS = 5
S5_VALUE_OFFSET = 1
SEMANTIC_KEY_NOISE_LAW = "semantic_key_noise"
SEMANTIC_KEY_COORD_STRATEGIES = ("fixed", "cyclic", "hash")
SEMANTIC_KEY_APPLY_TO = ("partial_perm_image",)
SEMANTIC_KEY_CONFIG_KEYS = (
    "enabled",
    "coord_strategy",
    "fixed_coord",
    "seed",
    "include_clean_value",
    "eligible_values",
    "apply_to",
    "one_key_per_block",
)


@dataclass(frozen=True)
class SemanticKeyNoiseConfig:
    enabled: bool = True
    coord_strategy: str = "cyclic"
    fixed_coord: int = 0
    seed: int = 1337
    include_clean_value: bool = True
    eligible_values: tuple[int, ...] = (1, 2, 3, 4, 5)
    apply_to: str = "partial_perm_image"
    one_key_per_block: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["eligible_values"] = list(self.eligible_values)
        payload["eligible_token_ids"] = eligible_token_ids_from_values(self.eligible_values)
        return payload


def semantic_key_noise_config_from_obj(value: Mapping[str, Any] | object | None) -> SemanticKeyNoiseConfig:
    if value is None:
        return SemanticKeyNoiseConfig()
    if isinstance(value, SemanticKeyNoiseConfig):
        validate_semantic_key_noise_config(value)
        return value
    if isinstance(value, Mapping):
        raw = {key: value[key] for key in SEMANTIC_KEY_CONFIG_KEYS if key in value}
    else:
        raw = {
            key: getattr(value, key)
            for key in SEMANTIC_KEY_CONFIG_KEYS
            if hasattr(value, key)
        }
    if "eligible_values" in raw and raw["eligible_values"] is not None:
        raw["eligible_values"] = tuple(int(x) for x in raw["eligible_values"])
    config = SemanticKeyNoiseConfig(**raw)
    validate_semantic_key_noise_config(config)
    return config


def validate_semantic_key_noise_config(config: SemanticKeyNoiseConfig) -> None:
    if not config.enabled:
        raise ValueError("semantic_key_noise.enabled must be true when using semantic_key_noise")
    if config.coord_strategy not in SEMANTIC_KEY_COORD_STRATEGIES:
        raise ValueError(
            f"unknown semantic_key_noise coord_strategy={config.coord_strategy!r}; "
            f"expected one of {SEMANTIC_KEY_COORD_STRATEGIES}"
        )
    if not 0 <= int(config.fixed_coord) < S5_NUM_COORDS:
        raise ValueError(
            f"semantic_key_noise.fixed_coord={config.fixed_coord} must be in [0, {S5_NUM_COORDS - 1}]"
        )
    if not config.include_clean_value:
        raise ValueError("semantic_key_noise.include_clean_value must remain true for this teacher law")
    if config.apply_to not in SEMANTIC_KEY_APPLY_TO:
        raise ValueError(
            f"unsupported semantic_key_noise.apply_to={config.apply_to!r}; "
            f"expected one of {SEMANTIC_KEY_APPLY_TO}"
        )
    if not config.one_key_per_block:
        raise ValueError("semantic_key_noise.one_key_per_block must remain true")
    if len(config.eligible_values) == 0:
        raise ValueError("semantic_key_noise.eligible_values must be non-empty")
    invalid_values = [int(value) for value in config.eligible_values if int(value) not in range(1, 6)]
    if invalid_values:
        raise ValueError(
            f"semantic_key_noise.eligible_values must be S5 values 1..5, got {invalid_values}"
        )
    if sorted(set(int(value) for value in config.eligible_values)) != [1, 2, 3, 4, 5]:
        raise ValueError(
            "semantic_key_noise.eligible_values must include all S5 values [1, 2, 3, 4, 5] "
            "so the clean value is always in the uniform support"
        )


def eligible_token_ids_from_values(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(stoi[str(int(value))] for value in values)


def default_eligible_token_ids() -> tuple[int, ...]:
    return tuple(int(token_id) for token_id in CORRUPTIBLE_IDS)


def target_block_coord_for_positions(
    target_len: int,
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = torch.arange(int(target_len), dtype=torch.long, device=device)
    block_idx = positions // S5_BLOCK_LEN
    offset = positions % S5_BLOCK_LEN
    coord_idx = offset - S5_VALUE_OFFSET
    is_value = (coord_idx >= 0) & (coord_idx < S5_NUM_COORDS)
    return block_idx, coord_idx, is_value


def selected_coordinate_for_block(
    block_idx: int,
    *,
    prompt_ids: torch.Tensor | None,
    config: SemanticKeyNoiseConfig,
) -> int:
    strategy = config.coord_strategy
    if strategy == "fixed":
        return int(config.fixed_coord)
    if strategy == "cyclic":
        return int(block_idx) % S5_NUM_COORDS
    if strategy == "hash":
        if prompt_ids is None:
            raise ValueError("hash semantic_key_noise coordinate selection requires prompt_ids")
        return _hash_selected_coordinate(prompt_ids, block_idx=int(block_idx), seed=int(config.seed))
    raise ValueError(f"unknown semantic_key_noise coord_strategy={strategy!r}")


def semantic_key_mask(
    prompt_ids: torch.Tensor,
    target_len: int,
    config: SemanticKeyNoiseConfig | Mapping[str, Any] | object | None = None,
) -> torch.Tensor:
    config = semantic_key_noise_config_from_obj(config)
    prompt_ids = prompt_ids.detach().to(device="cpu", dtype=torch.long)
    if prompt_ids.ndim != 2:
        raise ValueError(f"prompt_ids must be rank-2 [B, prompt_len], got shape {tuple(prompt_ids.shape)}")
    target_len = int(target_len)
    if target_len < 0:
        raise ValueError(f"target_len must be non-negative, got {target_len}")

    block_idx, coord_idx, is_value = target_block_coord_for_positions(target_len, device="cpu")
    mask = torch.zeros((prompt_ids.size(0), target_len), dtype=torch.bool)
    if target_len == 0:
        return mask

    if config.coord_strategy == "hash":
        for row_idx in range(prompt_ids.size(0)):
            for pos in range(target_len):
                if not bool(is_value[pos].item()):
                    continue
                block = int(block_idx[pos].item())
                selected = selected_coordinate_for_block(
                    block,
                    prompt_ids=prompt_ids[row_idx],
                    config=config,
                )
                mask[row_idx, pos] = int(coord_idx[pos].item()) == selected
        return mask

    selected_by_position = torch.empty(target_len, dtype=torch.long)
    for pos in range(target_len):
        block = int(block_idx[pos].item())
        selected_by_position[pos] = selected_coordinate_for_block(
            block,
            prompt_ids=None,
            config=config,
        )
    row_mask = is_value & coord_idx.eq(selected_by_position)
    mask[:, :] = row_mask.view(1, target_len)
    return mask


def semantic_key_mask_for_step(
    prompt_ids: torch.Tensor,
    step: int,
    config: SemanticKeyNoiseConfig | Mapping[str, Any] | object | None = None,
) -> torch.Tensor:
    config = semantic_key_noise_config_from_obj(config)
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    block = int(step) // S5_BLOCK_LEN
    coord = int(step) % S5_BLOCK_LEN - S5_VALUE_OFFSET
    prompt_cpu = prompt_ids.detach().to(device="cpu", dtype=torch.long)
    if prompt_cpu.ndim != 2:
        raise ValueError(f"prompt_ids must be rank-2 [B, prompt_len], got shape {tuple(prompt_cpu.shape)}")
    if coord < 0 or coord >= S5_NUM_COORDS:
        return torch.zeros(prompt_cpu.size(0), dtype=torch.bool, device=prompt_ids.device)
    if config.coord_strategy == "hash":
        selected = torch.tensor(
            [
                selected_coordinate_for_block(
                    block,
                    prompt_ids=prompt_cpu[row_idx],
                    config=config,
                )
                for row_idx in range(prompt_cpu.size(0))
            ],
            dtype=torch.long,
            device=prompt_ids.device,
        )
        return selected.eq(coord)
    selected_coord = selected_coordinate_for_block(block, prompt_ids=None, config=config)
    return torch.full(
        (prompt_cpu.size(0),),
        bool(coord == selected_coord),
        dtype=torch.bool,
        device=prompt_ids.device,
    )


def _hash_selected_coordinate(prompt_ids_row: torch.Tensor, *, block_idx: int, seed: int) -> int:
    row = prompt_ids_row.detach().to(device="cpu", dtype=torch.long).flatten().tolist()
    digest = hashlib.blake2b(digest_size=8)
    digest.update(int(seed).to_bytes(8, byteorder="little", signed=True))
    digest.update(int(block_idx).to_bytes(8, byteorder="little", signed=False))
    for token_id in row:
        digest.update(int(token_id).to_bytes(2, byteorder="little", signed=False))
    return int.from_bytes(digest.digest(), byteorder="little") % S5_NUM_COORDS

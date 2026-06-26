from __future__ import annotations

import random
from collections.abc import Sequence

import torch


def _normalize_id_sequence(
    token_ids: Sequence[int] | torch.Tensor,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    return torch.as_tensor(list(token_ids), dtype=torch.long, device=device)


def corrupt_token(
    tok_id: int,
    eta: float,
    *,
    corruptible_token_ids: Sequence[int],
    replacement_token_ids: Sequence[int] | None = None,
) -> int:
    if eta <= 0:
        return int(tok_id)
    replacement_ids = tuple(
        corruptible_token_ids if replacement_token_ids is None else replacement_token_ids
    )
    if int(tok_id) in set(int(x) for x in corruptible_token_ids) and random.random() < eta:
        return int(random.choice(replacement_ids))
    return int(tok_id)


def corrupt_ids(
    ids: torch.Tensor,
    eta: float,
    *,
    corruptible_token_ids: Sequence[int] | torch.Tensor,
    replacement_token_ids: Sequence[int] | torch.Tensor | None = None,
) -> torch.Tensor:
    if eta <= 0:
        return ids

    corruptible_ids = _normalize_id_sequence(corruptible_token_ids, device=ids.device)
    if corruptible_ids.numel() == 0:
        return ids

    should_corrupt = (
        ids.unsqueeze(-1).eq(corruptible_ids.view(*([1] * ids.ndim), -1)).any(dim=-1)
        & (torch.rand(ids.shape, device=ids.device) < eta)
    )
    if not torch.any(should_corrupt):
        return ids

    replacement_ids = (
        corruptible_ids
        if replacement_token_ids is None
        else _normalize_id_sequence(replacement_token_ids, device=ids.device)
    )

    corrupted = ids.clone()
    replacements = replacement_ids[
        torch.randint(
            low=0,
            high=int(replacement_ids.numel()),
            size=(int(should_corrupt.sum().item()),),
            device=ids.device,
        )
    ].to(dtype=ids.dtype)
    corrupted[should_corrupt] = replacements
    return corrupted

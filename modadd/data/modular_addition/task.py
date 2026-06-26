"""Modular-addition synthetic task used in the paper experiments.

See `modadd/README.md` for the paper-to-code map. This file defines the
symbolic vocabulary, clean CoT target, corruption support, and clean-task
evaluation metrics for the p=7, m=31 suite.
"""

from __future__ import annotations

import random

import torch

from data.common.corruption import (
    corrupt_ids as generic_corrupt_ids,
    corrupt_token as generic_corrupt_token,
)
from data.common.eval import (
    estimate_saved_clean_train_loss as shared_estimate_saved_clean_train_loss,
    evaluate_saved_clean_metrics,
    greedy_generate_target_ids_batched,
    teacher_forced_exact_batch,
)


def validate_task_params(*, p: int, m: int | None = None) -> None:
    if int(p) < 2:
        raise ValueError(f"p must be >= 2, got {p}")
    if m is not None and int(m) < 1:
        raise ValueError(f"m must be >= 1, got {m}")


def tokens_for_p(p: int) -> list[str]:
    validate_task_params(p=p)
    return [str(i) for i in range(p)] + ["="]


def stoi_for_p(p: int) -> dict[str, int]:
    return {token: idx for idx, token in enumerate(tokens_for_p(p))}


def itos_for_p(p: int) -> dict[int, str]:
    stoi = stoi_for_p(p)
    return {idx: token for token, idx in stoi.items()}


def vocab_size(p: int) -> int:
    validate_task_params(p=p)
    return int(p) + 1


def equals_token_id(p: int) -> int:
    validate_task_params(p=p)
    return int(p)


def corruptible_token_ids(p: int) -> tuple[int, ...]:
    # All residues are semantic answer/running-sum tokens; the equals token is
    # prompt scaffold and is never sampled as random-suffix feedback.
    validate_task_params(p=p)
    return tuple(range(int(p)))


def encode(tokens: list[str], *, p: int) -> list[int]:
    stoi = stoi_for_p(p)
    return [stoi[token] for token in tokens]


def decode(ids, *, p: int) -> list[str]:
    itos = itos_for_p(p)
    return [itos[int(i)] for i in ids]


def validate_mode(mode: str) -> None:
    if mode not in ("cot", "base"):
        raise ValueError(f"mode must be 'cot' or 'base', got {mode!r}")


def sample_target_ids_from_rng(rng: random.Random, *, p: int, m: int, mode: str) -> tuple[list[int], list[int]]:
    validate_task_params(p=p, m=m)
    validate_mode(mode)
    prompt_ids = []
    cot_ids = []
    running = 0
    for _ in range(m):
        value = rng.randrange(p)
        prompt_ids.append(value)
        running = (running + value) % p
        cot_ids.append(running)
    prompt_ids.append(equals_token_id(p))
    if mode == "cot":
        return prompt_ids, cot_ids
    return prompt_ids, [running]


def sample_cot_example_ids_from_rng(rng: random.Random, *, p: int, m: int) -> tuple[list[int], list[int]]:
    return sample_target_ids_from_rng(rng, p=p, m=m, mode="cot")


def sample_base_example_ids_from_rng(rng: random.Random, *, p: int, m: int) -> tuple[list[int], list[int]]:
    return sample_target_ids_from_rng(rng, p=p, m=m, mode="base")


def sample_cot_example(*, p: int, m: int) -> tuple[list[str], list[str]]:
    prompt_ids, cot_ids = sample_cot_example_ids_from_rng(random, p=p, m=m)
    return decode(prompt_ids, p=p), decode(cot_ids, p=p)


def sample_base_example(*, p: int, m: int) -> tuple[list[str], list[str]]:
    prompt_ids, target_ids = sample_base_example_ids_from_rng(random, p=p, m=m)
    return decode(prompt_ids, p=p), decode(target_ids, p=p)


def sample_xy(*, p: int = 7, m: int = 21, mode: str = "cot"):
    validate_task_params(p=p, m=m)
    validate_mode(mode)
    prompt_ids, target_ids = sample_target_ids_from_rng(random, p=p, m=m, mode=mode)
    seq = prompt_ids + target_ids

    x = torch.tensor(seq[:-1], dtype=torch.long)
    y = torch.tensor(seq[1:], dtype=torch.long)

    prompt_len = len(prompt_ids)
    y[:prompt_len - 1] = -1
    return x, y


def get_batch(batch_size, device, *, p: int = 7, m: int = 21, mode: str = "cot"):
    validate_task_params(p=p, m=m)
    validate_mode(mode)
    xs, ys = [], []
    for _ in range(batch_size):
        x, y = sample_xy(p=p, m=m, mode=mode)
        xs.append(x)
        ys.append(y)
    x_batch = torch.stack(xs).to(device)
    y_batch = torch.stack(ys).to(device)
    return x_batch, y_batch


def continuation_exact_from_logits(logits, y):
    pred = logits.argmax(dim=-1)
    mask = y != -1
    ok = torch.logical_or(pred.eq(y), ~mask)
    return ok.all(dim=1).float().mean().item()


def corrupt_token(tok_id, eta, *, p: int):
    validate_task_params(p=p)
    return generic_corrupt_token(
        tok_id,
        eta,
        corruptible_token_ids=corruptible_token_ids(p),
    )


def corrupt_ids(ids, eta, *, p: int):
    validate_task_params(p=p)
    return generic_corrupt_ids(
        ids,
        eta,
        corruptible_token_ids=corruptible_token_ids(p),
    )


def _sample_eval_batch_from_rng(rng, batch_n, *, p: int, m: int, mode: str = "cot"):
    validate_task_params(p=p, m=m)
    validate_mode(mode)
    prompt_batches = []
    target_batches = []
    for _ in range(batch_n):
        prompt_ids, target_ids = sample_target_ids_from_rng(rng, p=p, m=m, mode=mode)
        prompt_batches.append(prompt_ids)
        target_batches.append(target_ids)
    prompt_ids_batch = torch.tensor(prompt_batches, dtype=torch.long)
    target_ids_batch = torch.tensor(target_batches, dtype=torch.long)
    return prompt_ids_batch, target_ids_batch


@torch.no_grad()
def evaluate_clean_modadd_metrics(
    model,
    *,
    device,
    p: int = 7,
    m: int = 21,
    n_eval=256,
    seed=123,
    batch_size=256,
    mode: str = "cot",
):
    validate_task_params(p=p, m=m)
    validate_mode(mode)
    rng = random.Random(seed)

    tf_full_ok = 0
    ar_full_ok = 0
    ar_final_ok = 0

    for start in range(0, n_eval, batch_size):
        end = min(start + batch_size, n_eval)
        prompt_ids_batch, target_ids_batch = _sample_eval_batch_from_rng(rng, end - start, p=p, m=m, mode=mode)

        tf_ok = teacher_forced_exact_batch(model, prompt_ids_batch, target_ids_batch, device)
        tf_full_ok += int(tf_ok.sum().item())

        pred_ids_batch = greedy_generate_target_ids_batched(
            model,
            prompt_ids_batch,
            target_ids_batch.size(1),
            device,
        )
        ar_full_ok += int(pred_ids_batch.eq(target_ids_batch).all(dim=1).sum().item())
        ar_final_ok += int(pred_ids_batch[:, -1:].eq(target_ids_batch[:, -1:]).all(dim=1).sum().item())

    return {
        "cot_exact": tf_full_ok / n_eval,
        "clean_full_exact": ar_full_ok / n_eval,
        "clean_final_exact": ar_final_ok / n_eval,
    }


@torch.no_grad()
def evaluate_saved_clean_modadd_metrics(
    model,
    *,
    device,
    data_dir,
    n_eval=None,
    batch_size=256,
):
    return evaluate_saved_clean_metrics(
        model,
        device=device,
        data_dir=data_dir,
        final_answer_len=1,
        n_eval=n_eval,
        batch_size=batch_size,
    )


@torch.no_grad()
def estimate_saved_clean_train_loss(
    model,
    *,
    device,
    data_dir,
    eval_iters,
    batch_size,
    subset_size=None,
):
    return shared_estimate_saved_clean_train_loss(
        model,
        device=device,
        data_dir=data_dir,
        eval_iters=eval_iters,
        batch_size=batch_size,
        subset_size=subset_size,
    )

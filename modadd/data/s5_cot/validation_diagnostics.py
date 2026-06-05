from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from data.s5_cot.semantic_key_noise import (
    SEMANTIC_KEY_NOISE_LAW,
    semantic_key_mask,
    semantic_key_noise_config_from_obj,
)
from data.s5_cot.task import CORRUPTIBLE_IDS, decode
from data.synthetic.eval import greedy_generate_target_ids_batched
from data.synthetic.prompt_bank import build_xy_from_prompt_and_target


def decode_s5_ids(ids: torch.Tensor | Iterable[int]) -> str:
    if isinstance(ids, torch.Tensor):
        values = ids.detach().cpu().to(dtype=torch.long).flatten().tolist()
    else:
        values = [int(x) for x in ids]
    return "".join(decode(values))


def _nanmean(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return math.nan
    return float(values.float().mean().item())


def _nanmedian(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return math.nan
    return float(values.float().median().item())


def _safe_mean_masked(values: torch.Tensor, mask: torch.Tensor) -> float:
    if not torch.any(mask):
        return math.nan
    return float(values[mask].float().mean().item())


def _safe_acc(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    if not torch.any(mask):
        return math.nan
    return float(pred[mask].eq(target[mask]).float().mean().item())


def s5_corruptible_mask(target_ids: torch.Tensor) -> torch.Tensor:
    corruptible = torch.as_tensor(
        CORRUPTIBLE_IDS,
        dtype=target_ids.dtype,
        device=target_ids.device,
    )
    return target_ids.unsqueeze(-1).eq(
        corruptible.view(*([1] * target_ids.ndim), -1)
    ).any(dim=-1)


def extract_supervised_targets(y: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Return the contiguous supervised suffix from an offline-BC label tensor."""
    if y.ndim != 2:
        raise ValueError(f"expected Y to be rank-2 [B, T], got shape {tuple(y.shape)}")
    if y.size(0) == 0:
        raise ValueError("cannot extract targets from an empty batch")

    mask = y.ne(-1)
    if not torch.any(mask):
        raise ValueError("Y contains no supervised targets")
    first_row_mask = mask[0]
    true_positions = torch.nonzero(first_row_mask, as_tuple=False).flatten()
    target_start = int(true_positions[0].item())
    expected = torch.zeros_like(first_row_mask, dtype=torch.bool)
    expected[target_start:] = True
    if not torch.equal(first_row_mask, expected):
        raise ValueError("offline BC targets must be a contiguous suffix of Y")
    if not torch.equal(mask, expected.unsqueeze(0).expand_as(mask)):
        raise ValueError("all rows in an offline BC batch must share the same target mask")
    return y[:, target_start:].to(dtype=torch.long), target_start


def _token_stats_from_logits(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    final_answer_len: int,
    loss_threshold: float,
    greedy_targets: torch.Tensor | None = None,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    clean_targets, target_start = extract_supervised_targets(y)
    target_logits = logits[:, target_start:target_start + clean_targets.size(1), :]
    if target_logits.shape[:2] != clean_targets.shape:
        raise ValueError(
            f"logits target slice shape {tuple(target_logits.shape[:2])} does not "
            f"match clean target shape {tuple(clean_targets.shape)}"
        )

    log_probs = F.log_softmax(target_logits.float(), dim=-1)
    probs = log_probs.exp()
    clean_log_probs = log_probs.gather(2, clean_targets.unsqueeze(-1)).squeeze(-1)
    clean_probs = clean_log_probs.exp()
    token_loss = -clean_log_probs
    pred = probs.argmax(dim=-1)
    top2_probs, top2_ids = probs.topk(k=min(2, probs.size(-1)), dim=-1)
    if top2_probs.size(-1) == 1:
        top1_margin = top2_probs[..., 0]
    else:
        top1_margin = top2_probs[..., 0] - top2_probs[..., 1]
    entropy = -(probs * log_probs).sum(dim=-1)

    target_len = int(clean_targets.size(1))
    final_answer_len = min(max(int(final_answer_len), 0), target_len)
    final_mask = torch.zeros_like(clean_targets, dtype=torch.bool)
    if final_answer_len > 0:
        final_mask[:, target_len - final_answer_len:] = True
    cot_body_mask = ~final_mask
    corruptible_mask = s5_corruptible_mask(clean_targets)
    noncorruptible_mask = ~corruptible_mask

    seq_nll = token_loss.mean(dim=1)
    full_exact = None
    final_exact = None
    if greedy_targets is not None:
        if greedy_targets.shape != clean_targets.shape:
            raise ValueError(
                f"greedy target shape {tuple(greedy_targets.shape)} does not match "
                f"clean target shape {tuple(clean_targets.shape)}"
            )
        token_exact = greedy_targets.to(clean_targets.device).eq(clean_targets)
        full_exact = token_exact.all(dim=1)
        final_exact = (
            token_exact[:, -final_answer_len:].all(dim=1)
            if final_answer_len > 0
            else token_exact.all(dim=1)
        )

    flat_loss = token_loss.flatten()
    flat_probs = clean_probs.flatten()
    flat_log_probs = clean_log_probs.flatten()
    flat_margin = top1_margin.flatten()
    flat_entropy = entropy.flatten()

    metrics: dict[str, float] = {
        "teacher_forced_token_acc": float(pred.eq(clean_targets).float().mean().item()),
        "clean_token_prob_mean": _nanmean(flat_probs),
        "clean_token_prob_median": _nanmedian(flat_probs),
        "clean_token_logprob_mean": _nanmean(flat_log_probs),
        "top1_margin_mean": _nanmean(flat_margin),
        "top1_margin_median": _nanmedian(flat_margin),
        "top1_entropy_mean": _nanmean(flat_entropy),
        "top1_entropy_median": _nanmedian(flat_entropy),
        "clean_ce_loss": _nanmean(flat_loss),
        "corruptible_token_loss": _safe_mean_masked(token_loss, corruptible_mask),
        "noncorruptible_token_loss": _safe_mean_masked(token_loss, noncorruptible_mask),
        "corruptible_token_acc": _safe_acc(pred, clean_targets, corruptible_mask),
        "noncorruptible_token_acc": _safe_acc(pred, clean_targets, noncorruptible_mask),
        "corruptible_clean_token_prob_mean": _safe_mean_masked(clean_probs, corruptible_mask),
        "noncorruptible_clean_token_prob_mean": _safe_mean_masked(clean_probs, noncorruptible_mask),
        "cot_body_loss": _safe_mean_masked(token_loss, cot_body_mask),
        "final_answer_loss": _safe_mean_masked(token_loss, final_mask),
        "cot_body_token_acc": _safe_acc(pred, clean_targets, cot_body_mask),
        "final_answer_token_acc": _safe_acc(pred, clean_targets, final_mask),
        "cot_body_clean_token_prob_mean": _safe_mean_masked(clean_probs, cot_body_mask),
        "final_answer_clean_token_prob_mean": _safe_mean_masked(clean_probs, final_mask),
        "sequence_nll_mean": _nanmean(seq_nll),
        "sequence_nll_median": _nanmedian(seq_nll),
    }

    if full_exact is not None and final_exact is not None:
        metrics["sequence_nll_when_full_exact"] = _safe_mean_masked(seq_nll, full_exact)
        metrics["sequence_nll_when_final_exact"] = _safe_mean_masked(seq_nll, final_exact)
        full_exact_count = int(full_exact.sum().item())
        metrics["fraction_full_exact_with_loss_gt_threshold"] = (
            math.nan
            if full_exact_count == 0
            else float(
                (full_exact & seq_nll.gt(float(loss_threshold))).sum().item()
                / full_exact_count
            )
        )

    tensors = {
        "clean_targets": clean_targets.detach().cpu(),
        "clean_probs": clean_probs.detach().cpu(),
        "clean_log_probs": clean_log_probs.detach().cpu(),
        "token_loss": token_loss.detach().cpu(),
        "pred": pred.detach().cpu(),
        "top1_margin": top1_margin.detach().cpu(),
        "top2_probs": top2_probs.detach().cpu(),
        "top2_ids": top2_ids.detach().cpu(),
        "entropy": entropy.detach().cpu(),
        "seq_nll": seq_nll.detach().cpu(),
        "corruptible_mask": corruptible_mask.detach().cpu(),
        "final_mask": final_mask.detach().cpu(),
    }
    if full_exact is not None:
        tensors["full_exact"] = full_exact.detach().cpu()
    if final_exact is not None:
        tensors["final_exact"] = final_exact.detach().cpu()
    return metrics, tensors


def _metrics_from_token_tensors(
    token_tensors: dict[str, torch.Tensor],
    *,
    loss_threshold: float,
) -> dict[str, float]:
    clean_targets = token_tensors["clean_targets"]
    clean_probs = token_tensors["clean_probs"]
    clean_log_probs = token_tensors["clean_log_probs"]
    token_loss = token_tensors["token_loss"]
    pred = token_tensors["pred"]
    top1_margin = token_tensors["top1_margin"]
    entropy = token_tensors["entropy"]
    seq_nll = token_tensors["seq_nll"]
    corruptible_mask = token_tensors["corruptible_mask"]
    noncorruptible_mask = ~corruptible_mask
    final_mask = token_tensors["final_mask"]
    cot_body_mask = ~final_mask

    metrics: dict[str, float] = {
        "teacher_forced_token_acc": float(pred.eq(clean_targets).float().mean().item()),
        "clean_token_prob_mean": _nanmean(clean_probs.flatten()),
        "clean_token_prob_median": _nanmedian(clean_probs.flatten()),
        "clean_token_logprob_mean": _nanmean(clean_log_probs.flatten()),
        "top1_margin_mean": _nanmean(top1_margin.flatten()),
        "top1_margin_median": _nanmedian(top1_margin.flatten()),
        "top1_entropy_mean": _nanmean(entropy.flatten()),
        "top1_entropy_median": _nanmedian(entropy.flatten()),
        "clean_ce_loss": _nanmean(token_loss.flatten()),
        "corruptible_token_loss": _safe_mean_masked(token_loss, corruptible_mask),
        "noncorruptible_token_loss": _safe_mean_masked(token_loss, noncorruptible_mask),
        "corruptible_token_acc": _safe_acc(pred, clean_targets, corruptible_mask),
        "noncorruptible_token_acc": _safe_acc(pred, clean_targets, noncorruptible_mask),
        "corruptible_clean_token_prob_mean": _safe_mean_masked(clean_probs, corruptible_mask),
        "noncorruptible_clean_token_prob_mean": _safe_mean_masked(clean_probs, noncorruptible_mask),
        "cot_body_loss": _safe_mean_masked(token_loss, cot_body_mask),
        "final_answer_loss": _safe_mean_masked(token_loss, final_mask),
        "cot_body_token_acc": _safe_acc(pred, clean_targets, cot_body_mask),
        "final_answer_token_acc": _safe_acc(pred, clean_targets, final_mask),
        "cot_body_clean_token_prob_mean": _safe_mean_masked(clean_probs, cot_body_mask),
        "final_answer_clean_token_prob_mean": _safe_mean_masked(clean_probs, final_mask),
        "sequence_nll_mean": _nanmean(seq_nll),
        "sequence_nll_median": _nanmedian(seq_nll),
    }
    if "full_exact" in token_tensors and "final_exact" in token_tensors:
        full_exact = token_tensors["full_exact"]
        final_exact = token_tensors["final_exact"]
        metrics["sequence_nll_when_full_exact"] = _safe_mean_masked(seq_nll, full_exact)
        metrics["sequence_nll_when_final_exact"] = _safe_mean_masked(seq_nll, final_exact)
        full_exact_count = int(full_exact.sum().item())
        metrics["fraction_full_exact_with_loss_gt_threshold"] = (
            math.nan
            if full_exact_count == 0
            else float(
                (full_exact & seq_nll.gt(float(loss_threshold))).sum().item()
                / full_exact_count
            )
        )
    return metrics


def deterministic_uniform_corruption_entropy(eta: float, k: int) -> float:
    eta = float(eta)
    k = int(k)
    if eta <= 0.0 or k <= 1:
        return 0.0
    clean_prob = 1.0 - eta + eta / k
    other_prob = eta / k
    entropy = 0.0
    if clean_prob > 0.0:
        entropy -= clean_prob * math.log(clean_prob)
    if other_prob > 0.0:
        entropy -= (k - 1) * other_prob * math.log(other_prob)
    return float(entropy)


def noisy_teacher_entropy_baseline(
    *,
    clean_targets: torch.Tensor,
    prompt_ids: torch.Tensor,
    final_answer_len: int,
    eta: float,
    teacher_law: str,
    meta: dict[str, Any] | None = None,
) -> dict[str, float]:
    target_len = int(clean_targets.size(1))
    final_answer_len = min(max(int(final_answer_len), 0), target_len)
    final_mask = torch.zeros_like(clean_targets, dtype=torch.bool)
    if final_answer_len > 0:
        final_mask[:, target_len - final_answer_len:] = True
    corruptible = s5_corruptible_mask(clean_targets)
    entropy_per_noisy_position = deterministic_uniform_corruption_entropy(
        float(eta),
        len(CORRUPTIBLE_IDS),
    )

    if teacher_law in {"distributional_noise", "corrupted_greedy"}:
        noisy_position_mask = corruptible
    elif teacher_law == SEMANTIC_KEY_NOISE_LAW:
        semantic_cfg = semantic_key_noise_config_from_obj(
            None if meta is None else meta.get("semantic_key_noise")
        )
        noisy_position_mask = semantic_key_mask(prompt_ids, target_len, semantic_cfg)
    else:
        return {
            "noisy_teacher_entropy_mean": math.nan,
            "noisy_teacher_entropy_corruptible_mean": math.nan,
            "noisy_teacher_entropy_final_answer_mean": math.nan,
        }

    entropy = noisy_position_mask.to(dtype=torch.float32) * entropy_per_noisy_position
    return {
        "noisy_teacher_entropy_mean": _nanmean(entropy.flatten()),
        "noisy_teacher_entropy_corruptible_mean": _safe_mean_masked(entropy, corruptible),
        "noisy_teacher_entropy_final_answer_mean": _safe_mean_masked(entropy, final_mask),
    }


def _example_records_from_batch(
    *,
    row_offset: int,
    prompt_ids: torch.Tensor,
    clean_targets: torch.Tensor,
    greedy_targets: torch.Tensor,
    token_tensors: dict[str, torch.Tensor],
    final_answer_len: int,
    loss_threshold: float,
) -> list[dict[str, Any]]:
    seq_nll = token_tensors["seq_nll"]
    full_exact = token_tensors["full_exact"]
    final_exact = token_tensors["final_exact"]
    clean_probs = token_tensors["clean_probs"]
    token_loss = token_tensors["token_loss"]
    top2_probs = token_tensors["top2_probs"]
    top2_ids = token_tensors["top2_ids"]
    corruptible_mask = token_tensors["corruptible_mask"]
    final_mask = token_tensors["final_mask"]

    examples: list[dict[str, Any]] = []
    candidates = torch.nonzero(
        full_exact & seq_nll.gt(float(loss_threshold)),
        as_tuple=False,
    ).flatten()
    for local_idx in candidates.tolist():
        token_rows = []
        for pos in range(clean_targets.size(1)):
            top1_id = int(top2_ids[local_idx, pos, 0].item())
            top1_prob = float(top2_probs[local_idx, pos, 0].item())
            if top2_ids.size(-1) > 1:
                top2_id = int(top2_ids[local_idx, pos, 1].item())
                top2_prob = float(top2_probs[local_idx, pos, 1].item())
            else:
                top2_id = top1_id
                top2_prob = top1_prob
            token_rows.append(
                {
                    "pos": int(pos),
                    "region": "final" if bool(final_mask[local_idx, pos].item()) else "cot_body",
                    "corruptible": bool(corruptible_mask[local_idx, pos].item()),
                    "clean_token": decode_s5_ids([int(clean_targets[local_idx, pos].item())]),
                    "greedy_token": decode_s5_ids([int(greedy_targets[local_idx, pos].item())]),
                    "clean_prob": float(clean_probs[local_idx, pos].item()),
                    "loss": float(token_loss[local_idx, pos].item()),
                    "top1_token": decode_s5_ids([top1_id]),
                    "top1_prob": top1_prob,
                    "top2_token": decode_s5_ids([top2_id]),
                    "top2_prob": top2_prob,
                    "top1_is_clean": top1_id == int(clean_targets[local_idx, pos].item()),
                }
            )
        examples.append(
            {
                "row": int(row_offset + local_idx),
                "sequence_nll": float(seq_nll[local_idx].item()),
                "clean_full_exact": bool(full_exact[local_idx].item()),
                "clean_final_exact": bool(final_exact[local_idx].item()),
                "prompt": decode_s5_ids(prompt_ids[local_idx]),
                "clean_target": decode_s5_ids(clean_targets[local_idx]),
                "greedy_generated_target": decode_s5_ids(greedy_targets[local_idx]),
                "final_answer_len": int(final_answer_len),
                "tokens": token_rows,
            }
        )
    return examples


@torch.no_grad()
def evaluate_s5_offline_validation_diagnostics(
    model,
    *,
    device: str | torch.device,
    data_dir: str | os.PathLike[str],
    n_eval: int | None = None,
    batch_size: int = 256,
    final_answer_len: int | None = None,
    loss_threshold: float = 1.0,
    autocast_context=nullcontext(),
    collect_examples: bool = False,
    max_examples: int = 5,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    data_dir = Path(data_dir)
    with open(data_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    prompt_ids_all = torch.load(data_dir / "clean_val_prompt_ids.pt", map_location="cpu").long()
    cot_ids_all = torch.load(data_dir / "clean_val_cot_ids.pt", map_location="cpu").long()
    if n_eval is not None:
        prompt_ids_all = prompt_ids_all[: int(n_eval)]
        cot_ids_all = cot_ids_all[: int(n_eval)]
    n = int(prompt_ids_all.size(0))
    if n == 0:
        raise ValueError(f"{data_dir} has no clean validation rows")

    resolved_final_answer_len = (
        int(final_answer_len)
        if final_answer_len is not None
        else int(meta.get("final_answer_len", meta.get("answer_len", 7)))
    )
    eta = float(meta.get("eta", 0.0))
    teacher_law = str(meta.get("teacher_law", "distributional_noise"))

    clean_targets_all: list[torch.Tensor] = []
    prompt_all: list[torch.Tensor] = []
    batch_token_tensors: list[dict[str, torch.Tensor]] = []
    exact_examples: list[dict[str, Any]] = []

    for start in range(0, n, int(batch_size)):
        end = min(start + int(batch_size), n)
        prompt_ids = prompt_ids_all[start:end]
        cot_ids = cot_ids_all[start:end]
        x, y = build_xy_from_prompt_and_target(prompt_ids, cot_ids)
        x = x.to(device=device, dtype=torch.long, non_blocking=True)
        y_device = y.to(device=device, dtype=torch.long, non_blocking=True)
        with autocast_context:
            logits, _ = model(x, y_device)

        greedy_targets = greedy_generate_target_ids_batched(
            model,
            prompt_ids,
            cot_ids.size(1),
            device,
        )
        _, token_tensors = _token_stats_from_logits(
            logits,
            y_device,
            final_answer_len=resolved_final_answer_len,
            loss_threshold=loss_threshold,
            greedy_targets=greedy_targets.to(device=y_device.device),
        )
        batch_token_tensors.append(token_tensors)
        clean_targets = token_tensors["clean_targets"]
        clean_targets_all.append(clean_targets)
        prompt_all.append(prompt_ids.detach().cpu())
        if collect_examples:
            exact_examples.extend(
                _example_records_from_batch(
                    row_offset=start,
                    prompt_ids=prompt_ids,
                    clean_targets=clean_targets,
                    greedy_targets=greedy_targets.detach().cpu(),
                    token_tensors=token_tensors,
                    final_answer_len=resolved_final_answer_len,
                    loss_threshold=loss_threshold,
                )
            )

    combined_token_tensors = {
        key: torch.cat([batch[key] for batch in batch_token_tensors], dim=0)
        for key in batch_token_tensors[0]
    }
    metrics = _metrics_from_token_tensors(
        combined_token_tensors,
        loss_threshold=loss_threshold,
    )
    clean_targets_cat = torch.cat(clean_targets_all, dim=0)
    prompt_cat = torch.cat(prompt_all, dim=0)
    entropy_metrics = noisy_teacher_entropy_baseline(
        clean_targets=clean_targets_cat,
        prompt_ids=prompt_cat,
        final_answer_len=resolved_final_answer_len,
        eta=eta,
        teacher_law=teacher_law,
        meta=meta,
    )
    metrics.update(entropy_metrics)
    if "clean_ce_loss" in metrics and "noisy_teacher_entropy_mean" in metrics:
        metrics["clean_ce_minus_noisy_entropy_estimate"] = (
            float(metrics["clean_ce_loss"]) - float(metrics["noisy_teacher_entropy_mean"])
        )
    metrics["diagnostic_num_examples"] = float(n)
    metrics["diagnostic_loss_threshold"] = float(loss_threshold)

    exact_examples.sort(key=lambda record: float(record["sequence_nll"]), reverse=True)
    if max_examples >= 0:
        exact_examples = exact_examples[: int(max_examples)]
    return metrics, exact_examples

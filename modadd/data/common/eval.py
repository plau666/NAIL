from __future__ import annotations

import os

import torch

from data.common.prompt_bank import build_xy_from_prompt_and_target


@torch.no_grad()
def greedy_generate_target_ids_batched(model, prompt_ids_batch, max_new_tokens, device):
    prompt = prompt_ids_batch.to(device=device, dtype=torch.long)
    generated = torch.empty((prompt.size(0), max_new_tokens), dtype=torch.long, device=device)
    input_ids = prompt
    past_key_values = None

    for step in range(max_new_tokens):
        logits, _, past_key_values = model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        next_id = torch.argmax(logits[:, -1, :], dim=-1)
        generated[:, step] = next_id
        input_ids = next_id.unsqueeze(1)

    return generated.to(device="cpu", dtype=torch.long)


@torch.no_grad()
def teacher_forced_exact_batch(model, prompt_ids_batch, cot_ids_batch, device):
    seq = torch.cat((prompt_ids_batch, cot_ids_batch), dim=1).to(device=device, dtype=torch.long)
    x = seq[:, :-1].clone()
    y = seq[:, 1:].clone()
    y[:, :prompt_ids_batch.size(1) - 1] = -1

    logits, _ = model(x, y)
    pred = logits.argmax(dim=-1)
    mask = y != -1
    return torch.logical_or(pred.eq(y), ~mask).all(dim=1).to(device="cpu")


@torch.no_grad()
def evaluate_saved_clean_metrics(
    model,
    *,
    device,
    data_dir,
    final_answer_len: int,
    n_eval=None,
    batch_size=256,
):
    prompt_ids_all = torch.load(os.path.join(data_dir, "clean_val_prompt_ids.pt"), map_location="cpu").long()
    cot_ids_all = torch.load(os.path.join(data_dir, "clean_val_cot_ids.pt"), map_location="cpu").long()

    if n_eval is not None:
        prompt_ids_all = prompt_ids_all[:n_eval]
        cot_ids_all = cot_ids_all[:n_eval]

    tf_full_ok = 0
    ar_full_ok = 0
    ar_final_ok = 0
    n = prompt_ids_all.size(0)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        prompt_ids_batch = prompt_ids_all[start:end]
        cot_ids_batch = cot_ids_all[start:end]

        tf_ok = teacher_forced_exact_batch(model, prompt_ids_batch, cot_ids_batch, device)
        tf_full_ok += int(tf_ok.sum().item())

        pred_ids_batch = greedy_generate_target_ids_batched(
            model,
            prompt_ids_batch,
            cot_ids_batch.size(1),
            device,
        )
        ar_full_ok += int(pred_ids_batch.eq(cot_ids_batch).all(dim=1).sum().item())
        ar_final_ok += int(
            pred_ids_batch[:, -final_answer_len:].eq(cot_ids_batch[:, -final_answer_len:]).all(dim=1).sum().item()
        )

    return {
        "cot_exact": tf_full_ok / n,
        "clean_full_exact": ar_full_ok / n,
        "clean_final_exact": ar_final_ok / n,
    }


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
    prompt_ids_all = torch.load(os.path.join(data_dir, "clean_train_prompt_ids.pt"), map_location="cpu").long()
    cot_ids_all = torch.load(os.path.join(data_dir, "clean_train_cot_ids.pt"), map_location="cpu").long()

    if subset_size is not None and subset_size > 0:
        prompt_ids_all = prompt_ids_all[:subset_size]
        cot_ids_all = cot_ids_all[:subset_size]

    n = prompt_ids_all.size(0)
    losses = torch.zeros(eval_iters)

    for k in range(eval_iters):
        idx = torch.randint(n, (batch_size,))
        prompt_ids = prompt_ids_all.index_select(0, idx)
        cot_ids = cot_ids_all.index_select(0, idx)
        x, y = build_xy_from_prompt_and_target(prompt_ids, cot_ids)
        x = x.to(device=device, dtype=torch.long)
        y = y.to(device=device, dtype=torch.long)
        _, loss = model(x, y)
        losses[k] = loss.item()

    return float(losses.mean().item())

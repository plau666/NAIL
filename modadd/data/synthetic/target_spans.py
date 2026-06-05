from __future__ import annotations

from typing import Iterable

import torch

from data.synthetic.prompt_bank import PromptBank


def canonical_target_len(prompt_bank: PromptBank) -> int:
    target_len = int(prompt_bank.target_len)
    cot_len = int(prompt_bank.cot_len)
    if target_len != cot_len:
        raise ValueError(
            f"target_len={target_len} does not match cot_len={cot_len}; "
            "synthetic continuation supervision expects one canonical span"
        )
    return target_len


def _ids_to_list(ids: torch.Tensor | Iterable[int]) -> list[int]:
    if isinstance(ids, torch.Tensor):
        return [int(x) for x in ids.detach().cpu().to(dtype=torch.long).flatten().tolist()]
    return [int(x) for x in ids]


def decode_ids_for_task(ids: torch.Tensor | Iterable[int], *, task: str, p: int) -> str:
    token_ids = _ids_to_list(ids)
    if task == "s5":
        from data.s5_cot.task import decode

        return "".join(decode(token_ids))
    if task == "modadd":
        from data.modular_addition.task import decode

        return " ".join(decode(token_ids, p=p))
    return " ".join(str(token_id) for token_id in token_ids)


def target_ids_from_y_row(y_row: torch.Tensor) -> torch.Tensor:
    return y_row.detach().cpu()[y_row.detach().cpu().ne(-1)].to(dtype=torch.long)


def print_target_span_diagnostic(
    *,
    method_name: str,
    task: str,
    p: int,
    prompt_len: int,
    cot_len: int,
    final_answer_len: int,
    actual_target_len: int,
    total_sequence_len: int,
    prompt_ids: torch.Tensor,
    target_ids: torch.Tensor,
    target_description: str,
) -> None:
    if actual_target_len != cot_len:
        raise ValueError(
            f"{method_name} actual target_len={actual_target_len} does not match "
            f"cot_len={cot_len}"
        )
    if int(target_ids.numel()) != actual_target_len:
        raise ValueError(
            f"{method_name} decoded target has {int(target_ids.numel())} tokens, "
            f"expected target_len={actual_target_len}"
        )

    answer_len = min(int(final_answer_len), int(target_ids.numel()))
    answer_ids = target_ids[-answer_len:] if answer_len > 0 else target_ids[:0]
    decoded_prompt = decode_ids_for_task(prompt_ids, task=task, p=p)
    decoded_target = decode_ids_for_task(target_ids, task=task, p=p)
    decoded_answer = decode_ids_for_task(answer_ids, task=task, p=p)

    print(
        "\n".join(
            [
                f"{method_name} target span diagnostic:",
                f"  prompt_len={prompt_len}",
                f"  cot_len={cot_len}",
                f"  final_answer_len={final_answer_len}",
                f"  actual_training_target_len={actual_target_len}",
                f"  training_x_y_seq_len={total_sequence_len}",
                f"  loss_token_count={actual_target_len}",
                f"  target_description={target_description}",
                f"  example_prompt={decoded_prompt!r}",
                f"  example_training_continuation={decoded_target!r}",
                f"  example_final_answer_suffix={decoded_answer!r}",
            ]
        )
    )


def print_prompt_bank_target_span_diagnostic(
    *,
    method_name: str,
    prompt_bank: PromptBank,
    actual_target_len: int,
    total_sequence_len: int,
    target_description: str,
    row: int = 0,
) -> None:
    row = min(int(row), max(int(prompt_bank.clean_train_prompt_ids.size(0)) - 1, 0))
    print_target_span_diagnostic(
        method_name=method_name,
        task=prompt_bank.task,
        p=prompt_bank.p,
        prompt_len=prompt_bank.prompt_len,
        cot_len=prompt_bank.cot_len,
        final_answer_len=prompt_bank.final_answer_len,
        actual_target_len=actual_target_len,
        total_sequence_len=total_sequence_len,
        prompt_ids=prompt_bank.clean_train_prompt_ids[row],
        target_ids=prompt_bank.clean_train_cot_ids[row].to(dtype=torch.long),
        target_description=target_description,
    )

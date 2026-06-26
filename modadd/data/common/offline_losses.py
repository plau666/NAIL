from __future__ import annotations

import torch

from nanogpt.methods.student_prefix import forward_kl_full_loss


def extract_teacher_prob_student_logits(
    logits: torch.Tensor,
    y: torch.Tensor,
    teacher_probs: torch.Tensor,
) -> torch.Tensor:
    if teacher_probs is None:
        raise ValueError("teacher_probs batch is required for offline teacher-prob loss")
    target_len = int(teacher_probs.size(1))
    if logits.size(1) < target_len:
        raise ValueError(
            f"Model logits length {logits.size(1)} is shorter than teacher_probs "
            f"target_len {target_len}"
        )
    prefix_len = logits.size(1) - target_len
    expected_mask = torch.zeros_like(y, dtype=torch.bool)
    expected_mask[:, prefix_len:] = True
    actual_mask = y.ne(-1)
    if not torch.equal(actual_mask, expected_mask):
        raise ValueError(
            "Offline teacher-prob targets require the continuation region to be "
            "the final contiguous suffix of Y"
        )
    student_logits = logits[:, prefix_len:, :]
    if student_logits.size(2) != teacher_probs.size(2):
        raise ValueError(
            f"Teacher probs vocab size {teacher_probs.size(2)} does not match "
            f"model vocab size {student_logits.size(2)}"
        )
    return student_logits


def offline_teacher_prob_loss_from_logits(
    logits: torch.Tensor,
    y: torch.Tensor,
    teacher_probs: torch.Tensor,
):
    student_logits = extract_teacher_prob_student_logits(logits, y, teacher_probs)
    loss, stats = forward_kl_full_loss(
        student_logits,
        teacher_probs=teacher_probs,
    )
    return student_logits, loss, stats

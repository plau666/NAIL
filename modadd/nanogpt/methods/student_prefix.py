from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F

from data.s5_cot.prompt_bank import PromptBank, build_xy_from_prompt_and_target
from data.s5_cot.task import CORRUPTIBLE_IDS, LPAREN_ID, RPAREN_ID
from data.s5_cot.semantic_key_noise import (
    SEMANTIC_KEY_NOISE_LAW,
    S5_BLOCK_LEN,
    S5_NUM_COORDS,
    S5_VALUE_OFFSET,
    default_eligible_token_ids,
    eligible_token_ids_from_values,
    semantic_key_mask,
    semantic_key_noise_config_from_obj,
)
from data.synthetic.random_suffix_noise import (
    RANDOM_SUFFIX_AFTER_ERROR_LAW,
    compute_poisoned_before,
    effective_trigger_eta,
    random_suffix_after_error_probs,
    random_suffix_noise_config_from_obj,
    validate_random_suffix_applies_to_task,
)

# Student-prefix method helpers. The paper separates "which prefixes are
# visited" from "which divergence is optimized on those prefixes"; this file is
# where that separation is made concrete. See the paper appendix for the table
# mapping NAIL-F/R and OPD-F/R to these switches.

IMPLEMENTATION_BACKEND = "student_prefix"

DEFAULT_ROLLOUT_TEMPERATURE = {
    # OPD-R / OPD-F collect sampled student prefixes by default.
    "opd": 1.0,
    # NAIL-F / NAIL-R collect greedy student prefixes by default.
    "nail": 0.0,
}

# ---------------------------------------------------------------------------
# Legacy checkpoint/config aliases. Do not use in new configs.
#
# Older runs encoded the local loss and teacher feedback type in a single
# objective string, and sometimes used one student-temperature field for both
# rollout and loss-side distributions. New launches should use:
#   task.loss = forward / reverse / mixed / jsd
#   task.teacher_signal = mc / full
#   task.rollout_temperature_override
#   task.loss_temperature_override
# The aliases below are kept so old checkpoints, run_meta.json files, and
# analysis artifacts normalize to the canonical fields at load time.
# ---------------------------------------------------------------------------
LEGACY_OBJECTIVE_MAP = {
    "forward_kl_simple": ("mc", "forward"),
    "forward_kl_full": ("full", "forward"),
    "reverse_kl_simple": ("mc", "reverse"),
    "reverse_kl_tm": ("mc", "reverse"),
    "reverse_kl_full": ("full", "reverse"),
}


def _lookup(value: Mapping[str, Any] | object, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def default_rollout_temperature(method_family: str) -> float:
    try:
        return DEFAULT_ROLLOUT_TEMPERATURE[method_family]
    except KeyError as exc:
        raise ValueError(f"unknown method_family {method_family!r}") from exc


def rollout_policy_name(rollout_temperature: float) -> str:
    return "greedy" if float(rollout_temperature) == 0.0 else "sampled"


def resolved_paper_method_name(
    *,
    method_family: str,
    loss: str,
    rollout_temperature: float,
) -> str:
    """Infer the paper-facing method name from backend knobs.

    The forward objective is NAIL-F on greedy learner prefixes and OPD-F on
    sampled learner prefixes. Reverse MC uses the backend family as well because
    NAIL-R draws a fresh auxiliary student action, while OPD-R reuses the
    sampled rollout action.
    """
    policy = rollout_policy_name(rollout_temperature)
    if loss == "forward":
        return "NAIL-F" if policy == "greedy" else "OPD-F"
    if loss == "reverse" and method_family == "opd":
        return "OPD-R" if policy == "sampled" else "OPD-R (greedy-rollout surrogate)"
    if loss == "reverse" and method_family == "nail":
        return "NAIL-R" if policy == "greedy" else "NAIL-R (sampled-rollout surrogate)"
    if loss == "mixed":
        return "Mixed KL"
    if loss == "jsd":
        return "JSD"
    return f"student_prefix/{loss}"


def legacy_objective_to_teacher_signal_loss(objective: str) -> tuple[str, str]:
    try:
        return LEGACY_OBJECTIVE_MAP[objective]
    except KeyError as exc:
        raise ValueError(f"unknown legacy objective {objective!r}") from exc


def resolve_method_family_for_reverse_loss(
    *,
    loss: str,
    rollout_temperature: float | None,
    default_method_family: str | None,
) -> str:
    if loss in {"forward", "mixed", "jsd"}:
        return "nail"
    if default_method_family is not None:
        return default_method_family
    if rollout_temperature is not None and float(rollout_temperature) == 0.0:
        return "nail"
    return "opd"


def normalize_student_prefix_method(
    value: Mapping[str, Any] | object,
    *,
    default_method_family: str | None = None,
) -> dict[str, Any]:
    """Canonicalize Hydra knobs into the paper-facing student-prefix method.

    NAIL-F/R and OPD-F/R are selected by prefix policy, local loss direction,
    and teacher signal. This helper also normalizes legacy checkpoint objective
    strings into canonical `loss` and `teacher_signal` fields. It does not touch
    tensors or RNG; it only resolves metadata and launch-time defaults.
    """
    method_family = _lookup(value, "method_family") or default_method_family
    teacher_signal = _lookup(value, "teacher_signal")
    loss = _lookup(value, "loss")
    kl_beta = _optional_float(_lookup(value, "kl_beta"))
    rollout_override = _optional_float(_lookup(value, "rollout_temperature_override"))
    loss_override = _optional_float(_lookup(value, "loss_temperature_override"))

    objective = _lookup(value, "objective")
    legacy_loss_temperature = _optional_float(_lookup(value, "student_temperature"))
    legacy_rollout_temperature = _optional_float(_lookup(value, "student_rollout_temperature"))

    if teacher_signal is None or loss is None:
        if objective in (None, ""):
            teacher_signal, loss = ("mc", "reverse")
        else:
            teacher_signal, loss = legacy_objective_to_teacher_signal_loss(str(objective))

    resolved_rollout = rollout_override
    if resolved_rollout is None:
        if legacy_rollout_temperature is not None:
            resolved_rollout = legacy_rollout_temperature
        elif objective not in (None, "") and legacy_loss_temperature is not None:
            # Older configs used one temperature knob for both rollout and loss.
            resolved_rollout = legacy_loss_temperature

    method_family = method_family or resolve_method_family_for_reverse_loss(
        loss=str(loss),
        rollout_temperature=resolved_rollout,
        default_method_family=default_method_family,
    )

    if resolved_rollout is None:
        resolved_rollout = default_rollout_temperature(str(method_family))

    resolved_loss = loss_override
    if resolved_loss is None and objective not in (None, "") and str(loss) == "forward":
        resolved_loss = legacy_loss_temperature

    resolved_rollout = float(resolved_rollout)
    state = {
        "method_family": str(method_family),
        "teacher_signal": str(teacher_signal),
        "loss": str(loss),
        "kl_beta": kl_beta,
        "rollout_temperature_override": rollout_override,
        "loss_temperature_override": loss_override,
        "resolved_rollout_temperature": resolved_rollout,
        "resolved_loss_temperature": None if resolved_loss is None else float(resolved_loss),
        "resolved_rollout_policy": rollout_policy_name(resolved_rollout),
        "resolved_method_name": resolved_paper_method_name(
            method_family=str(method_family),
            loss=str(loss),
            rollout_temperature=resolved_rollout,
        ),
        "implementation_backend": IMPLEMENTATION_BACKEND,
    }
    if objective not in (None, ""):
        state["legacy_objective"] = str(objective)
    return state


def format_temperature_tag(temperature: float | None) -> str:
    if temperature is None:
        return "default"
    if float(temperature) == 0.0:
        return "greedy"
    return f"t{str(float(temperature)).replace('.', 'p')}"


class FixedPromptCycle:
    def __init__(
        self,
        prompt_ids: torch.Tensor,
        *,
        order: torch.Tensor | None = None,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 1337,
    ):
        self.prompt_ids = prompt_ids.to(device="cpu", dtype=torch.uint8).contiguous()
        if order is None:
            order = torch.arange(self.prompt_ids.size(0), dtype=torch.long)
        self.base_order = order.to(device="cpu", dtype=torch.long).contiguous()
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.n = int(self.base_order.numel())
        self.epoch = 0
        self.pos = 0
        self.order = self._make_order()

    def _make_order(self) -> torch.Tensor:
        if not self.shuffle:
            return self.base_order.clone()
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        perm = torch.randperm(self.n, generator=generator)
        return self.base_order.index_select(0, perm)

    def state_dict(self) -> dict[str, torch.Tensor | int | bool]:
        return {
            "base_order": self.base_order.clone(),
            "pos": int(self.pos),
            "n": int(self.n),
            "epoch": int(self.epoch),
            "batch_size": int(self.batch_size),
            "shuffle": bool(self.shuffle),
            "seed": int(self.seed),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor | int | bool]) -> None:
        self.base_order = state["base_order"].to(device="cpu", dtype=torch.long).clone()
        self.pos = int(state["pos"])
        self.n = int(state["n"])
        self.epoch = int(state["epoch"])
        self.batch_size = int(state["batch_size"])
        self.shuffle = bool(state["shuffle"])
        self.seed = int(state["seed"])
        self.order = self._make_order()

    def _advance_epoch(self) -> None:
        self.epoch += 1
        self.pos = 0
        self.order = self._make_order()

    def has_remaining_in_epoch(self) -> bool:
        return self.pos < self.n

    def next_batch_indices_no_wrap(self) -> torch.Tensor:
        if self.pos >= self.n:
            return self.base_order[:0].clone()
        take = min(self.batch_size, self.n - self.pos)
        idx = self.order[self.pos:self.pos + take]
        self.pos += take
        return idx.clone()

    def next_batch_no_wrap(self) -> torch.Tensor:
        idx = self.next_batch_indices_no_wrap()
        return self.prompt_ids.index_select(0, idx)

    def next_batch_indices(self) -> torch.Tensor:
        batches = []
        remaining = self.batch_size

        while remaining > 0:
            if self.pos >= self.n:
                self._advance_epoch()

            take = min(remaining, self.n - self.pos)
            idx = self.order[self.pos:self.pos + take]
            batches.append(idx.clone())
            self.pos += take
            remaining -= take

        if len(batches) == 1:
            return batches[0]
        return torch.cat(batches, dim=0)

    def next_batch(self) -> torch.Tensor:
        idx = self.next_batch_indices()
        return self.prompt_ids.index_select(0, idx)


@torch.no_grad()
def rollout_student(
    model,
    prompt_ids: torch.Tensor,
    *,
    target_len: int,
    temperature: float,
    device: str | torch.device,
    autocast_context=nullcontext(),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect stopped learner prefixes for NAIL-F/R and OPD-F/R.

    `temperature=0.0` gives the greedy rollout policy used by NAIL-F/R;
    positive temperatures sample OPD-style learner prefixes. The returned
    actions/log_q come from the rollout distribution only. Gradients do not flow
    through this sampling path; the trainer later recomputes loss-side logits on
    the fixed prefixes.
    """
    prompt = prompt_ids.to(device=device, dtype=torch.long, non_blocking=True)
    batch_size, prompt_len = prompt.shape
    full_seq = torch.empty((batch_size, prompt_len + target_len), dtype=torch.long, device=device)
    full_seq[:, :prompt_len] = prompt
    actions = full_seq[:, prompt_len:]
    log_q = torch.empty((batch_size, target_len), dtype=torch.float32, device=device)
    q_temperature = temperature if temperature > 0 else None
    input_ids = prompt
    past_key_values = None

    for step in range(target_len):
        with autocast_context:
            logits, _, past_key_values = model(
                input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
        next_logits = logits[:, -1, :]
        if temperature > 0:
            probs = F.softmax(next_logits.float() / temperature, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_ids = torch.argmax(next_logits, dim=-1)
        actions[:, step] = next_ids
        log_q[:, step] = gather_action_log_probs(
            next_logits.unsqueeze(1),
            next_ids.unsqueeze(1),
            temperature=q_temperature,
        ).squeeze(1)
        input_ids = next_ids.unsqueeze(1)

    return full_seq, actions, log_q


def gather_action_log_probs(
    logits: torch.Tensor,
    actions: torch.Tensor,
    *,
    temperature: float | None = None,
) -> torch.Tensor:
    log_probs = log_probs_from_logits(logits, temperature=temperature)
    return log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)


def log_probs_from_logits(
    logits: torch.Tensor,
    *,
    temperature: float | None = None,
) -> torch.Tensor:
    work_logits = logits.float()
    if temperature is not None and temperature > 0:
        work_logits = work_logits / temperature
    return F.log_softmax(work_logits, dim=-1)


def extract_answer_logits(
    full_logits: torch.Tensor,
    *,
    prompt_len: int,
    target_len: int,
) -> torch.Tensor:
    return full_logits[:, prompt_len - 1:prompt_len + target_len - 1, :]


def distributional_noisy_teacher_probs(
    clean_logits: torch.Tensor,
    *,
    eta: float,
    corruptible_token_ids: tuple[int, ...] | list[int] | torch.Tensor = CORRUPTIBLE_IDS,
) -> torch.Tensor:
    """Distribution-level counterpart of offline `sample_then_corrupt`.

    This is the standard noisy expert law in the paper: redistribute mass among
    eligible semantic tokens with mixing weight `eta`.
    """
    clean_probs = F.softmax(clean_logits.float(), dim=-1)
    if eta <= 0:
        return clean_probs

    corruptible_ids = torch.as_tensor(corruptible_token_ids, dtype=torch.long, device=clean_probs.device)
    num_digits = int(corruptible_ids.numel())
    if num_digits == 0:
        return clean_probs

    selected_probs = clean_probs.index_select(dim=-1, index=corruptible_ids)
    noisy_probs = clean_probs.clone()
    noisy_selected_probs = (
        (1.0 - eta) * selected_probs
        + (eta / num_digits) * selected_probs.sum(dim=-1, keepdim=True)
    )
    noisy_probs.index_copy_(dim=-1, index=corruptible_ids, source=noisy_selected_probs)
    return noisy_probs


def corrupted_greedy_teacher_probs(
    clean_logits: torch.Tensor,
    *,
    eta: float,
    corruptible_token_ids: tuple[int, ...] | list[int] | torch.Tensor = CORRUPTIBLE_IDS,
) -> torch.Tensor:
    """Distribution-level counterpart of offline `greedy_then_corrupt`."""
    greedy_actions = torch.argmax(clean_logits, dim=-1)
    corruptible_ids = torch.as_tensor(corruptible_token_ids, dtype=torch.long, device=clean_logits.device)
    is_greedy_digit = greedy_actions.unsqueeze(-1).eq(
        corruptible_ids.view(*([1] * greedy_actions.ndim), -1)
    ).any(dim=-1)
    probs = torch.zeros_like(clean_logits, dtype=torch.float32)
    num_digits = int(corruptible_ids.numel())
    if num_digits > 0:
        uniform_noise = is_greedy_digit.unsqueeze(-1).to(dtype=torch.float32).expand(*greedy_actions.shape, num_digits)
        probs.index_copy_(dim=-1, index=corruptible_ids, source=uniform_noise * (eta / num_digits))
    base_mass = torch.where(
        is_greedy_digit,
        torch.full_like(greedy_actions, 1.0 - eta, dtype=torch.float32),
        torch.ones_like(greedy_actions, dtype=torch.float32),
    )
    probs.scatter_add_(2, greedy_actions.unsqueeze(-1), base_mass.unsqueeze(-1))
    return probs


def _normalize_teacher_key_mask(
    key_mask: torch.Tensor,
    *,
    teacher_probs: torch.Tensor,
) -> torch.Tensor:
    mask = key_mask.to(device=teacher_probs.device, dtype=torch.bool)
    expected_shape = teacher_probs.shape[:-1]
    if tuple(mask.shape) == tuple(expected_shape):
        return mask
    if teacher_probs.ndim >= 3 and tuple(mask.shape) == tuple(expected_shape[:-1]):
        return mask.unsqueeze(-1).expand(expected_shape)
    if mask.ndim == 0 and len(expected_shape) == 0:
        return mask
    raise ValueError(
        f"semantic_key_noise key_mask shape {tuple(mask.shape)} is incompatible "
        f"with teacher_probs shape {tuple(teacher_probs.shape)}"
    )


def semantic_key_noise_probs(
    teacher_probs: torch.Tensor,
    *,
    eta: float,
    key_mask: torch.Tensor,
    eligible_token_ids: tuple[int, ...] | list[int] = default_eligible_token_ids(),
) -> torch.Tensor:
    clean_probs = teacher_probs.float()
    if eta <= 0:
        return clean_probs

    key_mask = _normalize_teacher_key_mask(key_mask, teacher_probs=clean_probs)
    if not torch.any(key_mask):
        return clean_probs

    eligible_ids = torch.as_tensor(eligible_token_ids, dtype=torch.long, device=clean_probs.device)
    num_eligible = int(eligible_ids.numel())
    if num_eligible == 0:
        raise ValueError("semantic_key_noise requires at least one eligible token id")

    uniform_noise = torch.zeros_like(clean_probs)
    uniform_noise.index_fill_(dim=-1, index=eligible_ids, value=1.0 / num_eligible)
    mixed = (1.0 - float(eta)) * clean_probs + float(eta) * uniform_noise
    return torch.where(key_mask.unsqueeze(-1), mixed, clean_probs)


def compute_teacher_token_probs(
    clean_logits: torch.Tensor,
    *,
    eta: float,
    teacher_law: str,
    corruptible_token_ids: tuple[int, ...] | list[int] | torch.Tensor = CORRUPTIBLE_IDS,
    key_mask: torch.Tensor | None = None,
    eligible_token_ids: tuple[int, ...] | list[int] | None = None,
) -> torch.Tensor:
    """Return the noisy expert next-token distribution for clean teacher logits.

    Stateless laws can be computed from the current clean logits alone. The
    absorbing random-suffix law is stateful, so online code must call
    `cached_teacher_token_probs` with the clean target to infer whether the
    student prefix has already become poisoned.
    """
    if teacher_law == "distributional_noise":
        return distributional_noisy_teacher_probs(
            clean_logits,
            eta=eta,
            corruptible_token_ids=corruptible_token_ids,
        )
    if teacher_law == "corrupted_greedy":
        return corrupted_greedy_teacher_probs(
            clean_logits,
            eta=eta,
            corruptible_token_ids=corruptible_token_ids,
        )
    if teacher_law == "semantic_key_noise":
        if key_mask is None:
            raise ValueError("semantic_key_noise requires key_mask")
        clean_probs = F.softmax(clean_logits.float(), dim=-1)
        return semantic_key_noise_probs(
            clean_probs,
            eta=eta,
            key_mask=key_mask,
            eligible_token_ids=(
                default_eligible_token_ids()
                if eligible_token_ids is None
                else eligible_token_ids
            ),
        )
    if teacher_law == RANDOM_SUFFIX_AFTER_ERROR_LAW:
        raise NotImplementedError(
            f"{RANDOM_SUFFIX_AFTER_ERROR_LAW} is stateful; use "
            "cached_teacher_token_probs with clean_target_ids so poison state can "
            "be inferred from the rollout prefix."
        )
    raise ValueError(f"Unknown teacher_law: {teacher_law}")


def compute_teacher_log_probs(
    clean_logits: torch.Tensor,
    actions: torch.Tensor,
    *,
    eta: float,
    teacher_law: str,
    corruptible_token_ids: tuple[int, ...] | list[int] | torch.Tensor = CORRUPTIBLE_IDS,
    key_mask: torch.Tensor | None = None,
    eligible_token_ids: tuple[int, ...] | list[int] | None = None,
    eps: float = 1e-10,
) -> torch.Tensor:
    teacher_probs = compute_teacher_token_probs(
        clean_logits,
        eta=eta,
        teacher_law=teacher_law,
        corruptible_token_ids=corruptible_token_ids,
        key_mask=key_mask,
        eligible_token_ids=eligible_token_ids,
    )
    teacher_action_probs = teacher_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)
    return torch.log(teacher_action_probs.clamp_min(eps))


def sample_teacher_actions(teacher_probs: torch.Tensor) -> torch.Tensor:
    batch_size, target_len, vocab_size = teacher_probs.shape
    flat_samples = torch.multinomial(
        teacher_probs.reshape(batch_size * target_len, vocab_size),
        num_samples=1,
    )
    return flat_samples.reshape(batch_size, target_len)


def _semantic_config_from_random_suffix_config(config) -> object:
    return semantic_key_noise_config_from_obj(
        {
            "enabled": True,
            "coord_strategy": config.coord_strategy,
            "fixed_coord": config.fixed_coord,
            "seed": config.seed,
            "include_clean_value": True,
            "eligible_values": config.eligible_values,
            "apply_to": "partial_perm_image",
            "one_key_per_block": config.one_key_per_block,
        }
    )


def _s5_random_suffix_online_masks(
    prompt: torch.Tensor,
    *,
    target_len: int,
    config,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
    # In S5, only one selected value coordinate per CoT block is the semantic
    # "fork" token. Parentheses are scaffold tokens and can be forced to remain
    # syntactically valid after poisoning.
    torch_device = torch.device(device)
    semantic_config = _semantic_config_from_random_suffix_config(config)
    key_mask = semantic_key_mask(prompt, target_len, semantic_config).to(
        device=torch_device,
        dtype=torch.bool,
    )
    offsets = torch.arange(int(target_len), dtype=torch.long, device=torch_device) % S5_BLOCK_LEN
    semantic_row = (
        (offsets >= S5_VALUE_OFFSET)
        & (offsets < S5_VALUE_OFFSET + S5_NUM_COORDS)
    )
    scaffold_row = torch.where(
        offsets.eq(0),
        torch.full_like(offsets, LPAREN_ID),
        torch.full_like(offsets, RPAREN_ID),
    )
    semantic_mask = semantic_row.view(1, target_len).expand(prompt.size(0), target_len)
    scaffold_token_ids = scaffold_row.view(1, target_len).expand(prompt.size(0), target_len)
    return (
        key_mask,
        semantic_mask,
        scaffold_token_ids,
        eligible_token_ids_from_values(config.eligible_values),
    )


def _random_suffix_online_masks(
    *,
    task_name: str,
    prompt: torch.Tensor,
    actions: torch.Tensor,
    config,
    corruptible_token_ids: tuple[int, ...] | list[int] | torch.Tensor,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, tuple[int, ...] | list[int]]:
    target_len = int(actions.size(1))
    if task_name == "s5":
        return _s5_random_suffix_online_masks(
            prompt,
            target_len=target_len,
            config=config,
            device=device,
        )
    if task_name == "modadd":
        # In modular addition every target token is a running-sum token, so the
        # key mask and semantic mask are both all true. This is the paper's
        # p=7, m=31 random-suffix experiment.
        mask = torch.ones_like(actions, dtype=torch.bool, device=torch.device(device))
        return mask, mask, None, tuple(int(token_id) for token_id in corruptible_token_ids)
    raise ValueError(
        f"{RANDOM_SUFFIX_AFTER_ERROR_LAW} online support expects task_name "
        f"'s5' or 'modadd', got {task_name!r}"
    )


@torch.no_grad()
def sample_student_aux_actions(
    student_logits: torch.Tensor,
    *,
    temperature: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample NAIL-R auxiliary actions from the fixed-prefix student policy.

    Inputs are loss-side logits recomputed on already-collected prefixes. NAIL-R
    uses these samples for the reverse-KL estimator instead of reusing the
    greedy rollout token. Returned actions/log_q are detached sampling records;
    gradients later flow through the reverse loss's current `student_logits`,
    not through this draw.
    """
    if temperature is not None and temperature <= 0:
        raise ValueError("sample_student_aux_actions temperature must be > 0 when set.")

    detached_logits = student_logits.detach().float()
    batch_size, target_len, vocab_size = detached_logits.shape
    work_logits = detached_logits if temperature is None else detached_logits / temperature
    probs = F.softmax(work_logits, dim=-1)
    flat_samples = torch.multinomial(
        probs.reshape(batch_size * target_len, vocab_size),
        num_samples=1,
    )
    actions = flat_samples.reshape(batch_size, target_len)
    log_q = gather_action_log_probs(detached_logits, actions, temperature=temperature)
    return actions, log_q


def teacher_cross_entropy(
    teacher_probs: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    temperature: float | None = None,
) -> torch.Tensor:
    student_log_probs = log_probs_from_logits(student_logits, temperature=temperature)
    return -(teacher_probs * student_log_probs).sum(dim=-1)


def teacher_forward_kl(
    teacher_probs: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    temperature: float | None = None,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact per-token KL(teacher || student) for full teacher distributions."""
    teacher_log_probs = torch.log(teacher_probs.clamp_min(eps))
    ce = teacher_cross_entropy(
        teacher_probs,
        student_logits,
        temperature=temperature,
    )
    entropy = -(teacher_probs * teacher_log_probs).sum(dim=-1)
    return ce - entropy, ce, entropy


def reverse_kl_tm_loss(
    student_logits: torch.Tensor,
    actions: torch.Tensor,
    *,
    log_q: torch.Tensor,
    teacher_probs: torch.Tensor,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """MC reverse-KL score-function estimator for NAIL-R and OPD-R.

    `actions` and `log_q` come from the student sampling distribution used by
    the estimator: OPD-R usually reuses sampled rollout actions, while NAIL-R
    passes fresh auxiliary actions. Teacher probabilities are fixed feedback on
    the visited prefixes. Gradients flow only through `log_p` from current
    loss-side logits; rollout log_q and teacher feedback are treated as stopped.
    """
    teacher_action_probs = teacher_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)
    log_teacher = torch.log(teacher_action_probs.clamp_min(eps))
    advantage = log_teacher - log_q
    log_p = gather_action_log_probs(student_logits, actions)
    importance_weight = torch.exp(log_p - log_q.detach())
    loss = -(importance_weight * advantage.detach()).mean()
    return loss, {
        "log_p": log_p,
        "log_teacher": log_teacher,
        "advantage": advantage,
        "importance_weight": importance_weight,
    }


def reverse_kl_mc_loss(
    student_logits: torch.Tensor,
    actions: torch.Tensor,
    *,
    log_q: torch.Tensor,
    teacher_probs: torch.Tensor,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Canonical name for the MC reverse-KL score-function helper."""
    return reverse_kl_tm_loss(
        student_logits,
        actions,
        log_q=log_q,
        teacher_probs=teacher_probs,
        eps=eps,
    )


def reverse_kl_full_loss(
    student_logits: torch.Tensor,
    *,
    teacher_probs: torch.Tensor,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Full-distribution reverse KL on fixed learner prefixes.

    This exact `KL(student || teacher)` variant uses the full noisy teacher
    distribution at each visited prefix. The prefixes and teacher probabilities
    are stopped; gradients flow through the current student distribution.
    """
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    student_probs = student_log_probs.exp()
    teacher_log_probs = torch.log(teacher_probs.clamp_min(eps))
    reverse_kl = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
    student_teacher_ce = -(student_probs * teacher_log_probs).sum(dim=-1)
    student_entropy = -(student_probs * student_log_probs).sum(dim=-1)
    loss = reverse_kl.mean()
    return loss, {
        "reverse_kl": reverse_kl,
        "student_teacher_ce": student_teacher_ce,
        "student_entropy": student_entropy,
    }


def reverse_full_kl_loss(
    student_logits: torch.Tensor,
    *,
    teacher_probs: torch.Tensor,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Canonical name for exact per-prefix KL(student || teacher)."""
    return reverse_kl_full_loss(
        student_logits,
        teacher_probs=teacher_probs,
        eps=eps,
    )


def forward_kl_simple_loss(
    student_logits: torch.Tensor,
    teacher_targets: torch.Tensor,
    *,
    teacher_probs: torch.Tensor,
    temperature: float | None = None,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """MC forward loss for NAIL-F and OPD-F on visited prefixes.

    `teacher_targets` are sampled from the noisy teacher distribution at fixed
    learner prefixes. Greedy prefixes make this NAIL-F; sampled prefixes make it
    OPD-F. Gradients flow through the student log-probability of the sampled
    teacher token, not through teacher sampling or prefix rollout.
    """
    teacher_target_probs = teacher_probs.gather(2, teacher_targets.unsqueeze(-1)).squeeze(-1)
    log_teacher_target = torch.log(teacher_target_probs.clamp_min(eps))
    log_student_target = gather_action_log_probs(
        student_logits,
        teacher_targets,
        temperature=temperature,
    )
    loss = -log_student_target.mean()
    return loss, {
        "log_student_target": log_student_target,
        "log_teacher_target": log_teacher_target,
    }


def forward_mc_loss(
    student_logits: torch.Tensor,
    teacher_targets: torch.Tensor,
    *,
    teacher_probs: torch.Tensor,
    temperature: float | None = None,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Canonical name for the MC teacher-sample forward update."""
    return forward_kl_simple_loss(
        student_logits,
        teacher_targets,
        teacher_probs=teacher_probs,
        temperature=temperature,
        eps=eps,
    )


def mixed_kl_loss_from_components(
    forward_loss: torch.Tensor,
    reverse_loss: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    return (1.0 - float(beta)) * forward_loss + float(beta) * reverse_loss


def jsd_mc_loss(
    student_logits: torch.Tensor,
    teacher_targets: torch.Tensor,
    student_actions: torch.Tensor,
    *,
    teacher_probs: torch.Tensor,
    beta: float,
    temperature: float | None = None,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Monte Carlo Jensen-Shannon-style mixture loss on fixed rollout prefixes."""
    beta = float(beta)
    student_log_probs = log_probs_from_logits(
        student_logits,
        temperature=temperature,
    )
    student_probs = student_log_probs.exp()
    mixture_probs = (1.0 - beta) * teacher_probs + beta * student_probs
    log_mixture = torch.log(mixture_probs.clamp_min(eps))

    log_mixture_teacher_target = log_mixture.gather(
        2,
        teacher_targets.unsqueeze(-1),
    ).squeeze(-1)
    teacher_target_probs = teacher_probs.gather(
        2,
        teacher_targets.unsqueeze(-1),
    ).squeeze(-1)
    log_teacher_target = torch.log(teacher_target_probs.clamp_min(eps))
    teacher_to_mix = -log_mixture_teacher_target

    log_student_action = student_log_probs.gather(
        2,
        student_actions.unsqueeze(-1),
    ).squeeze(-1)
    log_mixture_student_action = log_mixture.gather(
        2,
        student_actions.unsqueeze(-1),
    ).squeeze(-1)
    student_to_mix = log_student_action - log_mixture_student_action

    teacher_to_mix_loss = teacher_to_mix.mean()
    student_to_mix_loss = student_to_mix.mean()
    loss = (1.0 - beta) * teacher_to_mix_loss + beta * student_to_mix_loss
    return loss, {
        "teacher_to_mix": teacher_to_mix,
        "student_to_mix": student_to_mix,
        "teacher_to_mix_loss": teacher_to_mix_loss,
        "student_to_mix_loss": student_to_mix_loss,
        "log_teacher_target": log_teacher_target,
        "log_mixture_teacher_target": log_mixture_teacher_target,
        "log_student_action": log_student_action,
        "log_mixture_student_action": log_mixture_student_action,
    }


def forward_kl_full_loss(
    student_logits: torch.Tensor,
    *,
    teacher_probs: torch.Tensor,
    temperature: float | None = None,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Full-distribution forward KL for NAIL-F/OPD-F-style prefixes.

    `teacher_probs` is the complete noisy teacher distribution on fixed learner
    prefixes. Gradients flow through the current student logits under the
    optional loss temperature; neither teacher probabilities nor prefix rollout
    are differentiated.
    """
    token_kl, teacher_ce, teacher_entropy = teacher_forward_kl(
        teacher_probs,
        student_logits,
        temperature=temperature,
        eps=eps,
    )
    loss = token_kl.mean()
    return loss, {
        "forward_kl": token_kl,
        "teacher_ce": teacher_ce,
        "teacher_entropy": teacher_entropy,
    }


def forward_full_kl_loss(
    student_logits: torch.Tensor,
    *,
    teacher_probs: torch.Tensor,
    temperature: float | None = None,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Canonical name for NAIL-F/OPD-F full-distribution forward KL.

    Inputs are the same fixed-prefix logits and full teacher probabilities as
    `forward_kl_full_loss`; this wrapper exists so reader-facing docs can avoid
    legacy naming while keeping checkpoint/test compatibility.
    """
    return forward_kl_full_loss(
        student_logits,
        teacher_probs=teacher_probs,
        temperature=temperature,
        eps=eps,
    )


@torch.no_grad()
def cached_teacher_token_probs(
    model,
    prompt_ids: torch.Tensor,
    actions: torch.Tensor,
    *,
    eta: float,
    teacher_law: str,
    corruptible_token_ids: tuple[int, ...] | list[int] = CORRUPTIBLE_IDS,
    semantic_key_noise_config: dict[str, object] | object | None = None,
    clean_target_ids: torch.Tensor | None = None,
    random_suffix_noise_config: dict[str, object] | object | None = None,
    task_name: str = "s5",
    device: str | torch.device,
    autocast_context=nullcontext(),
) -> torch.Tensor:
    """Query `pi*_eta` on fixed learner prefixes for all online methods.

    `prompt_ids` plus rollout `actions` define the learner-induced prefixes:
    greedy for NAIL-F/R, sampled for OPD-F/R. The clean teacher is frozen and
    queried under `torch.no_grad()`, then `teacher_law` and `eta` turn clean
    logits into noisy teacher probabilities. No gradients flow through teacher
    queries or noisy-law application.
    """
    torch_device = torch.device(device)
    prompt = prompt_ids.to(device=torch_device, dtype=torch.long, non_blocking=True)
    actions = actions.to(device=torch_device, dtype=torch.long, non_blocking=True)
    target_len = int(actions.size(1))
    corruptible_token_ids_tensor = torch.as_tensor(
        corruptible_token_ids,
        dtype=torch.long,
        device=torch_device,
    )
    teacher_probs = torch.empty(
        (*actions.shape, model.config.vocab_size),
        dtype=torch.float32,
        device=torch_device,
    )
    input_ids = prompt
    past_key_values = None
    eligible_token_ids = None
    semantic_key_masks = None
    if teacher_law == SEMANTIC_KEY_NOISE_LAW:
        semantic_config = semantic_key_noise_config_from_obj(
            semantic_key_noise_config,
        )
        eligible_token_ids = eligible_token_ids_from_values(semantic_config.eligible_values)
        semantic_key_masks = semantic_key_mask(prompt, target_len, semantic_config).to(
            device=torch_device,
            dtype=torch.bool,
        )
    random_suffix_config = None
    random_suffix_key_mask = None
    random_suffix_semantic_mask = None
    random_suffix_scaffold_token_ids = None
    random_suffix_eligible_token_ids = None
    poisoned_before = None
    random_suffix_eta = float(eta)
    if teacher_law == RANDOM_SUFFIX_AFTER_ERROR_LAW:
        random_suffix_config = random_suffix_noise_config_from_obj(random_suffix_noise_config)
        validate_random_suffix_applies_to_task(random_suffix_config, task_name=task_name)
        if clean_target_ids is None:
            raise ValueError(
                f"{RANDOM_SUFFIX_AFTER_ERROR_LAW} online teacher queries require "
                "clean_target_ids so poisoned prefixes can be inferred."
            )
        clean_targets = clean_target_ids.to(
            device=torch_device,
            dtype=torch.long,
            non_blocking=True,
        )
        if clean_targets.ndim != 2 or clean_targets.size(0) != actions.size(0):
            raise ValueError(
                "clean_target_ids must have shape [B, T] with the same batch size "
                f"as actions; got {tuple(clean_targets.shape)} for actions "
                f"{tuple(actions.shape)}"
            )
        if clean_targets.size(1) < actions.size(1):
            raise ValueError(
                f"clean_target_ids length {clean_targets.size(1)} is shorter than "
                f"actions length {actions.size(1)}"
            )
        clean_targets = clean_targets[:, :target_len]
        (
            random_suffix_key_mask,
            random_suffix_semantic_mask,
            random_suffix_scaffold_token_ids,
            random_suffix_eligible_token_ids,
        ) = _random_suffix_online_masks(
            task_name=task_name,
            prompt=prompt,
            actions=actions,
            config=random_suffix_config,
            corruptible_token_ids=corruptible_token_ids_tensor,
            device=torch_device,
        )
        # Online random-suffix feedback has no rendered teacher trajectory.
        # The poisoned state is inferred from learner actions versus the clean
        # expert target, so later semantic feedback becomes uniform after the
        # first previous key-token mistake.
        poisoned_before = compute_poisoned_before(
            actions,
            clean_targets,
            random_suffix_key_mask,
        )
        random_suffix_eta = effective_trigger_eta(float(eta), random_suffix_config)

    for step in range(target_len):
        with autocast_context:
            logits, _, past_key_values = model(
                input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
        if random_suffix_config is not None:
            assert poisoned_before is not None
            assert random_suffix_key_mask is not None
            assert random_suffix_semantic_mask is not None
            assert random_suffix_eligible_token_ids is not None
            scaffold_ids = (
                None
                if random_suffix_scaffold_token_ids is None
                else random_suffix_scaffold_token_ids[:, step]
            )
            teacher_probs[:, step, :] = random_suffix_after_error_probs(
                F.softmax(logits[:, -1:, :].float(), dim=-1),
                eta=random_suffix_eta,
                poisoned=poisoned_before[:, step],
                key_mask=random_suffix_key_mask[:, step],
                semantic_mask=random_suffix_semantic_mask[:, step],
                eligible_token_ids=random_suffix_eligible_token_ids,
                scaffold_token_ids=scaffold_ids,
                keep_format_tokens=random_suffix_config.keep_format_tokens,
            ).squeeze(1)
        else:
            key_mask = None
            if semantic_key_masks is not None:
                key_mask = semantic_key_masks[:, step]
            teacher_probs[:, step, :] = compute_teacher_token_probs(
                logits[:, -1:, :],
                eta=eta,
                teacher_law=teacher_law,
                corruptible_token_ids=corruptible_token_ids_tensor,
                key_mask=key_mask,
                eligible_token_ids=eligible_token_ids,
            ).squeeze(1)
        input_ids = actions[:, step:step + 1]

    return teacher_probs


@torch.no_grad()
def evaluate_clean_ce_loss(
    model,
    prompt_bank: PromptBank,
    *,
    batch_size: int,
    device: str | torch.device,
    autocast_context=nullcontext(),
) -> float:
    total_loss = 0.0
    total_examples = 0

    for start in range(0, prompt_bank.clean_val_prompt_ids.size(0), batch_size):
        end = min(start + batch_size, prompt_bank.clean_val_prompt_ids.size(0))
        prompt_ids = prompt_bank.clean_val_prompt_ids[start:end]
        cot_ids = prompt_bank.clean_val_cot_ids[start:end]
        x, y = build_xy_from_prompt_and_target(prompt_ids, cot_ids)
        x = x.to(device=device, dtype=torch.long, non_blocking=True)
        y = y.to(device=device, dtype=torch.long, non_blocking=True)
        with autocast_context:
            _, loss = model(x, y)
        batch_n = end - start
        total_loss += float(loss.item()) * batch_n
        total_examples += batch_n

    return total_loss / max(total_examples, 1)

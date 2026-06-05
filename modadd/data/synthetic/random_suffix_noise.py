from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F


# Shared implementation of the absorbing "random suffix after error" teacher
# law described in the paper appendix. Offline
# rendering samples the poison state while generating a teacher trajectory;
# student-prefix methods infer the poison state from student-prefix mistakes.
RANDOM_SUFFIX_AFTER_ERROR_LAW = "random_suffix_after_error"
RANDOM_SUFFIX_KEY_POSITION_CHOICES = ("semantic_key",)
RANDOM_SUFFIX_MODE_CHOICES = ("valid_tokens",)
RANDOM_SUFFIX_APPLY_TO_CHOICES = ("s5", "modadd", "both")
RANDOM_SUFFIX_CONFIG_KEYS = (
    "enabled",
    "key_positions",
    "trigger_eta",
    "random_suffix_mode",
    "keep_format_tokens",
    "seed",
    "apply_to",
    "coord_strategy",
    "fixed_coord",
    "eligible_values",
    "one_key_per_block",
)


@dataclass(frozen=True)
class RandomSuffixNoiseConfig:
    """Config for the paper's absorbing random-suffix noisy teacher law."""

    enabled: bool = True
    # Key tokens are semantic decision points that can trigger the absorbing
    # poisoned state. For modadd every target token is a key token; for S5 this
    # is one selected value coordinate per CoT block.
    key_positions: str = "semantic_key"
    # Optional trigger rate separate from eta, useful for ablations. When null,
    # eta both triggers first errors and controls the unpoisoned key mixture.
    trigger_eta: float | None = None
    random_suffix_mode: str = "valid_tokens"
    # S5 can keep parentheses syntactically valid after poisoning while making
    # later value-token feedback uninformative.
    keep_format_tokens: bool = True
    seed: int = 1337
    apply_to: str = "both"
    coord_strategy: str = "cyclic"
    fixed_coord: int = 0
    eligible_values: tuple[int, ...] = (1, 2, 3, 4, 5)
    one_key_per_block: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["eligible_values"] = list(self.eligible_values)
        return payload


@dataclass(frozen=True)
class RandomSuffixStepSpec:
    key_mask: torch.Tensor
    semantic_mask: torch.Tensor
    scaffold_token_ids: torch.Tensor | None = None


def random_suffix_noise_config_from_obj(
    value: Mapping[str, Any] | object | None,
) -> RandomSuffixNoiseConfig:
    if value is None:
        config = RandomSuffixNoiseConfig()
    elif isinstance(value, RandomSuffixNoiseConfig):
        config = value
    elif isinstance(value, Mapping):
        raw = {key: value[key] for key in RANDOM_SUFFIX_CONFIG_KEYS if key in value}
        config = RandomSuffixNoiseConfig(**_normalize_raw_config(raw))
    else:
        raw = {
            key: getattr(value, key)
            for key in RANDOM_SUFFIX_CONFIG_KEYS
            if hasattr(value, key)
        }
        config = RandomSuffixNoiseConfig(**_normalize_raw_config(raw))
    validate_random_suffix_noise_config(config)
    return config


def _normalize_raw_config(raw: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(raw)
    if "eligible_values" in normalized and normalized["eligible_values"] is not None:
        normalized["eligible_values"] = tuple(int(x) for x in normalized["eligible_values"])
    if "trigger_eta" in normalized and normalized["trigger_eta"] in ("", "null"):
        normalized["trigger_eta"] = None
    return normalized


def validate_random_suffix_noise_config(config: RandomSuffixNoiseConfig) -> None:
    if not config.enabled:
        raise ValueError(
            "random_suffix_noise.enabled must be true when using "
            f"{RANDOM_SUFFIX_AFTER_ERROR_LAW}"
        )
    if config.key_positions not in RANDOM_SUFFIX_KEY_POSITION_CHOICES:
        raise ValueError(
            f"unknown random_suffix_noise.key_positions={config.key_positions!r}; "
            f"expected one of {RANDOM_SUFFIX_KEY_POSITION_CHOICES}"
        )
    if config.trigger_eta is not None and not 0.0 <= float(config.trigger_eta) <= 1.0:
        raise ValueError(
            f"random_suffix_noise.trigger_eta={config.trigger_eta} must be in [0, 1]"
        )
    if config.random_suffix_mode not in RANDOM_SUFFIX_MODE_CHOICES:
        raise ValueError(
            f"unknown random_suffix_noise.random_suffix_mode={config.random_suffix_mode!r}; "
            f"expected one of {RANDOM_SUFFIX_MODE_CHOICES}"
        )
    if config.apply_to not in RANDOM_SUFFIX_APPLY_TO_CHOICES:
        raise ValueError(
            f"unknown random_suffix_noise.apply_to={config.apply_to!r}; "
            f"expected one of {RANDOM_SUFFIX_APPLY_TO_CHOICES}"
        )
    if config.coord_strategy not in {"fixed", "cyclic", "hash"}:
        raise ValueError(
            f"unknown random_suffix_noise.coord_strategy={config.coord_strategy!r}; "
            "expected one of ('fixed', 'cyclic', 'hash')"
        )
    if not 0 <= int(config.fixed_coord) < 5:
        raise ValueError("random_suffix_noise.fixed_coord must be in [0, 4]")
    if not config.one_key_per_block:
        raise ValueError("random_suffix_noise.one_key_per_block must remain true")
    if len(config.eligible_values) == 0:
        raise ValueError("random_suffix_noise.eligible_values must be non-empty")
    invalid = [int(value) for value in config.eligible_values if int(value) not in range(1, 6)]
    if invalid:
        raise ValueError(
            "random_suffix_noise.eligible_values are S5 values and must be in "
            f"1..5, got {invalid}"
        )


def validate_random_suffix_applies_to_task(
    config: RandomSuffixNoiseConfig,
    *,
    task_name: str,
) -> None:
    if config.apply_to == "both" or config.apply_to == task_name:
        return
    raise ValueError(
        f"teacher_law={RANDOM_SUFFIX_AFTER_ERROR_LAW!r} was requested for "
        f"task={task_name!r}, but random_suffix_noise.apply_to={config.apply_to!r}"
    )


def effective_trigger_eta(eta: float, config: RandomSuffixNoiseConfig) -> float:
    return float(eta if config.trigger_eta is None else config.trigger_eta)


def random_suffix_noise_meta(
    config: RandomSuffixNoiseConfig,
    *,
    eta: float,
    task_name: str,
    eligible_token_ids: Sequence[int],
) -> dict[str, Any]:
    payload = config.to_dict()
    payload["effective_trigger_eta"] = effective_trigger_eta(eta, config)
    payload["task"] = task_name
    payload["eligible_token_ids"] = [int(token_id) for token_id in eligible_token_ids]
    payload["law"] = RANDOM_SUFFIX_AFTER_ERROR_LAW
    return payload


def compute_poisoned_before(
    actions: torch.Tensor,
    clean_targets: torch.Tensor,
    key_mask: torch.Tensor,
) -> torch.Tensor:
    """Return whether each prefix already has a previous key-token mismatch.

    This is the online counterpart of the paper's poisoned flag: after a prior
    semantic/key error, later semantic feedback is uniform and no longer
    informative about the clean continuation.
    """
    if actions.ndim != 2:
        raise ValueError(f"actions must have shape [B, T], got {tuple(actions.shape)}")
    if clean_targets.ndim != 2:
        raise ValueError(
            f"clean_targets must have shape [B, T], got {tuple(clean_targets.shape)}"
        )
    if tuple(clean_targets.shape) != tuple(actions.shape):
        raise ValueError(
            f"clean_targets shape {tuple(clean_targets.shape)} must match "
            f"actions shape {tuple(actions.shape)}"
        )

    batch_size, target_len = actions.shape
    mask = key_mask.to(device=actions.device, dtype=torch.bool)
    if mask.ndim == 1:
        if int(mask.numel()) != target_len:
            raise ValueError(
                f"1D key_mask length {int(mask.numel())} must match target_len={target_len}"
            )
        mask = mask.view(1, target_len).expand(batch_size, target_len)
    elif mask.ndim == 2:
        if tuple(mask.shape) != tuple(actions.shape):
            raise ValueError(
                f"2D key_mask shape {tuple(mask.shape)} must match "
                f"actions shape {tuple(actions.shape)}"
            )
    else:
        raise ValueError(f"key_mask must be rank 1 or 2, got shape {tuple(mask.shape)}")

    key_mismatch = mask & actions.ne(clean_targets.to(device=actions.device, dtype=actions.dtype))
    poisoned_before = torch.zeros_like(key_mismatch)
    if target_len > 1:
        poisoned_before[:, 1:] = key_mismatch[:, :-1].cumsum(dim=1).gt(0)
    return poisoned_before


def make_random_suffix_generator(
    *,
    device: str | torch.device,
    seed: int,
) -> torch.Generator:
    torch_device = torch.device(device)
    try:
        generator = torch.Generator(device=torch_device)
    except TypeError:
        generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def random_suffix_after_error_probs(
    clean_probs: torch.Tensor,
    *,
    eta: float,
    poisoned: torch.Tensor,
    key_mask: torch.Tensor,
    semantic_mask: torch.Tensor,
    eligible_token_ids: Sequence[int],
    scaffold_token_ids: torch.Tensor | None = None,
    keep_format_tokens: bool = True,
) -> torch.Tensor:
    """Apply the absorbing random-suffix law to one teacher-query step.

    `clean_probs` is the clean expert distribution. Before poisoning, key
    positions mix it with uniform mass over corruptible semantic tokens using
    eta. After poisoning, semantic positions are uniform over those tokens;
    optional scaffold tokens are forced to remain well-formed.
    """
    probs = clean_probs.float()
    had_token_dim = probs.ndim == 3
    if had_token_dim:
        if probs.size(-2) != 1:
            raise ValueError(
                "random_suffix_after_error_probs expects per-step probabilities "
                f"with singleton time dim, got shape {tuple(probs.shape)}"
            )
        probs_2d = probs.squeeze(-2)
    elif probs.ndim == 2:
        probs_2d = probs
    else:
        raise ValueError(f"clean_probs must be rank 2 or 3, got shape {tuple(probs.shape)}")

    batch_size, vocab_size = probs_2d.shape
    device = probs_2d.device
    poisoned = _normalize_step_mask(poisoned, batch_size=batch_size, device=device, name="poisoned")
    key_mask = _normalize_step_mask(key_mask, batch_size=batch_size, device=device, name="key_mask")
    semantic_mask = _normalize_step_mask(
        semantic_mask,
        batch_size=batch_size,
        device=device,
        name="semantic_mask",
    )

    eligible_ids = torch.as_tensor(eligible_token_ids, dtype=torch.long, device=device)
    if eligible_ids.numel() == 0:
        raise ValueError(f"{RANDOM_SUFFIX_AFTER_ERROR_LAW} requires eligible_token_ids")
    if int(eligible_ids.min().item()) < 0 or int(eligible_ids.max().item()) >= vocab_size:
        raise ValueError(
            f"eligible_token_ids={eligible_ids.tolist()} are outside vocab_size={vocab_size}"
        )

    uniform = torch.zeros_like(probs_2d)
    uniform.index_fill_(dim=-1, index=eligible_ids, value=1.0 / float(eligible_ids.numel()))
    out = probs_2d.clone()

    unpoisoned_key = (~poisoned) & key_mask
    if torch.any(unpoisoned_key):
        mixed = (1.0 - float(eta)) * probs_2d + float(eta) * uniform
        out[unpoisoned_key] = mixed[unpoisoned_key]

    poisoned_semantic = poisoned & semantic_mask
    if torch.any(poisoned_semantic):
        out[poisoned_semantic] = uniform[poisoned_semantic]

    poisoned_format = poisoned & ~semantic_mask
    if keep_format_tokens and torch.any(poisoned_format) and scaffold_token_ids is not None:
        scaffold_ids = scaffold_token_ids.to(device=device, dtype=torch.long).flatten()
        if scaffold_ids.numel() == 1 and batch_size != 1:
            scaffold_ids = scaffold_ids.expand(batch_size)
        if scaffold_ids.numel() != batch_size:
            raise ValueError(
                "scaffold_token_ids must be scalar or length batch_size, got "
                f"{scaffold_ids.numel()} for batch_size={batch_size}"
            )
        if int(scaffold_ids.min().item()) < 0 or int(scaffold_ids.max().item()) >= vocab_size:
            raise ValueError("scaffold_token_ids contain ids outside the vocabulary")
        forced = torch.zeros_like(probs_2d)
        forced.scatter_(1, scaffold_ids.unsqueeze(1), 1.0)
        out[poisoned_format] = forced[poisoned_format]

    return out.unsqueeze(-2) if had_token_dim else out


@torch.inference_mode()
def generate_random_suffix_after_error_targets(
    model,
    prompt_ids: torch.Tensor,
    *,
    target_len: int,
    eta: float,
    rollout_mode: str,
    target_mode: str,
    device: str | torch.device,
    config: RandomSuffixNoiseConfig,
    eligible_token_ids: Sequence[int],
    step_spec_fn: Callable[[int, torch.Tensor, torch.device], RandomSuffixStepSpec],
    generator: torch.Generator | None = None,
    saved_teacher_probs_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
    """Render fixed offline trajectories for LogLossBC under the absorbing law.

    The realized tokens are fed back into later teacher queries, so once a key
    token visibly differs from the clean teacher argmax, the remaining semantic
    suffix is sampled from the valid-token uniform distribution.
    """
    if rollout_mode not in {"greedy_then_corrupt", "sample_then_corrupt"}:
        raise ValueError(f"unknown rollout_mode={rollout_mode!r}")
    if target_mode not in {"tokens", "teacher_probs"}:
        raise ValueError(f"unknown target_mode={target_mode!r}")

    torch_device = torch.device(device)
    if generator is None:
        generator = make_random_suffix_generator(device=torch_device, seed=config.seed)

    input_ids = prompt_ids.to(device=torch_device, dtype=torch.long, non_blocking=True)
    generated = torch.empty(
        (prompt_ids.size(0), int(target_len)),
        dtype=prompt_ids.dtype,
        device=torch_device,
    )
    teacher_probs = None
    if target_mode == "teacher_probs":
        teacher_probs = torch.empty(
            (prompt_ids.size(0), int(target_len), model.config.vocab_size),
            dtype=saved_teacher_probs_dtype,
            device="cpu",
        )

    poisoned = torch.zeros(prompt_ids.size(0), dtype=torch.bool, device=torch_device)
    first_poison_positions = torch.full(
        (prompt_ids.size(0),),
        -1,
        dtype=torch.long,
        device=torch_device,
    )
    effective_eta = effective_trigger_eta(eta, config)
    past_key_values = None

    for step in range(int(target_len)):
        outputs = model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        if isinstance(outputs, tuple):
            step_logits, _, past_key_values = outputs
        elif hasattr(outputs, "logits") and hasattr(outputs, "past_key_values"):
            step_logits = outputs.logits
            past_key_values = outputs.past_key_values
        else:
            raise TypeError(
                "teacher model outputs must be either a tuple "
                "(logits, loss, past_key_values) or an object with "
                ".logits and .past_key_values"
            )

        step_logits = step_logits[:, -1:, :]
        clean_probs = F.softmax(step_logits.float(), dim=-1)
        step_spec = step_spec_fn(step, prompt_ids, torch_device)
        step_probs = random_suffix_after_error_probs(
            clean_probs,
            eta=effective_eta,
            poisoned=poisoned,
            key_mask=step_spec.key_mask,
            semantic_mask=step_spec.semantic_mask,
            eligible_token_ids=eligible_token_ids,
            scaffold_token_ids=step_spec.scaffold_token_ids,
            keep_format_tokens=config.keep_format_tokens,
        )
        if teacher_probs is not None:
            teacher_probs[:, step, :] = step_probs.squeeze(1).to(
                device="cpu",
                dtype=saved_teacher_probs_dtype,
            )

        if effective_eta <= 0.0 and rollout_mode == "greedy_then_corrupt":
            next_ids = torch.argmax(step_logits[:, -1, :], dim=-1)
        else:
            next_ids = torch.multinomial(
                step_probs.squeeze(1),
                num_samples=1,
                generator=generator,
            ).squeeze(1)

        clean_argmax = torch.argmax(step_logits[:, -1, :], dim=-1)
        key_mask = step_spec.key_mask.to(device=torch_device, dtype=torch.bool).flatten()
        new_poison = (~poisoned) & key_mask & next_ids.ne(clean_argmax)
        if torch.any(new_poison):
            first_poison_positions[new_poison] = int(step)
            poisoned = poisoned | new_poison

        next_ids = next_ids.to(dtype=prompt_ids.dtype)
        generated[:, step] = next_ids
        input_ids = next_ids.unsqueeze(1).to(dtype=torch.long)

    diagnostics = {
        "poisoned": poisoned.to(device="cpu"),
        "first_poison_positions": first_poison_positions.to(device="cpu"),
    }
    return generated.to(device="cpu", dtype=prompt_ids.dtype), teacher_probs, diagnostics


def _normalize_step_mask(
    value: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    mask = value.to(device=device, dtype=torch.bool).flatten()
    if mask.numel() == 1 and batch_size != 1:
        return mask.expand(batch_size)
    if mask.numel() != batch_size:
        raise ValueError(
            f"{name} must be scalar or length batch_size, got {mask.numel()} "
            f"for batch_size={batch_size}"
        )
    return mask

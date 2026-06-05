from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


def _float_tag(value: object) -> str:
    return str(value).replace(".", "p").replace("-", "neg")


def _temp_tag(value: object) -> str:
    numeric = float(value)
    if numeric == 0:
        return "greedy"
    return f"t{_float_tag(value)}"


def _student_prefix_temp_suffix(
    method_family: object,
    rollout_temperature_override: object,
    loss_temperature_override: object,
) -> str:
    tags: list[str] = []
    rollout_text = "" if rollout_temperature_override is None else str(rollout_temperature_override)
    loss_text = "" if loss_temperature_override is None else str(loss_temperature_override)
    default_rollout = 1.0 if str(method_family) == "opd" else 0.0
    if rollout_text != "" and float(rollout_temperature_override) != default_rollout:
        tags.append(f"roll{_temp_tag(rollout_temperature_override)}")
    if loss_text != "":
        tags.append(f"loss{_temp_tag(loss_temperature_override)}")
    if not tags:
        return ""
    return "-" + "-".join(tags)


def _student_prefix_beta_suffix(loss: object, kl_beta: object = None) -> str:
    if kl_beta is None:
        return ""
    text = str(kl_beta)
    if text in {"", "None", "none", "null"}:
        return ""
    if str(loss) == "jsd":
        return f"-jsd_beta{_float_tag(kl_beta)}"
    return f"-beta{_float_tag(kl_beta)}"


def _rollout_tag(value: object) -> str:
    text = str(value)
    if text == "greedy_then_corrupt":
        return "greedy"
    if text == "sample_then_corrupt":
        return "sample"
    return text.replace("_", "-")


def _rollout_suffix(
    value: object,
    default: object = "greedy_then_corrupt",
    separator: object = "-",
) -> str:
    if str(value) == str(default):
        return ""
    return f"{separator}{_rollout_tag(value)}"


def _path_join(*parts: object) -> str:
    cleaned = [str(part) for part in parts if part is not None and str(part) != ""]
    if not cleaned:
        return ""
    return str(Path(cleaned[0]).joinpath(*cleaned[1:]))


def _out_name(name: object) -> str:
    return f"out-{name}"


def _seed_suffix(seed: object, label: object = "seed", default: object = 1337) -> str:
    numeric_seed = int(seed)
    numeric_default = int(default)
    if numeric_seed == numeric_default:
        return ""
    return f"_{label}{numeric_seed}"


def _depth_tag(depth: object) -> str:
    return f"depth{int(depth)}"


def _epoch_steps(sample_count: object, batch_size: object) -> int:
    n = int(sample_count)
    batch = int(batch_size)
    if batch <= 0:
        raise ValueError("batch_size must be positive")
    return (n + batch - 1) // batch


def _modadd_teacher_root(p: object, m: object, teacher_seed: object) -> str:
    return f"reruns/modadd_p{int(p)}_m{int(m)}_teacher{int(teacher_seed)}"


def _s5_teacher_root(m: object, teacher_seed: object) -> str:
    return f"reruns/s5_m{int(m)}_teacher{int(teacher_seed)}"


def _s5_suite_root(
    m: object,
    teacher_seed: object,
    render_seed: object,
    train_seed: object,
) -> str:
    return (
        f"reruns/s5_m{int(m)}_teacher{int(teacher_seed)}_"
        f"render{int(render_seed)}_train{int(train_seed)}"
    )


def _modadd_suite_root(
    p: object,
    m: object,
    teacher_seed: object,
    render_seed: object,
    train_seed: object,
) -> str:
    return (
        f"reruns/modadd_p{int(p)}_m{int(m)}_teacher{int(teacher_seed)}_"
        f"render{int(render_seed)}_train{int(train_seed)}"
    )


def _s5_block_size(m: object, mode: object = "cot") -> int:
    length = int(m)
    normalized_mode = str(mode)
    if normalized_mode == "base":
        return 7 * length + 7
    if normalized_mode in {"cot", "offline", "noisy", "opd"}:
        return 14 * length
    raise ValueError(f"unsupported S5 mode {normalized_mode!r}")


def _modadd_block_size(m: object, mode: object = "cot") -> int:
    length = int(m)
    normalized_mode = str(mode)
    if normalized_mode == "base":
        return length + 1
    if normalized_mode in {"cot", "offline", "noisy", "opd"}:
        return 2 * length
    raise ValueError(f"unsupported modular-addition mode {normalized_mode!r}")


def _modadd_prompt_bank_name(
    p: object,
    m: object,
    n_train: object,
    n_val: object,
    bank_seed: object = 1337,
) -> str:
    return (
        f"modadd_clean_prompt_bank_p{int(p)}_m{int(m)}_n{int(n_train)}_val{int(n_val)}"
        f"{_seed_suffix(bank_seed)}"
    )


def _s5_prompt_bank_name(
    m: object,
    n_train: object,
    n_val: object,
    bank_seed: object = 1337,
) -> str:
    return (
        f"s5_clean_prompt_bank_m{int(m)}_n{int(n_train)}_val{int(n_val)}"
        f"{_seed_suffix(bank_seed)}"
    )


def _s5_clean_dataset_name(
    m: object,
    n_train: object,
    render_seed: object = 1337,
) -> str:
    return (
        f"s5_clean_offline_m{int(m)}_n{int(n_train)}"
        f"{_seed_suffix(render_seed)}"
    )


def _s5_target_suffix(
    target_mode: object,
    default: object = "tokens",
    separator: object = "-",
) -> str:
    normalized = str(target_mode)
    if normalized == str(default):
        return ""
    if normalized == "teacher_probs":
        return f"{separator}full-dist"
    return f"{separator}{normalized.replace('_', '-')}"


def _s5_dataset_prefix(
    rollout_mode: object = "greedy_then_corrupt",
    target_mode: object = "tokens",
    render_seed: object = 1337,
    teacher_law: object = "distributional_noise",
) -> str:
    prefix = "s5_noisy_offline"
    if str(teacher_law) != "distributional_noise":
        prefix += f"_{str(teacher_law).replace('-', '_')}"
    if str(target_mode) == "teacher_probs":
        prefix += "_full_dist"
    if str(rollout_mode) != "greedy_then_corrupt":
        prefix += f"_{rollout_mode}"
    prefix += _seed_suffix(render_seed)
    return prefix


def _s5_dataset_name(
    dataset_prefix: object,
    m: object,
    subset_size: object,
    eta: object,
) -> str:
    return f"{dataset_prefix}_m{int(m)}_n{int(subset_size)}_eta_{_float_tag(eta)}"


def _s5_expert_run_name(
    m: object,
    depth: object,
    seed: object,
    mode: object = "cot",
) -> str:
    normalized_mode = str(mode)
    if normalized_mode not in {"cot", "base"}:
        raise ValueError(f"unsupported S5 expert mode {normalized_mode!r}")
    return f"s5-{normalized_mode}-m{int(m)}-{_depth_tag(depth)}-seed{int(seed)}"


def _s5_clean_offline_run_name(
    m: object,
    subset_size: object,
    seed: object,
) -> str:
    subset_n = int(subset_size)
    if subset_n > 0:
        return f"s5-clean-offline-bc-m{int(m)}-n{subset_n}-seed{int(seed)}"
    return f"s5-clean-offline-bc-m{int(m)}-seed{int(seed)}"


def _s5_noisy_bc_run_name(
    m: object,
    subset_size: object,
    eta: object,
    rollout_mode: object,
    target_mode: object,
    seed: object,
    teacher_law: object = "distributional_noise",
) -> str:
    law_suffix = ""
    if str(teacher_law) != "distributional_noise":
        law_suffix = f"-{str(teacher_law).replace('_', '-')}"
    return (
        f"s5-noisy-bc-m{int(m)}-n{int(subset_size)}-eta{_float_tag(eta)}"
        f"{_s5_target_suffix(target_mode)}"
        f"{_rollout_suffix(rollout_mode)}{law_suffix}-seed{int(seed)}"
    )


def _s5_student_prefix_run_name(
    method_family: object,
    loss: object,
    teacher_signal: object,
    m: object,
    subset_size: object,
    eta: object,
    teacher_law: object,
    rollout_temperature_override: object,
    loss_temperature_override: object,
    seed: object,
    kl_beta: object = None,
) -> str:
    beta_suffix = _student_prefix_beta_suffix(loss, kl_beta)
    return (
        f"s5-{method_family}-{loss}-{teacher_signal}-m{int(m)}-n{int(subset_size)}-"
        f"eta{_float_tag(eta)}-{teacher_law}"
        f"{_student_prefix_temp_suffix(method_family, rollout_temperature_override, loss_temperature_override)}-"
        f"{beta_suffix.lstrip('-') + '-' if beta_suffix else ''}"
        f"seed{int(seed)}"
    )


def _modadd_noisy_dataset_prefix(
    rollout_mode: object,
    render_seed: object = 1337,
    teacher_law: object = "distributional_noise",
) -> str:
    prefix = "modadd_noisy_offline"
    if str(teacher_law) != "distributional_noise":
        prefix += f"_{str(teacher_law).replace('-', '_')}"
    prefix += f"_{rollout_mode}"
    prefix += _seed_suffix(render_seed)
    return prefix


def _modadd_noisy_dataset_name(
    dataset_prefix: object,
    p: object,
    m: object,
    subset_size: object,
    eta: object,
) -> str:
    return (
        f"{dataset_prefix}_p{int(p)}_m{int(m)}_n{int(subset_size)}_eta_{_float_tag(eta)}"
    )


def _modadd_expert_run_name(
    p: object,
    m: object,
    depth: object,
    seed: object,
    mode: object = "cot",
) -> str:
    normalized_mode = str(mode)
    if normalized_mode not in {"cot", "base"}:
        raise ValueError(f"unsupported modular-addition expert mode {normalized_mode!r}")
    return (
        f"modadd-{normalized_mode}-p{int(p)}-m{int(m)}-{_depth_tag(depth)}-seed{int(seed)}"
    )


def _modadd_noisy_bc_run_name(
    p: object,
    m: object,
    subset_size: object,
    eta: object,
    rollout_mode: object,
    seed: object,
    teacher_law: object = "distributional_noise",
) -> str:
    law_suffix = ""
    if str(teacher_law) != "distributional_noise":
        law_suffix = f"-{str(teacher_law).replace('_', '-')}"
    return (
        f"modadd-noisy-bc-p{int(p)}-m{int(m)}-n{int(subset_size)}-eta{_float_tag(eta)}"
        f"{_rollout_suffix(rollout_mode)}{law_suffix}-seed{int(seed)}"
    )


def _modadd_student_prefix_run_name(
    method_family: object,
    loss: object,
    teacher_signal: object,
    p: object,
    m: object,
    subset_size: object,
    eta: object,
    teacher_law: object,
    rollout_temperature_override: object,
    loss_temperature_override: object,
    seed: object,
    kl_beta: object = None,
) -> str:
    beta_suffix = _student_prefix_beta_suffix(loss, kl_beta)
    return (
        f"modadd-{method_family}-{loss}-{teacher_signal}-p{int(p)}-m{int(m)}-n{int(subset_size)}-"
        f"eta{_float_tag(eta)}-{teacher_law}"
        f"{_student_prefix_temp_suffix(method_family, rollout_temperature_override, loss_temperature_override)}-"
        f"{beta_suffix.lstrip('-') + '-' if beta_suffix else ''}"
        f"seed{int(seed)}"
    )


def register_resolvers() -> None:
    OmegaConf.register_new_resolver("float_tag", _float_tag, replace=True)
    OmegaConf.register_new_resolver("temp_tag", _temp_tag, replace=True)
    OmegaConf.register_new_resolver("rollout_tag", _rollout_tag, replace=True)
    OmegaConf.register_new_resolver("rollout_suffix", _rollout_suffix, replace=True)
    OmegaConf.register_new_resolver("path_join", _path_join, replace=True)
    OmegaConf.register_new_resolver("out_name", _out_name, replace=True)
    OmegaConf.register_new_resolver("seed_suffix", _seed_suffix, replace=True)
    OmegaConf.register_new_resolver("depth_tag", _depth_tag, replace=True)
    OmegaConf.register_new_resolver("epoch_steps", _epoch_steps, replace=True)
    OmegaConf.register_new_resolver("s5_teacher_root", _s5_teacher_root, replace=True)
    OmegaConf.register_new_resolver("s5_suite_root", _s5_suite_root, replace=True)
    OmegaConf.register_new_resolver("s5_block_size", _s5_block_size, replace=True)
    OmegaConf.register_new_resolver("s5_prompt_bank_name", _s5_prompt_bank_name, replace=True)
    OmegaConf.register_new_resolver("s5_clean_dataset_name", _s5_clean_dataset_name, replace=True)
    OmegaConf.register_new_resolver("s5_target_suffix", _s5_target_suffix, replace=True)
    OmegaConf.register_new_resolver("s5_dataset_prefix", _s5_dataset_prefix, replace=True)
    OmegaConf.register_new_resolver("s5_dataset_name", _s5_dataset_name, replace=True)
    OmegaConf.register_new_resolver("s5_expert_run_name", _s5_expert_run_name, replace=True)
    OmegaConf.register_new_resolver("s5_clean_offline_run_name", _s5_clean_offline_run_name, replace=True)
    OmegaConf.register_new_resolver("s5_noisy_bc_run_name", _s5_noisy_bc_run_name, replace=True)
    OmegaConf.register_new_resolver("s5_student_prefix_run_name", _s5_student_prefix_run_name, replace=True)
    OmegaConf.register_new_resolver("modadd_teacher_root", _modadd_teacher_root, replace=True)
    OmegaConf.register_new_resolver("modadd_suite_root", _modadd_suite_root, replace=True)
    OmegaConf.register_new_resolver("modadd_block_size", _modadd_block_size, replace=True)
    OmegaConf.register_new_resolver("modadd_prompt_bank_name", _modadd_prompt_bank_name, replace=True)
    OmegaConf.register_new_resolver("modadd_noisy_dataset_prefix", _modadd_noisy_dataset_prefix, replace=True)
    OmegaConf.register_new_resolver("modadd_noisy_dataset_name", _modadd_noisy_dataset_name, replace=True)
    OmegaConf.register_new_resolver("modadd_expert_run_name", _modadd_expert_run_name, replace=True)
    OmegaConf.register_new_resolver("modadd_noisy_bc_run_name", _modadd_noisy_bc_run_name, replace=True)
    OmegaConf.register_new_resolver("modadd_student_prefix_run_name", _modadd_student_prefix_run_name, replace=True)

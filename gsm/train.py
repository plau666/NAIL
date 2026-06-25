"""Launch GSM LoRA experiments from YAML configs.

Examples:
    python train.py configs/nail_f.yaml
    python train.py configs/nail_mixed.yaml BETA=0.25 EXPERT_TEMP=4.0 GPU=1
    METHOD=nail-r python train.py
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


GSM_DIR = Path(__file__).resolve().parent
CONFIG_DIR = GSM_DIR / "configs"

METHOD_CONFIG = {
    "offline-bc": "offline_bc.yaml",
    "loglossbc": "offline_bc.yaml",
    "obc": "offline_bc.yaml",
    "nail-f": "nail_f.yaml",
    "opd-f": "opd_f.yaml",
    "nail-r": "nail_r.yaml",
    "opd-r": "opd_r.yaml",
    "nail-mixed": "nail_mixed.yaml",
    "mixed": "nail_mixed.yaml",
}

KEY_ALIASES = {
    "student": "student_model",
    "expert": "expert_model",
    "student_temp": "student_temperature",
    "expert_temp": "expert_temperature",
    "gsm8k_eval_loss_batch_size": "gsm8k_eval_loss_bsz",
}

ENV_KEYS = {
    "method",
    "student_model",
    "student_short",
    "expert_model",
    "expert_short",
    "train_data",
    "prompt_field",
    "student_temperature",
    "expert_temperature",
    "beta",
    "aux_sample",
    "epochs",
    "bsz",
    "grad_accum",
    "lr",
    "warmup_ratio",
    "weight_decay",
    "logging_steps",
    "max_grad_norm",
    "max_new_tokens",
    "max_length",
    "save_steps",
    "save_total_limit",
    "seed",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "gpu",
    "wandb_project",
    "gsm8k_eval_loss_data",
    "gsm8k_eval_loss_bsz",
    "vllm_gpu_mem_util",
    "run_name",
    "output_dir",
    "resume_from_checkpoint",
    "skip_initial_eval",
}


def canonical_key(key: str) -> str:
    normalized = key.strip().lower().replace("-", "_")
    return KEY_ALIASES.get(normalized, normalized)


def canonical_method(value: Any) -> str:
    return str(value).strip().lower().replace("_", "-")


def parse_scalar(value: str) -> Any:
    parsed = yaml.safe_load(value)
    return value if parsed is None and value.strip().lower() != "null" else parsed


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def temp_tag(value: Any) -> str:
    s = str(value)
    return "greedy" if s in {"0", "0.0"} else f"t{s.replace('.', 'p')}"


def beta_tag(value: Any) -> str:
    return f"beta{str(value).replace('.', 'p')}"


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Config must be a mapping: {path}")
    return {canonical_key(k): v for k, v in data.items()}


def split_args(argv: list[str]) -> tuple[Path | None, dict[str, Any], bool]:
    config_path = None
    overrides: dict[str, Any] = {}
    dry_run = False

    for arg in argv:
        if arg == "--dry-run":
            dry_run = True
        elif "=" in arg:
            key, value = arg.split("=", 1)
            overrides[canonical_key(key)] = parse_scalar(value)
        elif config_path is None:
            config_path = Path(arg)
        else:
            raise SystemExit(f"Unexpected argument: {arg}")

    return config_path, overrides, dry_run


def default_config_path(config_path: Path | None, overrides: dict[str, Any]) -> Path:
    if config_path is not None:
        return config_path if config_path.is_absolute() else GSM_DIR / config_path

    method = overrides.get("method") or os.environ.get("METHOD") or "nail-mixed"
    method = canonical_method(method)
    if method not in METHOD_CONFIG:
        valid = ", ".join(["offline-bc", "nail-f", "opd-f", "nail-r", "opd-r", "nail-mixed"])
        raise SystemExit(f"Unknown METHOD={method}. Valid methods: {valid}")
    return CONFIG_DIR / METHOD_CONFIG[method]


def apply_env_overrides(config: dict[str, Any]) -> None:
    for env_key, value in os.environ.items():
        key = canonical_key(env_key)
        if key in ENV_KEYS:
            config[key] = parse_scalar(value)


def require(config: dict[str, Any], key: str) -> Any:
    if key not in config or config[key] is None or config[key] == "":
        raise SystemExit(f"Missing required config key: {key}")
    return config[key]


def maybe_add(cmd: list[str], flag: str, config: dict[str, Any], key: str) -> None:
    value = config.get(key)
    if value is not None and value != "":
        cmd.extend([flag, str(value)])


def build_run_name(config: dict[str, Any]) -> str:
    method = canonical_method(require(config, "method"))
    rank = config.get("lora_rank", 128)
    seed = config.get("seed", 42)
    student = config.get("student_short", "student")

    if method == "offline-bc":
        return f"offline_bc_r{rank}_{student}_seed{seed}"

    expert = config.get("expert_short", "expert")
    s_tag = temp_tag(config.get("student_temperature", 1.0))
    e_tag = temp_tag(config.get("expert_temperature", 1.0))
    if method == "nail-mixed":
        return f"nail-mixed_r{rank}_{student}_{expert}_s{s_tag}_e{e_tag}_{beta_tag(config.get('beta', 0.5))}_seed{seed}"
    return f"{method}_r{rank}_{student}_{expert}_s{s_tag}_e{e_tag}_seed{seed}"


def common_env(config: dict[str, Any], uses_vllm: bool) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = str(config.get("gpu", 0))
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["WANDB_PROJECT"] = str(config.get("wandb_project", "NAIL"))
    if uses_vllm:
        env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    return env


def add_shared_lora_args(cmd: list[str], config: dict[str, Any]) -> None:
    maybe_add(cmd, "--lora_rank", config, "lora_rank")
    maybe_add(cmd, "--lora_alpha", config, "lora_alpha")
    maybe_add(cmd, "--lora_dropout", config, "lora_dropout")


def add_common_eval_args(cmd: list[str], config: dict[str, Any]) -> None:
    maybe_add(cmd, "--gsm8k_eval_loss_data", config, "gsm8k_eval_loss_data")
    maybe_add(cmd, "--gsm8k_eval_loss_batch_size", config, "gsm8k_eval_loss_bsz")
    if bool_value(config.get("skip_initial_eval", False)):
        cmd.append("--skip_initial_eval")


def build_command(config: dict[str, Any]) -> tuple[list[str], bool]:
    method = canonical_method(require(config, "method"))
    if method in {"loglossbc", "obc"}:
        method = "offline-bc"
    if method == "mixed":
        method = "nail-mixed"
    config["method"] = method

    config.setdefault("run_name", build_run_name(config))
    config.setdefault("output_dir", f"output/{config['run_name']}")

    if method == "offline-bc":
        cmd = [
            sys.executable,
            "trainers/offline_bc.py",
            "--student_model",
            str(require(config, "student_model")),
            "--train_data",
            str(require(config, "train_data")),
            "--output_dir",
            str(require(config, "output_dir")),
            "--name",
            str(require(config, "run_name")),
            "--wandb_project",
            str(config.get("wandb_project", "NAIL")),
            "--seed",
            str(config.get("seed", 42)),
            "--num_train_epochs",
            str(config.get("epochs", 1)),
            "--per_device_train_batch_size",
            str(config.get("bsz", 8)),
            "--gradient_accumulation_steps",
            str(config.get("grad_accum", 8)),
            "--learning_rate",
            str(config.get("lr", "1e-4")),
            "--warmup_ratio",
            str(config.get("warmup_ratio", 0.1)),
            "--weight_decay",
            str(config.get("weight_decay", 0.01)),
            "--logging_steps",
            str(config.get("logging_steps", 10)),
            "--save_steps",
            str(config.get("save_steps", 200)),
            "--save_total_limit",
            str(config.get("save_total_limit", 50)),
            "--bf16",
            "--max_length",
            str(config.get("max_length", 768)),
        ]
        add_shared_lora_args(cmd, config)
        add_common_eval_args(cmd, config)
        maybe_add(cmd, "--resume_from_checkpoint", config, "resume_from_checkpoint")
        return cmd, False

    trainer = {
        "nail-f": "trainers/forward_kl.py",
        "opd-f": "trainers/forward_kl.py",
        "nail-r": "trainers/reverse_kl.py",
        "opd-r": "trainers/reverse_kl.py",
        "nail-mixed": "trainers/mixed_kl.py",
    }.get(method)
    if trainer is None:
        valid = ", ".join(["offline-bc", "nail-f", "opd-f", "nail-r", "opd-r", "nail-mixed"])
        raise SystemExit(f"Unknown method: {method}. Valid methods: {valid}")

    cmd = [
        sys.executable,
        trainer,
        "--student_model",
        str(require(config, "student_model")),
        "--expert_model",
        str(require(config, "expert_model")),
        "--train_data",
        str(require(config, "train_data")),
        "--prompt_field",
        str(config.get("prompt_field", "question")),
        "--output_dir",
        str(require(config, "output_dir")),
        "--name",
        str(require(config, "run_name")),
        "--wandb_project",
        str(config.get("wandb_project", "NAIL")),
        "--seed",
        str(config.get("seed", 42)),
        "--num_train_epochs",
        str(config.get("epochs", 1)),
        "--batch_size",
        str(config.get("bsz", 2)),
        "--gradient_accumulation_steps",
        str(config.get("grad_accum", 32)),
        "--learning_rate",
        str(config.get("lr", "1e-4")),
        "--warmup_ratio",
        str(config.get("warmup_ratio", 0.1)),
        "--weight_decay",
        str(config.get("weight_decay", 0.01)),
        "--max_grad_norm",
        str(config.get("max_grad_norm", 1.0)),
        "--logging_steps",
        str(config.get("logging_steps", 50)),
        "--save_steps",
        str(config.get("save_steps", 200)),
        "--save_total_limit",
        str(config.get("save_total_limit", 50)),
        "--bf16",
        "--max_new_tokens",
        str(config.get("max_new_tokens", 512)),
        "--student_temperature",
        str(require(config, "student_temperature")),
        "--expert_temperature",
        str(require(config, "expert_temperature")),
    ]
    if method == "nail-mixed":
        maybe_add(cmd, "--beta", config, "beta")
    add_shared_lora_args(cmd, config)
    add_common_eval_args(cmd, config)
    maybe_add(cmd, "--vllm_gpu_mem_util", config, "vllm_gpu_mem_util")
    maybe_add(cmd, "--resume_from_checkpoint", config, "resume_from_checkpoint")
    if bool_value(config.get("aux_sample", False)):
        cmd.append("--aux_sample")
    return cmd, True


def main(argv: list[str]) -> int:
    config_arg, cli_overrides, dry_run = split_args(argv)
    config_path = default_config_path(config_arg, cli_overrides)
    config = load_config(config_path)
    apply_env_overrides(config)
    config.update(cli_overrides)

    cmd, uses_vllm = build_command(config)
    train_data = GSM_DIR / str(require(config, "train_data"))
    if not dry_run and not train_data.exists():
        hint = "Generate teacher rollouts first with data/teacher_rollout.py." if config["method"] == "offline-bc" else \
            "Unpack data/tinygsm/tinygsm_400k.jsonl.gz first."
        raise SystemExit(f"Train data not found: {config['train_data']}\n{hint}")

    output_dir = GSM_DIR / str(config["output_dir"])
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    print("=== GSM LoRA training ===")
    print(f"  Config:    {config_path.relative_to(GSM_DIR) if config_path.is_relative_to(GSM_DIR) else config_path}")
    print(f"  Method:    {config['method']}")
    print(f"  Student:   {config['student_model']}")
    if config["method"] != "offline-bc":
        print(f"  Expert:    {config['expert_model']}")
    print(f"  Train:     {config['train_data']}")
    print(f"  GPU:       {config.get('gpu', 0)}")
    print(f"  Eff BSZ:   {int(config.get('bsz', 1)) * int(config.get('grad_accum', 1))} "
          f"(bsz={config.get('bsz')}, grad_accum={config.get('grad_accum')})")
    print(f"  LR:        {config.get('lr')}")
    print(f"  Output:    {config['output_dir']}")
    print(f"  Command:   {shlex.join(cmd)}")

    if dry_run:
        return 0

    return subprocess.run(cmd, cwd=GSM_DIR, env=common_env(config, uses_vllm)).returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

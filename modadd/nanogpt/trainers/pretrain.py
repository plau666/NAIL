from __future__ import annotations

import json
import os
import runpy
import shlex
import subprocess
from pathlib import Path

from nanogpt.trainers.configs import PretrainConfig
from nanogpt.utils.repo import command_env, repo_root, write_launch_metadata


_RUN_CONFIG_ENV = "NANOGPT_RUN_CONFIG"


def _torchrun_bin(python_bin: str) -> str:
    candidate = Path(python_bin).resolve().parent / "torchrun"
    if candidate.exists():
        return str(candidate)
    return "torchrun"


def _worker_config_path(cfg: PretrainConfig) -> Path:
    return Path(cfg.out_dir) / "worker_config.json"


def _launcher_command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _write_worker_config(cfg: PretrainConfig) -> Path:
    path = _worker_config_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg.worker_payload(), f, indent=2)
    return path


def _local_pretrain_payload(cfg: PretrainConfig | dict[str, object]) -> dict[str, object]:
    if isinstance(cfg, PretrainConfig):
        return cfg.worker_payload()
    return dict(cfg)


def run_pretrain_local(cfg: PretrainConfig | dict[str, object]) -> None:
    runpy.run_module(
        "nanogpt.workers.pretrain_body",
        init_globals={"INJECTED_CONFIG": _local_pretrain_payload(cfg)},
        run_name="__main__",
    )


def run_pretrain(cfg: PretrainConfig, *, launcher_command: list[str]) -> None:
    write_launch_metadata(Path(cfg.out_dir), cfg=cfg, command=launcher_command)
    torchrun = cfg.torchrun
    if torchrun.nproc_per_node <= 1 and torchrun.nnodes <= 1:
        run_pretrain_local(cfg)
        return

    config_path = _write_worker_config(cfg)
    command = [_torchrun_bin(cfg.python_bin)]
    if torchrun.standalone and torchrun.nnodes == 1:
        command.append("--standalone")
    command.extend(
        [
            f"--nproc_per_node={torchrun.nproc_per_node}",
            f"--nnodes={torchrun.nnodes}",
            f"--node_rank={torchrun.node_rank}",
            f"--master_addr={torchrun.master_addr}",
            f"--master_port={torchrun.master_port}",
            "-m",
            "nanogpt.workers.pretrain",
        ]
    )
    launch_file = Path(cfg.out_dir) / "launcher_command.txt"
    with open(launch_file, "a", encoding="utf-8") as f:
        f.write(f"\n# internal torchrun worker bootstrap\n{_launcher_command_text(command)}\n")
    env = command_env()
    env[_RUN_CONFIG_ENV] = str(config_path)
    subprocess.run(command, cwd=repo_root(), env=env, check=True)


def load_worker_config_from_env() -> dict[str, object]:
    config_path = os.environ.get(_RUN_CONFIG_ENV)
    if not config_path:
        raise RuntimeError(f"{_RUN_CONFIG_ENV} is required for the internal pretrain worker")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

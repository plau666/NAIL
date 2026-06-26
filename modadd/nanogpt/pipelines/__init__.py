from __future__ import annotations

from nanogpt.config_schema import AppConfig
from nanogpt.pipelines.modadd_data import run_modadd_prompt_bank, run_modadd_render
from nanogpt.trainers import run_nail, run_opd, run_pretrain
from nanogpt.trainers.configs import (
    project_nail_config,
    project_opd_config,
    project_pretrain_config,
)


def run_pipeline(cfg: AppConfig, *, launcher_command: list[str]) -> None:
    # Hydra experiments pick one of these pipelines. `pipeline=student_prefix`
    # is the neutral online backend; `pipeline=nail` is a legacy alias for the
    # same code path.
    if cfg.pipeline.name == "pretrain":
        run_pretrain(project_pretrain_config(cfg), launcher_command=launcher_command)
        return
    if cfg.pipeline.name == "opd":
        run_opd(project_opd_config(cfg), launcher_command=launcher_command)
        return
    if cfg.pipeline.name in {"nail", "student_prefix"}:
        run_nail(project_nail_config(cfg), launcher_command=launcher_command)
        return
    if cfg.pipeline.name == "modadd_prompt_bank":
        run_modadd_prompt_bank(cfg, launcher_command=launcher_command)
        return
    if cfg.pipeline.name == "modadd_render":
        run_modadd_render(cfg, launcher_command=launcher_command)
        return
    raise ValueError(f"unsupported pipeline {cfg.pipeline.name!r}")

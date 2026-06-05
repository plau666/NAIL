from __future__ import annotations

from nanogpt.trainers.configs import OpdConfig
from nanogpt.trainers.native_student_prefix import (
    run_student_prefix,
    validate_config,
    validate_resume_metadata as _validate_resume_metadata,
)


def run_opd(cfg: OpdConfig, *, launcher_command: list[str]) -> None:
    """Compatibility entrypoint for OPD-R over the student-prefix backend.

    OPD-R uses sampled learner prefixes and reuses rollout actions for the MC
    reverse estimator when rollout/loss distributions match. The implementation
    delegates to `run_student_prefix`, so gradients and RNG behavior are shared
    with the other student-prefix methods.
    """
    run_student_prefix(cfg, launcher_command=launcher_command)


validate_args = validate_config


def validate_resume_metadata(out_dir, metadata) -> None:
    _validate_resume_metadata(out_dir, metadata, default_method_family="opd")

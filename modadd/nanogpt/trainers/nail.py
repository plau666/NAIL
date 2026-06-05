from __future__ import annotations

from nanogpt.trainers.configs import NailConfig
from nanogpt.trainers.native_student_prefix import (
    run_student_prefix,
    validate_config,
    validate_resume_metadata as _validate_resume_metadata,
)


def run_nail(cfg: NailConfig, *, launcher_command: list[str]) -> None:
    """Legacy NAIL entrypoint for the shared student-prefix backend.

    New neutral configs may use `pipeline=student_prefix`; both paths delegate
    to `run_student_prefix` and preserve the same greedy-default method family.
    """
    run_student_prefix(cfg, launcher_command=launcher_command)


validate_args = validate_config


def validate_resume_metadata(out_dir, metadata) -> None:
    _validate_resume_metadata(out_dir, metadata, default_method_family="nail")

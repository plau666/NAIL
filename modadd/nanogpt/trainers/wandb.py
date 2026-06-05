from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def maybe_init_wandb(
    *,
    enabled: bool,
    project: str,
    run_name: str | None,
    run_id: str | None,
    out_dir: str | Path,
    init_from: str,
    init_timeout: int,
    config: dict[str, object],
):
    if not enabled:
        return None
    import wandb

    out_dir = Path(out_dir)
    state_path = out_dir / "wandb_state.json"
    fallback_id = hashlib.sha1(f"{project}:{out_dir.resolve()}".encode("utf-8")).hexdigest()[:16]
    fallback_pattern = re.compile(r"^[0-9a-f]{16}$")

    def is_fallback(candidate: str | None) -> bool:
        return candidate is not None and bool(fallback_pattern.fullmatch(candidate)) and candidate == fallback_id

    has_saved_run_id = False
    effective_run_id = run_id
    if is_fallback(effective_run_id):
        print(
            "warning: ignoring deterministic fallback W&B run id passed explicitly; "
            "will use resume='allow' instead."
        )
        effective_run_id = None
    if effective_run_id is None and state_path.exists():
        with open(state_path, "r", encoding="utf-8") as f:
            saved_state = json.load(f)
        effective_run_id = saved_state.get("run_id")
        if is_fallback(effective_run_id):
            print(
                "warning: ignoring stale deterministic fallback W&B run id from wandb_state.json; "
                "looking for a real resumable run id instead."
            )
            effective_run_id = None
        has_saved_run_id = effective_run_id is not None
    if effective_run_id is None:
        effective_run_id = fallback_id
        if init_from == "resume":
            print(
                "warning: no saved W&B run id found; using a deterministic fallback id. "
                "This may create a new W&B run instead of resuming the original graph."
            )
    has_explicit_resume_id = run_id is not None or has_saved_run_id
    resume_mode = "must" if init_from == "resume" and has_explicit_resume_id else "allow"
    try:
        wandb.init(
            project=project,
            name=run_name,
            id=effective_run_id,
            resume=resume_mode,
            config=config,
            settings=wandb.Settings(init_timeout=init_timeout),
        )
    except Exception as exc:
        deleted_run_id_error = "previously created and deleted" in str(exc)
        can_retry_fresh = (
            deleted_run_id_error
            and init_from != "resume"
            and not has_explicit_resume_id
        )
        if can_retry_fresh:
            print(
                "warning: deterministic fallback W&B run id refers to a deleted run; "
                "retrying with a fresh anonymous run id."
            )
            try:
                wandb.init(
                    project=project,
                    name=run_name,
                    config=config,
                    settings=wandb.Settings(init_timeout=init_timeout),
                )
            except Exception as retry_exc:
                print(
                    "warning: wandb.init retry failed "
                    f"({retry_exc}). Continuing with wandb logging disabled for this process."
                )
                return None
        else:
            print(
                f"warning: wandb.init failed ({exc}). Continuing with wandb logging disabled for this process."
            )
            return None
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": wandb.run.id,
                "project": project,
                "name": run_name,
            },
            f,
            indent=2,
        )
    return wandb

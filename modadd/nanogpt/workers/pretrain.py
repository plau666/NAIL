from __future__ import annotations

from nanogpt.trainers.pretrain import load_worker_config_from_env, run_pretrain_local


def main() -> None:
    run_pretrain_local(load_worker_config_from_env())


if __name__ == "__main__":
    main()

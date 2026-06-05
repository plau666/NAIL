from __future__ import annotations

import sys

import hydra
from omegaconf import DictConfig

from nanogpt.config_schema import materialize_config
from nanogpt.pipelines import run_pipeline
from nanogpt.utils.resolvers import register_resolvers


# Paper-reader orientation: `modadd/README.md` maps paper method names
# (LogLossBC, NAIL-F/R, OPD-F/R, NAIL-Mixed) to Hydra entrypoints and
# config fields.
register_resolvers()


@hydra.main(version_base=None, config_path="../hydra_configs", config_name="config")
def main(raw_cfg: DictConfig) -> None:
    cfg = materialize_config(raw_cfg)
    launcher_command = [cfg.runtime.python_bin or sys.executable, "-m", "nanogpt.run", *sys.argv[1:]]
    run_pipeline(cfg, launcher_command=launcher_command)


if __name__ == "__main__":
    main()

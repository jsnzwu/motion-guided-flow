from __future__ import annotations

import os

import pytest

from wickit.config import load_task_config


def test_cfg_assignment_is_rejected() -> None:
    cfg = load_task_config("config/entry/single_moflow_debug")
    with pytest.raises(Exception):
        cfg.job_name = "changed"


def test_cfg_container_mutation_is_rejected() -> None:
    cfg = load_task_config("config/entry/single_moflow_debug")
    with pytest.raises(Exception):
        cfg.runtime.cuda_visible_devices.append(2)  # tuple field has no append


def test_runtime_rejects_yaml_config_spec() -> None:
    with pytest.raises(Exception) as exc_info:
        _ = load_task_config("config/entry/single_moflow_debug.yaml")
    assert "unsupported config spec suffix" in str(exc_info.value)


def test_runner_path_cfg_write_violation_is_caught() -> None:
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    from wickit.runner.runner import Runner

    cfg = load_task_config("config/entry/single_moflow_debug")
    runner = Runner(cfg, None, resume=False)
    with pytest.raises(Exception):
        runner.config.runtime.device = "cuda:0"

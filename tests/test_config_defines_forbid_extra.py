from __future__ import annotations

import pytest
from pydantic import ValidationError

from config_defines import MFRRTaskConfig


def _base_config_dict() -> dict:
    return {
        "entry": "MFRRTaskConfig",
        "job_name": "forbid_extra_case",
        "runner": {"entry": "RunnerDummy"},
        "dataset": {"entry": "DatasetDummy"},
        "model": {"entry": "ModelDummy"},
        "train_parameter": {"batch_size": 1},
    }


def test_mfrr_task_config_rejects_unknown_field() -> None:
    data = _base_config_dict()
    data["unknown_field"] = 123
    with pytest.raises(ValidationError) as exc_info:
        _ = MFRRTaskConfig.from_dict(data)
    msg = str(exc_info.value)
    assert "unknown_field" in msg
    assert "extra_forbidden" in msg


def test_mfrr_task_config_rejects_type_mismatch() -> None:
    data = _base_config_dict()
    data["runner"] = {"entry": "RunnerDummy", "num_gpu": "oops"}
    with pytest.raises(ValidationError) as exc_info:
        _ = MFRRTaskConfig.from_dict(data)
    msg = str(exc_info.value)
    assert "runner.num_gpu" in msg
    assert "valid integer" in msg


from __future__ import annotations

from config_defines.moflow_config_utils import parse_config
from wickit.config import TaskConfig


def build_config() -> TaskConfig:
    return parse_config("config/dataset/infer_dataset_v6_moflow_ess.yaml", root_path="")

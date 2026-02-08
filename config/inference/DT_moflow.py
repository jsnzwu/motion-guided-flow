from __future__ import annotations

from config_defines.moflow_config_utils import parse_config
from wickit.config import TaskConfig


def build_config() -> TaskConfig:
    return parse_config("config/inference/DT_moflow.yaml", root_path="")

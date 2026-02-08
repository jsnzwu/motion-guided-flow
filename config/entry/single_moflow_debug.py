from __future__ import annotations

from config_defines.moflow_config_utils import parse_config
from wickit.config import TaskConfig


def build_config() -> TaskConfig:
    return parse_config("config/entry/single_moflow_debug.yaml", root_path="")

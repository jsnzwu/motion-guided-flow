from wickit.config.config_utils import (
    load_yaml_with_replacements,
    parse_config as _parse_config,
    parse_config_to_dict,
)
from config.moflow_components import MFRRTaskConfig


def parse_config(path: str, root_path: str = "") -> MFRRTaskConfig:
    # Ensure project configs are registered before resolving entry.
    return _parse_config(path, root_path=root_path)


__all__ = [
    "load_yaml_with_replacements",
    "parse_config",
    "parse_config_to_dict",
]

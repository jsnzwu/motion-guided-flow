from wickit.config.config_utils import (
    load_yaml_with_replacements,
    parse_config_to_dict,
)
from config_defines.moflow_components import MFRRTaskConfig


def parse_config(path: str, root_path: str = "") -> MFRRTaskConfig:
    # Bridge path: keep legacy yaml parser for dict materialization only.
    # Runtime entry must stay on pyconfig loader and should not call this helper directly.
    config_dict = parse_config_to_dict(path, root_path=root_path)
    return MFRRTaskConfig.model_validate(config_dict)


__all__ = [
    "load_yaml_with_replacements",
    "parse_config",
    "parse_config_to_dict",
]

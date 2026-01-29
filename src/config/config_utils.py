from wickit.config.config_utils import load_yaml_with_replacements, parse_config_to_dict

from config.components import MFRRTaskConfig


def parse_config(path: str, root_path: str = "") -> MFRRTaskConfig:
    config_dict = parse_config_to_dict(path, root_path=root_path)
    return MFRRTaskConfig.from_dict(config_dict)


__all__ = [
    "load_yaml_with_replacements",
    "parse_config",
    "parse_config_to_dict",
]

# config

Project-specific configuration dataclasses and helpers.

## Contents
- `moflow_components.py`: Defines `MFRRTaskConfig` and related dataclasses (e.g., `FGDatasetConfig`, `FGModelConfig`, `MFRRRunnerConfig`), registered in `wickit.config.CONFIGS`.
- `moflow_config_utils.py`: Wrapper helpers returning the registry-resolved config (e.g., `MFRRTaskConfig`) from parsed YAML (`entry`-based).

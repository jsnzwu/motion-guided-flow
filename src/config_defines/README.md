# config_defines

Project-specific configuration dataclasses and legacy YAML bridge helpers.

## Contents
- `moflow_components.py`: Defines `MFRRTaskConfig` and related dataclasses (e.g., `FGDatasetConfig`, `FGModelConfig`, `MFRRRunnerConfig`), registered in `wickit.config.CONFIGS`.
- `moflow_config_utils.py`: YAML bridge helpers returning registry-resolved config objects (e.g., `MFRRTaskConfig`).

## Pyconfig Module Protocol
- Preferred protocol: implement `build_config() -> TaskConfig` in each pyconfig module.
- Compatible fallback: module-level `CONFIG: TaskConfig`.
- Priority rule: when both exist, loader always uses `build_config()`.

## `--config` Canonical Spec
- Canonical `--config` value is suffixless pyconfig spec, e.g. `config/inference/DT_moflow`.
- Loader only auto-appends `.py` for suffixless specs.
- `.py` specs are loaded as pyconfig modules directly.

## Compatibility Policy (#2)
- Optional compatibility adapter is allowed only inside loader implementation.
- If compatibility path is enabled, it must emit deprecation logs and remain auditable.
- Compatibility code must stay removable in later cleanup phases.

# config_defines

Project-specific config extensions for MFRR, based on `wickit.config` Pydantic models.

## Current State
- `moflow_components.py` defines `MFRRTaskConfig` and project sub-models on top of `ConfigStruct`.
- Runtime config model path is Pydantic-only.
- Unknown fields are rejected (`extra="forbid"`).

## Bridge Policy
- `moflow_config_utils.parse_config(...)` is bridge-only.
- Bridge implementation is limited to:
  - YAML parse to dict (`parse_config_to_dict`)
  - dict to typed model (`MFRRTaskConfig.model_validate`)
- Runtime main entry must not call bridge parse directly; runtime uses `load_task_config(...)`.

## Update Rule
- Config updates must use controlled API with re-validate semantics (`copy_update`).
- Do not use direct shortcut updates that bypass validation.

## Runtime Boundary
- Runtime `cfg` is read-only.
- Mutable runtime state must be isolated from config.

## Bridge Exit Criteria
- No runtime entry path calls `parse_config(...)` bridge.
- No training/inference launcher path depends on YAML bridge output.
- When both are satisfied, bridge helper can be removed.

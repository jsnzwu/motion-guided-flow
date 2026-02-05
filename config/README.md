# Config

Project configuration entry points and defaults.

## Logging
- `config/base.yaml` now uses Wickit `logging` routes/targets (TensorBoard, console, file, bar, checkpoint/model).
- Legacy `log` blocks are not consumed by the Wickit logging system; prefer `logging` only.
- `config/model-pipeline/moflow_loss.yaml` overrides `logging.scalar_items` to control which metrics appear in logs/TensorBoard.

## Notes
- Entry configs in `config/entry/` typically inherit `base.yaml` via `model-arch/*` configs.
- Entry `dataset` sections should define `train_scene`/`test_scene` at the top level.

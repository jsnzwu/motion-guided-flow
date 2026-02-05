# test

Integration and utility test scripts for the project.

## Notes
- `test_inference.py` uses `eval()` on `config.model.type` and `config.trainer.type`, so the referenced classes (e.g., `MFRRNetModel`, `MFRRNetRunner`) must be imported in that module.
- `test_trainer.py` creates `MFRRNetRunner` directly and expects it to implement the training flow.

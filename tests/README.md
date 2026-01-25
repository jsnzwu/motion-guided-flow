## Migration Validation

Run the migration validation command from the repository root:

```bash
conda run -n env-py311-cu121 python src/test/test_inference.py --config .\\config\\inference\\DT_moflow.yaml
```

# Source Code Documentation

## Overview

This directory contains the main source code for the Motion-Guided Flow Migration (MoFlow) project. The codebase has been migrated to use the **wickit** framework as its foundation, with domain-specific logic preserved in this repository.

## Architecture

```
src/
├── config/              # Configuration utilities
├── dataloaders/         # Data loading and preprocessing
├── datasets/            # Dataset implementations
├── lr_scheduler/        # Learning rate schedulers (deprecated, use wickit.lr_schedulers)
├── models/              # Model architectures
│   ├── blend_model_base.py    # Blend model base class
│   ├── conv_lstm/             # ConvLSTM implementation
│   ├── general/               # General-purpose modules (UNet, CommonStructure)
│   ├── loss/                  # Loss functions
│   └── mfrrnet/               # MFRRNet core architecture
├── samplers/           # Data samplers (deprecated, use wickit.samplers)
├── trainers/           # Training runners
├── utils/              # Utility functions
└── test/               # Test scripts
```

## Key Components

### Configuration System

The project uses Wickit config utilities with an adapter to convert YAML/config dicts to typed dataclasses:

- **`wickit.config.config_utils`**: Configuration loading utilities (YAML + includes/pipeline/base)
- **`utils/config_adapter.py`**: Dict-to-dataclass adapter (`DictToDataclassAdapter`)

### Data Loading Pipeline

1. **`dataloaders/patch_loader.py`**: Main data loader with caching
2. **`dataloaders/dataset_base.py`**: Wickit dataset base (kept in sync; project-specific logic lives in `datasets/mfrrnet_dataset.py`)
3. **`datasets/mfrrnet_dataset.py`**: MFRRNet-specific dataset

### Models

- **`models/mfrrnet/mfrrnet.py`**: Main MFRRNet model (~1290 lines)
- **`models/blend_model_base.py`**: Blend model base for hybrid rendering
- **`models/loss/flow_loss.py`**: Domain-specific flow loss functions

### Training

- **`trainers/mfrrnet_runner.py`**: MFRRNet training runner extending wickit.Runner
- **`trainers/fe_runner_base.py`**: Feature-extraction runner base extending wickit.Runner

## Type Annotation Guidelines

When contributing to this codebase, please follow these type annotation rules:

### DO

```python
# Use specific types for return values
def aces_tonemapper(x: torch.Tensor, inv_gamma: bool = False) -> torch.Tensor:
    ...

# Use specific types for parameters
def to_numpy(arr: torch.Tensor, detach: bool = True, cpu: bool = True) -> np.ndarray:
    ...

# Use TypedDict or dataclass for complex dict structures
@dataclass
class DatasetGlobalConfig:
    max_luminance: int = -1
    log_tonemapper__mu: float = 8.0
```

### DON'T

```python
# Avoid Any for return types
def process_data(data) -> Any:  # Bad
    ...

# Avoid dict for parameter types when structure is known
def train(config: dict):  # OK for generic config, but prefer typed configs
    ...

# Avoid untyped parameters
def func(data, mode, config):  # Bad
    ...
```

## Code Review Checklist

### 1. Type Annotations

- [ ] All function parameters have type annotations
- [ ] All function returns have type annotations
- [ ] Avoid using `Any` or `object` types
- [ ] Use specific collection types (`list[int]` instead of `list`)

### 2. Redundant Code

- [ ] No duplicate function implementations
- [ ] No unused helper functions
- [ ] No commented-out dead code
- [ ] Constants defined at module level if used across functions

### 3. Dataclass Conversion

Consider converting dict-based configs to dataclasses when:
- The dict has a fixed set of keys
- Keys have known types
- The config is used in multiple places

Example conversion:

```python
# Before (dict-based)
def set_dataset_global_config(buffer_config):
    max_luminance = buffer_config.get('max_luminance', -1)
    mu = buffer_config.get('mu', 8.0)

# After (dataclass)
@dataclass
class BufferConfig:
    max_luminance: int = -1
    mu: float = 8.0
    is_normalization: bool = False

def set_dataset_global_config(config: BufferConfig):
    ...
```

## Migration Notes

This codebase was migrated from a self-contained infrastructure to use wickit. Key changes:

1. **Removed**: `src/lr_scheduler/`, `src/samplers/` (use wickit equivalents)
2. **Removed**: `src/utils/str_utils.py`, `src/utils/timer.py` (use wickit equivalents)
3. **Added**: `utils/config_adapter.py` for configuration adaptation
4. **Added**: `trainers/mfrrnet_runner.py` extending wickit.Runner
5. **Modified**: Models to inherit from wickit.ModelBase
6. **Modified**: Datasets to inherit from wickit.DatasetBase

## Testing

Run the migration validation:

```bash
conda run -n env-py311-cu121 python src/test/test_inference.py --config ./config/inference/DT_moflow.yaml
```

## Imports

The project uses the following import patterns:

```python
# wickit imports (external framework)
from wickit.runner import Runner
from wickit.datasets import DatasetBase
from wickit.models import ModelBase

# Project imports (domain-specific)
from utils.config_adapter import DictToDataclassAdapter
from datasets.mfrrnet_dataset import MFRRNetDataset
from trainers.mfrrnet_runner import MFRRNetRunner
```

from wickit.datasets import DATASETS

from .mfrrnet_dataset import MFRRNetDataset

DATASETS.register_module(name="MFRRNetDataset")(MFRRNetDataset)

__all__ = [
    "MFRRNetDataset",
]

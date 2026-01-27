from __future__ import annotations

from torch.utils.data import Dataset
from wickit.dataloaders.asset_loader_base import AssetLoaderBase
from wickit.utils.log import log
from wickit.datasets.metadata import MetaData
from wickit.utils.enums import ForwardMode

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from wickit.config.components import DatasetConfig

class DatasetBase(Dataset):
    def __init__(self, dataset_name, metadata_list: list[MetaData], asset_loader:AssetLoaderBase, mode=ForwardMode.train, config: DatasetConfig = None):
        self.dataset_name = dataset_name
        self.metadata_list = metadata_list
        self.mode = mode
        self.asset_loader = asset_loader
        self.config = config
        self.len = len(self.metadata_list)
        log.info("dataset_name: {}, data_size: {}".format(
            self.dataset_name, self.__len__()))

    @staticmethod
    def preprocess(data, config: DatasetConfig = None):
        return data

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index) -> dict:
        return {}

from __future__ import annotations

import torch

from wickit.datasets import DatasetBase as WickitDatasetBase
from wickit.datasets.metadata import MetaData, MetaDataWithPath
from .metadata_task_utils import (
    create_meta_data_list,
    create_metadata_by_glob,
    dispatch_task_by_metadata,
    dispatch_task_by_part_name,
)
from wickit.utils.enums import ForwardMode
from utils.dataset_utils import DatasetGlobalConfig
from utils.log_tonemap_utils import tonemap_func
from utils.log import log


class CropMetaData(MetaData):
    def __init__(self, scene_name, index, global_index, skybox_ratio, discontinuity_ratio):
        super().__init__(scene_name, index)
        self.global_index = global_index
        self.skybox_ratio = skybox_ratio
        self.discontinuity_ratio = discontinuity_ratio

    def to_dict(self):
        return {
            'dataset_name': self.dataset_name,
            'index': self.index,
            'global_index': self.global_index,
            'skybox_ratio': self.skybox_ratio,
            'discontinuity_ratio': self.discontinuity_ratio,
        }


class DatasetBase(WickitDatasetBase):
    def __init__(self, dataset_name, metadatas: list[MetaDataWithPath], mode="train"):
        runner_mode = ForwardMode.train if mode == "train" else ForwardMode.test
        super().__init__(dataset_name, metadatas, asset_loader=None, mode=runner_mode)
        self.metadatas = metadatas
        self.mode = mode
        log.info("dataset_name: {}, data_size: {}".format(
            self.dataset_name, self.__len__()))

    @staticmethod
    def preprocess(data, config=None):
        ret = {}
        for name in data.keys():
            if isinstance(data[name], torch.Tensor) and ('scene_color' in name or 'sky_color' in name or 'st_color' in name):
                ret[name] = tonemap_func(data[name], use_global_settings=True, mean_map=DatasetGlobalConfig.log_tonemapper__color_mean_map)
            elif isinstance(data[name], torch.Tensor) and ('scene_light' in name):
                ret[name] = tonemap_func(data[name], use_global_settings=True, mean_map=DatasetGlobalConfig.log_tonemapper__light_mean_map)
            elif 'normal' in name:
                ret[name] = data[name] * 0.5 + 0.5
            else:
                ret[name] = data[name]
        return ret


__all__ = [
    "DatasetBase",
    "MetaData",
    "MetaDataWithPath",
    "CropMetaData",
    "create_meta_data_list",
    "create_metadata_by_glob",
    "dispatch_task_by_metadata",
    "dispatch_task_by_part_name",
]

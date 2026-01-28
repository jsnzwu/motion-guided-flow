import torch

from wickit.datasets import DatasetABC
from wickit.utils.enums import ForwardMode
from dataloaders.asset_loader import AssetLoader
from utils.dataset_utils import DatasetGlobalConfig
from utils.log_tonemap_utils import tonemap_func

start_offset = 0


class MFRRNetDataset(DatasetABC):
    def __init__(self, config, dataset_name, metadata, patch_loader: AssetLoader, mode):
        runner_mode = ForwardMode.train if mode == "train" else ForwardMode.test
        super().__init__(dataset_name, metadata, asset_loader=None, mode=runner_mode, config=getattr(config, "dataset", None))
        self.metadatas = metadata
        self.mode = mode
        self.config = config
        self.batch_size = config.train_parameter.batch_size
        self.patch_loader = patch_loader
        self.part_size = self.config.dataset.part_size if mode == 'train' else 1
        self.is_block = config.dataset.is_block
        self.is_block_part = config.dataset.is_block_part if self.is_block else False
        
    def __getitem__(self, index) -> list[dict]:
        datas = [self.patch_loader.load(self.metadatas[index].get_offset(i),
                                      history_config=self.config.dataset.get('history_config', None),
                                      future_config=self.config.dataset.get('future_config', None),  allow_skip=False)
                 for i in range(self.part_size)]
        for i, item in enumerate(datas):
            assert self.metadatas[index].get_offset(i).index == item['metadata']['index'] - start_offset
        return datas

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

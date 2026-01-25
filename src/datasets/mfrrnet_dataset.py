import copy
import json
import os
import torch
import torch.utils.data._utils

from dataloaders.dataset_base import DatasetBase, MetaData
from dataloaders.patch_loader import PatchLoader
from utils.buffer_utils import create_flip_data
import numpy as np
from utils.log import get_local_rank, log
from wickit.utils.basic.string import dict_to_string

start_offset = 0

class MFRRNetDataset(DatasetBase):
    def __init__(self, config, dataset_name, metadata, patch_loader: PatchLoader, mode):
        super().__init__(dataset_name, metadata, mode)
        self.config = config
        self.batch_size = config['train_parameter']['batch_size']
        self.patch_loader = patch_loader
        self.part_size = self.config['dataset']['part_size'] if mode == 'train' else 1
        self.is_block = config['dataset']['is_block']
        self.is_block_part = config['dataset']['is_block_part'] if self.is_block else False
        
    def __getitem__(self, index) -> list[dict]:
        datas = [self.patch_loader.load(self.metadatas[index].get_offset(i),
                                      history_config=self.config['dataset'].get('history_config', None),
                                      future_config=self.config['dataset'].get('future_config', None),  allow_skip=False)
                 for i in range(self.part_size)]
        for i, item in enumerate(datas):
            assert self.metadatas[index].get_offset(i).index == item['metadata']['index'] - start_offset
        return datas



from wickit.utils.log import configure_logging, log
import numpy as np
import torch
from wickit.utils.basic.string import dict_to_string
from utils.buffer_utils import create_flip_data
from utils.dataset_utils import DatasetGlobalConfig
from wickit.utils.basic.tensor import data_as_type, data_to_device
from trainers.trainer_base import TrainerBase

configure_logging()
''' create feature_0 and encoding_0 '''


def get_his_recurrent_list(cur_data_index, num_he, block_size=0):
    assert block_size > 0, block_size
    ''' when block_size == 1, it's performed not in sequence. '''
    ''' every single frame will use recurrent frame of he_id = 1 '''
    if block_size == 1:
        return [(he_id % 2 == 1) for he_id in range(num_he)]
    else:
        return [((cur_data_index - he_id) % block_size != 0) for he_id in range(num_he)]

class FETrainerBase(TrainerBase):

    def __init__(self, config, model, resume=False):
        super().__init__(config, model, resume)
        self.cur_data_index = -1
        self.output_buffer = []
        self.loss_buffer = []

    def flip_data(self, data):
        def get_flip_argument():
            vertical = False
            horizontal = False
            if np.random.random() > 0.5:
                vertical = True
            if np.random.random() > 0.5:
                horizontal = True
            return vertical, horizontal

        def flip_(data, batch_size, flip_datas):
            assert len(data.get('future_data_list', [])) == 0, f"flip_ is not implemented for future_data_list, {dict_to_string(data)}"
            for batch_id in range(batch_size):
                data = create_flip_data(
                    data, vertical=flip_datas[batch_id][0], horizontal=flip_datas[batch_id][1], use_batch=True, batch_mask=[batch_id])
                if 'history_data_list' in data.keys():
                    history_datas = data['history_data_list']
                    for he_id, he_data in enumerate(history_datas):
                        history_datas[he_id] = create_flip_data(
                            he_data, vertical=flip_datas[batch_id][0], horizontal=flip_datas[batch_id][1], use_batch=True, batch_mask=[batch_id])
                ''' TODO: add flip to future_data_list '''
            return data

        if self.cur_data_index == 0:
            self.last_flip_datas = [get_flip_argument() for _ in range(self.train_dataset.batch_size)]

        data = flip_(data, self.train_dataset.batch_size, self.last_flip_datas)
        data['metadata']['vertical_flip'] = torch.tensor(
            [item[0] for item in self.last_flip_datas], device=data['metadata']['index'].device)
        data['metadata']['horizontal_flip'] = torch.tensor(
            [item[1] for item in self.last_flip_datas], device=data['metadata']['index'].device)
        return data

    def apply_max_luminance(self, data):
        for name in data.keys():
            if ('scene_light' in name or 'scene_color' in name or 'sky_color' in name or 'st_color' in name)\
                and isinstance(data[name], torch.Tensor) and len(data[name].shape) == 4:
                if DatasetGlobalConfig.max_luminance > 0:
                    data[name].clamp_max_(DatasetGlobalConfig.max_luminance)

    def load_data(self, data, mode="test"):
        self.cur_data['cur_data_index'] = self.cur_data_index
        if self.use_cuda:
            self.cur_data: dict = data_to_device(data, self.config['device'], non_blocking=True)  # type: ignore
        if mode == 'train' and self.config['dataset']['flip'] and self.config['dataset']['is_block_part']:
            assert 'vertical_flip' not in self.cur_data['metadata'].keys()  # type: ignore
            self.cur_data = self.flip_data(self.cur_data)
        self.set_recurrent_data(mode=mode)
        if mode == 'train':
            self.cur_data = data_as_type(self.cur_data, self.dataset_train_precision_mode)
        elif mode == 'test':
            self.cur_data = data_as_type(self.cur_data, self.dataset_test_precision_mode)
        # log.debug(dict_to_string(self.cur_data))
        if not self.config['dataset']['augment_loader']:
            self.cur_data = self.model.get_augment_data(self.cur_data)
        # log.debug(dict_to_string(self.cur_data))
        self.apply_max_luminance(self.cur_data)
        ''' make sure cur_data_index not deleted by get_augment_data '''
        self.cur_data['cur_data_index'] = self.cur_data_index
        # log.debug(dict_to_string([mode, self.dataset_test_precision_mode, self.cur_data]))
        # log.debug(dict_to_string({k: self.cur_data[k] for k in ['metadata', 'cur_data_index']}, "data_to_input"))

    def update_one_batch(self, epoch_index=None, batch_index=None, mode="train"):
        self.set_recurrent_feature(mode=mode)
        self.update_forward(epoch_index=epoch_index,
                            batch_index=batch_index, mode=mode)
        if mode == "train":
            self.gather_execute_result(training=True, enable_loss=True)
        if mode == "test":
            self.gather_execute_result(enable_loss=True)

        self.update_backward(epoch_index=epoch_index,
                             batch_index=batch_index, mode=mode)
        # log.info(f"[ShadeTrainerBase] done update_backward, {self.step%self.step_per_epoch}/{self.step_per_epoch}")
        self.cache_one_batch_output(mode)
        # log.info(f"[ShadeTrainerBase] done _cache_one_batch_output, {self.step%self.step_per_epoch}/{self.step_per_epoch}")

    def get_block_size(self, mode) -> int:
           # log.debug(dict_to_string(self.config['trainer']))
        block_cfg = self.config['trainer'][f'recurrent_{mode}']['block_size']
        flag = False
        for stage in block_cfg:
            cur_epoch_index = self.epoch_index
            total_epoch = self.total_epoch
            start_epoch = stage['start']
            end_epoch = stage['end']
            if stage.get('ratio', False):
                start_epoch *= total_epoch
                end_epoch *= total_epoch
            if cur_epoch_index >= start_epoch and cur_epoch_index < end_epoch:
                if self.cur_data_index ==0 and cur_epoch_index-1 < start_epoch:
                    ''' if going to a new stage, we clear former min_loss '''
                    self.min_loss = 1e9
                return stage['value']
        assert flag, dict_to_string([block_cfg, self.epoch_index, self.total_epoch])
        return 1

    def set_recurrent_data(self, mode):
        ...
        
    def set_recurrent_feature(self, mode):
        ...

    def cache_one_batch_output(self, mode, epoch_index=None, batch_index=None):
        ...

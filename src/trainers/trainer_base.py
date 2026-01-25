from __future__ import annotations
import copy
import cv2
import random
import torch
from torch.amp.autocast_mode import autocast as autocast
from torch.amp.grad_scaler import GradScaler
import torch.utils.data.distributed
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
import gc
from utils.buffer_utils import add_text_to_image, to_ldr_numpy
from wickit.lr_schedulers import ConstantWarmup, CosineAnnealingWarmupRestarts
from wickit.samplers.distributed_partition_sampler import DistributedPartitionSampler
from dataloaders.dataset_base import create_meta_data_list
from dataloaders.patch_loader import PatchLoader
from datasets.mfrrnet_dataset import MFRRNetDataset
from utils.loss_utils import lpips, psnr, ssim
from utils.utils import add_at_dict_front, create_dir, get_file_component, get_tensor_mean_min_max_str, inline_assert, \
    remove_all_in_dir, write_text_to_file
from wickit.utils.basic.string import dict_to_string, dict_to_string_join
from wickit.utils.log import configure_logging, log
from utils.warp import warp
from utils.dataset_utils import get_input_filter_list, resize
from utils.utils import del_data, del_dict_item
from utils.buffer_utils import buffer_data_to_vis, gamma, to_numpy
from models.loss.loss import LossFunction
from wickit.models import ModelBase
from tqdm import tqdm
from wickit.utils.basic.tensor import (
    align_channel_buffer,
    data_as_type,
    data_as_type_dict,
    data_to_device,
    data_to_device_dict,
)
import itertools
import torch.autograd
from torch.optim import AdamW, Adam
from torch.utils.data import DataLoader
from datetime import datetime, timedelta, timezone
from glob import glob
import json
import os
import re
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn.functional as F
import time
import math
from config.config_utils import convert_to_dict

configure_logging()


def parse_tensor_type(type_str):
    type_map = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
    }
    return type_map[type_str]


def should_active_at(step, total_step, total: int, offset=0, last=-1e9) -> bool:
    # if step - last < max(1, total_step // total):
    #     return False
    if total == 0:
        ''' always disabled when total == 0 '''
        return False
    else:
        return (step - offset) % (max(1, total_step // max(total, 1))) == 0


def get_time_string_in_dir(path):
    dirs = glob(path + "/*")
    dirs = [str(d).replace('\\', '/') for d in dirs]
    log.debug(dirs)
    newest_stamp = ""
    find = []
    for d in dirs:
        # res = re.match("(*)?/([\d]+)-([\d]+)-([\d]+)_([\d]+)-([\d]+)-([\d]+)",dirs)
        res = re.search(
            "(.+)/([\d]+)-([\d]+)-([\d]+)_([\d]+)-([\d]+)-([\d]+)", str(d))
        if (res):
            tmp_stamp = "{}-{}-{}_{}-{}-{}".format(res.group(2), res.group(
                3), res.group(4), res.group(5), res.group(6), res.group(7))
            if tmp_stamp > newest_stamp:
                find.append(tmp_stamp)
                newest_stamp = tmp_stamp
    if len(newest_stamp) > 0:
        log.debug("find newest_stamp \"{}\" in {}".format(newest_stamp, dirs))
    else:
        raise ValueError(f'cant find proper newest_stamp. possibly there is no train result under "{[path]}"')
    return newest_stamp


step_log_interval = 4
step_print_interval = 15


def set_random_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TrainerBase:
    def __init__(self, config, model: ModelBase, resume=False):
        config['_trainer_config'] = copy.deepcopy(convert_to_dict(config))
        self.resume = resume
        self.config = config
        self.class_name = self.config['trainer']['class']
        self.step_log_interval = self.config['log'].get('step_log_interval', step_log_interval)
        self.step_print_interval = self.config['log'].get('step_print_interval', step_print_interval)
        self.model = model
        if self.model is not None:
            self.net = self.model.get_net()
        self.next_write_debug_info_step_offset = 1
        self.last_write_debug_info_step = -1e9
        self.local_rank = int(self.config['local_rank'])
        self.enable_log = self.local_rank <= 0 and self.config['log_to_file']
        self.enable_test = self.local_rank <= 0

        self.shuffle_loader = self.config['dataset']['shuffle_loader']
        self.dataset_scaled = self.config['dataset'].get('scale', 1) != 1
        self.train_config = config['train_parameter']
        self.gradient_accumulation_steps = self.train_config['gradient_accumulation_steps']
        self.job_name = config['job_name']
        self.output_path = config['output_root_path'] + self.job_name
        self.debug_data_flow = config['trainer']['debug_data_flow']
        self.train_precision_mode = parse_tensor_type(self.train_config['train_precision_mode'])
        self.test_precision_mode = parse_tensor_type(self.train_config['test_precision_mode'])
        self.dataset_train_precision_mode = parse_tensor_type(self.config['dataset']['train_precision_mode'])
        self.dataset_test_precision_mode = parse_tensor_type(self.config['dataset']['test_precision_mode'])

        if self.resume:
            if 'time_string' in config.keys():
                self.time_string = config['time_string']
            else:
                self.time_string = get_time_string_in_dir(
                    "{}/".format(self.output_path))
                log.debug(self.time_string)
        else:
            gmt_format = "%Y-%m-%d_%H-%M-%S"
            tz = timezone(timedelta(hours=+8))
            self.time_string = datetime.now(tz).strftime(gmt_format)
            self.config['time_string'] = self.time_string
            if self.config.get('clear_output_path', False) and self.enable_log:
                remove_all_in_dir(self.output_path)

        self.log_path = "{}/{}/log/".format(
            self.output_path, self.time_string)
        self.history_model_path = "{}/{}/history_models/".format(
            self.output_path, self.time_string)
        self.model_path = "{}/{}/model/".format(
            self.output_path, self.time_string)
        self.tensorboard_path = "{}/{}/tensorboard/".format(
            self.output_path, self.time_string)
        self.checkpoint_path = "{}/{}/checkpoint/".format(
            self.output_path, self.time_string)
        self.history_checkpoint_path = "{}/{}/history_checkpoints/".format(
            self.output_path, self.time_string)

        self.batch_size = self.train_config['batch_size']
        self.num_gpu = self.config['num_gpu']
        # self.lr = self.train_config['lr']
        self.use_cuda = self.config['use_cuda']
        self.start_epoch = self.train_config.get('start_epoch', 0)
        # self.epoch_loop = None
        # self.main_loop = None
        self.end_epoch: int = self.train_config['epoch']
        self.total_epoch = self.train_config['epoch']
        self.config['tensorboard_info_step'] = {'train': [], 'test': []}
        self.config['tensorboard_info_epoch'] = {'train': [], 'test': []}
        self.config['bar_info_step'] = {'train': [], 'test': []}
        self.config['bar_info_epoch'] = {'train': [], 'test': []}
        self.config['text_info_step'] = {'train': [], 'test': []}
        self.config['text_info_epoch'] = {'train': [], 'test': []}
        self.config['avg_info_epoch'] = {'train': [], 'test': []}
        for mode in ['train', 'test']:
            for source in self.config['log'][mode].keys():
                item = self.config['log'][mode][source]
                item['name'] = item.get('name', source)
                if item.get('log_step', False):
                    self.config['tensorboard_info_step'][mode].append(source)
                if item.get('log_epoch', False):
                    self.config['tensorboard_info_epoch'][mode].append(source)
                if item.get('bar_step', False):
                    self.config['bar_info_step'][mode].append(source)
                if item.get('bar_epoch', False):
                    self.config['bar_info_epoch'][mode].append(source)
                if item.get('text_step', False):
                    self.config['text_info_step'][mode].append(source)
                if item.get('text_epoch', False):
                    self.config['text_info_epoch'][mode].append(source)
                if item.get('bar_epoch', False) or item.get('log_epoch', False) or item.get('text_epoch', False):
                    self.config['avg_info_epoch'][mode].append(source)

        self.epoch_index = 0
        self.save_best_epoch = max(int(self.train_config.get('save_best_at', 0) * self.end_epoch), 1) - 1
        self.batch_index = 0
        self.step = 0
        self.step_per_epoch = 0
        self.total_step = 0
        self.timestamp = time.time()
        self.start_time = time.time()
        self.min_loss = 1e9
        self.latent_step = 0
        self.best_step = -1
        self.data_time_interval = 0
        self.infer_time_interval = 0
        self.model_loaded = False
        self.loss = data_to_device(torch.tensor(
            0.0), device=self.config['device'])
        # self.loss_func = None
        self.active_loss_funcs = []

        self.info_epoch_avg = {}
        self.info_count = {}
        self.log_image_step_count = 0
        self.log_scalar_step_count = 0

        self.cur_data = {}
        self.cur_output = {}
        self.cur_lr = 0.0
        self.cur_loss = {}
        self.cur_loss_debug = {}
        self.enable_amp = self.train_config.get('amp', False)
        # self.prepare()

    def is_accumulated_step_interval(self, interval):
        return (self.step+1) % self.gradient_accumulation_steps == 0 and self.get_accumulated_step() % interval == 0

    def get_accumulated_step(self):
        return self.step // self.gradient_accumulation_steps

    ''' TODO: multi test_dataset '''

    def create_train_dataset(self) -> None:
        self.train_dataset = eval(self.config['dataset']['class'])(
            self.config, 'train', self.train_meta_data_list, self.data_loader, mode="train")

    def create_test_dataset(self, epoch_index=0) -> None:
        test_config = copy.deepcopy(self.config)
        test_config['buffer_config']['crop_config']['enable'] = False
        # log.debug(self.config['dataset'])
        self.test_dataset = eval(self.config['dataset']['class'])(
            test_config, 'test', self.test_meta_data_lists[epoch_index], self.data_loader, mode="test")
        self.cur_test_dataset_scene_name = self.test_meta_data_lists[epoch_index][0].dataset_name

    def create_valid_dataset(self) -> None:
        valid_config = copy.deepcopy(self.config)
        valid_config['buffer_config']['crop_config']['enable'] = False
        if len(self.valid_meta_data_list) > 0:
            self.valid_dataset = eval(self.config['dataset']['class'])(
                valid_config, 'valid', self.valid_meta_data_list, self.data_loader, mode="valid")
        else:
            self.valid_dataset = None

    def create_dataset_metadatas(self) -> None:
        if not self.config['dataset'].get('enable', True):
            return
        self.train_meta_data_list, self.valid_meta_data_list, self.test_meta_data_lists = create_meta_data_list(
            self.config)
        if len(self.train_meta_data_list) <= 0:
            raise RuntimeError("train dataset not found.")
        # calc input filter
        require_list = self.config['dataset']['require_list']

        self.data_loader = PatchLoader(
            self.config['dataset']['part'],
            job_config={'export_path': self.config['job_config']['export_path'],
                        'dataset_path': self.config['job_config']['dataset_path'],
                        'dataset_format': self.config['job_config']['dataset_format']},
            buffer_config=self.config['buffer_config'],
            require_list=require_list,
            with_augment=self.config['dataset']['augment_loader']
        )

    def create_train_loader(self) -> None:
        if not self.config['dataset'].get('enable', True):
            return
        pin_memory = self.config['dataset']['pin_memory']
        if self.config['use_ddp']:
            if self.config['dataset']['is_block'] and self.config['dataset']['is_block_part']:
                self.train_sampler = DistributedPartitionSampler(self.train_dataset,
                                                                 num_replicas=self.config['world_size'],
                                                                 rank=self.config['local_rank'],
                                                                 shuffle=self.config['dataset']['shuffle_loader'])
            else:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset,
                                                                                     num_replicas=self.config['world_size'],
                                                                                     rank=self.config['local_rank'],
                                                                                     shuffle=self.config['dataset']['shuffle_loader'])
            self.train_sampler.set_epoch(self.epoch_index)
        collate_fn = None
        # if self.config['buffer_config']['dual']:
        #     collate_fn = dual_collate_fn
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.batch_size,
                                       num_workers=self.config['dataset']['train_num_worker'],
                                       pin_memory=pin_memory,
                                       drop_last=True,
                                       sampler=self.train_sampler,
                                       shuffle=self.shuffle_loader,
                                       collate_fn=collate_fn)
        #    shuffle=self.config['dataset']['shuffle_loader'])
        # if self.valid_dataset:
        #     self.valid_loader = DataLoader(self.valid_dataset,
        #                                    batch_size=1,
        #                                    num_workers=self.config['dataset']['test_num_worker'],
        #                                    pin_memory=False,
        #                                    shuffle=self.shuffle_loader)

    def create_test_loader(self) -> None:
        if not self.config['dataset'].get('enable', True):
            return
        collate_fn = None
        # if self.config['buffer_config']['dual']:
        #     collate_fn = dual_collate_fn
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=1,
                                      num_workers=self.config['dataset']['test_num_worker'],
                                      pin_memory=False,
                                      shuffle=False,
                                      collate_fn=collate_fn)

    def create_scaler(self):
        # self.scaler = GradScaler(device='cuda', init_scale=4096.0, growth_interval=self.step_per_epoch, enabled=self.enable_amp)
        self.scaler = GradScaler(device='cuda', growth_interval=self.step_per_epoch, enabled=self.enable_amp)
        self.scaler__scale = self.scaler.get_scale()

    def create_optimizer(self):
        # initial_lr = self.train_config['lr'] * self.train_config['batch_size'] * self.config.get("num_gpu", 1)
        betas = tuple(self.train_config['optimizer'].get('betas', (0.9, 0.999)))
        self.train_config['optimizer']['betas'] = betas
        wd = self.train_config['optimizer'].get('weight_decay', 0.0)
        self.train_config['optimizer']['weight_decay'] = wd
        if self.enable_amp:
            eps = 1e-7
        else:
            eps = 1e-7
        optim_cls_hdlr = {
            'AdamW': AdamW,
        }
        optim_cls = optim_cls_hdlr[self.train_config['optimizer']['class']]
        self.optimizer = optim_cls(
            itertools.chain(self.model.net.parameters()),  # type: ignore
            lr=self.train_config['lr_config']['initial_lr'],
            betas=betas,
            eps=eps,
            weight_decay=wd)
        for param_group in self.optimizer.param_groups:
            if 'initial_lr' not in param_group:
                param_group['initial_lr'] = param_group['lr']
        # self.optimizer = Adam(
        #     itertools.chain(self.model.net.parameters()),
        #     lr=1e-6,
        # )

    def create_scheduler(self):
        assert self.optimizer is not None
        lr_cfg = self.train_config['lr_config']
        if lr_cfg['class'] == "ConstantWarmup":
            if 'warmup_epoch' not in lr_cfg.keys():
                lr_cfg['warmup_epoch'] = 0.0
            if 'min_lr' not in lr_cfg.keys():
                lr_cfg['min_lr'] = lr_cfg['max_lr']
            if 'decay_at_epoch' not in lr_cfg.keys():
                lr_cfg['decay_at_epoch'] = -1
            if 'total_epoch' not in lr_cfg.keys():
                lr_cfg['total_epoch'] = self.end_epoch
            self.scheduler = ConstantWarmup(
                self.optimizer,
                max_lr=lr_cfg['max_lr'],
                min_lr=lr_cfg['min_lr'],
                warmup_epoch=lr_cfg['warmup_epoch'],
                decay_at_epoch=lr_cfg['decay_at_epoch'],
                total_epoch=lr_cfg['total_epoch'],
                last_epoch=self.start_epoch
            )
        elif lr_cfg['class'] == "CosineAnnealingWarmupRestarts":
            if 'gamma' not in lr_cfg.keys():
                lr_cfg['gamma'] = 1.0
            if 'warmup_epoch' not in lr_cfg.keys():
                lr_cfg['warmup_epoch'] = 0.0
            if 'cycle_mult' not in lr_cfg.keys():
                lr_cfg['cycle_mult'] = 1.0
            self.scheduler = CosineAnnealingWarmupRestarts(
                self.optimizer,
                max_lr=lr_cfg['max_lr'],
                min_lr=lr_cfg['min_lr'],
                first_cycle_epoch=lr_cfg['first_cycle_epochs'],
                cycle_mult=lr_cfg['cycle_mult'],
                warmup_epoch=lr_cfg['warmup_epoch'],
                gamma=lr_cfg['gamma'],
                last_epoch=self.start_epoch
            )
        else:
            raise NotImplementedError

        optim_cfg = self.config['train_parameter']['optimizer']
        cfg = {
            'lr_config': lr_cfg,
            'optimizer_config': optim_cfg
        }
        if self.enable_log:
            self.write_text_info(
                text_dict_to_file=cfg,
                log_name="log",
                file_line_end="\n",
                enable_log=True
            )

    def create_loss_func(self, mode='train') -> None:
        self.cur_data = copy.deepcopy(self.model.dummy_net_input)
        self.cur_data = data_as_type_dict(self.cur_data, self.train_precision_mode)
        self.model.set_eval()
        # log.debug(dict_to_string(self.cur_data))
        log.debug(self.train_precision_mode)
        with autocast(device_type="cuda", dtype=self.train_precision_mode, enabled=self.enable_amp):
            self.cur_output = self.model.update(self.cur_data)

        loss_config = {}
        if mode == 'train':
            loss_config['train_loss'] = self.config["loss"]['train_loss']
            loss_config['debug_loss'] = self.config["loss"]['debug_loss']
        elif mode == 'test':
            loss_config['train_loss'] = {}
            loss_config['debug_loss'] = self.config["loss"]['test_loss']
        else:
            assert False, f'mode "{mode}" is not supported.'
        self.loss_func: LossFunction = LossFunction(loss_config)
        self.gather_execute_result(training=True, enable_loss=True)
        tmp_output = self.cur_output
        tmp_output.update(self.cur_data)
        self.loss_func.check_data(tmp_output)
        self.active_loss_funcs = self.loss_func.get_active_loss_func_names()
        self.cur_data.clear()
        self.cur_output.clear()
        log.debug("active loss_func name: {}".format(self.active_loss_funcs))

    def set_step_per_epoch(self, mode, epoch=0):
        self.step_per_epoch = 0
        if mode == 'train':
            if self.train_meta_data_list:
                assert len(self.train_meta_data_list) % (self.batch_size * self.num_gpu) == 0
                self.step_per_epoch = len(self.train_meta_data_list) // self.batch_size // self.num_gpu
            if self.config['dataset']['is_block']:
                if not self.config['dataset']['is_block_part']:
                    self.step_per_epoch *= self.config['dataset']['block_size']
                else:
                    self.step_per_epoch *= self.config['dataset']['part_size']
        elif mode == 'test':
            if self.test_meta_data_lists:
                self.step_per_epoch = len(self.test_meta_data_lists[epoch])

    def prepare(self, mode='train') -> None:
        set_random_seed()
        if self.enable_amp:
            _ = torch.set_flush_denormal(False)
        if self.enable_log:
            create_dir(self.log_path)
            create_dir(self.tensorboard_path)

        if self.config['dataset'].get('enable', True):
            self.create_dataset_metadatas()
            log.info("[{}]: dataset created.".format(
                self.config['trainer']['class']))

            if mode == 'train':
                self.set_step_per_epoch(mode)
                self.total_step = self.step_per_epoch * \
                    (self.end_epoch)
                self.latent_step = 20 * self.step_per_epoch
                self.create_scaler()
                self.create_optimizer()

            elif mode == 'test':
                self.set_step_per_epoch(mode)
                self.total_step = self.step_per_epoch
                self.end_epoch = self.total_epoch = len(self.test_meta_data_lists)

        if self.resume:
            self.load_checkpoint()

        if self.model is not None:
            if not self.resume and "pre_model" in self.config.keys():
                self.model.load_by_path(self.config["pre_model"])
                self.model_loaded = True
                log.debug("pre_model loaded: {}".format(self.config["pre_model"]))

            if self.config['use_ddp']:
                log.debug(f"model to_ddp, local_rank:{self.config['local_rank']}")
                self.model.to_ddp(
                    device_ids=[self.config['local_rank']],
                    output_device=self.config['local_rank'])

            self.create_loss_func(mode=mode)
            ''' test tensorboard_write'''
            self.cur_data = copy.deepcopy(self.model.dummy_input)
            self.cur_output = self.model.update(self.cur_data)
            self.get_tensorboard_image('test')
            self.cur_data.clear()
            self.cur_output.clear()
            ''' end of test tensorboard_write'''

        ''' after resuming the optimizer state, now we can initialize lr_scheduler '''
        if mode == 'train':
            self.create_scheduler()
            # self.scheduler._initial_step(resume_step=self.start_epoch)

        if mode == 'test' and self.model is not None:
            assert self.model_loaded, f'running at "test" mode, must loaded a pretrained model'

        if self.enable_log:
            self.writer_dict = {
                'info': SummaryWriter(self.tensorboard_path, filename_suffix="__info"),
                'train': SummaryWriter(self.tensorboard_path, filename_suffix="__train"),
                'test': SummaryWriter(self.tensorboard_path, filename_suffix="__test")
            }
            if not self.resume:
                config_file_path = self.log_path + \
                    "input_{}.json".format("{}_{}".format(
                        self.config['job_name'].replace("/", "_"), "runtime"))

                write_text_to_file(config_file_path, json.dumps(self.config['_input_config'], indent=4), "w")

                config_file_path = self.log_path + \
                    "trainer_{}.json".format("{}_{}".format(
                        self.config['job_name'].replace("/", "_"), "runtime"))

                write_text_to_file(config_file_path, json.dumps(self.config['_trainer_config'], indent=4), "w")

                self.write_text_info(
                    text_dict_to_tensorboard=self.get_info_dict('tensorboard', 'initial'),
                    text_dict_to_file=self.get_info_dict('text', 'initial'),
                    log_name="info",
                    tensorboard_mode='info',
                    enable_log=True
                )

            step_info = {
                "dict_context": "step_info",
                'step': self.step,
                'step_per_epoch': self.step_per_epoch,
                'total_step': self.total_step,
                'total_epoch': self.total_epoch,
                'step_log_interval': self.step_log_interval,
                'step_print_interval': self.step_print_interval,
            }
            self.write_text_info(
                text_dict_to_file=step_info,
                log_name="log",
                file_line_end="\n",
                enable_log=True
            )
            log.debug(dict_to_string(step_info))

    def load_data(self, data, mode='test'):
        if self.use_cuda:
            # log.debug(dict_to_string(data))
            # log.debug(dict_to_string(self.config['device']))
            self.cur_data = data_to_device_dict(data, self.config['device'], non_blocking=True)
        # log.debug(dict_to_string(self.cur_data))
        if mode == 'train':
            self.cur_data = data_as_type_dict(self.cur_data, self.dataset_train_precision_mode)
        elif mode == 'test':
            self.cur_data = data_as_type_dict(self.cur_data, self.dataset_test_precision_mode)
        # log.debug(dict_to_string(self.cur_data))
        # log.debug(dict_to_string([mode, self.dataset_train_precision_mode, self.dataset_test_precision_mode]))
        if not self.config['dataset']['augment_loader']:
            self.cur_data = self.model.get_augment_data(self.cur_data)
        # self.cur_data = data_as_type(self.cur_data, self.train_precision_mode) # type: ignore

    def update_forward(self, epoch_index=None, batch_index=None, mode="train"):
        self.update_step_before(mode=mode)
        # log.debug("{} forward".format(self.config['job_name']))
        if epoch_index is not None:
            self.epoch_index = epoch_index
        if batch_index is not None:
            self.batch_index = batch_index
        # log.debug(dict_to_string(self.cur_data, mmm=True))
        self.cur_data['enable_step_print'] = self.enable_step_print
        if mode == "train":
            self.cur_data = data_as_type_dict(self.cur_data, self.train_precision_mode)  # type: ignore
            with autocast(device_type="cuda", dtype=self.train_precision_mode, enabled=self.enable_amp):
                self.execute_model(training=True)
        if mode == "test":
            self.cur_data = data_as_type_dict(self.cur_data, self.train_precision_mode)  # type: ignore
            with autocast(device_type="cuda", dtype=self.train_precision_mode, enabled=self.enable_amp):
                self.execute_model(training=False)

    def update_backward(self, epoch_index=None, batch_index=None, mode="train"):
        if epoch_index is not None:
            self.epoch_index = epoch_index
        if batch_index is not None:
            self.batch_index = batch_index
        if mode == "test":
            # self.after_infer()
            with autocast(device_type="cuda", dtype=self.test_precision_mode, enabled=self.enable_amp):
                self.calc_loss_func(mode)
        if mode == "train":
            with autocast(device_type="cuda", dtype=self.train_precision_mode, enabled=self.enable_amp):
                self.calc_loss_func(mode)
            self.backward()
            # log.info(f"[TrainerBase] done backward, {self.step%self.step_per_epoch}/{self.step_per_epoch}")
        self.update_step(mode)
        # log.info(f"[TrainerBase] done update_step, {self.step%self.step_per_epoch}/{self.step_per_epoch}")
        self.step += 1

    def update(self, data, epoch_index=None, batch_index=None, mode="train"):
        self.load_data(data, mode=mode)
        self.update_forward(epoch_index=epoch_index,
                            batch_index=batch_index, mode=mode)
        if mode == "train":
            self.gather_execute_result(training=True, enable_loss=True)
        if mode == "test":
            self.gather_execute_result(enable_loss=True)
        self.update_backward(epoch_index=epoch_index,
                             batch_index=batch_index, mode=mode)

    def get_epoch_description_str(self, mode):
        if mode == 'train':
            return "{}train:{}/{},(BS{}/NW{}/NDV{}\"{}\"),S{}/I{}".format(
                "RK:{} ".format(
                    self.config['local_rank']) if self.config['local_rank'] >= 0 else "",
                self.step,
                self.total_step,
                self.batch_size,
                self.config['dataset']['train_num_worker'],
                self.num_gpu,
                os.environ.get(
                    'CUDA_VISIBLE_DEVICES', ""),
                self.log_scalar_step_count,
                self.log_image_step_count)
        elif mode == 'test':
            return "{}test:{}/{},(NDV{}\"{}\"),S{}/I{}".format(
                "RK:{} ".format(
                    self.config['local_rank']) if self.config['local_rank'] >= 0 else "",
                self.step,
                self.total_step,
                self.num_gpu,
                os.environ.get(
                    'CUDA_VISIBLE_DEVICES', ""),
                self.log_scalar_step_count,
                self.log_image_step_count)

    def train(self):
        self.prepare('train')
        if self.start_epoch >= self.end_epoch:
            return
        if self.config['detect_anomaly']:
            torch.autograd.set_detect_anomaly(True)
        log.info("start training, start_epoch:{}, end_epoch:{}".format(
            self.start_epoch, self.end_epoch))
        with tqdm(range(self.start_epoch, self.end_epoch),
                  position=0, leave=True, disable=not (self.enable_log)) as self.main_loop:
            #   position=0, leave=True, disable=True) as self.main_loop:
            for self.epoch_index in self.main_loop:
                # self.train_loader.sampler.set_epoch(self.epoch_index)
                self.reset_info_accumulator("train")
                self.main_loop.set_description_str(
                    "[{}] epoch: {}".format(self.config['job_name'], self.epoch_index))
                # with tqdm(range(self.train_loader.__len__()), position=1, leave=True) as test_epoch_loop:
                #     for _ in test_epoch_loop:
                #         log.debug("epoch: {} in-epoch step:{}".format(self.epoch_index, _))
                if self.dataset_scaled:
                    self.create_dataset_metadatas()
                self.create_train_dataset()
                self.create_train_loader()
                log.debug(
                    f'starting training epoch {self.epoch_index}, data size: {self.train_loader.__len__()}, step_per_epoch: {self.step_per_epoch}')
                with tqdm(self.train_loader, position=1, leave=True, disable=not (self.enable_log)) as self.epoch_loop:
                    # with tqdm(self.train_loader, position=1, leave=True, disable=True) as self.epoch_loop:
                    for self.batch_index, data in enumerate(self.epoch_loop):
                        # log.debug(dict_to_string(data, mmm=True))
                        ''' temp commented out '''
                        if self.enable_log:
                            self.epoch_loop.set_description_str(self.get_epoch_description_str('train'))
                        self.update(data, mode="train")
                        # log.info(f"[TrainerBase] done update, {self.step%self.step_per_epoch}/{self.step_per_epoch}")
                        self.cur_data.clear()
                        self.cur_loss.clear()
                        self.cur_loss_debug.clear()
                        self.cur_output.clear()
                        # torch.cuda.empty_cache()
                        # gc.collect()
                        # log.debug(dict_to_string(self.info_epoch_avg))
                self.update_epoch(mode="train")
                # torch.cuda.empty_cache()
                # log.debug('waiting for end barrier of epoch')
                # torch.cuda.empty_cache()
                # gc.collect()
        del self.train_loader
        torch.cuda.empty_cache()
        log.debug('waiting for end barrier of training')
        self.add_barrier()

    def test(self):
        self.prepare(mode='test')
        ''' reset self.step to 0, in case of mis-modification in self.prepare('test') '''
        log.debug("test start")
        if not (self.enable_test):
            return
        if self.resume:
            self.load_model(name="best")
        self.step = 0
        self.log_scalar_step_count = self.log_image_step_count = 0
        self.start_epoch = 0
        self.end_epoch = len(self.test_meta_data_lists)
        with tqdm(range(self.start_epoch, self.end_epoch),
                  position=0, leave=True, disable=not (self.enable_log)) as self.main_loop:
            for self.epoch_index in self.main_loop:
                # self.train_loader.sampler.set_epoch(self.epoch_index)
                self.create_test_dataset(self.epoch_index)
                self.create_test_loader()
                self.set_step_per_epoch('test', self.epoch_index)
                self.total_step = self.step_per_epoch
                self.main_loop.set_description_str(
                    "[{}] epoch: {}, scene_name:{}".format(self.config['job_name'],
                                                           self.epoch_index, self.cur_test_dataset_scene_name))
                log.debug(
                    f'starting test epoch {self.epoch_index}, data size: {self.test_loader.__len__()}, step_per_epoch: {self.step_per_epoch}')
                self.reset_info_accumulator("test")
                with tqdm(self.test_loader, position=1, leave=True, disable=not (self.enable_log)) as self.epoch_loop:
                    for self.batch_index, data in enumerate(self.epoch_loop):
                        self.cur_data.clear()
                        self.cur_loss.clear()
                        self.cur_loss_debug.clear()
                        if self.enable_log:
                            # bs: batch_size, nw: num_worker, dv: device
                            self.epoch_loop.set_description_str(self.get_epoch_description_str('test'))
                            epoch_info_bar = self.get_info_dict('bar', 'epoch', trainer_mode='test')
                            self.main_loop.set_postfix(epoch_info_bar)
                        with torch.no_grad():
                            self.update(data, mode="test")
                self.update_epoch("test")
                self.cur_output.clear()
                torch.cuda.empty_cache()

    def load_model(self, path=None, name="best"):
        if path is None:
            path = self.model_path
        file_path = "{}/{}.pt".format(path, name)
        self.model.load_by_path(file_path)
        self.model_loaded = True
        log.info("loaded model: {}".format(file_path))

    def save_model(self, file_name, path=None):
        if path is None:
            path = self.model_path
        file_path = path + "{}.pt".format(file_name)
        info_path = path + "{}.log".format(file_name)
        if create_dir(path):
            log.info("generate model save_dir: \"{}\".".format(path))
        self.model.save(file_path)
        f = open(info_path, "w")
        loss = self.get_avg_info("loss")
        model_loss = self.get_model_loss()
        f.write("epoch: {}, step: {} loss: {:.6f} model_loss: {:.6f} last_update_step: {}".format(
            self.epoch_index,
            self.step,
            loss,
            model_loss,
            self.best_step))
        log.info("saved model: {}.".format(file_path))

    def save_checkpoint(self, state_dict=True):
        if create_dir(self.checkpoint_path):
            log.info("generate checkpoint save_dir: \"{}\".".format(
                self.checkpoint_path))
        if create_dir(self.history_checkpoint_path):
            log.info("generate history checkpoint save_dir: \"{}\".".format(
                self.history_checkpoint_path))
        if remove_all_in_dir(self.checkpoint_path):
            log.debug("remove all files in save_dir: \"{}\".".format(
                self.checkpoint_path))
        checkpoint_dict = {
            'step': self.step,
            'epoch': self.epoch_index,
            'loss': self.get_model_loss(),
            'optimizer_state_dict': self.optimizer.state_dict(), # Save optimizer's state_dict
            'model_state_dict': self.model.get_net().state_dict(), # Save model's state_dict
            'best_step': self.best_step,
        }
        if self.scaler is not None:
            checkpoint_dict['scaler_state_dict'] = self.scaler.state_dict() # Save scaler's state_dict
        torch.save(checkpoint_dict, "{}/checkpoint_{}_{}.pt".format(self.checkpoint_path, self.epoch_index, self.step))
        torch.save(checkpoint_dict, "{}/checkpoint_{}_{}.pt".format(self.history_checkpoint_path, self.epoch_index, self.step))
        log.info("saved checkpoint: {}.".format(self.checkpoint_path))

    def load_checkpoint(self) -> None:
        path_template = "{}/checkpoint_{{}}_{{}}.pt".format(
            self.checkpoint_path)
        files = glob(path_template.format("[0-9]*", "[0-9]*"))
        if len(files) != 1:
            raise FileNotFoundError(
                "file ({}) of {} is not satisfied: {}".format(path_template, self.job_name, files))
        file_path = files[-1]
        file_res = get_file_component(file_path)
        res = re.match("(.*)[_](.*)[_](.*)", file_res['filename'])
        if res:
            epoch = res.groups()[1]
            step = res.groups()[2]
            load_path = path_template.format(epoch, step)
            checkpoint = torch.load(load_path, weights_only=True)
            self.step = checkpoint['step']
            self.start_epoch = checkpoint['epoch'] + 1
            if hasattr(self, 'optimizer') and self.optimizer is not None:
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                else:
                    log.warn(f"Optimizer state_dict not found in checkpoint: {load_path}. Skipping optimizer load.")
            
            # Load scaler state_dict
            if hasattr(self, 'scaler') and self.scaler is not None:
                if 'scaler_state_dict' in checkpoint:
                    # Check if scaler was enabled when saved
                    # A typical way to check if an AMP scaler was "enabled" is to check its state_dict contents.
                    # This might require inspecting the specific structure of GradScaler's state_dict.
                    # For simplicity here, we assume if present, it was enabled.
                    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                else:
                    log.warn(f'Scaler state_dict not found in checkpoint: {load_path}. Skipping scaler load.')

            # Load model state_dict
            if 'model_state_dict' in checkpoint:
                self.model.load_by_dict(checkpoint['model_state_dict']) # Assumes self.model.load_by_dict accepts state_dict
            else:
                raise KeyError(f"Model state_dict ('model_state_dict') not found in checkpoint: {load_path}")
            
            self.model_loaded = True
            log.info(f"loaded checkpoint {self.model.model_name}. step:{checkpoint['step']}, epoch:{checkpoint['epoch']}, loss:{checkpoint['loss']}.")
            # --- MODIFICATION END ---
        else:
            raise Exception(f'wrong name of checkpoint file_name, please check path "{file_path}"')

    def before_infer(self) -> None:
        self.data_time_interval = time.time() - self.timestamp
        self.timestamp = time.time()

    def after_infer(self) -> None:
        torch.cuda.synchronize()
        self.infer_time_interval = time.time() - self.timestamp
        self.timestamp = time.time()

    def reset_info_accumulator(self, mode="train") -> None:
        if not self.enable_log:
            return
        config = self.config["avg_info_epoch"][mode]
        for scalar_name in config:
            self.info_epoch_avg[scalar_name] = 0
            self.info_count[scalar_name] = 0

    def step_info_accumulator(self, mode="train") -> None:
        # log.debug(self.config["avg_info_epoch"])
        config = self.config["avg_info_epoch"][mode]
        for scalar_name in config:
            info = self.get_info(scalar_name)
            if info is None or torch.isnan(torch.tensor(info)) or torch.isinf(torch.tensor(info)):
                pass
            else:
                self.info_epoch_avg[scalar_name] += info
                self.info_count[scalar_name] += 1

    def get_avg_info(self, name, item=False) -> torch.Tensor | float | None:
        if name not in self.info_count.keys() or self.info_count[name] == 0:
            return None
        ret = self.info_epoch_avg[name] / self.info_count[name] if self.info_count[name] else 0
        if isinstance(ret, torch.Tensor) and item:
            ret = ret.item()
        return ret

    def update_step_before(self, mode="train"):
        self.enable_step_log = True
        self.enable_step_print = True
        if mode == "train":
            self.enable_step_log = self.is_accumulated_step_interval(self.step_log_interval)
            self.enable_step_print = self.is_accumulated_step_interval(self.step_print_interval)
        if mode == "test":
            self.enable_step_log = True
        self.__update_step_before_checked = True

    def update_step(self, mode="train"):
        assert self.__update_step_before_checked
        if not self.enable_log:
            # log.debug(f'skip write_tensorboard_step: {self.step%self.step_per_epoch}/{self.step_per_epoch}')
            return
        # log.debug(f'before write_tensorboard_step: {self.step%self.step_per_epoch}/{self.step_per_epoch}')
        self.write_tensorboard_step(mode=mode)
        # log.debug(f'after write_tensorboard_step: {self.step%self.step_per_epoch}/{self.step_per_epoch}')
        if self.enable_step_print and self.debug_data_flow:
            self.log_data_step()
        # step_info_bar = self.get_step_bar(mode=mode)
        step_info_bar = self.get_info_dict('bar', 'step', trainer_mode=mode)
        if self.epoch_loop is not None:
            self.epoch_loop.set_postfix(step_info_bar)
        if self.enable_step_log:
            step_info_text = {'step': self.step}
            step_info_text.update(self.get_info_dict('text', 'step', trainer_mode=mode))
            self.write_text_info(
                text_dict_to_file=step_info_text,
                file_line_mid=", ",
                log_name="{}_step".format(mode),
            )
        self.step_info_accumulator(mode=mode)
        self.__update_step_before_checked = False

    def get_model_loss(self):
        return self.get_avg_info("loss")

    def update_epoch(self, mode="train"):
        log.debug('entering update_epoch')
        if not self.enable_log:
            log.debug('skip update_epoch')
            return
        if mode == "train":
            if self.config['local_rank'] <= 0:
                loss = self.get_model_loss()
                assert loss is not None
                if self.epoch_index >= self.save_best_epoch and self.min_loss > loss:
                    self.min_loss = loss
                    self.best_step = self.step
                    self.save_model("best")
                self.save_model("new")
                self.save_model(f"model_e{self.epoch_index}", self.history_model_path)
                self.save_checkpoint()

            # log.debug('before epoch_index_update')
            epoch_info_bar = self.get_info_dict('bar', 'epoch', trainer_mode=mode)
            epoch_info_bar = add_at_dict_front(
                epoch_info_bar, "epoch", self.epoch_index)
            if self.main_loop is not None:
                self.main_loop.set_postfix(epoch_info_bar)
            epoch_info_text = {'epoch': self.epoch_index}
            epoch_info_text.update(self.get_info_dict('text', 'epoch', trainer_mode=mode))
            self.write_text_info(
                text_dict_to_file=epoch_info_text,
                file_line_mid=", ",
                log_name="{}_epoch".format(mode),
                enable_log=True
            )
        if mode == "test":
            epoch_info_bar = self.get_info_dict('bar', 'epoch', trainer_mode=mode)
            epoch_info_text = self.get_info_dict('text', 'epoch', trainer_mode=mode)
            log.info("test epoch: {}".format(epoch_info_text))
            self.write_text_info(
                text_dict_to_tensorboard=self.get_info_dict('tensorboard', 'epoch', trainer_mode=mode),
                tensorboard_mode=mode,
                text_dict_to_file=epoch_info_text,
                log_name="test",
                file_line_mid=", ",
                step=self.epoch_index,
                enable_log=True
            )
        self.write_to_tensorboard_scalar(step_mode='epoch', mode=mode)
        log.debug('leaving update_epoch')

    def _get_float_in_cur_loss(self, name, item=False):
        data = self.cur_loss[name]
        if isinstance(self.cur_loss[name], torch.Tensor) and item:
            data = self.cur_loss[name].item()
            self.cur_loss[name] = data
        return data

    def get_info(self, name, item=False):
        if name == 'lr':
            return self.cur_lr
        elif self.cur_loss is not None and name in self.cur_loss.keys():
            return self._get_float_in_cur_loss(name, item=item)
        else:
            return None

    def get_info_dict(self, info_mode, step_mode, trainer_mode='train') -> dict:
        '''
        info_mode: 'bar', 'text', 'tensorboard',
        step_mode: 'step', 'epoch', 'initial'
        trainer_mode: 'train', 'test'
        '''
        ret = {}
        if step_mode == 'initial':
            assert info_mode == 'text' or info_mode == 'tensorboard'
            ret['batch_size'] = self.batch_size
            if self.model is not None:
                ret['infer_time'] = self.model.get_infer_time()
                ret['net_parameter_num'] = self.model.get_net_parameter_num()
            if info_mode == 'text':
                ret['active_loss'] = self.active_loss_funcs
                if self.model is not None:
                    ret['net_struct'] = self.model.get_net()
            return ret
        if info_mode == 'bar':
            if trainer_mode == 'train':
                ret['LUS'] = "{:d}".format(self.best_step)
            # if trainer_mode == "test" and step_mode == 'step':
            #     ret['infer_time'] = "{:.3g}ms".format(
            #         self.infer_time_interval * 1000)
        if info_mode == 'text':
            if trainer_mode == 'test':
                if step_mode in ["step", "epoch"]:
                    ret['scene_name'] = self.cur_data['metadata']['scene_name'][0]  # type:ignore
                if step_mode == "step":
                    ret['index'] = self.cur_data['metadata']['index'][0].item()  # type:ignore
        print_list = self.config[f'{info_mode}_info_{step_mode}'][trainer_mode]
        if info_mode == 'text':
            for _src in self.config[f'bar_info_{step_mode}'][trainer_mode]:
                if _src not in print_list:
                    print_list.append(_src)
            for _src in self.config[f'tensorboard_info_{step_mode}'][trainer_mode]:
                if _src not in print_list:
                    print_list.append(_src)
        for source in print_list:
            if step_mode == 'step':
                value = self.get_info(source, item=True)
            elif step_mode == 'epoch':
                value = self.get_avg_info(source)
            else:
                raise Exception(f'with wrong step_mode: {step_mode}, only "step" and "epoch" supported')
            if value is None:
                continue
            name = self.config['log'][trainer_mode][source]['name']
            fmt = self.config['log'][trainer_mode][source]['fmt']
            ret[name] = fmt.format(value)
        return ret

    def gather_execute_result(self, training=False, enable_loss=False):
        for k in self.cur_output.keys():
            assert not k.endswith('_loss')
        if enable_loss:
            if self.loss_func is None:
                raise Exception('self.loss_func is None')
            if training:
                losses = self.loss_func.loss_funcs + self.loss_func.debug_loss_funcs
            else:
                losses = self.loss_func.debug_loss_funcs
            for item in losses:
                if not item.enable:
                    continue
                for arg in item.args:
                    if arg not in self.cur_output.keys() and arg in self.cur_data.keys():
                        self.cur_output[arg] = self.loss_data_process(
                            arg, self.cur_data)

    def execute_model(self, training=False):
        if training:
            self.model.set_train()
        else:
            self.model.set_eval()
        self.cur_output = self.model.update(self.cur_data)
        # assert self.cur_data['scene_color_no_st'].dtype == torch.float32
        # log.debug(dict_to_string(self.cur_data, mmm=True))
        # log.debug(dict_to_string(self.cur_output, mmm=True))

    def loss_data_process(self, arg: str, data: dict):
        return data[arg]

    def calc_loss_func(self, mode='train') -> None:
        # with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.enable_amp):
        # with autocast(device_type="cuda", dtype=self.train_precision_mode, enabled=self.enable_amp):

        if mode == 'train':
            self.cur_loss_debug = self.calc_loss_debug()
            self.cur_loss = self.calc_loss_train()
            if self.loss is None:
                self.loss = self.cur_loss['loss']
            else:
                self.loss += self.cur_loss['loss']
            self.cur_loss.update(self.cur_loss_debug)
        elif mode == 'test':
            self.cur_loss_debug = self.calc_loss_debug(force_full_precision=True)
            self.cur_loss = self.cur_loss_debug

    def calc_loss_train(self) -> dict:
        if self.loss_func is None:
            raise Exception('self.loss_func is None')
        ''' some complex loss should be calculated inner model scope '''
        self.model.calc_loss(self.cur_data, self.cur_output)
        ''' standard loss can be calculated through LossFunction via cur_output '''
        ret = self.loss_func.update_loss_info()
        ret = self.loss_func.forward(self.cur_output)
        return ret

    def calc_loss_debug(self, cpu=False, force_full_precision=False) -> dict:
        if self.loss_func is None:
            raise Exception('self.loss_func is None')
        with torch.no_grad():
            ret = self.loss_func.forward_debug(
                self.cur_output, cpu=cpu, force_full_precision=force_full_precision)
        return ret

    def write_text_info(self, text_dict_to_tensorboard=None, tensorboard_mode="info", text_dict_to_file=None,
                        log_name="info", file_line_mid="\n", file_line_end="", step=0, enable_log=False):
        if text_dict_to_tensorboard is not None:
            tmp_lines = dict_to_string_join(
                text_dict_to_tensorboard, sep="  \n")
            assert self.writer_dict is not None
            self.writer_dict[tensorboard_mode].add_text(log_name, tmp_lines, global_step=step)
            log.info("[{}]: add ({}) to tensorboard as tag \"{}\"".format(
                self.class_name, list(text_dict_to_tensorboard.keys()), log_name))
        if text_dict_to_file is not None:
            tmp_lines = dict_to_string_join(text_dict_to_file, mid=file_line_mid) + file_line_end
            file_path = "{}/{}.log".format(self.log_path, log_name)
            write_text_to_file(file_path, tmp_lines, "a")
            if enable_log:
                log.info("[{}]: add text to file: \"{}\"".format(
                    self.class_name, file_path))

    def write_tensorboard_step(self, mode="train") -> None:
        if mode == "train":
            if should_active_at(self.step, self.step_per_epoch, self.config['log']['train_image_epoch_sum'],
                                offset=self.step_per_epoch - max(self.batch_size//self.num_gpu, 1),
                                last=self.last_write_debug_info_step):
                self.write_to_tensorboard_image(mode=mode)
                self.last_write_debug_info_step = self.step
                log.info(f'write_to_tensorboard_step, {self.step%self.step_per_epoch}/{self.step_per_epoch}')
            if should_active_at(self.step, self.step_per_epoch, self.config['log']['train_scalar_epoch_sum']):
                self.write_to_tensorboard_scalar('step', mode=mode)
        elif mode == "test":
            # log.debug(self.config['log']['test_image_epoch_sum'])
            # log.debug(f"{self.step} {self.total_step}")
            # num_test_scene = len(self.test_meta_data_lists)
            if should_active_at(self.batch_index, self.step_per_epoch, self.config['log']['test_image_epoch_sum'], offset=self.config['log']['test_image_epoch_offset']):
                self.write_to_tensorboard_image(mode=mode)
            self.write_to_tensorboard_scalar('step', mode=mode)

    def write_to_tensorboard_scalar(self, step_mode, mode="train", step=None):
        '''
        step_mode: 'step', 'epoch',
        mode: 'train', 'test'
        '''
        assert step_mode == 'step' or step_mode == 'epoch'
        assert mode == 'train' or mode == 'test'
        if mode == 'test' and step_mode == 'epoch':
            step = 0
        if step_mode == 'step':
            if step is None:
                if mode == 'train':
                    step = self.get_accumulated_step()
                else:
                    step = self.step
        elif step_mode == 'epoch':
            if step is None:
                step = self.epoch_index
        config = self.config[f"tensorboard_info_{step_mode}"][mode]
        written_scalar_name = []
        for scalar_name in config:
            if step_mode == 'step':
                info = self.get_info(scalar_name, item=True)
            elif step_mode == 'epoch':
                info = self.get_avg_info(scalar_name)
            if info is None:
                pass
            else:
                assert self.writer_dict is not None
                self.writer_dict[mode].add_scalar(
                    "{}_{}_{}".format(step_mode, mode, scalar_name), info, global_step=step)
                written_scalar_name.append(scalar_name)
                if step_mode == 'step':
                    self.log_scalar_step_count += 1
        if step_mode == 'epoch':
            log.debug("write_epoch_{}: {} to tb".format(mode, written_scalar_name))

    def get_tensorboard_image(self, mode='train') -> dict:
        self.images = []
        self.image_texts = []
        self.prefix_texts = []
        self.debug_images = []
        self.debug_image_texts = []
        self.debug_prefix_texts = []
        self.gather_tensorboard_image(mode=mode)
        self.gather_tensorboard_image_debug(mode=mode)
        with torch.no_grad():
            device = self.images[0].device
            for img_id in range(len(self.images)):
                self.images[img_id] = add_text_to_image(
                    self.images[img_id], self.image_texts[img_id])
            for img_id in range(len(self.debug_images)):
                self.debug_images[img_id] = add_text_to_image(
                    self.debug_images[img_id], self.debug_image_texts[img_id])

            assert len(self.images) > 0
            # imgs = torch.cat(self.images, dim=1)
            imgs = np.concatenate(self.images, axis=1)
            if len(self.debug_images) > 0:
                # debug_imgs = torch.cat(self.debug_images, dim=1)
                debug_imgs = np.concatenate(self.debug_images, axis=1)
            else:
                debug_imgs = None

            text = ""
            debug_text = ""
            if mode == 'train':
                text = 'train/e({}/{}_s{})'.format(self.epoch_index, self.end_epoch, self.step)
                text += ' batch:{}/{}'.format(self.batch_index, self.train_loader.__len__())
            elif mode == 'test':
                text = 'test/'

            debug_text = 'debug_' + text
            scene_name = self.cur_data['metadata']['scene_name'][0]  # type: ignore
            index = self.cur_data['metadata']['index'][0]  # type: ignore
            text += f" img_{scene_name}_{index}"
            debug_text += f" img_{scene_name}_{index}"

            if mode in ['train', 'test']:
                text += " ({})".format(", ".join(self.prefix_texts + self.image_texts))
                debug_text += " ({})".format(", ".join(self.debug_prefix_texts + self.debug_image_texts))

        return {
            'imgs': imgs,
            'text': text,
            'debug_imgs': debug_imgs,
            'debug_text': debug_text
        }

    def gather_tensorboard_image_debug(self, mode='train'):
        pass

    def write_to_tensorboard_image(self, mode="train") -> None:
        image_data = self.get_tensorboard_image(mode=mode)
        assert self.writer_dict is not None
        if image_data['imgs'] is not None:
            self.writer_dict[mode].add_image(
                image_data['text'],
                image_data['imgs'],
                self.step,
                dataformats='HWC')
            # log.debug("write_img {} {}".format(
            # self.config['job_name'], self.step))
            self.log_image_step_count += 1

        if image_data.get('debug_imgs', None) is not None:
            self.writer_dict[mode].add_image(
                image_data['debug_text'],
                image_data['debug_imgs'],
                self.step,
                dataformats='HWC')
            self.log_image_step_count += 1

    def log_data_step(self):
        log.debug(dict_to_string(self.cur_data, "cur_data", mmm=True))
        log.debug(dict_to_string(self.cur_output, "cur_output", mmm=True))
        log.debug(dict_to_string(self.cur_loss, "cur_loss", mmm=True))

        # self.cur_data['history_warped_scene_color_no_st_0'] = warp(
        #     self.cur_data['history_scene_color_no_st_0'], self.cur_data['merged_motion_vector_0'])

        # def write_data(data):
        #     scene_name = data['metadata']['scene_name'][0]
        #     index = data['metadata']['index'][0]
        #     write_path = f"../output/images/trainer_debug/{scene_name}_{index}/"
        #     create_dir(write_path)
        #     for k in data.keys():
        #         if not (isinstance(data[k], torch.Tensor)) or len(data[k].shape) != 4:
        #             continue
        #         if data[k].shape[1] == 1:
        #             write_buffer("{}/{}_{}.exr".format(write_path, k, index),
        #                          align_channel_buffer(
        #                 data[k][0], channel_num=3, mode="repeat"))
        #         else:
        #             write_buffer("{}/{}_{}.exr".format(write_path, k, index),
        #                          data[k][0])

        # write_data(self.cur_data)
        # self.cur_output['metadata'] = self.cur_data['metadata']
        # write_data(self.cur_output)
        
    def backward(self, skip_step:bool = False, loss_rescale:float = 1.0) -> None:
        return self.backward_inner(skip_step, loss_rescale)
    def backward_inner(self, skip_step:bool = False, loss_rescale:float = 1.0) -> None:
        '''
        skip_step: force skip to optimizer.step
        loss_rescale: force rescale the loss
        '''
        def reduce_tensor(loss):
            ws = self.num_gpu
            with torch.no_grad():
                torch.distributed.reduce(loss, dst=0)
            return loss / ws

        # if self.scheduler is None:
        #     self.cur_lr = get_learning_rate(self)
        #     assert self.optimizer is not None
        #     for params_group in self.optimizer.param_groups:
        #         params_group['lr'] = self.cur_lr

        self.cur_lr = self.scheduler.get_lr()[0]
        if self.loss is not None:
            if float(self.loss) < 1e-9:
                self.loss = None
                return
            
            assert self.optimizer is not None
            if self.gradient_accumulation_steps > 1:
                self.loss = self.loss / self.gradient_accumulation_steps
            if loss_rescale != 1.0:
                self.loss = self.loss * loss_rescale
            self.scaler.scale(self.loss).backward()  # type: ignore

            ''' test code to examine the zero grads '''
            # for name, param in self.model.get_net().named_parameters():
            #     if param.grad is not None:
            #         # print(f"{name} gradient: {param.grad}")
            #         if torch.all(param.grad.eq(0)):
            #             log.debug(f"{name} gradient is all zeros!")
            #         else:
            #             log.debug(dict_to_string(param.grad, name, mmm=True))
            #     else:
            #         log.debug(f"{name} has no gradient.")

            # log.debug(dict_to_string({'self.gradient_accumulation_steps': self.gradient_accumulation_steps}))
            # self.loss.backward()
            # self.optimizer.step()
            if (self.step + 1) % self.gradient_accumulation_steps == 0 and not skip_step:
                # log.debug(f"step: {self.step}")
                scaler_step_retval = self.scaler.step(self.optimizer)  # type: ignore
                # log.debug(dict_to_string([scaler_step_retval]))
                # assert self.scheduler.optimizer._step_count == self.scheduler._step_count
                self.scaler.update()  # type: ignore
                # log.debug("after step")
                self.scheduler.step(float(self.step)/self.step_per_epoch)
                if (new_scale := self.scaler.get_scale()) < self.scaler__scale:
                    if self.enable_log:
                        msg = {'step': self.step,
                               'self.scaler__scale': self.scaler__scale,
                               'new_scale': new_scale,
                               'msg': 'skip backward, becuase of nan or inf in gradient.'}
                        self.write_text_info(
                            text_dict_to_file=msg,
                            file_line_mid=", ",
                            log_name="backward_step_debug",
                        )
                        log.warn(dict_to_string(msg))
                self.scaler__scale = new_scale
                flag = torch.isnan(self.loss).any() or torch.isinf(self.loss).any()
                if self.enable_log and flag:
                    if self.debug_data_flow:
                        log.debug("=" * 20 + "loss is nan" + "=" * 20)
                        log.warn("epoch: {}, step: {}, loss is nan, skip.".format(
                            self.epoch_index, self.step))
                        self.log_data_step()
                    msg = {'step': self.step,
                        'self.scaler__scale': self.scaler__scale,
                        'msg': 'detected nan from loss, becuase of nan or inf in gradient.',
                        'data': dict_to_string(self.cur_data, k='cur_data', mmm=True),
                        'output': dict_to_string(self.cur_output, k='cur_output', mmm=True),
                        'loss': dict_to_string(self.cur_loss, k='cur_loss', mmm=True),
                        }
                    self.write_text_info(
                        text_dict_to_file=msg,
                        file_line_mid=", ",
                        log_name="loss_nan_debug",
                    )
                if self.enable_amp and new_scale < 128.0:
                    assert False
                # if self.scheduler.optimizer._step_count >= self.scheduler._step_count:
                #     self.scheduler.step(float(self.step)/self.step_per_epoch)
                #     # log.debug(float(self.step)/self.step_per_epoch)
                # else:
                #     self.scheduler._update_step(float(self.step)/self.step_per_epoch)
                #     if self.enable_log:
                #         msg = {'step': self.step,
                #                'scale': self.scaler._scale,
                #                'self.scheduler.optimizer._step_count': self.scheduler.optimizer._step_count,
                #                'self.scheduler._step_count': self.scheduler._step_count,
                #                'msg': 'skip backward, becuase of nan or inf in gradient.'}
                #         self.write_text_info(
                #             text_dict_to_file=msg,
                #             file_line_mid=", ",
                #             log_name="backward_step_debug",
                #         )
                #         log.warn(dict_to_string(msg))
                self.optimizer.zero_grad(set_to_none=True)
            else:
                pass
                # log.debug(f"skip: {self.step}")
            ''' end of optimizer update '''
        self.loss = None

    def add_diff_buffer(self, name1, name2, scale=10, allow_skip=True, debug=False, cur_scale=1.0):
        buffer1 = self.get_buffer(name1, allow_skip=allow_skip)
        buffer2 = self.get_buffer(name2, allow_skip=allow_skip)
        if buffer1 is not None and buffer2 is not None:
            diff = scale * (torch.abs(buffer1 - buffer2))
            self.add_render_buffer(
                "{}*l1({},{})".format(scale, name1, name2), diff, debug=debug, cur_scale=cur_scale)
        else:
            assert allow_skip

    def gather_tensorboard_image(self, mode='train'):
        diff_scale = 10
        self.add_render_buffer("pred")
        self.add_render_buffer("gt")
        pred = self.get_buffer("pred", allow_skip=False, device="cuda")
        gt = self.get_buffer("gt", allow_skip=False, device="cuda")
        if pred is not None and gt is not None:
            diff = diff_scale * ((pred - gt)**2)
            # log.debug(dict_to_string([pred, gt]))
            self.add_render_buffer(f"diff ({diff_scale}x", buffer=diff)
            self.prefix_texts.insert(0, f'lpips: {float(lpips(pred, gt)):.4g}')
            self.prefix_texts.insert(0, f'ssim: {ssim(pred, gt):.4g}')
            self.prefix_texts.insert(0, f'psnr: {psnr(pred, gt):.4g}')

    def get_buffer(self, name, allow_skip=True, device="cuda") -> torch.Tensor | None:
        buffer = None
        if name in self.cur_output.keys():
            buffer = self.cur_output[name]
        if buffer is None and name in self.cur_data.keys():
            buffer = self.cur_data[name]
        if isinstance(buffer, torch.Tensor):
            ret = torch.narrow(buffer, 0, 0, 1).detach().float()
            device_handlers = {
                'cpu': lambda x: x.cpu(),
                'cuda': lambda x: x if inline_assert(x.is_cuda) else None,
            }
            assert device in device_handlers.keys()
            return device_handlers[device](ret)
        else:
            if not allow_skip:
                raise ValueError(f"Buffer '{name}' is not found.")

    def add_render_buffer(self, name, buffer=None, buffer_type="base_color", debug=False, cur_scale=1.0):
        if buffer is None:
            buffer_gpu = self.get_buffer(name, device='cuda')
            if buffer_gpu is None:
                return
        elif isinstance(buffer, torch.Tensor):
            buffer_gpu = buffer.detach()
        else:
            raise Exception(f'buffer type is "{type(buffer)}", but only torch.Tensor was supported!')
        buffer_gpu = buffer_gpu.float()
        if len(buffer_gpu.shape) == 4:
            buffer_gpu = buffer_gpu[0]
        if debug:
            buffer_gpu = resize(buffer_gpu.unsqueeze(0), 0.5/cur_scale)[0]
        else:
            buffer_gpu = resize(buffer_gpu.unsqueeze(0), 1.0/cur_scale)[0]
        if buffer_gpu.shape[0] == 1:
            buffer_gpu = align_channel_buffer(buffer_gpu, channel_num=3, mode="repeat")

        buffer_gpu = buffer_data_to_vis(
            align_channel_buffer(buffer_gpu), buffer_type)
        if buffer_type in ['base_color', 'scene_color', 'scene_light']:
            buffer_gpu = gamma(buffer_gpu)
        if debug:
            self.debug_images.append(buffer_gpu)
            self.debug_image_texts.append(name)
        else:
            self.images.append(buffer_gpu)
            self.image_texts.append(name)

    def add_barrier(self):
        if self.config['use_ddp']:
            torch.distributed.barrier()

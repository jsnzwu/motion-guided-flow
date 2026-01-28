from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from config.components import TaskConfig
from dataloaders.metadata_task_utils import create_meta_data_list
from dataloaders.asset_loader import AssetLoader
from datasets.mfrrnet_dataset import MFRRNetDataset
from trainers.fe_runner_abc import get_his_recurrent_list
from utils.buffer_utils import (aces_tonemapper, buffer_data_to_vis,
                                create_flip_data, inv_log_tonemapper, to_numpy)
from utils.dataset_utils import DatasetGlobalConfig
from utils.loss_utils import lpips, psnr, ssim
from wickit.runner import Runner
from wickit.utils.basic.tensor import (align_channel_buffer, data_as_type,
                                       data_to_device)
from wickit.utils.enums import ForwardMode
from wickit.utils.log import log


class MFRRNetRunner(Runner):
    def __init__(self, config: dict | TaskConfig, model, resume: bool = False):
        if isinstance(getattr(model, "trainer_config", None), TaskConfig):
            config = model.trainer_config
        elif not isinstance(config, TaskConfig):
            if isinstance(config, dict):
                config = TaskConfig.from_dict(config)
            else:
                raise TypeError(f"config must be dict or TaskConfig, got {type(config)}")
        super().__init__(config, model, resume)
        self.output_cache = None
        self.last_output = []
        self.last_scene_name = ""
        self.last_index = -1
        self.cur_data_index = -1
        self.use_cuda = self.device_context.use_gpu
        self.disable_debug_images = True

    def _normalize_mode(self, mode: ForwardMode | str) -> ForwardMode:
        if isinstance(mode, ForwardMode):
            return mode
        if isinstance(mode, str):
            return ForwardMode.from_str(mode)
        return ForwardMode.train

    def prepare(self, mode: ForwardMode | str) -> None:
        super().prepare(self._normalize_mode(mode))

    def update_forward(self, epoch_index: int | None = None, batch_index: int | None = None, mode: ForwardMode = ForwardMode.train) -> None:
        super().update_forward(epoch_index=epoch_index, batch_index=batch_index, mode=self._normalize_mode(mode))

    def update_backward(self, epoch_index: int | None = None, batch_index: int | None = None, mode: ForwardMode = ForwardMode.train) -> None:
        super().update_backward(epoch_index=epoch_index, batch_index=batch_index, mode=self._normalize_mode(mode))

    def update(self, data, epoch_index: int | None = None, batch_index: int | None = None, mode: ForwardMode = ForwardMode.train) -> None:
        mode = self._normalize_mode(mode)
        data_list = data if isinstance(data, list) else [data]
        for item in data_list:
            is_same_block = self.last_index > 0 and self.last_scene_name == item['metadata']['scene_name'][0] \
                and int(self.last_index) == int(item['metadata']['index'][0]) - 1
            self.last_scene_name = item['metadata']['scene_name'][0]
            self.last_index = item['metadata']['index'][0]
            if is_same_block:
                self.cur_data_index += 1
            else:
                self.cur_data_index = 0
                del self.last_output
                self.last_output = []
            self.load_data(item, mode)
            self.update_one_batch(epoch_index=epoch_index, batch_index=batch_index, mode=mode)

    def get_dataset_metadatas(self) -> None:
        if not self.config.dataset.enable:
            return
        self.train_meta_data_list, self.valid_meta_data_list, self.test_meta_data_lists = create_meta_data_list(
            self.config)
        if len(self.train_meta_data_list) <= 0:
            raise RuntimeError("train dataset not found.")
        require_list = self.config['dataset'].get('require_list', [])
        self.patch_loader = AssetLoader(
            self.config['dataset']['part'],
            job_config={
                'export_path': self.config['job_config']['export_path'],
                'dataset_path': self.config['job_config']['dataset_path'],
                'dataset_format': self.config['job_config']['dataset_format'],
            },
            buffer_config=self.config['buffer_config'],
            require_list=require_list,
            with_augment=self.config['dataset']['augment_loader']
        )

    def create_train_dataset(self) -> None:
        self.train_dataset = MFRRNetDataset(
            self.config,
            ForwardMode.train.name,
            self.train_meta_data_list,
            self.patch_loader,
            mode=ForwardMode.train.name,
        )

    def create_test_dataset(self, epoch_index: int = 0) -> None:
        test_config = copy.deepcopy(self.config)
        test_config.unfreeze()
        test_config.buffer_config['crop_config']['enable'] = False
        self.test_dataset = MFRRNetDataset(
            test_config,
            ForwardMode.test.name,
            self.test_meta_data_lists[epoch_index],
            self.patch_loader,
            mode=ForwardMode.test.name,
        )
        self.cur_test_dataset_scene_name = self.test_meta_data_lists[epoch_index][0].dataset_name

    def create_valid_dataset(self) -> None:
        valid_config = copy.deepcopy(self.config)
        valid_config.unfreeze()
        valid_config.buffer_config['crop_config']['enable'] = False
        if len(self.valid_meta_data_list) > 0:
            self.valid_dataset = MFRRNetDataset(
                valid_config,
                "valid",
                self.valid_meta_data_list,
                self.patch_loader,
                mode="valid",
            )
        else:
            self.valid_dataset = None

    def get_model_loss(self):
        log.debug("get model loss using \"-psnr\"")
        loss = self.get_avg_info("psnr")
        assert loss is not None
        return loss * -1.0

    def gather_tensorboard_image(self, mode: ForwardMode = ForwardMode.train):
        diff_scale = 4
        self.add_render_buffer("pred", buffer_type='scene_color')
        self.add_render_buffer("gt", buffer_type='scene_color')
        pred = aces_tonemapper(self.get_buffer("pred", allow_skip=False))
        gt = aces_tonemapper(self.get_buffer("gt", allow_skip=False))
        diff = diff_scale * ((pred - gt) ** 2)
        self.add_render_buffer(f"diff ({diff_scale}x)", buffer=diff)
        if 'pred_st_color' in self.cur_output.keys():
            pred_no_st = aces_tonemapper(self.get_buffer("pred_scene_color_no_st", allow_skip=False))
            gt_no_st = aces_tonemapper(self.get_buffer("scene_color_no_st", allow_skip=False))
            diff = diff_scale * ((pred_no_st - gt_no_st) ** 2)
            self.add_render_buffer("pred_scene_color_no_st", buffer_type='scene_color')
            self.add_render_buffer("scene_color_no_st", buffer_type='scene_color')
            self.add_render_buffer(f"diff_no_st ({diff_scale}x)", buffer=diff)
            pred_st = aces_tonemapper(self.get_buffer("pred_st_color", allow_skip=False))
            gt_st = aces_tonemapper(self.get_buffer("st_color", allow_skip=False))
            diff = diff_scale * ((pred_st - gt_st) ** 2)
            self.add_render_buffer("pred_st_color", buffer_type='scene_color')
            self.add_render_buffer("st_color", buffer_type='scene_color')
            self.add_render_buffer(f"diff_st ({diff_scale}x)", buffer=diff)
            if mode == ForwardMode.test:
                self.prefix_texts.insert(0, f'lpips_st: {float(lpips(pred_st, gt_st)):.4g}')
            self.prefix_texts.insert(0, f'ssim_st: {ssim(pred_st, gt_st):.4g}')
            self.prefix_texts.insert(0, f'psnr_st: {psnr(pred_st, gt_st):.4g}')
            if mode == ForwardMode.test:
                self.prefix_texts.insert(0, f'lpips_no_st: {float(lpips(pred_no_st, gt_no_st)):.4g}')
            self.prefix_texts.insert(0, f'ssim_no_st: {ssim(pred_no_st, gt_no_st):.4g}')
            self.prefix_texts.insert(0, f'psnr_no_st: {psnr(pred_no_st, gt_no_st):.4g}')
        if mode == ForwardMode.test:
            self.prefix_texts.insert(0, f'lpips: {float(lpips(pred, gt)):.4g}')
        self.prefix_texts.insert(0, f'ssim: {ssim(pred, gt):.4g}')
        self.prefix_texts.insert(0, f'psnr: {psnr(pred, gt):.4g}')

    def gather_tensorboard_image_debug(self, mode: ForwardMode = ForwardMode.train) -> None:
        if self.disable_debug_images:
            return
        with torch.no_grad():
            num_he = int(self.model.get_net().num_history_encoder)  # type: ignore
            num_dec = int(self.model.get_net().num_shade_decoder_layer)  # type: ignore

            net = self.model.get_net()
            self.add_render_buffer("pred", buffer_type="scene_color", debug=True)
            self.add_render_buffer("scene_color", buffer_type="scene_color", debug=True)

            if net.enable_demodulate:
                albedo = self.get_buffer('dmdl_color', allow_skip=False)
                self.add_render_buffer(f"dmdl_color({self.config['dataset']['demodulation_mode']})", albedo, debug=True)
            else:
                albedo = None

            if self.model.get_net().method in ["residual", "shade"]:
                self.add_render_buffer("pred_scene_light_no_st", debug=True)
                self.add_render_buffer("scene_light_no_st", debug=True)
                self.add_render_buffer("disc_mask", buffer_type="depth", debug=True)
                self.add_render_buffer("residual_mask", buffer_type="depth", debug=True)
                residual_item = self.get_buffer("residual_item", allow_skip=False)
                self.add_render_buffer("abs(residual)", buffer=torch.abs(-(residual_item) +  # type: ignore
                                                                         self.get_buffer("pred_scene_light_no_st", allow_skip=False)),
                                       buffer_type="depth", debug=True)
                self.add_render_buffer("pred_warped_scene_color_no_st", buffer=residual_item, buffer_type="scene_color", debug=True)
                self.add_diff_buffer("gt_comp", "gt", debug=True)
                self.add_render_buffer(f'pred_layer_{0}_tmv_{0}', buffer_type="motion_vector_8", debug=True)

            if self.model.get_net().enable_st:
                self.add_render_buffer("pred_st_color", buffer_type="scene_color", debug=True)
                self.add_render_buffer("st_color", buffer_type="scene_color", debug=True)
                self.add_render_buffer("pred_st_alpha", debug=True)
                self.add_render_buffer("st_alpha", debug=True)
                self.add_render_buffer("pred_sky_color", buffer_type="scene_color", debug=True)
                self.add_render_buffer("sky_color", buffer_type="scene_color", debug=True)
                self.add_render_buffer("skybox_mask", debug=True)
                self.add_render_buffer("pred_comp_color_before_sky_st", buffer_type="scene_color", debug=True)
                self.add_render_buffer("pred_comp_color_sky", buffer_type="scene_color", debug=True)

            def get_pyramid_buffer(layer_id, he_id, in_name):
                if f'pred_layer_{layer_id}_{in_name}_{he_id}' not in self.cur_output.keys():
                    return None
                if i == num_dec:
                    mv = self.cur_output[f'pred_layer_{layer_id}_{in_name}_{he_id}'][0]
                else:
                    ratio = 2 ** (layer_id)
                    mv = F.interpolate(self.cur_output[f'pred_layer_{layer_id}_{in_name}_{he_id}'][:1], scale_factor=ratio)[0]
                return mv

            for he_id in range(num_he):
                if self.model.get_net().enable_lmv_res:
                    for i in range(num_dec):
                        if he_id == 0:
                            if self.model.get_net().enable_lmv_res:
                                self.add_render_buffer(f'l{i}_lmv_res_{he_id}', buffer_type="motion_vector_64",
                                                       buffer=get_pyramid_buffer(i, he_id, "lmv_res"), debug=True)
                            if self.model.get_net().enable_st_lmv_res:
                                self.add_render_buffer(f'l{i}_st_lmv_res_{he_id}', buffer_type="motion_vector_64",
                                                       buffer=get_pyramid_buffer(i, he_id, "st_lmv_res"), debug=True)

                for i in range(num_dec):
                    if self.model.get_net().enable_feature_warp:
                        self.add_render_buffer(f'l{i}_tmv_{he_id}', buffer_type="motion_vector_64",
                                               buffer=get_pyramid_buffer(i, he_id, "tmv"), debug=True)
                    if self.model.get_net().enable_st_feature_warp:
                        self.add_render_buffer(f'l{i}_st_tmv_{he_id}', buffer_type="motion_vector_64",
                                               buffer=get_pyramid_buffer(i, he_id, "st_tmv"), debug=True)

            if albedo is not None:
                self.add_render_buffer("dmdl_color", debug=True)
            self.add_render_buffer("pred_comp_color_before_sky", debug=True)
            self.add_render_buffer("pred_comp_color_before_sky_st", debug=True)
            self.add_render_buffer("pred_comp_color_sky", debug=True)
            self.add_render_buffer("pred_mflow", debug=True)
            self.add_render_buffer("pred_scene_light_no_st", debug=True)
            self.add_render_buffer("scene_light_no_st", debug=True)
            self.add_render_buffer("pred_scene_color_no_st", debug=True)
            self.add_render_buffer("scene_color_no_st", debug=True)
            self.add_render_buffer("diff_no_st", debug=True)
            self.add_render_buffer("dmdl_color", debug=True)
            self.add_render_buffer("pred_st_color", debug=True)
            self.add_render_buffer("st_color", debug=True)
            self.add_render_buffer("diff_st", debug=True)
            self.add_render_buffer("pred_st_alpha", debug=True)
            self.add_render_buffer("st_alpha", debug=True)
            self.add_render_buffer("sky_color", buffer_type='scene_color', debug=True)
            self.add_render_buffer("pred_sky_color", buffer_type='scene_color', debug=True)
            self.add_render_buffer("skybox_mask", debug=True)
            self.add_render_buffer("pred_warped_scene_color_no_st", debug=True)
            self.add_render_buffer("pred_warped_st_color", debug=True)
            self.add_render_buffer("pred_warped_st_alpha", debug=True)

    def flip_data(self, data):
        def get_flip_argument():
            vertical = False
            horizontal = False
            if torch.rand(1).item() > 0.5:
                vertical = True
            if torch.rand(1).item() > 0.5:
                horizontal = True
            return vertical, horizontal

        def flip_(data, batch_size, flip_datas):
            assert len(data.get('future_data_list', [])) == 0, f"flip_ is not implemented for future_data_list, {data}"
            for batch_id in range(batch_size):
                data = create_flip_data(
                    data, vertical=flip_datas[batch_id][0], horizontal=flip_datas[batch_id][1], use_batch=True, batch_mask=[batch_id])
                if 'history_data_list' in data.keys():
                    history_datas = data['history_data_list']
                    for he_id, he_data in enumerate(history_datas):
                        history_datas[he_id] = create_flip_data(
                            he_data, vertical=flip_datas[batch_id][0], horizontal=flip_datas[batch_id][1], use_batch=True, batch_mask=[batch_id])
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
            if ('scene_light' in name or 'scene_color' in name or 'sky_color' in name or 'st_color' in name) \
                and isinstance(data[name], torch.Tensor) and len(data[name].shape) == 4:
                if DatasetGlobalConfig.max_luminance > 0:
                    data[name].clamp_max_(DatasetGlobalConfig.max_luminance)

    def load_data(self, data, mode: ForwardMode = ForwardMode.test) -> None:
        mode = self._normalize_mode(mode)
        self.cur_data['cur_data_index'] = self.cur_data_index
        if self.use_cuda:
            self.cur_data = data_to_device(data, self.config.runtime.device, non_blocking=True)
        else:
            self.cur_data = data
        if mode == ForwardMode.train and self.config.dataset.flip and self.config.dataset.is_block_part:
            assert 'vertical_flip' not in self.cur_data['metadata'].keys()  # type: ignore
            self.cur_data = self.flip_data(self.cur_data)
        self.set_recurrent_data(mode=mode)
        if mode == ForwardMode.train:
            self.cur_data = data_as_type(self.cur_data, self.dataset_train_precision_mode)
        elif mode == ForwardMode.test:
            self.cur_data = data_as_type(self.cur_data, self.dataset_test_precision_mode)
        if not self.config.dataset.augment_loader:
            self.cur_data = self.model.get_augment_data(self.cur_data)
        self.apply_max_luminance(self.cur_data)
        self.cur_data['cur_data_index'] = self.cur_data_index

    def update_one_batch(self, epoch_index=None, batch_index=None, mode: ForwardMode = ForwardMode.train):
        mode = self._normalize_mode(mode)
        self.set_recurrent_feature(mode=mode)
        self.update_forward(epoch_index=epoch_index, batch_index=batch_index, mode=mode)
        self.gather_execute_result(mode, enable_loss=True)
        self.update_backward(epoch_index=epoch_index, batch_index=batch_index, mode=mode)
        self.cache_one_batch_output(mode)

    def get_block_size(self, mode: ForwardMode) -> int:
        mode = self._normalize_mode(mode)
        mode_name = mode.name
        block_cfg = self.config['trainer'][f'recurrent_{mode_name}']['block_size']
        for stage in block_cfg:
            cur_epoch_index = self.epoch_index
            total_epoch = self.total_epoch
            start_epoch = stage['start']
            end_epoch = stage['end']
            if stage.get('ratio', False):
                start_epoch *= total_epoch
                end_epoch *= total_epoch
            if cur_epoch_index >= start_epoch and cur_epoch_index < end_epoch:
                if self.cur_data_index == 0 and cur_epoch_index - 1 < start_epoch:
                    self.min_loss = 1e9
                return stage['value']
        raise RuntimeError(f"block_size stage not found for {mode_name}")

    def __recurrent_gbuffer_layer_d2e(self, he_id):
        num_dec = int(self.net.num_shade_decoder_layer)  # type: ignore
        for layer_id in range(1, num_dec):
            self.cur_data[f"history_{he_id}_ge_sc_layers_{layer_id}"] = self.last_output[-(
                he_id) - 1][f'ge_sc_layers_{layer_id}'].detach()

    def __recurrent_history_layer_d2e(self, he_id, pf=""):
        num_dec = int(self.net.num_shade_decoder_layer)  # type: ignore
        for layer_id in range(1, num_dec):
            self.cur_data[f'history_{he_id}_{pf}d2e_sc_layers_{layer_id}'] = self.last_output[-(
                he_id) - 1][f'{pf}d2e_sc_layers_{layer_id}'].detach()

    def __recurrent_layer_d2e(self, he_id):
        self.cur_data['recurrent_d2e_he_id'] = he_id
        self.__recurrent_gbuffer_layer_d2e(he_id)
        self.__recurrent_history_layer_d2e(he_id)

    def __recurrent_one_batch_data(self, he_id):
        history_datas = self.cur_data['history_data_list']
        pred_buffers = ['scene_color_no_st']
        if self.net.enable_st:
            pred_buffers += ['st_color', 'st_alpha', 'sky_color']
        for buffer_name in pred_buffers:
            history_datas[he_id][f'{buffer_name}'] = self.last_output[-1 - he_id][f'pred_{buffer_name}'].detach()  # type: ignore
        self.cur_data[f'recurrent_pred_{he_id}'] = True
        he_pf = self.net.he_pfs[he_id]  # type: ignore
        assert not f"{he_pf}sc_layers" in self.cur_data.keys()

    def set_recurrent_data(self, mode: ForwardMode = ForwardMode.train):
        mode = self._normalize_mode(mode)
        mode_name = mode.name
        num_he = self.config['model']['history_encoders']['num']
        full_rendered = True
        recurrent_pred = self.config['trainer'][f'recurrent_{mode_name}']
        his_recurrent_list = get_his_recurrent_list(cur_data_index=self.cur_data_index,
                                                    num_he=num_he,
                                                    block_size=self.get_block_size(mode))
        for he_id in range(num_he):
            he_pf = self.net.he_pfs[he_id]  # type: ignore
            self.cur_data[he_pf + "prob"] = 0  # type: ignore
            if self.cur_data_index <= he_id:
                continue
            if his_recurrent_list[he_id]:
                if recurrent_pred:
                    start_recurrent_epoch = int(self.end_epoch * self.config['trainer']['recurrent_train_start'])
                    if self.epoch_index == start_recurrent_epoch and self.batch_index == 0:
                        self.min_loss = 1e9
                    if not (mode == ForwardMode.train and self.epoch_index < start_recurrent_epoch):
                        self.__recurrent_one_batch_data(he_id)
                        self.cur_data[he_pf + "prob"] = 1  # type: ignore
                        full_rendered = False
            else:
                ...
        self.cur_data['rendered_prob'] = 1 if full_rendered else 0  # type: ignore
        self.cur_data['trainer_mode'] = mode_name

    def set_recurrent_feature(self, mode: ForwardMode = ForwardMode.train):
        mode = self._normalize_mode(mode)
        num_he = self.config['model']['history_encoders']['num']
        his_recurrent_list = get_his_recurrent_list(cur_data_index=self.cur_data_index,
                                                    num_he=num_he,
                                                    block_size=self.get_block_size(mode))

        if self.net.enable_recurrent_d2e:
            self.cur_data['recurrent_d2e_he_id'] = -1
        for he_id in range(num_he):
            if self.cur_data_index <= he_id:
                continue
            if his_recurrent_list[he_id]:
                if self.net.enable_recurrent_d2e and self.cur_data['recurrent_d2e_he_id'] == -1:
                    self.__recurrent_layer_d2e(he_id)
            else:
                ...

    def cache_one_batch_output(self, mode: ForwardMode, epoch_index=None, batch_index=None):
        mode = self._normalize_mode(mode)
        mode_name = mode.name
        recurrent_pred = self.config['trainer'][f'recurrent_{mode_name}']
        num_he = int(self.net.num_history_encoder)  # type: ignore
        num_dec = int(self.net.num_shade_decoder_layer)  # type: ignore
        if len(self.last_output) > num_he:
            del self.last_output[0]
        cache_output = {}
        if recurrent_pred:
            cache_output.update({'pred_' + key: self.cur_output['pred_' + key] for key in self.config['model']['pred_buffers']})
            cache_output['pred_scene_color_no_st'] = self.cur_output['pred_scene_color_no_st']
        for layer_id in range(1, num_dec):
            if self.net.enable_recurrent_d2e:
                cache_output[f'd2e_sc_layers_{layer_id}'] = self.cur_output[f'd2e_sc_layers_{layer_id}']
                cache_output[f'ge_sc_layers_{layer_id}'] = self.cur_output[f'ge_sc_layers_{layer_id}']

        cache_output['metadata'] = copy.deepcopy(self.cur_data['metadata'])  # type: ignore
        self.last_output.append(cache_output)

    def calc_frame_losses(self, mode: ForwardMode = ForwardMode.train):
        mode = self._normalize_mode(mode)
        self.calc_loss_func(mode)
        return self.cur_loss

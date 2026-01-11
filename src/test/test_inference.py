
import includes.importer
import argparse
import math
from typing import List
import torch
import numpy as np
import os
import sys
import copy
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from utils.flip_loss import compute_ldrflip
from torch.amp.autocast_mode import autocast as autocast
from utils.config_enhancer import enhance_buffer_config, enhance_train_config, update_config
from utils.flow_vis import mv_to_image
from utils.buffer_utils import aces_tonemapper, gamma, inv_gamma, inv_log_tonemapper
from utils.buffer_utils import to_numpy, to_torch
from utils.flow_vis import flow_to_image
from utils.buffer_utils import flow_to_motion_vector
from utils.parser_utils import create_py_parser
from models.loss.loss import LossFunction, ssim_per_pixel
from dataloaders.dataset_base import MetaData
from dataloaders.raw_data_importer import UE4RawDataLoader, get_augmented_buffer
from utils.dataset_utils import create_de_color
from utils.utils import del_dict_item
from utils.buffer_utils import motion_vector_to_flow
from utils.dataset_utils import get_input_filter_list
from utils.utils import create_dir, remove_all_in_dir
from utils.buffer_utils import log_tonemapper
from utils.dataset_utils import resize
from utils.dataset_utils import create_warped_buffer
from trainers.mfrrnet_trainer import MFRRNetTrainer
import torch.distributed as dist
from utils.warp import warp
from utils.utils import write_text_to_file
from models.mfrrnet.mfrrnet import MFRRNetModel
from utils.buffer_utils import align_channel_buffer
from utils.buffer_utils import write_buffer
from utils.utils import Accumulator
from utils.utils import del_data
from dataloaders.patch_loader import PatchLoader
from config.config_utils import parse_config
from utils.str_utils import dict_to_string
from utils.log import log
from utils.buffer_utils import log_tonemapper
from utils.warp import warp
import cv2
import numpy as np
from matplotlib import pyplot as plt
num_he = 2
ldr_mode = False
index = 0
global_start = 0
global_end = 0


def to_vis(in_mv, flow_type="mv"):
    in_mv = torch.nn.functional.sigmoid((torch.abs(in_mv) ** 0.5)*torch.sign(in_mv))
    if flow_type == "mv":
        return mv_to_image(in_mv-0.5).float() / 255.0
    elif flow_type == "flow":
        return flow_to_image(in_mv-0.5).float() / 255.0
    else:
        assert False


def convert_cmap(data, color_map='inferno'):
    colormap = plt.get_cmap(color_map)
    heatmap = colormap(data.cpu())
    color_cmap = to_torch(heatmap[0, :, :, :3]).type(torch.float32)
    return color_cmap


def inference():
    global index
    global cur_idx
    global mode
    global writer
    global ldr_mode
    log.debug(f"{'='*20} start inference {'='*20}")
    with tqdm(dataset_trainer.test_loader) as epoch_loop:
        for dataset_trainer.batch_index, part in enumerate(epoch_loop):
            if mode not in ['msn', 'moflow', 'lmv', 'dfasr', 'fusesr', 'extranet', 'rrc']:
                part = [part]
            for data_part in part:
                for ind, tnr in enumerate(trainers):
                    # suffix = "png"
                    suffix = "png"
                    ''' png should add gamma when being exported '''
                    is_gamma = (suffix == 'png')

                    def easy_get(buffer_name, buffer=None, allow_skip=True):
                        if buffer is None and buffer_name in tnr.cur_output.keys():
                            return tnr.cur_output[buffer_name][0]
                        if buffer is None and buffer_name in tnr.cur_data.keys():
                            return tnr.cur_data[buffer_name][0]
                        if not allow_skip:
                            assert buffer is not None
                        return buffer

                    def easy_write(buffer_name, buffer=None, mv_vis=False, tonemap=False, suffix='png'):
                        buffer = easy_get(buffer_name, buffer)
                        if buffer is None:
                            return
                        if len(buffer.shape) == 4:
                            buffer = buffer[0]
                        assert int(mv_vis) + int(tonemap) <= 1
                        if mv_vis:
                            buffer = to_vis(buffer)
                        elif tonemap:
                            buffer = aces_tonemapper(buffer)

                        write_buffer(tnr.config['write_path'] +
                                     # f"{buffer_name}/{buffer_name}.{suffix}", buffer, mkdir=True, is_gamma=suffix == 'png')
                                     f"{buffer_name}/{buffer_name}_{str(out_idx).zfill(4)}.{suffix}", buffer, mkdir=True, is_gamma=suffix == 'png')
                    # log.debug(dict_to_string(data_part))
                    tnr.cur_data_index = index
                    data = copy.deepcopy(data_part)
                    # log.debug(dict_to_string(data))
                    out_idx = data['metadata']['index'][0].item()
                    epoch_loop.set_description_str(
                        f"[inference: {inference_name}_{tnr.config['model']['model_name']}_{tnr.config['vars']['block_size']}]")
                    # log.debug(dict_to_string(data))
                    tnr.load_data(data)
                    # log.debug(dict_to_string(tnr.cur_data, mmm=True))
                    if mode == "moflow":
                        tnr.set_recurrent_feature(mode="test")
                    tnr.update_forward(mode='test')
                    tnr.cache_one_batch_output(mode='test')

                    tmp_metric = {}
                    data_skip = False
                    gt = tnr.cur_output['gt'][0]
                    # log.debug(dict_to_string(tnr.cur_data, mmm=True))
                    # log.debug(dict_to_string(tnr.cur_output, mmm=True))

                    if mode in ['moflow'] and tnr.config['vars']['skip_pred']:
                        if tnr.config['vars']['block_size'] > 1 and index > num_he and index % tnr.config['vars']['block_size'] == 0:
                            pred = tnr.cur_data['scene_color'][0]
                            data_skip = True
                            # log.debug(f"data skip at index {index}")
                            for item in metric:
                                tmp_metric[item] = tnr.config['vars'][f"{item}_acc"].last_data
                        else:
                            pred = tnr.cur_output['pred'][0]
                    else:
                        pred = tnr.cur_output['pred'][0]

                    if not ldr_mode:
                        pred = aces_tonemapper(pred)
                        gt = aces_tonemapper(gt)
                    block_size = tnr.config['vars']['block_size']
                    if not data_skip:
                        ''' deal with special case of infinite block_size (0) '''
                        for item in metric:
                            ''' convert to float to avoid inaccuray in float16 '''
                            tmp_metric[item] = float(LossFunction.single_ops[item]([pred.float(), gt.float()]).mean().item())
                            tnr.config['vars'][f"{item}_acc"].add(tmp_metric[item])
                            if block_size > 1:
                                tnr.config['vars'][f"{item}_{index%block_size}_acc"].add(tmp_metric[item])
                        if mode == 'rrc':
                            tmp_metric['render_mask_ratio'] = float(tnr.cur_output['render_mask'].mean())
                            tnr.config['vars']['render_mask_ratio'].add(tmp_metric['render_mask_ratio'])
                    ''' write step '''
                    step_dict = {}
                    step_dict['metadata'] = f"{tnr.cur_data['metadata']['scene_name'][0]}_{tnr.cur_data['metadata']['index'][0]}"
                    step_dict['metric'] = {}
                    for item in metric:
                        tnr.config['vars']['writer'].add_scalar(f"step_{item}", tmp_metric[item], global_step=index)
                        step_dict['metric'][item] = tmp_metric[item] if not data_skip else -1

                    step_dict['step'] = index
                    step_dict['model_name'] = tnr.config['model']['model_name']
                    write_text_to_file(tnr.config['vars']['step_log_file'], str(step_dict) + '\n', "a")

                    ''' write epoch '''
                    write_text_to_file(tnr.config['vars']['epoch_log_file'], "", "w")
                    epoch_dict = {}
                    for i in range(block_size+1):
                        if i == block_size:
                            block_info = ""
                        else:
                            block_info = f'_{i}'
                        epoch_dict['model_name'] = tnr.config['model']['model_name']
                        epoch_dict['num_steps'] = index
                        epoch_dict['num_preds'] = tnr.config['vars'][f"psnr{block_info}_acc"].cnt
                        epoch_dict['metric'] = {}
                        for item in metric:
                            # epoch_dict['metric'][item] = tnr.config['vars'][f"{item}_acc"].get()
                            epoch_dict['metric'][item+block_info] = tnr.config['vars'][f"{item}{block_info}_acc"].get()
                        write_text_to_file(tnr.config['vars']['epoch_log_file'], str(epoch_dict) + "\n", "a+")
                    log.debug(dict_to_string(epoch_dict))
                    log.debug(f"data_skip: {data_skip}")
                    # log.debug(dict_to_string([list(data.keys()), data]))
                    log.debug([export_image, out_idx, global_start, global_end])
                    if not export_video and not export_image:
                        continue
                    elif export_image and (out_idx < global_start or out_idx > global_end):
                        continue
                    # suffix = "exr"
                    # log.debug(float(LossFunction.single_ops["psnr"]([pred, gt]).mean().item()))
                    error_4x = 4 * torch.abs(pred - gt)
                    with autocast(device_type="cuda", dtype=tnr.test_precision_mode, enabled=tnr.enable_amp):
                        error_flip = compute_ldrflip(pred.unsqueeze(0), gt.unsqueeze(0))[0]

                    flip_map_torch = convert_cmap(error_flip * 4)
                    ssim_map = ssim_per_pixel(pred, gt)
                    # log.debug(dict_to_string([error_flip, flip_map_torch], mmm=True))
                    # write_buffer(
                    #     tnr.config['write_path']+f"error_flip/error_flip_{str(out_idx).zfill(4)}.{suffix}", flip_map_torch, mkdir=True, is_gamma=is_gamma)
                    # write_buffer(tnr.config['write_path']+f"error_ssim/error_ssim_{str(out_idx).zfill(4)}.{suffix}", ssim_map, mkdir=True, is_gamma=is_gamma)
                    # write_buffer(tnr.config['write_path']+f"error/error_{str(out_idx).zfill(4)}.{suffix}", error_4x/4, mkdir=True, is_gamma=is_gamma)

                    easy_write('pred', buffer=pred, suffix=suffix)
                    easy_write('gt', buffer=gt, suffix=suffix)
                    easy_write('error_flip', buffer=error_flip, suffix=suffix)

                    log.debug(tnr.config['write_path']+f"pred/pred_{str(out_idx).zfill(4)}.{suffix}")
                    # warped_input = aces_tonemapper(warp(tnr.cur_data['history_scene_color_0'], tnr.cur_data['merged_motion_vector_0'])[0])
                    # error_flip = compute_ldrflip(warped_input.unsqueeze(0), gt.unsqueeze(0))[0]
                    # torch.abs(warped_input - gt)
                    # flip_map_torch = convert_cmap(error_flip.cpu() * 4)
                    # write_buffer(tnr.config['write_path']+f"warped_input/warped_input_{str(out_idx).zfill(4)}.{suffix}", warped_input, mkdir=True, is_gamma=is_gamma)
                    # write_buffer(tnr.config['write_path']+f"error_warped_input/error_warped_input_{str(out_idx).zfill(4)}.{suffix}", torch.abs(warped_input - gt), mkdir=True, is_gamma=is_gamma)
                    # write_buffer(tnr.config['write_path']+f"error_flip_warped_input/error_flip_warped_input_{str(out_idx).zfill(4)}.{suffix}", flip_map_torch, mkdir=True, is_gamma=is_gamma)
                    if mode in ['moflow']:
                        mv = tnr.cur_output['pred_layer_0_tmv_0'][0]
                    else:
                        assert False, mode
                    # write_buffer(tnr.config['write_path']+f"mv_raw/mv_raw_{str(out_idx).zfill(4)}.exr",
                    #              align_channel_buffer(mv, channel_num=3, mode="value", value=0.0), mkdir=True)
                    easy_write(mv, mv, mv_vis=True, suffix=suffix)
                    # log.debug(dict_to_string(tnr.cur_data))
                    # log.debug(dict_to_string(tnr.cur_output))
                    # easy_write('mv_raw', mv, mv_vis=False, suffix='exr')
                    # easy_write('mv', mv, mv_vis=True, suffix='png')
                    # if mode == 'msn':
                    #     rmv = tnr.cur_data['merged_motion_vector_0'][0]
                    # write_buffer(tnr.config['write_path']+f"rmv_raw/rmv_raw_{str(out_idx).zfill(4)}.exr",
                    #             align_channel_buffer(rmv, channel_num=3, mode="value", value=0.0), mkdir=True)
                    # write_buffer(tnr.config['write_path']+f"rmv/rmv_{str(out_idx).zfill(4)}.{suffix}",
                    #              align_channel_buffer(to_vis(rmv)), mkdir=True)
            index += 1


def update_inference_config(config):
    config['_input_config'] = copy.deepcopy(config)
    update_config(config)
    config['local_rank'] = 0
    config['use_ddp'] = False
    config["use_cuda"] = config['num_gpu'] > 0
    config['device'] = "cuda:0" if config["use_cuda"] else "cpu"


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description="trainer")
    parser.add_argument("--mode", default="", help="inference mode")
    parser.add_argument("--block", type=int, default=1, help="block_size")
    parser.add_argument("--same-block", default=False, action='store_true', help="block_size")
    parser.add_argument("--video", default=False, action='store_true', help="block_size")
    parser.add_argument("--image", default=False, action='store_true', help="block_size")
    parser.add_argument("--st", default=False, action='store_true', help="block_size")
    parser.add_argument("--scene", default="", help="block_size")
    parser.add_argument("--config", default="", help="inference config path")
    args = parser.parse_args()
    mode = args.mode
    input_mode = args.mode
    assert num_he == 2
    multiple_test = args.block > 2 or args.block == 0
    block_size = args.block
    same_block = args.same_block
    export_image = args.image
    export_video = args.video
    cmd_scene = args.scene

    assert mode in ['moflow'], "--mode must be moflow"
    assert cmd_scene, "--scene must be specified in cmd"

    metric = ['psnr', 'ssim', 'lpips']
    if mode == "moflow":
        dataset_cfg = parse_config("config/dataset/infer_dataset_v6_moflow_ess.yaml")
        if args.config:
            config_path = [args.config]
        else:
            config_path = [
                "config/inference/DT_moflow.yaml",
                # "config/inference/FC_moflow.yaml",
                # "config/inference/multi_3_DT_moflow.yaml",
            ]
    else:
        raise NotImplementedError(f"{mode} is not supported (list: moflow)")

    scene_name = cmd_scene.split("_")[0] + "_T"
    path_alias = scene_name
    inference_name = cmd_scene
    frame_idx = 32

    global_start = frame_idx - 8
    global_end = frame_idx + 8
    dataset_cfg['dataset']['train_scene'] = [
        {"name": f"{scene_name}/{inference_name}_720", "config": {"indice": [], "path_alias": path_alias}},
    ]
    if export_image:
        dataset_cfg['dataset']['test_scene'] = [
            {"name": f"{scene_name}/{inference_name}_720",
                "config": {"indice": [global_start-1, global_end+1], "path_alias": path_alias}},
        ]
    else:
        dataset_cfg['dataset']['test_scene'] = [
            {"name": f"{scene_name}/{inference_name}_720", "config": {"indice": [], "path_alias": path_alias}},
        ]
    log.debug(dict_to_string(dataset_cfg['dataset']['test_scene']))
    update_inference_config(dataset_cfg)
    enhance_train_config(dataset_cfg)
    if export_image:
        inference_name = mode + "_" + inference_name + f"_figure_{frame_idx}"
    else:
        inference_name = mode + "_" + inference_name
    # inference_name = mode + "_" + inference_name  + "_temporal"
    if multiple_test:
        inference_name = "multiple_" + inference_name
    # dataset_cfg['dataset']['train_scene'] = [
    #     {"name":"FC/FC_04", "config":{"indice":[]}},
    # ]
    dataset_cfg['log_to_file'] = False
    dataset_trainer = eval(dataset_cfg['trainer']['class'])(
        dataset_cfg, None, resume=False)
    dataset_trainer.prepare('test')
    dataset_trainer.create_test_dataset(0)
    dataset_trainer.create_test_loader()
    from torch.utils.tensorboard.writer import SummaryWriter
    write_path = "../output/inference/"
    log.warn(f"no --video or --image added. no images will be written to {write_path}")

    configs = []
    models = []
    trainers = []
    render_targets = []
    num_cfg = len(config_path)
    if multiple_test:
        if same_block:
            block_sizes = [block_size for _ in range(num_cfg)]
        else:
            block_sizes = [block_size + _ for _ in range(num_cfg)]
    else:
        block_sizes = [block_size for _ in range(num_cfg)]
    for i in range(len(config_path)):
        tmp_config = parse_config(config_path[i], root_path="")
        # log.debug(dict_to_string(tmp_config['model']['input_buffer']))
        update_inference_config(tmp_config)
        enhance_train_config(tmp_config)
        configs.append(tmp_config)
        config_train = copy.deepcopy(tmp_config)
        config_train['model']['input_buffer'] = dataset_cfg['model']['input_buffer']
        config_train['dataset']['enable'] = False
        config_train['initial_inference'] = False
        # log.debug(dict_to_string(dataset_cfg['model']['input_buffer']))
        tmp_model = eval(config_train['model']['class'])(config_train)

        def print_model_weights_dtype(model):
            for name, param in model.named_parameters():
                print(f"Parameter: {name}, Data type: {param.dtype}")

        # 假设 tmp_model 是一个 nn.Module 对象
        models.append(tmp_model)
        # resume = True
        # if mode == "interp" or mode == "extrap":
        resume = False
        config_train['log_to_file'] = False
        tmp_trainer = eval(config_train['trainer']['class'])(
            config_train, tmp_model, resume=resume)
        tmp_trainer.prepare("test")
        tmp_trainer.config['vars'] = {}
        tmp_trainer.config['vars']['skip_pred'] = False
        tmp_trainer.config['vars']['skip_pred'] = block_sizes[i] > 1
        tmp_trainer.config['vars']['block_size'] = block_sizes[i]
        tmp_trainer.config['trainer']['recurrent_test'] = {
            "block_size": [
                {'start': 0, 'end': 1, 'value': block_sizes[i], 'ratio': True},
            ]
        }
        for item in metric:
            tmp_trainer.config['vars'][f"{item}_acc"] = Accumulator()
            for j in range(block_sizes[i]):
                tmp_trainer.config['vars'][f"{item}_{j}_acc"] = Accumulator()
        tmp_trainer.config['write_path'] = write_path + \
            f"{inference_name}/{tmp_trainer.config['model']['model_name']}_{str(block_sizes[i])}/"
        tmp_trainer.config['vars']['writer'] = SummaryWriter(log_dir=tmp_trainer.config['write_path'])
        tmp_trainer.config['vars']['step_log_file'] = f"{tmp_trainer.config['write_path']}/step.log"
        tmp_trainer.config['vars']['epoch_log_file'] = f"{tmp_trainer.config['write_path']}/epoch.log"

        tmp_trainer.last_output = []
        create_dir(tmp_trainer.config['write_path'])
        remove_all_in_dir(tmp_trainer.config['write_path'])
        trainers.append(tmp_trainer)
        render_targets.append(None)

    require_list = get_input_filter_list({
        'input_config': dataset_cfg,
        'input_buffer': dataset_cfg['model']['require_data']
    })
    loader = PatchLoader(
        dataset_cfg['dataset']['path'],
        job_config={'export_path': dataset_cfg['dataset']['path'], 'dataset_format': 'npz'},
        buffer_config=dataset_cfg['buffer_config'],
        require_list=require_list)

    with torch.no_grad():
        inference()

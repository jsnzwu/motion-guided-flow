import copy
from enum import Enum, auto
from glob import glob
import json
import math
import os
from select import select
import threading
import time
import re
import multiprocessing as mp
from typing import List
from tqdm import tqdm
import torch
from utils.model_utils import get_2d_dim
from utils.dataset_utils_ess import create_dmdl_color_ess
# from utils.dataset_utils_ess import create_dmdl_color_ess
from utils.utils import is_item_all_in_another, create_dir, del_dict_item, get_file_component, remove_all_in_dir, write_text_to_file
from wickit.utils.metadata_task_utils import dispatch_task_by_metadata
from utils.dataset_utils import create_dmdl_color_brdf, create_de_color, create_future_frame, \
    create_history_frame, create_history_warped_buffer, create_scene_color, \
    create_scene_color_no_st, create_sky_color, create_skybox_mask, create_st_color, \
    create_warped_buffer, get_continuity_mask, write_npz

from wickit.utils.log import log
from wickit.utils.basic.string import dict_to_string
from utils.buffer_utils import aces_tonemapper, buffer_data_to_vis, buffer_raw_to_data, fix_dmdl_color_zero_value
from wickit.utils.basic.tensor import tensor_as_type_str
from wickit.utils.io.imageio import read_image
from wickit.utils.ext.warp import get_merged_motion_vector_from_last, warp
from wickit.dataloaders.metadata import MetaData
from utils.parser_utils import parse_buffer_name, parse_find_dict, parse_flat_dict


def get_augmented_buffer(augmented_output, buffer_config, data, last_data=[], next_data=[], allow_skip=False, with_batch=False, history_data_check=True) -> None:
    if augmented_output is None:
        return
    # log.debug(augmented_output)
    # log.debug(buffer_config)
    # test_base_color = False
    demodulation_mode = buffer_config['demodulation_mode']
    gt_name = buffer_config.get('gt_name', 'scene_color')
    max_luminance = buffer_config.get('max_luminance', -1)
    dmdl_color_use_skybox_mask = buffer_config.get('dmdl_color_use_skybox_mask', True)
    # max_luminance = -1
    augmented_data_recipe = buffer_config['augmented_data_recipe']

    if max_luminance > 0:
        clamp_list = ['scene_color_no_st']
        for k in clamp_list:
            if k in data.keys():
                data[k].clamp_(max=max_luminance)

    for key_augmented in augmented_output:
        ret = parse_buffer_name(key_augmented)
        pref = ret['sample']
        buffer_name = ret['buffer_name']
        frame_id = ret['frame_id']
        postf = ret['postf']
        # if ret['method'] == '_extranet':
        # log.debug(ret['buffer_name'])
        p_buffer_name = pref + buffer_name
        p_buffer_name_p = pref + buffer_name + postf
        # log.debug(buffer_name)
        # log.debug(f'{key_augmented}: pref: "{pref}", buffer_name: "{buffer_name}", his_id: "{his_id}", postf: "{postf}"')
        # log.debug(dict_to_string(data, key_augmented))
        # if not(test_base_color) and 'base_color' in data.keys():
        #     data['base_color'] = (data['base_color'] + 1.0/(math.exp(5.0)-1)) / (1+1.0/(math.exp(5.0)-1))
        #     test_base_color = True
        if key_augmented in data.keys():
            continue

        # log.debug([buffer_name, ret])
        # log.debug(f'start check: "{allow_skip}"')

        if p_buffer_name not in augmented_data_recipe.keys():
            raise Exception(f'{p_buffer_name} can\'t be found in augmented_data_recipe ({list(augmented_data_recipe.keys())})')
        num_history = augmented_data_recipe[pref + buffer_name].get('num_history', 0)

        if p_buffer_name not in augmented_data_recipe.keys():
            raise Exception("creating {}, found \"{}\" is not in recipe.keys:{}".format(
                key_augmented, p_buffer_name, list(augmented_data_recipe.keys())))
        flag, invalid_item = is_item_all_in_another(
            augmented_data_recipe[p_buffer_name]['dep'], data.keys())

        # log.debug(dict_to_string([augmented_data_recipe, augmented_data_recipe[p_buffer_name]]))
        err_msg = ""
        if not flag:
            err_msg = "creating {}, found {} (all_dep:{}) isn\'t in data.keys: {}".format(
                key_augmented, invalid_item, augmented_data_recipe[p_buffer_name]['dep'], data.keys())

        if history_data_check:
            if (frame_id is not None and frame_id > max(len(last_data), num_history)) or (frame_id is None and flag and len(last_data) < num_history):
                flag = False
                err_msg = "creating {}, the length of last_data is {}, but frame_id is {} and requirement is {}".format(
                    key_augmented, len(last_data), frame_id, num_history)

        for i in range(num_history):
            if not history_data_check:
                break
            if flag:
                # log.debug("{} {}".format(augmented_data_recipe[buffer_name + pf]['dep_history'][i], last_data[i].keys()))
                if buffer_name.startswith("history_"):
                    frame_str = f"_{frame_id}" if frame_id is not None else ""
                    key_augmented = f'{pref}{buffer_name}{frame_str}'
                    if (frame_id is not None and i == frame_id) or frame_id is None:
                        flag, invalid_item = is_item_all_in_another(
                            augmented_data_recipe[p_buffer_name]['dep_history'][i], last_data[i].keys())
                else:
                    flag, invalid_item = is_item_all_in_another(
                        augmented_data_recipe[p_buffer_name]['dep_history'][i], last_data[i].keys())
                if not flag:
                    err_msg = "creating \"{}\" as \"{}\", found {} isn\'t in last_data[{}].keys: {}".format(
                        key_augmented, p_buffer_name, invalid_item, i, last_data[i].keys())
            else:
                break

        if not flag:
            if not allow_skip:
                raise Exception(
                    f"\n{err_msg}\naugmented_output:\n{augmented_output}\ndata_dict:" +
                    f"\n{dict_to_string(data)}\nrecipe:\n{dict_to_string(augmented_data_recipe[p_buffer_name])}")
            else:
                # log.warn("skip {}. err_msg: {}".format(key_augmented, err_msg))
                continue

        augmented_data = None

        # log.debug("{} {} {}".format(key_augmented, buffer_name, augmented_data_recipe.keys()))

        # if buffer_name == 'camera_position':
        #     augmented_data = transform_position_image(
        #         data[pref + 'world_position' + postf], data['camera__view_matrix' + postf])

        if buffer_name == "black3":
            augmented_data = torch.zeros_like(data['base_color' + postf])
        elif buffer_name == "white1":
            augmented_data = torch.ones_like(data['base_color' + postf][:1])
        elif buffer_name == "stencil":
            augmented_data = torch.zeros_like(data['base_color' + postf][:1])

        # elif buffer_name == 'camera_normal':
        #     augmented_data = transform_direction_image(
        #         data[pref + 'world_normal' + postf], data['camera__view_matrix' + postf])
# only single eye
        # elif buffer_name == "history_masked_warped_scene_light_0_extranet":
        #     augmented_data = data['history_warped_scene_light_0' + postf] * data['history_occlusion_mask_0_extranet' + postf]
        elif buffer_name.startswith("history_warped_"):
            # log.debug(buffer_name)
            name = buffer_name.replace("history_warped_", "")
            if frame_id is not None:
                # log.debug(dict_to_string(last_data))
                # log.debug(dict_to_string(data))
                data[f"{pref}{buffer_name}_{frame_id}{postf}"] = create_warped_buffer(
                    last_data[frame_id][pref + name + postf], data[f'{pref}merged_motion_vector_{frame_id}{postf}'],
                    mode="bilinear", padding_mode="border", with_batch=with_batch)
            else:
                for i in range(0, num_history, 1):
                    for i in buffer_config['index']:
                        if i < len(last_data) - 1:
                            continue
                        data[f"{pref}{buffer_name}_{i}{postf}"] = create_warped_buffer(
                            last_data[i][pref + name + postf], data[f'{pref}merged_motion_vector_{i}{postf}'],
                            mode="bilinear", padding_mode="border")

        elif buffer_name.startswith("history_"):
            history_name = buffer_name
            buffer_name = buffer_name.replace("history_", "")
            if frame_id is not None:
                data[f"{pref}{history_name}_{frame_id}{postf}"] = create_history_frame(
                    last_data, f'{pref}{buffer_name}{postf}', index=frame_id)
                # log.debug(dict_to_string({
                #     f"{history_name}_{his_id}{postf}": data[f"{history_name}_{his_id}{postf}"]
                # }))
            else:
                for i in buffer_config['index']:
                    if i > len(last_data) - 1:
                        continue
                    data[f"{pref}{history_name}_{i}{postf}"] = create_history_frame(
                        last_data, f'{pref}{buffer_name}{[postf]}', index=i)
                    # log.debug(dict_to_string({
                    #     f"{history_name}_{his_id}{postf}": data[f"{history_name}_{his_id}{postf}"]
                    # }))
        elif buffer_name.startswith("future_"):
            future_name = buffer_name
            buffer_name = buffer_name.replace("future_", "")
            # log.debug(history_name)
            if frame_id is not None:
                # log.debug(dict_to_string([next_data, frame_id]))
                data[f"{pref}{future_name}_{frame_id}{postf}"] = create_future_frame(
                    next_data, f'{pref}{buffer_name}{postf}', index=frame_id)
            else:
                assert False, 'must specify his_id of future_frame'

        elif buffer_name == "merged_motion_vector":
            # data[f'{pref}merged_motion_vector_0{postf}'] = data[f'{pref}motion_vector{postf}']
            # if len(last_data) > num_history:
            #     num_history = len(last_data)
            # for history_ind in range(1, num_history + 1):
            #     mvs = [data[f'{pref}merged_motion_vector_0{postf}']]
            #     for i in range(history_ind):
            #         mv = get_merged_motion_vector_from_last(
            #             last_data[i][f'{pref}motion_vector{postf}'], mvs[-1], residual=mvs[-1])
            #         mvs.append(mv)
            #     if len(data[f'{pref}motion_vector{postf}'].shape) == 4:
            #         data[f'{pref}merged_motion_vector_{history_ind}{postf}'] = mvs[-1]
            #     else:
            #         data[f'{pref}merged_motion_vector_{history_ind}{postf}'] = mvs[-1][0]

            mv = data[f'{pref}merged_motion_vector_0{postf}'] = data[f'{pref}motion_vector{postf}']
            num_history = len(last_data)
            for history_ind in range(1, num_history):
                mv = get_merged_motion_vector_from_last(
                    last_data[history_ind-1][f'{pref}motion_vector{postf}'], mv, residual=mv)
                if with_batch:
                    data[f'{pref}merged_motion_vector_{history_ind}{postf}'] = mv
                else:
                    data[f'{pref}merged_motion_vector_{history_ind}{postf}'] = mv[0]

        elif buffer_name.startswith("history_multi_warped_"):
            name = buffer_name.replace("history_multi_warped_", "")
            # if his_id is not None:
            #     augmented_data = None
            for i in range(num_history):
                data[f'{pref}{buffer_name}_{i}{postf}'] = create_history_warped_buffer(
                    data, last_data, i, name, prefix=pref, postfix=postf,
                    mode="bilinear", padding_mode="border")

        # elif buffer_name == 'cross_sample':
        #     samples = create_cross_sample(data)
        #     data[buffer_name + "_l"] = samples[0]
        #     data[buffer_name + "_r"] = samples[1]

        # elif buffer_name == "history_warped_scene_light_cross":
        #     # input: "world_to_clip_r"\"world_to_clip_l"
        #     imgs = create_history_warped_scene_color_cross(data, last_data)
        #     data[buffer_name + "_l"] = imgs[0]
        #     data[buffer_name + "_r"] = imgs[1]

        # elif buffer_name == "warped_scene_light_cross":
        #     # input: "world_to_clip_r"\"world_to_clip_l"
        #     imgs = create_history_warped_scene_color_cross(
        #         data, last_data, historical=False)
        #     data[buffer_name + "_l"] = imgs[0]
        #     data[buffer_name + "_r"] = imgs[1]
        elif buffer_name == "scene_color":
            augmented_data = create_scene_color(data[pref + 'scene_color_no_st' + postf],
                                                data[pref + 'st_color' + postf], data[pref + 'st_alpha' + postf])
            if max_luminance > 0.0:
                augmented_data.clamp_(max=max_luminance)

        elif buffer_name == "scene_color_no_st":
            augmented_data = create_scene_color_no_st(data[pref + 'scene_color' + postf],
                                                      data[pref + 'st_color' + postf], data[pref + 'st_alpha' + postf])
            if max_luminance > 0.0:
                augmented_data.clamp_(max=max_luminance)

        elif buffer_name == "st_color":
            # augmented_data = torch.zeros_like(data[pref + 'depth' + postf])
            augmented_data = create_st_color(data[pref + 'scene_color' + postf],
                                             data[pref + 'scene_color_no_st' + postf], data[pref + 'st_alpha' + postf])
            # if max_luminance > 0.0:
            #     augmented_data.clamp_(max=max_luminance)

        elif buffer_name == "st_alpha":
            augmented_data = torch.ones_like(data[pref + 'depth' + postf])

        elif buffer_name == 'brdf_color':
            augmented_data = create_dmdl_color_brdf(data[pref + 'roughness' + postf], data[pref + 'nov' + postf],
                                                    data[pref + 'base_color' + postf], data[pref +
                                                                                            'metallic' + postf], data[pref + 'specular' + postf],
                                                    skybox_mask=data[pref + 'skybox_mask' + postf])
        elif buffer_name == 'dmdl_color_brdf':
            skybox_mask = data[pref + 'skybox_mask' + postf] if dmdl_color_use_skybox_mask else None
            augmented_data = create_dmdl_color_brdf(data[pref + 'roughness' + postf],
                                                    data[pref + 'nov' + postf],
                                                    data[pref + 'base_color' + postf],
                                                    data[pref + 'metallic' + postf],
                                                    data[pref + 'specular' + postf],
                                                    skybox_mask=skybox_mask)
        elif buffer_name == 'dmdl_color_ess':
            skybox_mask = data[pref + 'skybox_mask' + postf] if dmdl_color_use_skybox_mask else None
            augmented_data = create_dmdl_color_ess(data[pref + 'base_color' + postf],
                                                   data[pref + 'specular' + postf],
                                                   data[pref + 'metallic' + postf],
                                                   skybox_mask=skybox_mask)
        elif buffer_name == 'dmdl_color':
            data['demodulation_mode'] = demodulation_mode
            if demodulation_mode == 'base':
                augmented_data = fix_dmdl_color_zero_value(data[pref + 'base_color'])
            elif demodulation_mode == 'brdf':
                key_brdf_color = key_augmented.replace("dmdl_color", "dmdl_color_brdf")
                # assert key_augmented.replace("dmdl_color", "dmdl_color_ess") not in data.keys()
                if key_brdf_color in data.keys():
                    augmented_data = data[key_brdf_color]
                else:
                    skybox_mask = data[pref + 'skybox_mask' + postf] if dmdl_color_use_skybox_mask else None
                    augmented_data = create_dmdl_color_brdf(data[pref + 'roughness' + postf],
                                                            data[pref + 'nov' + postf],
                                                            data[pref + 'base_color' + postf],
                                                            data[pref + 'metallic' + postf],
                                                            data[pref + 'specular' + postf],
                                                            skybox_mask=skybox_mask)
            elif demodulation_mode == 'ess' or demodulation_mode == 'extranet':
                key_brdf_color = key_augmented.replace("dmdl_color", "dmdl_color_ess")
                # assert key_augmented.replace("dmdl_color", "dmdl_color_brdf") not in data.keys()
                # log.debug('using ess demodulation')
                if key_brdf_color in data.keys():
                    augmented_data = data[key_brdf_color]
                else:
                    skybox_mask = data[pref + 'skybox_mask' + postf] if dmdl_color_use_skybox_mask else None
                    augmented_data = create_dmdl_color_ess(data[pref + 'base_color' + postf],
                                                           data[pref + 'specular' + postf],
                                                           data[pref + 'metallic' + postf],
                                                           skybox_mask=skybox_mask)
            else:
                raise Exception(
                    f'dmdl_color only supports "base", "brdf" and "ess", but buffer_config.demodulation_mode="{demodulation_mode}"')

            # log.debug(f'demodulation_mode: "{demodulation_mode}" enabled.')

        elif buffer_name == 'scene_light_no_st':
            sc = data[pref + buffer_name.replace('scene_light', 'scene_color') + postf]
            # augmented_data = create_de_color(sc, data[pref + 'dmdl_color' + postf],
            #                                  skybox_mask=data[pref + 'skybox_mask' + postf], sky_color=data[pref + 'sky_color' + postf], fix=True)
            augmented_data = create_de_color(sc, data[pref + 'dmdl_color' + postf], fix=True)
            if max_luminance > 0.0:
                augmented_data.clamp_(max=max_luminance)
            # log.debug(f'demodulation_mode: "{demodulation_mode}" enabled.')
        elif buffer_name == 'sky_color':
            if pref == 'aa_':
                data[pref + 'sky_color' + postf] = data['sky_color']
            else: 
                augmented_data = create_sky_color(
                    data[pref + 'scene_color_no_st' + postf], data[pref + 'skybox_mask' + postf])
                if max_luminance > 0.0:
                    augmented_data.clamp_(max=max_luminance)

        elif buffer_name == 'scene_light':
            sc = data[pref + buffer_name.replace('scene_light', 'scene_color') + postf]
            augmented_data = create_de_color(sc, data[pref + 'dmdl_color' + postf],
                                             skybox_mask=data[pref + 'skybox_mask' + postf], sky_color=data[pref + 'sky_color' + postf], fix=True)
            if max_luminance > 0.0:
                augmented_data.clamp_(max=max_luminance)
            # log.debug(f'demodulation_mode: "{demodulation_mode}" enabled.')

        elif buffer_name == 'skybox_mask':
            sky_depth = data.get(pref + 'sky_depth' + postf, None)
            augmented_data = create_skybox_mask(
                data[pref + 'depth' + postf], data[pref + 'base_color' + postf], sky_depth=sky_depth, with_batch=with_batch)

        elif buffer_name == 'continuity_mask':
            def get_contin_mask(data):
                warp_mode = 'bilinear'
                warp_padding_mode = 'border'
                assert f'history_{gt_name}_0' in data.keys(), f'history_{gt_name}_0'
                warped_gt = warp(data[f'history_{gt_name}_0'], data['merged_motion_vector_0'],
                                 mode=warp_mode, padding_mode=warp_padding_mode)
                assert gt_name in data.keys()
                gt = data[gt_name]
                ret = get_continuity_mask(aces_tonemapper(warped_gt), aces_tonemapper(gt))
                # log.debug(dict_to_string(ret))
                return ret
            data['debug_continuity_mask_from'] = gt_name
            augmented_data = get_contin_mask(data)

        # elif buffer_name == "shadow":
        #     augmented_data = data["scene_color_no_st" + pf] / \
        #         data["scene_color_no_shadow" + pf]
        #     augmented_data[data["scene_color_no_shadow"+pf]<=0] = 1.0
        #     augmented_data[torch.isnan(augmented_data)] = 0.0
        #     augmented_data[torch.isinf(augmented_data)] = 0.0

        # elif buffer_name == "shadow_y":
        #     augmented_data = create_y_color(data['shadow'+pf])

        # elif key_augmented == "occlusion_mask":
        #     data[key_augmented] = create_occlusion_mask(
        #         data, last_data[0])

        # elif buffer_name == 'discontinuity_mask':
        #     augmented_data = 1-get_continuity_mask(data['history_warped_gt_no_st_0'], data['gt_no_st'])
            # scene_color = data['scene_color' + pf]
            # warped = data['history_warped_scene_color_0' + pf]
            # if buffer_config['demodulate']:
            #     scene_color = gamma_log(scene_color)
            #     warped = gamma_log(warped)
            # augmented_data = create_discontinuity_mask(
            #     scene_color, warped,
            #     ratio=buffer_config['augmented_data_recipe']['discontinuity_mask'+pf]['config']['diff'])

        # elif buffer_name == "shadow_discontinuity_mask":
        #     warped_shadow_mask = create_warped_buffer(
        #         last_data[0]['shadow_mask' + postf], data['motion_vector' + postf])
        #     # warped_shadow_mask = last_data[0]['shadow_mask' + postfix]
        #     shadow_diff = torch.abs(
        #         data['shadow_mask' + postf] - warped_shadow_mask)
        #     augmented_data = create_shadow_discontinuity_mask(
        #         shadow_diff,
        #         ratio=buffer_config['augmented_data_recipe']['shadow_discontinuity_mask' + postf]['config']['ratio'])

        # elif key_augmented == "light__directional_light_camera_direction":
        #     data[key_augmented] = transform_direction(
        #         data['light__directional_light_world_direction'], data['camera__view_matrix'])
        else:
            raise NotImplementedError(
                "{} is not supported for augmented_buffer".format(key_augmented))

        if augmented_data is not None:
            data[key_augmented] = augmented_data


def compress_buffer(data: dict, data_type='fp16'):
    '''
    convert all values of a dict to specific data_type
    '''
    raw_data = copy.deepcopy(data)
    for k in raw_data.keys():
        if not (isinstance(raw_data[k], torch.Tensor)):
            continue
        raw_data[k] = tensor_as_type_str(raw_data[k], type_str=data_type)
    return raw_data


def merge_buffer(data: dict, buffer_names: List[str]) -> torch.Tensor:
    # NOTE: Retained for project-specific tooling and future data pipeline extensions.
    merged_tensor = torch.cat([data[name] for name in buffer_names], dim=0)
    return merged_tensor


def split_buffer(tensor: torch.Tensor, buffer_names: List[str]) -> dict:
    dims = [get_2d_dim(name) for name in buffer_names]
    res = torch.split(tensor, dims, dim=0)
    ret = {}
    for ind, name in enumerate(buffer_names):
        ret[name] = res[ind]
    return ret


def get_extend_buffer(data: dict, part_name: str, buffer_config: dict, last_datas=[], start_cutoff=5) -> dict:
    '''
    ### input:
    `data`: input buffer dict

    `part_name`: part to extend

    `buffer_config`: config for extend rules

    `last_data`: history input buffer dict

    `start_cutoff`: not perform because of lack of history data

    ### output:
    `0: extended data, like {'<buffer_name1>': torch.Tensor, '<buffer_name2>': torch.Tensor}
    '''
    metadata = MetaData(data['metadata']["scene_name"], data['metadata']['index'])
    # log.debug(dict_to_string(data, "original data", mmm=True))
    ret = {}
    if metadata.index < start_cutoff:
        log.debug("metadata: {} doesnt have previous data, skip.".format(
            metadata.__str__()))
        return ret

    part = buffer_config['part'][part_name]

    # log.debug(dict_to_string(data))

    get_augmented_buffer(part.get('augmented_data', []) + part['buffer_name'],
                         buffer_config,
                         data, last_data=last_datas,
                         allow_skip=False)

    ret = {}
    for buffer_name in part['buffer_name']:
        if buffer_name not in data.keys():
            raise KeyError(f'{buffer_name} not in data.keys(), keys:{list(data.keys())}')
        if isinstance(data[buffer_name], torch.Tensor):
            ret[buffer_name] = tensor_as_type_str(
                data[buffer_name], part['type'])
        else:
            ret[buffer_name] = data[buffer_name]
        if isinstance(ret[buffer_name], torch.Tensor) and\
                (torch.isinf(ret[buffer_name]).any() or torch.isnan(ret[buffer_name]).any()):
            log.debug(dict_to_string(data, mmm=True))
            log.debug(dict_to_string(ret, mmm=True))
            log.warn(f"{'!'*10} an nan or inf occur in motion vector in {metadata.__str__()} {'!'*10}")
        # log.debug(dict_to_string(ret, "after augmented in part data", mmm=True))
    return ret


def task_wrapper(ins, scene, start_index, end_index, file_index, idx):
    # NOTE: Retained for project-specific tooling and future data pipeline extensions.
    log.debug("start wrapper {} {} {} {} {} {}".format(
        ins, scene, start_index, end_index, file_index, idx))
    ins.export_patch_range(scene, start_index, end_index, file_index, idx)


class DatasetFormat(Enum):
    NPZ = auto()
    HDF5 = auto()

    @staticmethod
    def get_by_str(name: str):
        return {
            'npz': DatasetFormat.NPZ,
            'hdf5': DatasetFormat.HDF5,
        }[name]


class UE4RawDataLoader:
    job_config = {}
    buffer_config = {}
    flow_estimator = None

    def __init__(self, in_buffer_config, in_job_config):
        log.info("start UE4RawDataLoader init")
        # log.debug(dict_to_string(in_job_config, "\njob_config"))
        # log.debug(dict_to_string(in_buffer_config, "\nbuffer_config"))
        self.job_config = in_job_config
        self.buffer_config = in_buffer_config
        self.data = dict()
        self.metadata = dict()

        self.dataset_format = DatasetFormat.get_by_str(in_job_config['dataset_format'])

        for s in self.job_config['scene']:
            path = self.job_config['import_path'] + s
            # log.debug(path)
            # num = 2
            # num = start_offset + 72
            # num = self.check_files(path, unsafe=True) - 5
            num = self.check_files(path)
            self.data[s] = []
            self.metadata[s] = []
            for i in tqdm(range(num), ncols=64):
                #     log.debug("processing directories: {}".format(d))
                #     tmp_data = self.parse_buffer(d, i)
                #     tmp_data.update(self.parse_scene(d, i))
                #     self.get_augmented_buffer(tmp_data)
                #     self.data[s].append(tmp_data)
                self.metadata[s].append(MetaData(s, i))
                # log.debug(dict_to_string(self.data[d][-1]))
            log.info("{} processed. total frame: {}.".format(
                s, len(self.metadata[s])))

    def check_files(self, path, unsafe=False):
        num = -1
        for item in self.buffer_config['engine_buffer'].keys():
            file_name_list = glob(
                "{}/{}/*[0-9].*".format(path, str(item)))
            cur_num = len(file_name_list)
            log.debug(f"{item}: {cur_num}")
            if num != -1 and cur_num != num:
                log.error("found error when exporting the {}".format(path))
                if not unsafe:
                    raise FileNotFoundError(
                        "{} length({}) is not same as required({}). ".format(item, cur_num, num))
            num = cur_num
            # log.debug(f'{num} frame(s) from "{"{}/{}/*[0-9].*".format(path, str(item))}"')
        log.info("path:{}, pre-check passed, num:{}.".format(path, num))
        return num

    def parse_buffer(self, buffers, directory, ind):
        tmp_data = {}
        for buffer_name in buffers.keys():
            origin = buffers[buffer_name]['origin']
            if origin not in self.buffer_config['engine_buffer'].keys():
                raise KeyError(
                    "{} is not in engine_buffer. keys:{}".format(origin, self.buffer_config['engine_buffer'].keys()))
            suffix = self.buffer_config['engine_buffer'][origin]['suffix']
            channel = buffers[buffer_name]['channel']
            path = self.job_config['pattern'].format(
                # directory, origin, ind+1, suffix)
                directory, origin, ind, suffix)
            buffer_data = read_image(path=path, channel=channel)

            if torch.isinf(buffer_data).any() or torch.isnan(buffer_data).any():
                log.warn(f'{"="*10 + "warning" + "="*10}\n there is inf or nan in "{path}"')
                # raise Exception(f'there is inf or nan in "{path}", abort.')

            buffer_data[torch.isinf(buffer_data)] = 0.0
            buffer_data[torch.isnan(buffer_data)] = 0.0
            tmp_data[buffer_name] = buffer_raw_to_data(
                buffer_data, buffer_name)

        return tmp_data

    def parse_scene(self, directory, ind):
        f = open(self.job_config['pattern'].format(
            directory, self.job_config['scene_info_name'], ind, "txt"))
        lines = [x.strip() for x in f.readlines()]
        # log.debug(lines)
        i = 0
        dict_name, i = parse_find_dict(lines, i)
        ret, i = parse_flat_dict(lines, "", i + 1)
        return ret

    def get_patch(self, scene, index, buffers=None):
        if buffers is None:
            buffers = self.buffer_config['output_buffer']
        else:
            buffers = {buffer_name: self.buffer_config['output_buffer'][buffer_name] for buffer_name in buffers}
        import_path = self.job_config['import_path'] + scene
        tmp_data = []
        tmp_data = self.parse_buffer(buffers, import_path, self.metadata[scene][index].index)
        tmp_data['metadata'] = self.metadata[scene][index].to_dict()
        # log.debug("scene:{} index:{}".format(scene, index))
        return tmp_data

    def export_patch_npz(self, metadata: MetaData):
        suffix = ""
        scene = metadata.dataset_name
        index = metadata.index
        file_path_template = "{}/{}{}/{{}}/{{}}.{{}}".format(
            self.job_config['export_path'], scene, suffix)
        overwrite = self.job_config.get('overwrite', False)

        ''' overwrite handler '''
        if not overwrite:
            assert self.dataset_format == DatasetFormat.NPZ
            metadata_part = self.buffer_config['metadata_part']
            if os.path.exists(file_path_template.format(metadata_part, index, "npz")):
                log.debug("{} exists.".format(
                    file_path_template.format(metadata_part, index, "npz")))
                return

        origin_data = self.get_patch(scene, index)
        ret = {}
        ''' extend process '''
        for part_name in self.buffer_config['basic_part_enable_list']:
            ret[part_name] = get_extend_buffer(origin_data, part_name, self.buffer_config, start_cutoff=0)

        ''' write buffer to disk'''
        export_path = ""
        for part_name in self.buffer_config['basic_part_enable_list']:
            if index == 0:
                log.info(dict_to_string(ret[part_name]))

            # export_path = file_path_template.format(part_name, index, "pt")
            export_path = file_path_template.format(part_name, index, "npz")
            path_comp = get_file_component(export_path)
            create_dir(path_comp['path'])
            scene_path = os.path.join(path_comp['path'], "..")
            if not os.path.exists(scene_path + "/" + "{}.txt".format(part_name)):
                write_text_to_file(scene_path + "/" + "{}.txt".format(part_name),
                                   json.dumps(list(ret[part_name].keys()), indent=4).replace(
                                       "true", "True").replace("false", "False").replace("null", "None"), "w")
            # log.debug(self.buffer_config['part'][part_name]['type'])
            ret[part_name] = compress_buffer(ret[part_name], self.buffer_config['part'][part_name]['type'])
            log.debug(dict_to_string(ret[part_name], mmm=True))

            write_npz(export_path, ret[part_name])

            # write_torch(export_path, ret[part_name])
        log.debug("patch \"{}\" exported. patch:{}".format(MetaData(scene, index), export_path))

    def export_npz(self):
        for s in self.job_config['scene']:
            tmp_metadatas = self.metadata[s]
            dispatch_task_by_metadata(self.export_patch_npz, tmp_metadatas,
                                      num_thread=self.job_config.get('num_thread', 0))

    def export(self):
        if self.dataset_format == DatasetFormat.NPZ:
            self.export_npz()
        elif self.dataset_format == DatasetFormat.HDF5:
            self.export_hdf5()
        else:
            assert False, self.job_config

    def test_task(self, metadata):
        log.debug(f"start range: {metadata}")

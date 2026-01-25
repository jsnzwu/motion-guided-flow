# import portalocker
import os
import torch
from utils.buffer_utils import fix_dmdl_color_zero_value, to_numpy
from wickit.utils.io.imageio import read_image
from utils.log import log
from wickit.utils.basic.string import dict_to_string
from wickit.utils.basic.tensor import data_to_device, data_to_device_dict
from utils.utils import create_dir, del_dict_item, get_tensor_mean_min_max_str
from utils.warp import warp
import numpy as np
from yacs.config import CfgNode
import torch.nn.functional as F
import re

# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
# import cv2


def rgb2gray(rgb):
    r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def get_input_filter_list(config: dict) -> list:
    ret = set()
    res = []
    tmp_res = []
    for k in config.keys():
        if k == 'input_buffer' and config.get('enable', True):
            for item in config[k]:
                if item not in ret:
                    ret.add(item)
                    res.append(item)
        elif type(config[k]) in [dict, CfgNode]:
            tmp_res += get_input_filter_list(config[k])
    for tmp_item in tmp_res:
        if tmp_item not in ret:
            ret.add(tmp_item)
            res.append(tmp_item)
    return res


def transform_position_image(pos, mat):
    c, h, w = pos.shape
    pos = pos.permute(1, 2, 0).reshape(-1, c)
    pos = torch.cat([pos, torch.ones(h * w, 1)], dim=1)
    # log.debug(pos.shape)
    # log.debug(mat)
    pos_trans = torch.mm(pos, mat)[..., :-1]
    pos_trans = pos_trans.reshape(h, w, c).permute(2, 0, 1)
    return pos_trans


def transform_direction_image(in_dir, mat):
    c, h, w = in_dir.shape
    in_dir = in_dir.permute(1, 2, 0).reshape(-1, c)
    in_dir = torch.cat([in_dir, torch.zeros(h * w, 1)], dim=1)
    # log.debug(dir.shape)
    # log.debug(mat)
    pos_trans = torch.mm(in_dir, mat)[..., :-1]
    pos_trans = pos_trans.reshape(h, w, c).permute(2, 0, 1)
    return pos_trans


def transform_position(vec, mat):
    tmp_vec = torch.cat([vec, torch.tensor([1])], dim=0)
    ret = torch.mm(tmp_vec, mat)[0, :-1]
    return ret


def transform_direction(vec, mat):
    tmp_vec = torch.cat([vec, torch.tensor([0])], dim=0).reshape(1, -1)
    ret = torch.mm(tmp_vec, mat)[0, :-1]
    return ret


def shadow_attention(data: list, config=None, **kwargs):
    op_dim = 1 if len(data[0].shape) == 4 else 0
    return (torch.abs(data[0] - data[1])) / (torch.mean(torch.min(data[0], data[1]), dim=op_dim, keepdim=True) + 1e-2)


def get_continuity_mask(pred, gt, alpha=3, beta=8):
    op_dim = 1 if len(pred.shape) == 4 else 0
    mask = torch.max(shadow_attention(
        [pred, gt]), dim=op_dim, keepdim=True).values
    mask = torch.exp(-alpha * mask)
    # log.debug(dict_to_string(beta*(mask - mask.mean())[0], mmm=True))
    # log.debug(dict_to_string(beta*(mask - 0.5)[0], mmm=True))
    mask = torch.sigmoid(beta * (mask - 0.5))
    return mask


def create_history_frame(last_data, name, index=0):
    ret = last_data[index][name]
    return ret


def create_future_frame(next_data, name, index=0):
    ret = next_data[index][name]
    return ret


lut = read_image("asset/precomputed_brdf_lut.exr")[:2, ...].unsqueeze(0)


def create_dmdl_color_brdf(roughness, nov, albedo, metallic, specular, skybox_mask=None, fix=False):
    global lut
    if lut.device != roughness.device:
        lut = lut.to(roughness.device)
    if len(nov.shape) == 4:
        op_dim = 1
    else:
        op_dim = 0
    uv = torch.cat([nov, roughness], dim=op_dim) * 2 - 1
    if op_dim == 0:
        uv = uv.unsqueeze(0)
    uv = uv.permute(0, 2, 3, 1)
    if lut.dtype != uv.dtype:
        lut = lut.type(uv.dtype)
    # log.debug(dict_to_string([lut, uv]))
    if op_dim == 0:
        input_lut = lut
    else:
        input_lut = lut.repeat(uv.shape[0], 1, 1, 1)
    precomputed = torch.nn.functional.grid_sample(input=input_lut, grid=uv,
                                                  mode="bilinear", padding_mode="border",
                                                  align_corners=True)
    specular_color = 0.08 * specular * albedo + (1 - 0.08 * specular) * metallic
    # log.debug(dict_to_string(data, mmm=True))
    if op_dim == 1:
        brdf_color = albedo * (1 - metallic) + precomputed[:, :1, ...] * specular_color + precomputed[:, 1:, ...]
    else:
        brdf_color = albedo * (1 - metallic) + precomputed[0, :1, ...] * specular_color + precomputed[0, 1:, ...]

    if skybox_mask is not None:
        brdf_color = torch.ones_like(brdf_color) * skybox_mask + brdf_color * (1 - skybox_mask)
    if fix:
        brdf_color = fix_dmdl_color_zero_value(brdf_color)
    return brdf_color


def create_scene_color_no_st(scene_color, st_color, st_alpha):
    scene_color_no_st = (scene_color - st_color) / st_alpha
    scene_color_no_st = torch.clamp(scene_color_no_st, min=0)
    scene_color_no_st = torch.where(st_alpha <= 0.01, st_color, scene_color_no_st)
    return scene_color_no_st


def create_scene_color(scene_color_no_st, st_color, st_alpha):
    return scene_color_no_st * st_alpha + st_color


def create_st_color(scene_color, scene_color_no_st, alpha):
    st_color = scene_color - scene_color_no_st * alpha
    st_color = torch.clamp(st_color, min=0)
    return st_color


def create_nov(world_normal, camera__forward):
    is_single_image = False
    if len(world_normal.shape) == 3:
        is_single_image = True
        world_normal = world_normal.unsqueeze(0)
    b, c, h, w = world_normal.shape
    camf_image = camera__forward.reshape(1, 3, 1, 1).repeat(b, 1, h, w) * -1
    nov = torch.sum(world_normal * camf_image, dim=1, keepdim=True)
    if is_single_image:
        nov = nov[0]
    return nov


def create_scene_color_no_sky(scene_color, sky_color, skybox_mask):
    scene_color_no_sky = (scene_color - sky_color*skybox_mask) / (1-skybox_mask)
    scene_color_no_sky = torch.where(skybox_mask == 1, torch.zeros_like(scene_color_no_sky), scene_color_no_sky)
    return torch.clamp(scene_color_no_sky, min=0)


def set_dataset_global_config(buffer_config):
    max_luminance = buffer_config.get('max_luminance', -1)
    mu = buffer_config.get('mu', 8.0)
    is_normalization = buffer_config.get('is_normalization', False)
    # mean_shift_value = buffer_config.get('is_mean_shift', False)
    DatasetGlobalConfig.max_luminance = max_luminance
    DatasetGlobalConfig.log_tonemapper__mu = mu
    DatasetGlobalConfig.log_tonemapper__is_normalization = is_normalization
    ''' TODO: add meanshift and normalization '''

from typing import Optional
from dataclasses import dataclass, field


@dataclass
class DatasetGlobalConfig:
    max_luminance = -1
    log_tonemapper__mu = 8.0
    log_tonemapper__is_normalization = False
    log_tonemapper__light_mean_map: Optional[torch.Tensor] = None
    log_tonemapper__color_mean_map: Optional[torch.Tensor] = None


def create_de_color(scene_color, dmdl_color, skybox_mask=None, sky_color=None, fix=False, max_luminance=False):
    # log.debug([max_luminance, global_max_luminance])
    for _ in [len(scene_color.shape) - 1, len(scene_color.shape) - 2]:
        assert scene_color.shape[_] == dmdl_color.shape[_]
    if fix:
        tmp_dmdl_color = fix_dmdl_color_zero_value(dmdl_color, skybox_mask)
    else:
        tmp_dmdl_color = dmdl_color
    if skybox_mask is not None:
        assert sky_color is not None
        scene_color = create_scene_color_no_sky(scene_color, sky_color, skybox_mask)
    scene_light = scene_color / tmp_dmdl_color
    sum_dim = len(tmp_dmdl_color.shape) - 3
    scene_light = torch.where(torch.sum(tmp_dmdl_color, dim=sum_dim, keepdim=True)
                              == 0, torch.zeros_like(scene_light), scene_light)
    if max_luminance:
        assert DatasetGlobalConfig.max_luminance > 0, f"global_max_luminance must be > 0, which is {DatasetGlobalConfig.max_luminance} now."
        scene_light = torch.clamp(scene_light, max=DatasetGlobalConfig.max_luminance)
    return scene_light

def compose_scene_color_no_st(scene_light, dmdl_color, skybox_mask=None, sky_color=None, fix=False):
    if fix:
        tmp_dmdl_color = fix_dmdl_color_zero_value(dmdl_color, skybox_mask)
    else:
        tmp_dmdl_color = dmdl_color
    scene_color = scene_light * tmp_dmdl_color
    if skybox_mask is not None:
        assert sky_color is not None
        scene_color = scene_color * (1-skybox_mask) + sky_color * skybox_mask
    return scene_color


def compose_scene_color(scene_light_no_st, dmdl_color, st_color, st_alpha, skybox_mask=None, sky_color=None, fix=False):
    scene_color_no_st = compose_scene_color_no_st(scene_light_no_st, dmdl_color, skybox_mask, sky_color, fix)
    scene_color = scene_color_no_st * st_alpha + st_color
    return scene_color


def create_sky_color(scene_color, skybox_mask):
    return skybox_mask * scene_color


def resize(image, scale_factor, mode='bilinear') -> torch.Tensor:
    shape3d = False
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
        shape3d = True
    if scale_factor != 1:
        ret = F.interpolate(image, scale_factor=scale_factor, mode="bilinear", align_corners=False)
        if shape3d:
            ret = ret[0]
        return ret.to(image.dtype)
    return image


def resize_by_size(image, size, mode='bilinear') -> torch.Tensor:
    shape3d = False
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
        shape3d = True
    ret = F.interpolate(image, size=size, mode="bilinear", align_corners=False)
    if shape3d:
        ret = ret[0]
    return ret


def create_aa_skybox_mask(skybox_mask, size, with_batch=False):
    if not with_batch:
        skybox_mask = skybox_mask.unsqueeze(0)
    origin_size = (skybox_mask.shape[-2], skybox_mask.shape[-1])
    blur_skybox_mask = resize_by_size(skybox_mask, size, mode='bicubic')
    blur_skybox_mask = resize_by_size(blur_skybox_mask, origin_size, mode='bicubic')
    # skybox_mask = torch.where(skybox_mask<=0, torch.zeros_like(blur_skybox_mask), blur_skybox_mask)
    if not with_batch:
        blur_skybox_mask = blur_skybox_mask[0]
    return blur_skybox_mask


def create_skybox_mask(depth, base_color, sky_depth=None, enable_aa=True, aa_sample=2, with_batch=False):
    # log.debug(dict_to_string(data))
    if with_batch:
        base_color = base_color.clone().sum(dim=1, keepdim=True)
    else:
        base_color = base_color.clone().sum(dim=0).unsqueeze(0)
    # log.debug(dict_to_string(base_color, "base_color", mmm=True))
    one_mask = torch.ones_like(depth)
    zero_mask = torch.zeros_like(depth)
    # log.debug("skybox_mask: depth_max:{}".format(depth.max()))
    # log.debug("depth mmm: {}".format(get_tensor_mean_min_max_str(depth)))

    # assert sky_depth is not None
    if sky_depth is None:
        max_value = 1.0 - 1e-6
        skybox_mask = torch.where(
            depth >= max_value, one_mask, zero_mask)
    else:
        # log.debug(dict_to_string(sky_depth/depth, mmm=True))
        eps = 1e-9
        skybox_mask = torch.where(
            (sky_depth / depth - one_mask * eps) <= 1, one_mask, zero_mask)
    # log.debug(dict_to_string(skybox_mask))
    # log.debug(dict_to_string(base_color))
    skybox_mask[torch.where(base_color > 0)] = 0.0

    # blur_skybox_mask = resize(skybox_mask.unsqueeze(0), 1/aa_sample**2)
    # blur_skybox_mask = resize(blur_skybox_mask, (aa_sample**2))[0]
    # skybox_mask = torch.where(skybox_mask<=0, zero_mask, blur_skybox_mask)
    return skybox_mask


def create_history_warped_buffer(data, last_data, idx, name, mode="bilinear", padding_mode="border",  prefix="", postfix="", with_batch=False):
    '''
    mode (str): sample mode for warp
        'nearest' | 'bilinear'. Default: 'zeros'
    padding_mode (str): padding mode for outside grid values
        'zeros' | 'border' | 'reflection'. Default: 'zeros'
    '''
    # log.debug('create_history_warped_scene_color: {}'.format(idx))
    ret = last_data[idx][prefix + name + postfix]
    for i in range(idx, 0, -1):
        ret = create_warped_buffer(
            ret, last_data[i - 1][prefix + 'motion_vector' + postfix], mode=mode, padding_mode=padding_mode, with_batch=with_batch)
    ret = create_warped_buffer(
        ret, data[prefix + 'motion_vector' + postfix], mode=mode, padding_mode=padding_mode, with_batch=with_batch)
    return ret


# def create_history_warped_scene_color(data, last_data, idx, postfix=""):
#     # log.debug('create_history_warped_scene_color: {}'.format(idx))
#     ret = last_data[idx]['scene_color' + postfix]
#     for i in range(idx, 0, -1):
#         ret = create_warped_buffer(
#             ret, last_data[i - 1]['motion_vector' + postfix], padding_mode="border")
#     ret = create_warped_buffer(
#         ret, data['motion_vector' + postfix], padding_mode="border")
#     return ret


def create_occlusion_mask(data, last_data):
    # d1 = warp(last_data['depth'], data['motion_vector'], padding_mode="border")[0]
    # d0 = data['depth']
    # wn1 = warp(last_data['world_normal'], data['motion_vector'], padding_mode="border")[0]
    # wn0 = data['world_normal']
    # bc1 = warp(last_data['base_color'], data['motion_vector'], padding_mode="border")[0]
    # bc0 = data['base_color']
    mv1 = data['merged_motion_vector_1'] - data['merged_motion_vector_0']
    mv0 = data['merged_motion_vector_0']
    # skybox_mask = data['skybox_mask']
    C, H, W = mv0.shape
    # wn_cos = torch.zeros_like(wn0[0,...])
    # for i in range(3):
    #     wn_cos += wn0[i, :, :] * wn1[i, :, :]
    mv_cos = torch.zeros_like(mv0[0, ...])
    for i in range(mv0.shape[0]):
        mv_cos += mv0[i, :, :] * mv1[i, :, :]

    mv_cos /= (mv0[0, ...]**2 + mv1[1, ...]**2 + 0.01)**0.5

    # mv_cos = torch.zeros_like(mv0[0,...])
    # for i in range(mv0.shape[0]):
    #     mv_cos += mv0[i, :, :] * mv1[i, :, :]

    # bc_diff = torch.max(torch.abs(bc0-bc1), dim=0, keepdim=True).values
    # cos_mv_diff = torch.max(mv_cos, dim=0, keepdim=True).values
    mv_diff = torch.max(torch.abs(mv1 - mv0), dim=0, keepdim=True).values

    one_mask = torch.ones(H, W).unsqueeze(0)
    zero_mask = torch.zeros(H, W).unsqueeze(0)
    mask = zero_mask
    # mask = torch.where(wn_cos < 0, one_mask, mask)
    # mask = torch.where(bc_diff > 0.03, one_mask, mask)
    # mask = torch.where(cos_mv_diff < 0.25, one_mask, mask)
    mask = torch.where(mv_diff > 8 / mv0.shape[-1], one_mask, mask)
    # mask = torch.where(mv_cos < 0, one_mask, mask)
    # mask = torch.where(1-skybox_mask > 0, mask, zero_mask)
    # log.debug("{} {}".format(mask.shape, skybox_mask_and_mask.shape))
    return mask


def create_cross_sample(data):
    rows = to_numpy(data['world_to_clip_l'])
    Proj = np.array([
        [rows[0, 0, 0], rows[0, 0, 1], 0, rows[0, 0, 2]],
        [rows[-1, 0, 0], rows[-1, 0, 1], 0, rows[-1, 0, 2]],
        [rows[0, -1, 0], rows[0, -1, 1], 0, rows[0, -1, 2]],
        [rows[-1, -1, 0], rows[-1, -1, 1], 1, rows[-1, -1, 2]],
    ])
    world_position_r_homo = to_numpy(data['world_position_r'])
    world_position_r_homo = np.concatenate(
        [world_position_r_homo, world_position_r_homo[..., 0:1] * 0.0 + 1.0], axis=2)
    # log.debug(data['world_position_r'].shape)
    # log.debug(data['world_position_r'][:, :2, :2])
    # log.debug(world_position_r_homo[:2, :2, :])
    # log.debug(Proj)
    cross_sample_l = np.matmul(world_position_r_homo, Proj)
    cross_sample_l = cross_sample_l.astype(np.float32)
    cross_sample_l[:, :, :2] = cross_sample_l[:,
                                              :, :2] / cross_sample_l[:, :, 3:4]
    cross_sample_l[:, :, 1] = cross_sample_l[:, :, 1] / cross_sample_l[0, 0, 1]
    cross_sample_l[:, :, 1] *= -1
    cross_sample_l = torch.from_numpy(
        cross_sample_l[:, :, :2]).permute(2, 0, 1)

    rows = to_numpy(data['world_to_clip_r'])
    Proj = np.array([
        [rows[0, 0, 0], rows[0, 0, 1], 0, rows[0, 0, 2]],
        [rows[-1, 0, 0], rows[-1, 0, 1], 0, rows[-1, 0, 2]],
        [rows[0, -1, 0], rows[0, -1, 1], 0, rows[0, -1, 2]],
        [rows[-1, -1, 0], rows[-1, -1, 1], 1, rows[-1, -1, 2]],
    ])
    world_position_l_homo = to_numpy(data['world_position_l'])
    world_position_l_homo = np.concatenate(
        [world_position_l_homo, np.ones_like(world_position_l_homo[..., 0:1])], axis=2)
    cross_sample_r = np.matmul(world_position_l_homo, Proj)
    cross_sample_r = cross_sample_r.astype(np.float32)
    cross_sample_r[:, :, :2] = cross_sample_r[:,
                                              :, :2] / cross_sample_r[:, :, 3:4]
    cross_sample_r[:, :, 1] = cross_sample_r[:, :, 1] / cross_sample_r[0, 0, 1]
    cross_sample_r[:, :, 1] *= -1
    cross_sample_r = torch.from_numpy(
        cross_sample_r[:, :, :2]).permute(2, 0, 1)

    return cross_sample_l, cross_sample_r


def create_history_warped_scene_color_cross(data, last_data, historical=True):
    if historical:
        history_warp_l = warp(
            last_data[0]['scene_color_l'], data['motion_vector_l'])[0]
        history_warp_r = warp(
            last_data[0]['scene_color_r'], data['motion_vector_r'])[0]
    else:
        history_warp_l = data['scene_color_l']
        history_warp_r = data['scene_color_r']
    # warp L to R
    history_warped_cross_r = \
        F.grid_sample(history_warp_l.unsqueeze(0), data['cross_sample_l'].permute(1, 2, 0).unsqueeze(
            0), mode="bilinear", align_corners=False)[0]

    # Warp R to L
    history_warped_cross_l = \
        F.grid_sample(history_warp_r.unsqueeze(0), data['cross_sample_r'].permute(1, 2, 0).unsqueeze(
            0), mode="bilinear", align_corners=False)[0]

    return history_warped_cross_l, history_warped_cross_r


def create_history_masked_occlusion_warped_scene_color(data, last_data, idx):
    ret = last_data[idx]['scene_color']
    if last_data[idx].get("occlusion_mask", None):
        last_data[idx]['occlusion_mask'] = create_occlusion_mask(
            last_data[idx], last_data[idx + 1])
    ret = ret * (1 - last_data[idx]['occlusion_mask'])
    for i in range(idx, 0, -1):
        if last_data[i - 1].get("occlusion_mask", None):
            last_data[i - 1]['occlusion_mask'] = create_occlusion_mask(
                last_data[i - 1], last_data[i])
        if last_data[i - 1].get("occlusion_motion_vector", None):
            last_data[i - 1]['occlusion_motion_vector'] = create_occlusion_motion_vector(
                last_data[i]['motion_vector'], last_data[i - 1]['motion_vector'], last_data[i - 1]['occlusion_mask'])
        ret = create_warped_buffer(
            ret, last_data[i - 1]['occlusion_motion_vector'])
        ret = ret * (1 - last_data[i - 1]['occlusion_mask'])
    ret = create_warped_buffer(
        ret, data['occlusion_motion_vector'])
    return ret


def create_occlusion_motion_vector(last_mv, mv, mask):
    occlu_mv = warp(last_mv, mv)[0]
    occlu_mv_mix = torch.where(mask > 0, occlu_mv, mv)
    return occlu_mv_mix


def create_warped_buffer(last_buffer, mv, mode="bilinear", padding_mode="zeros", with_batch=False):
    '''
    mode (str): sample mode for warp
        'nearest' | 'bilinear'. Default: 'zeros'
    padding_mode (str): padding mode for outside grid values
        'zeros' | 'border' | 'reflection'. Default: 'zeros'
    '''
    if with_batch:
        warped_scene_color = warp(
            last_buffer, mv, mode=mode, padding_mode=padding_mode)
    else:
        warped_scene_color = warp(
            last_buffer.unsqueeze(0), mv.unsqueeze(0), mode=mode, padding_mode=padding_mode)[0, ...]
    # log.debug(dict_to_string(warped_scene_color, "warped_scene_color"))
    return warped_scene_color

# (1,H,W)


def create_discontinuity_mask(scene_color, warped_scene_color, ratio=1):
    C, H, W = scene_color.shape
    # log.debug(dict_to_string(torch.sum(torch.abs(scene_color - warped_scene_color),axis=0), "discontinuity diff"))
    # log.debug(dict_to_string(torch.min(torch.cat([scene_color, warped_scene_color], axis=0), 0), "discontinuity min"))
    diff_map = torch.mean(torch.abs(scene_color - warped_scene_color) / torch.abs(scene_color.mean() - scene_color), dim=0)
    color_diff_rate = diff_map
    # log.debug(get_tensor_mean_min_max_str(diff_map, "diff_map"))
    # log.debug(get_tensor_mean_min_max_str(ratio_map, "ratio_map"))
    # log.debug(dict_to_string(color_diff_rate, "color_diff_rate"))
    # log.debug(dict_to_string(diff_map, "diff_map"))
    # log.debug(dict_to_string(ratio_map, "ratio_map"))
    one_mask = torch.ones(H, W)
    # one_mask[torch.where(diff_map == 0)] = 0.0
    mask = torch.zeros(H, W)
    # ratio = diff_map.max() - (diff_map.max() - diff_map.min()) * ratio
    # log.debug(ratio)
    mask = torch.where(color_diff_rate > ratio, one_mask, mask)
    # mask = torch.where(shadow_diff_mask[0, ...] > 0, one_mask, mask)
    return mask.unsqueeze(0)


def create_shadow_discontinuity_mask(shadow_diff, ratio=0.1):
    C, H, W = shadow_diff.shape
    one_mask = torch.ones(H, W)
    mask = torch.zeros(H, W)
    mask = torch.where(shadow_diff[0, ...] > 0.1, one_mask, mask)
    return mask.unsqueeze(0)


def _savez(file, args, kwds, compress, allow_pickle=True, pickle_kwargs=None):
    # Import is postponed to here since zipfile depends on gzip, an optional
    # component of the so-called standard library.
    import zipfile
    from numpy.lib import format

    if not hasattr(file, 'write'):
        from numpy.compat.py3k import os_fspath
        file = os_fspath(file)
        if not file.endswith('.npz'):
            file = file + '.npz'

    namedict = kwds
    for i, val in enumerate(args):
        key = 'arr_%d' % i
        if key in namedict.keys():
            raise ValueError(
                "Cannot use un-named variables and keyword %s" % key)
        namedict[key] = val

    if compress:
        compression = zipfile.ZIP_DEFLATED
    else:
        compression = zipfile.ZIP_STORED

    if 'compresslevel' in namedict:
        compresslevel = namedict['compresslevel']
        if not isinstance(compresslevel, int) or compresslevel < 1 or compresslevel > 9:
            compresslevel = None
        del namedict['compresslevel']
    else:
        compresslevel = None
    from numpy.lib.npyio import zipfile_factory  # type: ignore
    zipf = zipfile_factory(file, mode="w", compression=compression, compresslevel=compresslevel)

    for key, val in namedict.items():
        fname = key + '.npy'
        val = np.asanyarray(val)
        # always force zip64, gh-10776
        with zipf.open(fname, 'w', force_zip64=True) as fid:
            format.write_array(fid, val,
                               allow_pickle=allow_pickle,
                               pickle_kwargs=pickle_kwargs)

    zipf.close()


def _savez_wrapper(file, *args, **kwds):
    _savez(file, args, kwds, compress=True)


def write_npz(file_path, data):
    # log.debug(file_path)
    # log.debug(dict_to_string(data))
    # np.savez_compressed(file_path, data)
    data = data_to_device(data, 'cpu')
    _savez_wrapper(file_path, data, compresslevel=1)

    # np.savez(file_path, data)

def write_torch(file_path, data):
    log.debug(dict_to_string(data, mmm=True))
    torch.save(data, f=file_path)

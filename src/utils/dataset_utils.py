# import portalocker
import torch
from utils.buffer_utils import fix_dmdl_color_zero_value
from wickit.utils.io.imageio import read_image
from wickit.utils.basic.tensor import data_to_device
from wickit.utils.ext.warp import warp
import numpy as np
from collections.abc import Mapping
import torch.nn.functional as F

# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
# import cv2


def _is_mapping_like(value: object) -> bool:
    return isinstance(value, Mapping) or (hasattr(value, "keys") and hasattr(value, "__getitem__"))


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
        elif _is_mapping_like(config[k]):
            tmp_res += get_input_filter_list(config[k])
    for tmp_item in tmp_res:
        if tmp_item not in ret:
            ret.add(tmp_item)
            res.append(tmp_item)
    return res


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


def create_scene_color_no_sky(scene_color, sky_color, skybox_mask):
    scene_color_no_sky = (scene_color - sky_color*skybox_mask) / (1-skybox_mask)
    scene_color_no_sky = torch.where(skybox_mask == 1, torch.zeros_like(scene_color_no_sky), scene_color_no_sky)
    return torch.clamp(scene_color_no_sky, min=0)


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

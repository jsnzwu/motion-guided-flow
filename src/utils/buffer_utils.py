import math
import imghdr
import re
from wickit.utils.basic.string import dict_to_string
from wickit.utils.basic.tensor import align_channel_buffer
from wickit.utils.io.imageio import read_image
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from utils.parser_utils import parse_buffer_name
from utils.log import log
from tqdm import tqdm
import imageio
import skimage.io
albedo_min_clamp = 0.01

def hdr_to_ldr(img, use_gamma=False):
    if use_gamma:
        gamma_func = gamma
    else:
        gamma_func = lambda x:x
    ret = torch.round(gamma_func(aces_tonemapper(img)) * 255)
    return ret.type(torch.uint8)

def coord_to_grid_by_HW(coord, H, W) -> torch.Tensor:
    ret = coord * 1.0
    if len(ret.shape) == 3:
        ret[0] /= (W-1) / 2
        ret[1] /= (H-1) / 2
        return ret
    elif len(ret.shape) == 4:
        ret[:, 0] /= (W-1) / 2
        ret[:, 1] /= (H-1) / 2
        return ret
    else:
        raise Exception("flow must be a 3D or 4D tensor.")
    
def coord_to_grid(coord) -> torch.Tensor:
    ret = coord * 1.0
    if len(ret.shape) == 3:
        C, H, W = ret.shape
        ret[0] /= (W-1) / 2
        ret[1] /= (H-1) / 2
        return ret
    elif len(ret.shape) == 4:
        B, C, H, W = ret.shape
        ret[:, 0] /= (W-1) / 2
        ret[:, 1] /= (H-1) / 2
        return ret
    else:
        raise Exception("flow must be a 3D or 4D tensor.")

def flow_to_motion_vector(flow, align_corners=True) -> torch.Tensor:
    ret = flow * 1.0
    offset = 1 if align_corners else 0
    if len(ret.shape) == 3:
        C, H, W = ret.shape
        ret[0] /= (W-offset) / 2
        ret[1] /= (H-offset) / 2
        return ret
    elif len(ret.shape) == 4:
        B, C, H, W = ret.shape
        ret[:, 0] /= (W-offset) / 2
        ret[:, 1] /= (H-offset) / 2
        return ret
    else:
        raise Exception("flow must be a 3D or 4D tensor.")


def motion_vector_to_flow(mv, align_corners=True):
    ret = mv * 1.0
    offset = 1 if align_corners else 0
    if len(ret.shape) == 3:
        C, H, W = ret.shape
        ret[0] *= (W-offset) / 2
        ret[1] *= (H-offset) / 2
        return ret
    elif len(ret.shape) == 4:
        B, C, H, W = ret.shape
        ret[:, 0] *= (W-offset) / 2
        ret[:, 1] *= (H-offset) / 2
        return ret
    else:
        assert False


def get_buffer_filename(pattern, dir_path, buffer_name, index, suffix='png'):
    return pattern.format(dir_path, buffer_name, index, suffix)


def demodulate_buffer_name(buffers):
    selected_names = ['history_warped_scene_color_{}',
                      'warped_scene_color',
                      'scene_color_no_shadow',
                      'occlusion_warped_scene_color',
                      'masked_warped_scene_color',
                      'masked_occlusion_warped_scene_color']
    for i in range(len(buffers)):
        if buffers[i] in selected_names:
            buffers[i] = "de_" + buffers[i]
    return buffers


def fix_dmdl_color_zero_value(brdf_color, skybox_mask=None, sum_clamp=False):
    ret = brdf_color
    if skybox_mask is not None:
        ret = torch.ones_like(ret) * skybox_mask + ret * (1 - skybox_mask)
    return torch.clamp(ret, min=albedo_min_clamp)


def buffer_raw_to_data(data, buffer_name):
    ops = {
        'base_color': lambda x: x,  # fix_base_color_zero_value(x),
        'dmdl_color_brdf': lambda x: x,  # fix_base_color_zero_value(x),
        'dmdl_color_ess': lambda x: x,  # fix_base_color_zero_value(x),
        'dmdl_color': lambda x: x,  # fix_base_color_zero_value(x),
        'depth': lambda x: x,
        'nov': lambda x: x,
        'metallic': lambda x: x,
        'specular': lambda x: x,
        'roughness': lambda x: x,
        'stencil': lambda x: x,
        'shadow_mask': lambda x: x,
        'scene_color': lambda x: x,
        'scene_color_no_shadow': lambda x: x,
        'scene_light': lambda x: x,
        'skybox_mask': lambda x: x,
        'emissive_mask': lambda x: x,
        'scene_light_no_shadow': lambda x: x,
        'motion_vector': lambda x: x,
        'world_normal': lambda x: x,
        'world_position': lambda x: x,
        'world_to_clip': lambda x: x,
        "st_alpha": lambda x: x,
        "st_color": lambda x: x,
        "scene_color_no_st": lambda x: x,
        "sky_color": lambda x: x,
        "sky_depth": lambda x: x,
    }
    # res = re.search("(s\d+_)*(.+)(?:_[lr])*", buffer_name)
    # print(res)
    # if res:
    #     buffer_name = res.group(1)
    buffer_name = re.sub(r"((^(s|d|a|u)[\d]+_)|aa_)|(_[lr]$)", "", buffer_name)
    if buffer_name in ops.keys():
        ret = ops[buffer_name](data)
        if buffer_name == "motion_vector":
            ret[1, ...] *= -1
        if buffer_name == "depth":
            ret = torch.clamp(ret, max=65536.0)
            ret /= 65536.0
        return ret
    else:
        raise KeyError("{} is not in ops map".format(buffer_name))


def buffer_data_to_raw(data, buffer_name):
    ops = {
        'base_color': lambda x: x,
        'depth': lambda x: x,
        'nov': lambda x: x,
        'metallic': lambda x: x,
        'roughness': lambda x: x,
        'stencil': lambda x: x,
        'scene_light': lambda x: x,
        'motion_vector': lambda x: x,
        'skybox_mask': lambda x: x,
        'world_normal': lambda x: x,
        'world_position': lambda x: x
    }
    if buffer_name in ops.keys():
        ret = ops[buffer_name](data)
        # if buffer_name == "motion_vector":
        #     ret[1, ...] *= -1
        return ret
    else:
        raise KeyError("{} is not in ops map".format(buffer_name))


def aces_tonemapper(x, inv_gamma=False):
    dtype = x.dtype
    if dtype != torch.float32:
        x = x.type(torch.float32)
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    if inv_gamma:
        x = x ** (1 / 2.2)
    mapped = torch.clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)
    if dtype != torch.float32:
        mapped = mapped.type(dtype)
    return mapped


def gamma(image):
    return image ** (1 / 2.2)


def inv_gamma(image):
    return image ** (2.2)


def rein_tonemapper(x):
    return x / (1+x)

def inv_rein_tonemapper(y):
    return y / torch.clamp(1-y, min=1e-4)

def ldr_log_tonemapper(image, mu=8.0, mean_shift_value=1.0):
    # log.debug(dict_to_string(image, mmm=True))
    # image = image / max_value
    image = log_tonemapper(image, mu=mu)
    # log.debug(dict_to_string(image, mmm=True))
    if mean_shift_value != 1:
        image = image - mean_shift_value
    # log.debug(dict_to_string(image, mmm=True))
    return image
    
def inv_ldr_log_tonemapper(image, mu=8.0, mean_shift_value=1.0):
    # log.debug(dict_to_string(image, mmm=True))
    if mean_shift_value != 1:
        image = image + mean_shift_value
    # log.debug(dict_to_string([image, is_mean_shift, max_value], mmm=True))
    image = inv_log_tonemapper(image, mu=mu)
    # image = image * max_value
    # log.debug(dict_to_string(image, mmm=True))
    return image

def log_tonemapper(image, mu=8.0):
    dtype = image.dtype
    # if dtype != torch.float32:
    #     image = image.type(torch.float32)
    if isinstance(image, torch.Tensor):
        if mu == 0.0:
            one = torch.tensor(1.0, dtype=image.dtype, device=image.device)
            ret = torch.log(image + one)
        else:
            one = torch.tensor(1.0, dtype=image.dtype, device=image.device)
            t_mu = torch.tensor(mu, dtype=image.dtype, device=image.device)
            t_one_plus_mu = torch.tensor(math.log(1 + mu), dtype=image.dtype, device=image.device)
            # log.debug(dict_to_string([image, one, t_mu, t_one_plus_mu, torch.log(one + image * t_mu)], mmm=True))
            ret = torch.log(one + image * t_mu) / t_one_plus_mu
        if dtype != ret.dtype:
            ret = ret.to(dtype)
        # assert ret.dtype == image.dtype, f"ret.dtype: {ret.dtype}, image.dtype: {image.dtype}"
        return ret
    else:
        raise NotImplementedError(
            f'op \"log_tonemapper\": data type only support \"torch.Tensor\", but type of image is {type(image)}')


def inv_log_tonemapper(image, mu=8.0):
    # dtype = image.dtype
    # if dtype != torch.float32:
    #     image = image.type(torch.float32)
    if isinstance(image, torch.Tensor):
        if mu == 0.0:
            one = torch.tensor(1.0, dtype=image.dtype, device=image.device)
            ret = torch.exp(image) - one
        else:
            one = torch.tensor(1.0, dtype=image.dtype, device=image.device)
            t_mu = torch.tensor(mu, dtype=image.dtype, device=image.device)
            t_one_plus_mu = torch.tensor(math.log(1 + mu), dtype=image.dtype, device=image.device)
            ret = (torch.exp(image * t_one_plus_mu) - one) / t_mu
            # log.debug(dict_to_string([image, one, t_mu, t_one_plus_mu, torch.exp(image * t_one_plus_mu)], mmm=True))
        # if dtype != torch.float32:
            # log.debug(dict_to_string(ret))
            # ret = ret.type(dtype)
            # log.debug(dict_to_string(ret))
        # assert ret.dtype == image.dtype, f"init_dtype:{dtype}, ret.dtype: {ret.dtype}, image.dtype: {image.dtype}"
        return ret
    else:
        raise NotImplementedError(
            "op \"inv_log_tonemapper\": data type only support \"torch.Tensor\".")


def buffer_data_to_vis(data, buffer_name, scale=1.0) -> torch.Tensor:
    ops = {
        'base_color': lambda x: x,
        'depth': lambda x: x,
        'abs': lambda x: torch.abs(x),
        'nov': lambda x: x * 0.5 + 0.5,
        'metallic': lambda x: x,
        'roughness': lambda x: x,
        'stencil': lambda x: x,
        'world_normal': lambda x: x,
        'scene_light': lambda x: aces_tonemapper(x),
        'scene_color': lambda x: aces_tonemapper(x),
        'normal': lambda x: torch.clamp(16 * x, -0.5, 0.5) + 0.5,
        'normal_scale': lambda x: torch.clamp(scale * x, -0.5, 0.5) + 0.5,
        'normal_8': lambda x: torch.clamp(8 * x, -0.5, 0.5) + 0.5,
        'normal_64': lambda x: torch.clamp(64 * x, -0.5, 0.5) + 0.5,
        'motion_vector': lambda x: torch.clamp(x, -0.5, 0.5) + 0.5,
        'motion_vector_8': lambda x: torch.clamp(8 * x, -0.5, 0.5) + 0.5,
        'motion_vector_16': lambda x: torch.clamp(16 * x, -0.5, 0.5) + 0.5,
        'motion_vector_64': lambda x: torch.clamp(64 * x, -0.5, 0.5) + 0.5,
        'world_position': lambda x: torch.clamp(4096 * x, -4096, 4096) / 4096 * 0.5 * 20 + 0.5
    }
    if buffer_name in ops.keys():
        # buffer_name = parse_buffer_name(buffer_name)['buffer_name']
        ret = ops[buffer_name](data)
        # if buffer_name == "motion_vector_64":
        #     log.debug(dict_to_string([buffer_name, data, ret], mmm=True))
        return ret
    else:
        raise KeyError("{} is not in ops map".format(buffer_name))


def create_flip_data(data, vertical=True, horizontal=True, use_batch=True, batch_mask=None):
    if not use_batch:
        assert batch_mask is None
    if vertical or horizontal:
        if use_batch:
            target_pos = 1
        else:
            target_pos = 0
        for k in data.keys():
            if isinstance(data[k], torch.Tensor) and len(data[k].shape) == 3 + target_pos:
                flip_axis = []
                if vertical:
                    flip_axis.append(target_pos + 1)
                if horizontal:
                    flip_axis.append(target_pos + 2)
                if batch_mask is not None:
                    data[k][batch_mask,...] = torch.flip(data[k][batch_mask, ...], flip_axis)
                else:
                    data[k] = torch.flip(data[k], flip_axis)
                    
                if 'motion_vector' in k:
                    # log.debug(k)
                    # log.debug(dict_to_string(data))
                    if vertical:
                        if use_batch:
                            if batch_mask is not None:
                                data[k][batch_mask, 1, ...] *= -1
                            else:
                                data[k][:, 1, ...] *= -1
                        else:
                            data[k][1, ...] *= -1
                    if horizontal:
                        if use_batch:
                            if batch_mask is not None:
                                data[k][batch_mask, 0, ...] *= -1
                            else:
                                data[k][:, 0, ...] *= -1
                        else:
                            data[k][0, ...] *= -1
        return data
    return data


def show(img):
    plt.imshow(img)
    plt.show()




def to_numpy(arr, detach=True, cpu=True):
    assert len(arr.shape) == 3
    data = arr.permute(1, 2, 0)
    if detach:
        data = data.detach()
    if cpu:
        data = data.cpu()
    data = data.numpy()
    return data


def to_ldr_numpy(arr, normalize=255.0):
    img = (to_numpy(torch.clamp(arr,0,1)) * normalize).astype(np.uint8)
    return img


def save_to_img(arr, path):
    skimage.io.imsave(path, to_ldr_numpy(arr))


def add_text_to_image(img, text, multi_line=True) -> np.ndarray:
    if not isinstance(img, np.ndarray):
        # log.debug([img, text])
        img = to_ldr_numpy(img)
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    font_scale = 1
    thickness = 2
    H, W = img.shape[0], img.shape[1]
    position = (20, int(20 * W / H))
    if multi_line:
        line_height = int(cv2.getTextSize('Tg', font, font_scale, thickness)[0][1] * 1.5)
        current_line = ''
        max_width =  W - position[0]*2
        for char in text:
            # 临时行和宽度计算
            text_size = cv2.getTextSize(current_line+char, font, font_scale, thickness)[0]
            # 检查宽度是否超出
            if text_size[0] > max_width:
                '''                                                         B G  R     '''
                cv2.putText(img, current_line, position, font, font_scale, (0,0,255), thickness)
                current_line = ''
                position = (position[0], position[1] + line_height)
            current_line += char
        # 绘制最后一行
        if current_line:
            '''                                                         B G  R     '''
            cv2.putText(img, current_line, position, font, font_scale, (0,0,255), thickness)
    else:
        '''                                          B G  R     '''
        cv2.putText(img, text, position, font, font_scale, (0,0,255), thickness)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def torch_d2_to_d3(data):
    H = data.shape[1]
    W = data.shape[2]
    return torch.cat((data, torch.zeros(1, H, W, dtype=torch.float32).to(data.device)), 0)


def d3_to_d2(data):
    return data[:-1, :, :]


def export_video_in_path(path, image_files, output_path, fps, tonemap=False):
    log.debug("{} ... {}".format(str(image_files[:3]), str(image_files[-3:])))
    video = cv2.VideoWriter()
    image_0 = read_image(path + "/" + image_files[0])
    C, H, W = image_0.shape
    video.open(output_path, cv2.VideoWriter_fourcc(
        'm', 'p', '4', 'v'), fps, (W, H), True)
    # log.debug(image_files)
    for f in tqdm(image_files):
        # tmp_image = read_image(path + "/" + f)
        tmp_image = read_image(path + "/" + f)
        if tonemap:
            tmp_image = aces_tonemapper(tmp_image)
        tmp_image = to_numpy(align_channel_buffer(tmp_image, channel_num=3))
        if f.lower().endswith(".exr"):
            tmp_image = (
                gamma(tmp_image[:, :, [2, 1, 0]]) * 255.0).astype(np.uint8)
        video.write(tmp_image)
        # video.write(cv2.imread(os.path.join(path, f)))

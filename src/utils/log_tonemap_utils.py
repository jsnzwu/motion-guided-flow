import torch
import torch.nn.functional as F
from utils.warp import warp
from utils.dataset_utils import DatasetGlobalConfig
from utils.buffer_utils import inv_ldr_log_tonemapper, inv_log_tonemapper, ldr_log_tonemapper, log_tonemapper
from utils.log import log
from wickit.utils.basic.string import dict_to_string


max_value_storage = {}
def tonemap_func(image, use_global_settings=False, mu=8, max_luminance=1024.0, mean_map=None, is_normalization=False):
    if use_global_settings:
        max_luminance = DatasetGlobalConfig.max_luminance
        mu = DatasetGlobalConfig.log_tonemapper__mu
        is_normalization = DatasetGlobalConfig.log_tonemapper__is_normalization
    if max_luminance > 0:
        image = torch.clamp_max(image, max=max_luminance)
    if is_normalization:
        assert max_luminance != -1
    if is_normalization:
        assert False, "this setting should not be used"
        ret = ldr_log_tonemapper(image, mu=mu)
    else:
        ret = log_tonemapper(image, mu=mu)
    if mean_map is not None:
        ret = ret - mean_map
    return ret


def inv_tonemap_func(image, use_global_settings=False, mu=8, max_luminance=1024.0, mean_map=None, is_normalization=False):
    if use_global_settings:
        max_luminance = DatasetGlobalConfig.max_luminance
        mu = DatasetGlobalConfig.log_tonemapper__mu
        is_normalization = DatasetGlobalConfig.log_tonemapper__is_normalization
    if is_normalization:
        assert max_luminance != -1
    if mean_map is not None:
        image = image + mean_map
    if is_normalization:
        assert False, "this setting should not be used"
        ret = inv_ldr_log_tonemapper(image, mu=mu)
    else:
        ret = inv_log_tonemapper(image, mu=mu)
    return ret


import re
from wickit.utils.basic.string import dict_to_string
from utils.parser_utils import parse_buffer_name
from utils.log import log
import torch
import torch.nn as nn

def fix_the_size_with_dec(enc, dec):
    for ref_id in [2, 3]:
        if dec.shape[ref_id] != enc.shape[ref_id]:
            dec = dec.narrow(ref_id, 0, enc.shape[ref_id])
    return dec

def fix_the_size_with_dec_and_flow(enc, dec, flows=[]):
    for ref_id in [2, 3]:
        if dec.shape[ref_id] != enc.shape[ref_id]:
            dec = dec.narrow(ref_id, 0, enc.shape[ref_id])
            for i in range(len(flows)):
                flows[i] = flows[i].narrow(ref_id, 0, enc.shape[ref_id])
    return dec, flows

def retain_bn_float(net: nn.Module): 
    if isinstance(net, torch.nn.modules.batchnorm._BatchNorm) and net.affine is True:
        net.float()
    for child in net.children():
        retain_bn_float(child)
    return net
    

def model_to_half(net):
    net = net.half()
    return retain_bn_float(net)

def get_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    return total

dim2d_dict = {}

def get_2d_dim(item):
    if ('scene_color' in item or 'scene_light' in item):
        return 3
    # log.debug(dict_to_string(parse_buffer_name(item)))
    buffer_name = parse_buffer_name(item)['basic_element']
    if buffer_name not in dim2d_dict.keys():
        # log.warn("{} isnt in dim_dict, set dim = 0".format(item))
        # return 0
        raise KeyError("{} isnt in dim_dict.".format(buffer_name))
    return dim2d_dict[buffer_name]


dim1d_dict = {}


def get_1d_dim(item):
    # log.debug(item)
    if not (item in dim1d_dict.keys()):
        raise KeyError("{} isnt in dim_dict.".format(item))
    return dim1d_dict[item]


def calc_2d_dim(inputs):
    out_dim = 0
    for item in inputs:
        out_dim += get_2d_dim(item)
        # log.debug(dict_to_string([out_dim, item]))
    return out_dim


def calc_1d_dim(inputs):
    out_dim = 0
    # log.debug(inputs)
    for item in inputs:
        out_dim += get_1d_dim(item)
    return out_dim


def calc_dim(inputs):
    out_dim = 0
    for item in inputs:
        if item in dim1d_dict:
            out_dim += get_1d_dim(item)
        elif item in dim2d_dict:
            out_dim += get_2d_dim(item)
        else:
            raise KeyError("{} isnt in dim_dict.".format(item))
    return out_dim


def min_max_scalar(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-6)

from __future__ import annotations
from utils.log_tonemap_utils import inv_tonemap_func, tonemap_func
from models.mfrrnet.common import ConvLSTMCellV6
from wickit.utils.log import log
from wickit.utils.basic.string import dict_to_string
from wickit.models.model_base import ModelBaseEXT
from utils.config_adapter import DictToDataclassAdapter
from wickit.utils.ext.warp import get_merged_motion_vector_from_last, warp
from utils.buffer_utils import fix_dmdl_color_zero_value
from utils.dataset_utils import create_de_color, DatasetGlobalConfig
from utils.model_utils import get_1d_dim, get_2d_dim
from utils.parser_utils import parse_buffer_name
from wickit.utils.struct.struct_base import FlexDataStruct
from wickit.utils.enums import ForwardMode
from .loss.flow_loss import zero_flow_l1_loss, flow2_loss
from models.general.common_structure import NetBase
import copy
from datasets.mfrrnet_dataset import MFRRNetDataset
from utils.utils import TensorConcator, add_metaname
from dataloaders.patch_loader import history_extend
import torch.nn as nn
import torch
import torch.nn.functional as F
# from .archs_v6 import ConvLSTMCell


def fix_the_size_with(ref, res):
    for ref_id in [2, 3]:
        if res.shape[ref_id] != ref.shape[ref_id]:
            res = res.narrow(ref_id, 0, ref.shape[ref_id])
    return res


def data_to_input(data, config, cat_axis=1):
    # data = copy.deepcopy(data)
    data = MFRRNetDataset.preprocess(data)
    ret = {}
    cats = []
    for name in config['gbuffer_encoder']['input_buffer']:
        cats.append(data[name])
    ret['gbuffer_encoder_input'] = torch.cat(cats, dim=cat_axis)

    # log.debug(dict_to_string(data))
    def add_encoder_inputs(config):
        he_pfs = [item for item in config['output_prefixs']]
        for he_id, item in enumerate(config['inputs']):
            cats = []
            for name in item:
                tmp_data = data[name].clone()
                cats.append(tmp_data)
            ret[f'{he_pfs[he_id]}input'] = torch.cat(cats, dim=cat_axis)
            # ret[f'{he_pfs[he_id]}input_cats'] = cats

    add_encoder_inputs(config['history_encoders'])
    # for k in data.keys():
    #     if k in config['input_buffer']:
    #         ret[k] = data[k]
    #     elif 'sc_layers' in k:
    #         ret[k] = data[k]
    ret.update(data)
    return ret


def resize(x, scale_factor, mode='bilinear'):
    if scale_factor != 1:
        return F.interpolate(x, scale_factor=scale_factor, mode=mode)
    return x


def resize_by_resolution(x, size, mode='bilinear'):
    return F.interpolate(x, size=size, mode=mode)


def set_fake(buffer):
    zero_buffer = torch.zeros_like(buffer, dtype=buffer.dtype, device=buffer.device)
    return zero_buffer


def get_moflow(rmv: torch.Tensor | None, rmv_res: torch.Tensor | None, res_scale=1.0):
    if rmv is None:
        assert rmv_res is not None, f"smv_res should be not None: {rmv_res is not None}"
        return rmv_res
    if rmv_res is not None:
        residual = get_resized_mv(rmv_res, scale=res_scale)
        tmv0 = get_resized_mv(rmv) + residual
        return tmv0
    assert False, f"rmv and tmv0 should not both be None, rmv is {type(rmv)}, rmv_res is {type(rmv_res)}"

def get_resized_mv(mv, scale=1.0) -> torch.Tensor:
    if scale == 1.0:
        return mv
    else:
        return resize(mv, scale)


# def create_d2e_ru_encoders(encoder_config, decoder_config):
#     ret = nn.ModuleList([])
#     num_dec = len(decoder_config)
#     num_ru = num_dec - 1
#     ''' num_dec - 1: we dont recurent encoding at last decoder'''
#     for i in range(num_ru):
#         in_channel = int(decoder_config[i][-1])
#         out_channel = int(encoder_config[-(i + 1) - 1][-1])
#         # index of feature
#         re = RecurrentEncoder(in_channel, out_channel)
#         ret.append(re)
#     return ret


class ConvLSTMCellV6Wrapper(nn.Module):
    def __init__(self, in_channel, gbuffer_channel, out_channel, ks, norm=False):
        super().__init__()
        self.conv = ConvLSTMCellV6(in_channel + gbuffer_channel, out_channel, kernel_size=ks)
        self.out_channel = self.conv.out_channel
        self.gbuffer_channel = gbuffer_channel
        self.enable_norm = norm
        self.layer_norm = None

    def forward(self, input_tensor, cur_state):
        ret = self.conv(input_tensor, cur_state)
        if self.enable_norm:
            if self.layer_norm is None:
                self.layer_norm = torch.nn.LayerNorm(
                    ret.shape[1:], device=ret.device, dtype=ret.dtype, elementwise_affine=False)
            ret = self.layer_norm(ret)
        return ret


def create_d2e_ru_blocks(sc_cfg, g_cfg, shade_decoder_cfg, norm=False) -> nn.ModuleList:
    num_dec = len(shade_decoder_cfg['struct']['decoder'])
    num_ru = num_dec - 1
    ret_recurrent_blocks = nn.ModuleList([])
    for i in range(num_ru):
        ''' in_channel=24, gbuffer_channel=24'''
        # in_channel = sc_cfg.get('skip_layer_split', [item[-1] for item in sc_cfg['struct']['encoder']])[i]
        in_channel = sc_cfg.get('recurrent_d2e_channel', sc_cfg.get(
            'skip_layer_split', [item[-1] for item in sc_cfg['struct']['encoder']]))[i]
        gbuffer_channel = g_cfg.get('skip_layer_split', [item[-1]
                                    for item in g_cfg['struct']['encoder']])[i] if g_cfg is not None else 0
        out_channel = sc_cfg.get('recurrent_d2e_channel', sc_cfg.get(
            'skip_layer_split', [item[-1] for item in sc_cfg['struct']['encoder']]))[i]
        ks = shade_decoder_cfg['recurrent_d2e_kernel_size']

        if shade_decoder_cfg['recurrent_d2e_cell'] == 'lstm':
            ru = ConvLSTMCellV6Wrapper(in_channel, gbuffer_channel, out_channel, ks, norm=norm)
        else:
            assert False
        ret_recurrent_blocks.append(ru)
    return ret_recurrent_blocks


hidden_cache = {}


def initialize_zero_recurrent_encoding(data, ret, key_name, net: MRFFNet, channels=[]):
    sc_layers_key = f"{key_name}sc_layers_{{}}"
    # output_key = f"{key_name}sc_output_" + str(data['scene_color'].shape)
    device = data['merged_motion_vector_0'].device
    dtype = data['merged_motion_vector_0'].dtype
    bs = data['merged_motion_vector_0'].shape[0]
    if sc_layers_key.format(1) not in hidden_cache.keys():
        tmp_shape = list(data['merged_motion_vector_0'].shape)
        tmp_shape[2] = tmp_shape[2] // 2
        tmp_shape[3] = tmp_shape[3] // 2

    sc_layers = [torch.tensor(0) for _ in range(len(channels))]
    for i in range(len(channels)):
        layer_id = i+1
        layer_name = sc_layers_key.format(layer_id)
        cache_name = f"{device}_{dtype}_{bs}_" + layer_name
        if cache_name in hidden_cache.keys():
            sc_layers[i] = hidden_cache[cache_name]
        else:
            h_channel = channels[i]
            ''' h_channel = channels[-i-1] '''
            layer = torch.zeros(
                [tmp_shape[0], h_channel, tmp_shape[2], tmp_shape[3]], device=device, dtype=dtype)
            sc_layers[i] = hidden_cache[cache_name] = layer
            tmp_shape[2] = tmp_shape[2] // 2
            tmp_shape[3] = tmp_shape[3] // 2
        if not net.enable_timing:
            ret[layer_name] = sc_layers[i]
    return sc_layers


def encoder_op(encoder, input_dict, pf, enable_input_block=True):
    ret = {}
    tmp_output = encoder(input_dict)
    if enable_input_block:
        tmp_output['sc_layers'] = tmp_output['sc_layers'][1:]
    prefix_output = {}
    filter_list = ['sc_layers', 'output']
    for k in tmp_output.keys():
        if k not in filter_list:
            continue
        ''' add corresponding prefix to each output of he_encoder '''
        prefix_output[pf + k] = tmp_output[k]
    ret.update(prefix_output)
    return ret


def recurrent_d2e_pre_stage(he_id, d2e_he_id, data, ret, d2e_sc_layers, he_sc_layers, g_sc_layers, pf, recurrent_blocks: nn.ModuleList, net: MRFFNet, enable_gbuffer):
    num_ru = len(recurrent_blocks)
    if data['recurrent_d2e_he_id'] < 0:
        d2e_sc_layers[he_id] = initialize_zero_recurrent_encoding(data, ret, f"recurrent_{he_id}_{pf}d2e_", net,
                                                            channels=[recurrent_blocks[ru_ind].out_channel for ru_ind in range(num_ru)])

        if enable_gbuffer:
            g_sc_layers[he_id] = initialize_zero_recurrent_encoding(data, ret, f"recurrent_{he_id}_{net.ge_pf}", net,
                                                             channels=[recurrent_blocks[ru_ind].gbuffer_channel for ru_ind in range(num_ru)])
        if not net.enable_timing:
            ret[f'debug_recurrent_{he_id}_{pf}d2e_zero_encoding'] = True
    else:
        assert f"history_{he_id}_{pf}d2e_sc_layers_{1}" in data.keys(
        ), f"cur_data_idnex={data['cur_data_index']}, d2e_he_id={d2e_he_id}, history_{he_id}_{pf}d2e_sc_layers_1, {list(data.keys())}"
        assert f"history_{he_id}_{net.ge_pf}sc_layers_{1}" in data.keys(
        ), f"cur_data_idnex={data['cur_data_index']}, d2e_he_id={d2e_he_id}, history_{he_id}_{net.ge_pf}sc_layers_1, {list(data.keys())}"
        num_dec = net.num_shade_decoder_layer
        for layer_id in range(1, num_dec):
            d2e_sc_layers[he_id][layer_id-1] = data[f'history_{he_id}_{pf}d2e_sc_layers_{layer_id}']
            if enable_gbuffer:
                g_sc_layers[he_id][layer_id-1] = data[f"history_{he_id}_{net.ge_pf}sc_layers_{layer_id}"]
        if not net.enable_timing:
            ret[f'debug_recurrent_{he_id}_{pf}d2e_reuse_encoding'] = True
            for layer_id in range(1, num_dec):
                ret[f'recurrent_{he_id}_{pf}d2e_sc_layers_{layer_id}'] = d2e_sc_layers[he_id][layer_id-1]
                if enable_gbuffer:
                    ret[f"recurrent_{he_id}_{net.ge_pf}sc_layers_{layer_id}"] = g_sc_layers[he_id][layer_id-1]

def recurrent_d2e_block_stage(block_id, he_id, data, ret, sc_layers, d2e_sc_layer, d2e_g_sc_layer,
                              net: MRFFNet, recurrent_blocks: nn.ModuleList, pf, he_pfs, enable_gbuffer):
    layer_id = block_id + 1
    target_he_id = he_id
    # log.debug(str(recurrent_blocks[block_id]))
    # log.debug(dict_to_string([d2e_sc_layer, d2e_g_sc_layer, sc_layers[block_id]], mmm=True))
    sc_layers[block_id] = recurrent_blocks[block_id](
        [d2e_sc_layer] + ([d2e_g_sc_layer] if enable_gbuffer else []),
        sc_layers[block_id]
    )
    if not net.enable_timing:
        ret[f'{he_pfs[target_he_id]}sc_layers_{layer_id}'] = sc_layers[block_id]
        ret[f'debug_{he_pfs[target_he_id]}sc_layers_{layer_id}_{pf}d2e_recurrency'] = True


def recurrent_d2e_stage(he_id, data, ret, sc_layers, d2e_sc_layers, d2e_g_sc_layers, net: MRFFNet, recurrent_blocks: nn.ModuleList, pf, he_pfs, enable_gbuffer):
    num_ru = len(recurrent_blocks)
    for block_id in range(num_ru):
        if enable_gbuffer:
            recurrent_d2e_block_stage(block_id, he_id, data, ret,
                                        sc_layers[he_id], d2e_sc_layers[he_id][block_id], d2e_g_sc_layers[he_id][block_id],
                                        net, recurrent_blocks, pf, he_pfs, enable_gbuffer)
        else:
            recurrent_d2e_block_stage(block_id, he_id, data, ret,
                                        sc_layers[he_id], d2e_sc_layers[he_id][block_id], None,
                                        net, recurrent_blocks, pf, he_pfs, enable_gbuffer)


class MRFFNet(NetBase):
    class_name = "MRFFNet"
    cnt_instance = 0

    def __init__(self, config):
        add_metaname(self, MRFFNet)
        super().__init__(config)
        self.layer_flow_channel = 2
        self.layer_mask_channel = 1
        self.config = config
        self.gt_alias_name = self.config.gt_alias
        self.method = self.config.method
        self.arch = self.config.arch
        self.num_history_encoder = self.config.history_encoders['num']
        self.config.shade_decoder = self.config[f'shade_decoder__{self.method}']
        self.num_shade_decoder_layer = len(self.config.shade_decoder['struct']['decoder'])

        ''' feature configuration '''

        self.enable_demodulate = "demodulate" in self.config.feature
        self.enable_input_block = 'input_block' in self.config.feature
        self.enable_output_block = 'output_block' in self.config.feature

        self.enable_st = 'st' in self.config.feature

        self.enable_st_feature_warp = "st_feature_warp" in self.config.feature
        self.enable_st_lmv_res = "st_lmv_res" in self.config.feature
        self.enable_st_lmv_res_cat = "st_lmv_res_cat" in self.config.feature
        if self.enable_st_lmv_res_cat:
            assert self.enable_st_lmv_res_cat
        self.enable_his_st_lmv_res = "his_st_lmv_res" in self.config.feature
        self.enable_his_st_lmv_res_cat = "his_st_lmv_res_cat" in self.config.feature
        self.enable_st_initial_by_rmv = 'st_initial_by_rmv' in self.config.feature
        self.enable_initial_by_rmv = 'initial_by_rmv' in self.config.feature

        if self.enable_his_st_lmv_res_cat:
            assert self.enable_his_st_lmv_res
        if self.enable_his_st_lmv_res:
            assert self.enable_st_lmv_res

        # if self.enable_rmv_initial_layer_skip:
        #     self.enable_rmv_initial_layer_skip_at = self.config['feature_config']['rmv_initial_layer_skip_at']
        self.enable_flow_residual_scale = "flow_residual_scale" in self.config.feature
        if self.enable_flow_residual_scale:
            self.flow_residual_scale = self.config.feature_config['flow_residual_scale']
        self.enable_lmv_res = "lmv_res" in self.config.feature
        self.enable_lmv_res_cat = "lmv_res_cat" in self.config.feature
        if self.enable_lmv_res_cat:
            assert self.enable_lmv_res
        self.enable_his_lmv_res = "his_lmv_res" in self.config.feature
        self.enable_his_lmv_res_cat = "his_lmv_res_cat" in self.config.feature
        if self.enable_his_lmv_res_cat:
            assert self.enable_his_lmv_res
        if self.enable_his_lmv_res:
            self.enable_lmv_res
        self.enable_feature_warp = "feature_warp" in self.config.feature
        if self.enable_feature_warp:
            assert self.enable_his_lmv_res
        self.enable_tmv_skip_conn = "tmv_skip_conn" in self.config.feature

        self.enable_rmv_fuse = "rmv_fuse" in self.config.feature
        if self.enable_rmv_fuse:
            self.rmv_fuse_id = self.config.feature_config['rmv_fuse_id']
        if self.enable_rmv_fuse:
            assert self.enable_lmv_res
            if self.enable_feature_warp:
                assert self.enable_his_lmv_res

        self.enable_st_rmv_fuse = "st_rmv_fuse" in self.config.feature
        if self.enable_st_rmv_fuse:
            self.st_rmv_fuse_id = self.config.feature_config['st_rmv_fuse_id']
        if self.enable_st_rmv_fuse:
            assert self.enable_st_lmv_res
            if self.enable_st_feature_warp:
                assert self.enable_st_lmv_res
        self.enable_single_encoding_split_st = "single_encoding_split_st" in self.config.feature
        if self.enable_st_feature_warp:
            assert (self.enable_single_encoding_split_st) and (self.enable_his_st_lmv_res)

        self.enable_recurrent_d2e = "recurrent_d2e" in self.config.feature
        self.enable_recurrent_d2e_layer_norm = "recurrent_d2e_layer_norm" in self.config.feature
        ''' recurrent_d2e_st should be disabled, just for ablation test '''
        self.enable_recurrent_d2e_decoding_split = "recurrent_d2e_decoding_split" in self.config.feature
        self.enable_recurrent_d2e_init_feature_he0 = "recurrent_d2e_init_feature_he0" in self.config.feature
        if self.enable_recurrent_d2e_decoding_split:
            assert self.enable_st_feature_warp

        self.enable_zero_flow_loss = 'zero_flow' in self.config.loss
        self.enable_debug_output_clamp = 'output_clamp' in self.config.debug
        self.enable_debug_fake_feature_warp = 'fake_feature_warp' in self.config.debug
        self.enable_debug_pred_residual = 'pred_residual' in self.config.debug
        self.enable_timing = False

        self.feature_warp_mode = self.config.config.get('feature_warp_mode', 'bilinear')
        self.rmv_downsample_mode = self.config.config.get('rmv_downsample_mode', 'nearest')
        ''' lmv_warp_mode is for rmv warping by lmv '''
        self.lmv_warp_mode = self.config.config.get('lmv_warp_mode', 'nearest')
        ''' tmv_warp_mode is for image warping '''
        self.tmv_warp_mode = self.config.config.get('tmv_warp_mode', 'bilinear')

        ''' output channel configuration '''
        if not self.enable_input_block:
            del self.config.scene_color_encoder['struct']['input']
            del self.config.st_color_encoder['struct']['input']
            del self.config.gbuffer_encoder['struct']['input']

        ''' output channel configuration '''
        if not self.enable_output_block:
            del self.config.shade_decoder['struct']['output']
            out_channel = int(self.enable_lmv_res) + \
                int(self.enable_st_lmv_res)
            self.config.shade_decoder['out_channel'] = out_channel * 2

        ''' network name configuration '''
        self.ge_pf = self.config.gbuffer_encoder['output_prefix']
        # self.sce_pf = self.config['scene_color_encoder_output_prefix']
        self.he_pfs = [item for item in self.config.history_encoders['output_prefixs']]
        self.he_st_pfs = [item for item in self.config.history_st_encoders['output_prefixs']]

        self.he_mv_name = self.config.history_encoders['mv_name']
        self.he_st_mv_name = self.config.history_encoders['st_mv_name']
        self.he_ids = self.config.history_encoders['history_id']

        ''' network import '''
        from models.mfrrnet.archs import ShadeNetDecoder, ShadeNetEncoder, DecoderBlock

        if self.config.gbuffer_encoder['class'] == 'ShadeNetEncoder':
            self.gbuffer_encoder = ShadeNetEncoder(self.config.gbuffer_encoder)
        else:
            raise Exception('Wrong class name {} for ShadeNetEncoder!'.format(self.config.gbuffer_encoder['class']))
        # log.debug(dict_to_string([self.config['gbuffer_encoder'], self.gbuffer_encoder.in_channel]))

        if self.config.scene_color_encoder['class'] == 'ShadeNetEncoder':
            self.config.scene_color_encoder['gbuffer_channel'] = [channels[-1]
                                                                     for channels in self.config.gbuffer_encoder['struct']['encoder']]
            self.scene_color_encoder = ShadeNetEncoder(self.config.scene_color_encoder)
        else:
            raise Exception('Wrong class name {} for ShadeNetEncoder!'.format(self.config.scene_color_encoder['class']))
        
        # log.debug(dict_to_string([self.config['scene_color_encoder'], self.scene_color_encoder.in_channel]))

        ''' accumulate concat channel of history_encoders '''

        skip_first_layer = self.enable_input_block
        encoders_skip_channel = copy.deepcopy(self.config.gbuffer_encoder['encoders_skip_channel'][skip_first_layer:])
        # log.debug(encoders_skip_channel)
        for i in range(len(encoders_skip_channel)):
            extra_channel = 0
            ''' layer_id = i+1 '''
            layer_id = i+1
            ''' cat lmv_res, tmv from i layer to i-1 layer '''
            if i < len(encoders_skip_channel) - 1:
                rmv_fused = self.enable_rmv_fuse and layer_id < self.rmv_fuse_id
                st_rmv_fused = self.enable_st_rmv_fuse and layer_id < self.st_rmv_fuse_id
                if self.enable_tmv_skip_conn:
                    extra_channel = (int(self.enable_his_lmv_res) +
                                     int(self.enable_his_st_lmv_res)) * (self.num_history_encoder-1)
                    # log.debug(f'layer={layer_id}, extra_cat_channel={extra_channel}, rmv_fused:{rmv_fused}, st_rmv_fuse:{st_rmv_fused}')
                    extra_channel += (int(self.enable_lmv_res) +
                                      int(self.enable_st_lmv_res))
                    # log.debug(f'layer={layer_id}, extra_cat_channel={extra_channel}, rmv_fused:{rmv_fused}, st_rmv_fuse:{st_rmv_fused}')
                extra_channel -= (int(not self.enable_his_lmv_res_cat and self.enable_his_lmv_res)
                                  + int(not self.enable_his_st_lmv_res_cat and self.enable_his_st_lmv_res)) \
                    * (self.num_history_encoder-1)
                # log.debug(f'layer={layer_id}, extra_cat_channel={extra_channel}, rmv_fused:{rmv_fused}, st_rmv_fuse:{st_rmv_fused}')
                extra_channel -= (int(not self.enable_lmv_res_cat and self.enable_lmv_res)
                                  + int(not self.enable_st_lmv_res_cat and self.enable_st_lmv_res))
                # log.debug(f'layer={layer_id}, extra_cat_channel={extra_channel}, rmv_fused:{rmv_fused}, st_rmv_fuse:{st_rmv_fused}')
            extra_channel *= self.layer_flow_channel
            encoders_skip_channel[i] += extra_channel
            encoders_skip_channel[i] += (self.config.scene_color_encoder['encoders_skip_channel'][skip_first_layer:]
                                         [i]) * self.num_history_encoder

        # log.debug(dict_to_string(encoders_skip_channel))
        ''' add extra channel to the decoder layer in each pyramid'''
        decoder_struct = self.config.shade_decoder['struct']['decoder']
        if self.config.shade_decoder['enable_extra_channel']:
            # log.debug(dict_to_string(decoder_struct))
            for i in range(len(decoder_struct)):
                extra_channel = 0
                layer_id = len(decoder_struct) - i - 1
                is_last_layer = layer_id == 0
                rmv_fused = self.enable_rmv_fuse and layer_id < self.rmv_fuse_id
                st_rmv_fused = self.enable_st_rmv_fuse and layer_id < self.st_rmv_fuse_id
                if not is_last_layer:
                    extra_channel = (int(self.enable_his_lmv_res) + int(self.enable_his_st_lmv_res)) * (self.num_history_encoder-1)
                    # log.debug(f'layer={layer_id}, extra_out_channel={extra_channel}, rmv_fused:{rmv_fused}, st_rmv_fuse:{st_rmv_fused}')
                extra_channel += int(self.enable_lmv_res) + int(self.enable_st_lmv_res)
                # log.debug(f'layer={layer_id}, extra_out_channel={extra_channel}, rmv_fused:{rmv_fused}, st_rmv_fuse:{st_rmv_fused}')
                extra_channel *= self.layer_flow_channel
                decoder_struct[i][-1] += extra_channel

            # log.debug(dict_to_string(decoder_struct))

        self.config.shade_decoder['encoders_skip_channel'] = encoders_skip_channel
        self.config.shade_decoder['num_he'] = self.num_history_encoder
        
        if self.config.shade_decoder['class'] == 'ShadeNetDecoder':
            self.shade_decoder = ShadeNetDecoder(self.config.shade_decoder, DecoderBlock)
        else:
            raise Exception('Wrong class name {} for ShadeNetDecoder!'.format(self.config['shade_decoder']))

        #     self.shade_decoder = model_to_half(self.shade_decoder)
        if self.enable_recurrent_d2e:
            # self.recurrent_encoders = create_d2e_ru_encoders(
            #     self.config['scene_color_encoder']['struct']['encoder'],
            #     self.config['shade_decoder']['struct']['decoder'])

            self.recurrent_blocks = create_d2e_ru_blocks(
                self.config['scene_color_encoder'],
                self.config['gbuffer_encoder'],
                self.config['shade_decoder'],
                norm=self.enable_recurrent_d2e_layer_norm)
        self.get_feature_list()
        mode_key_list = ['feature_warp_mode', 'rmv_downsample_mode', 'lmv_warp_mode', 'tmv_warp_mode']
        log.debug(dict_to_string({mode_key: getattr(self, mode_key) for mode_key in mode_key_list}))
        log.debug(dict_to_string(self.enabled_features, f"[{self.__class__}] enabled feature: \n", full_name=False))
        log.debug(f"[{self.__class__}] disabled feature: {self.disabled_features}")

    def get_feature_list(self):
        self.enabled_features = []
        self.disabled_features = []
        for attr in dir(self):
            if attr.startswith("enable_"):
                if getattr(self, attr):
                    self.enabled_features.append(attr.replace("enable_", ""))
                else:
                    self.disabled_features.append(attr.replace("enable_", ""))

    def get_train_output(self, ret):
        return ret

    def get_debug_output(self, ret):
        return ret

    def forward(self, data):
        export_onnx = self.config.get("export_onnx", False)
        # onnx = False
        num_he = self.num_history_encoder
        num_dec = self.num_shade_decoder_layer
        last_lmv = [torch.tensor(0) for i in range(num_he)]
        last_lmv_res = [torch.tensor(0) for i in range(num_he)]
        last_st_lmv = [torch.tensor(0) for i in range(num_he)]
        last_st_lmv_res = [torch.tensor(0) for i in range(num_he)]
        last_tmv = [torch.tensor(0) for i in range(num_he)]
        last_st_tmv = [torch.tensor(0) for i in range(num_he)]
        outputs = [torch.tensor(0) for i in range(num_he)]
        warped_outputs = [torch.tensor(0) for i in range(num_he)]
        g_output = torch.tensor(0)
        sc_layers = [[torch.tensor(0) for enc_id in range(num_dec-1)] for i in range(num_he)]
        g_sc_layers = [torch.tensor(0) for enc_id in range(num_dec-1)]
        st_sc_layers = [[torch.tensor(0) for enc_id in range(num_dec-1)] for i in range(num_he)]
        d2e_sc_layers = [[torch.tensor(0) for enc_id in range(num_dec-1)] for i in range(num_he)]
        d2e_g_sc_layers = [[torch.tensor(0) for enc_id in range(num_dec-1)] for i in range(num_he)]
        ret = {}
        if self.enable_st_lmv_res:
            for he_id in range(num_he):
                frame_id = self.he_ids[he_id]
                if f'st_merged_motion_vector_{frame_id}' not in data:
                    data[f'st_merged_motion_vector_{frame_id}'] = data[f'merged_motion_vector_{frame_id}'].clone()
        self.enable_timing = self.enable_timing or self.config.get("export_onnx", False)

        ''' 
        gbuffer encoding
        '''
        g_output, g_sc_layers = self.gbuffer_encoder(data['gbuffer_encoder_input'])
        ''' se_ + output, sc_layers  '''
        if not self.enable_timing:
            for i in range(len(g_sc_layers)):
                ret[self.ge_pf + f'sc_layers_{i+1}'] = g_sc_layers[i]
            ret[self.ge_pf + 'output'] = g_output
        ''' 
        history encoding
        '''
        if self.arch == 'v6':
            from models.mfrrnet.archs import ShadeNetEncoder

        # log.debug(dict_to_string(self.config['scene_color_encoder']))
        def encoding_stage(encoder: ShadeNetEncoder, he_pfs, pf, local_outputs, local_sc_layers):
            for he_id in range(num_he):
                local_outputs[he_id], local_sc_layers[he_id] = encoder(data[f'{he_pfs[he_id]}input'])

                ''' tmp_output: he_$NUM_ + output, sc_layers '''
                ''' remove first sc_layer because no skip-connection from input_block '''

                if not self.enable_timing:
                    prefix_output = {}
                    ''' add corresponding prefix to each output of he_encoder '''
                    for layer_id in range(1, num_dec):
                        prefix_output[he_pfs[he_id] + f'sc_layers_{layer_id}'] = local_sc_layers[he_id][layer_id-1]
                    prefix_output[he_pfs[he_id] + 'output'] = local_outputs[he_id]
                    ''' cache the sc_layers and output '''
                    ret.update(prefix_output)

                ''' duplicate the encodings (only for time testing) '''
                if self.enable_timing:
                    for tmp_he_id in range(1, num_he):
                        local_outputs[tmp_he_id] = local_outputs[he_id]
                        local_sc_layers[tmp_he_id] = local_sc_layers[he_id]
                    break

        encoding_stage(self.scene_color_encoder, self.he_pfs, "",
                       outputs, sc_layers)

        # ''' onnx return encoding '''
        # return {
        #     'output': outputs[0],
        #     'g_output': g_output[0],
        # }
        '''
        recurrent
        '''

        ''' recurrent d2e '''
        if self.enable_recurrent_d2e:
            d2e_he_id = data['recurrent_d2e_he_id'] if not self.enable_timing else 0
            ''' if d2e_he_id == -1, then use the first history encoder to recurrent the zero feature to sc_layer_0 '''
            target_he_id = d2e_he_id if d2e_he_id >= 0 else 0
            assert target_he_id == 0 or target_he_id == 1, f"target_he_id={target_he_id}"
            for he_id in range(num_he):
                if he_id == target_he_id:
                    recurrent_d2e_pre_stage(he_id, d2e_he_id, data, ret, d2e_sc_layers,
                                            sc_layers, d2e_g_sc_layers,
                                            "", self.recurrent_blocks, self, True)
                    # log.debug(dict_to_string([he_id, sc_layers, d2e_sc_layers, d2e_g_sc_layers]))
                    recurrent_d2e_stage(he_id, data, ret,
                                        sc_layers, d2e_sc_layers, d2e_g_sc_layers,
                                        self, self.recurrent_blocks,
                                        "", self.he_pfs, True)
                    ''' recurrent_d2e_st should be disabled, just for ablation test '''
                    if self.enable_timing:
                        break

        '''
        decoding
        '''
        # def decoding_stage():
        ''' start of decoding stage '''

        ''' start decoding loop '''
        ''' dec_id = [0, 1, ..., num_dec-1(last layer of decoder), num_dec(output_block)] '''
        for dec_id in range(num_dec + 1):
            # if dec_id > 2:
            #     break
            offset = 0
            st_offset = 0
            if dec_id == num_dec:
                layer_id = 0
            else:
                layer_id = num_dec - dec_id
            concator = TensorConcator(f'decoding_cat_{dec_id}', use_cache=export_onnx)
            if dec_id == 0:
                def get_lowest_lmv(lmv_pf, feature_warp, local_last_tmv, local_last_lmv, local_last_lmv_res, is_rmv_fuse):
                    for he_id in range(num_he):
                        ''' TODO: make lowest lmv learanble by extract some channels to smv from lowest encoding '''
                        # final_mv = resize(data[self.he_mv_name.format(
                        #     self.he_ids[he_id])], 1 / (2**num_dec), mode=self.rmv_downsample_mode)
                        rmv = data[self.he_mv_name.format(self.he_ids[he_id])]
                        final_mv = resize_by_resolution(rmv, outputs[he_id].shape[-2:], mode=self.rmv_downsample_mode)

                        local_last_tmv[he_id] = final_mv
                        ''' warp the lowest encoding only by rendered motion vector '''
                        if not self.enable_timing:
                            ret[f"pred_layer_{layer_id}_{lmv_pf}tmv_{he_id}"] = final_mv
                        if is_rmv_fuse:
                            local_last_lmv_res[he_id] = final_mv
                            if not self.enable_timing:
                                assert self.enable_lmv_res
                                ret[f"pred_layer_{layer_id}_{lmv_pf}lmv_res_{he_id}"] = final_mv
                        else:
                            lmv = torch.zeros_like(final_mv)
                            local_last_lmv[he_id] = lmv
                            if not self.enable_timing:
                                ret[f"pred_layer_{layer_id}_{lmv_pf}lmv_{he_id}"] = ret[f"pred_layer_{layer_id}_{lmv_pf}lmv_res_{he_id}"] = lmv

                        if not feature_warp:
                            break

                get_lowest_lmv('', self.enable_feature_warp,
                               last_tmv, last_lmv, last_lmv_res,
                               self.enable_rmv_fuse and self.rmv_fuse_id == layer_id)

                if self.enable_st:
                    get_lowest_lmv('st_', self.enable_st_feature_warp,
                                   last_st_tmv, last_st_lmv, last_st_lmv_res,
                                   self.enable_st_rmv_fuse and self.st_rmv_fuse_id == layer_id)

                def cat_lowest_feature(he_pfs, local_warped_output, local_output, local_last_tmv, lmv_pf, enable_feature_warp):
                    mode = self.config['config'].get('feature_warp_mode', 'bilinear')
                    for he_id in range(num_he):
                        ''' TODO: make lowest lmv learnable by extract some channels to smv from lowest encoding '''
                        if enable_feature_warp:
                            local_warped_output[he_id] = warp(local_output[he_id], local_last_tmv[he_id],
                                                              padding_mode="border", mode=mode, flow_type="mv")
                            if not self.enable_timing:
                                he_pf = he_pfs[he_id]
                                ret[f'{he_pf}_warped_output'] = local_warped_output[he_id]
                            concator.add(local_warped_output[he_id])
                        else:
                            concator.add(local_output[he_id])

                cat_lowest_feature(self.he_pfs, warped_outputs, outputs, last_tmv,
                                   '', self.enable_feature_warp and not self.enable_debug_fake_feature_warp)

                concator.add(g_output)
                tmp_tensor = concator.get()
                # log.debug(dict_to_string([offset, layer_id, tmp_tensor]))
                ''' end of if dec_id == 0 '''

                # ''' onnx return dec0 '''
                # return {
                #     'output': tmp_tensor,
                # }
            elif dec_id > 0:
                no_st_feature_c = int(tmp_tensor.shape[1] * 0.5)
                ratio = 2 ** (num_dec - dec_id)
                ''' skip-connection will perform when i>0 and i<num_dec '''
                skip_conn = dec_id < num_dec
                ''' warp_last_dec: calculate lmv at output_block '''
                warp_last_dec = dec_id == num_dec
                ''' calculate lmv for each history encoding '''
                def calc_lmv_for_each_his_encoder(he_id, he_pfs, cur_offset, mv_name, residual_from_last_layer,
                                                  local_last_lmv_res, local_last_tmv, split_offset=0,
                                                  lmv_pf="", is_enable_his_lmv_res=True,
                                                  is_his_lmv_res_cat=True,
                                                  is_enable_lmv_res=True,
                                                  is_lmv_res_cat=True,
                                                  is_enable_feature_warp=True, 
                                                  is_skip_conn=True,
                                                  is_initial_by_rmv=True, 
                                                  is_rmv_fuse=False):
                    if (is_enable_lmv_res) and (he_id == 0 or is_enable_his_lmv_res):
                        def calculate_pyramid_flow(name, in_offset, in_flow_c, last_flow, is_enable_act=False):
                            # return tmp_tensor[:,0:2], in_offset + in_flow_c
                            ''' (i==1) calc smallest flow and upper residual '''
                            flow_raw = tmp_tensor[:, split_offset+in_offset:split_offset+in_offset + in_flow_c]
                            if is_enable_act:
                                flow = self.tanh(flow_raw)
                            else:
                                flow = flow_raw
                            if self.enable_flow_residual_scale:
                                flow *= self.flow_residual_scale

                            ''' (dec_id > 1) calc larger flow '''
                            # log.debug(dict_to_string(ret))
                            if residual_from_last_layer:
                                flow = flow + resize(last_flow[he_id], 2.0)

                            last_flow[he_id] = flow
                            if not self.enable_timing:
                                name = f'{lmv_pf}{name}'
                                ret[f'pred_layer_{layer_id}_{name}_{he_id}'] = flow
                                ret[f'pred_layer_{layer_id}_{name}_{he_id}_raw'] = flow_raw

                            return flow, in_offset + in_flow_c

                        rmv = None
                        if is_initial_by_rmv:
                            if lmv_pf == "":
                                assert not (self.enable_rmv_fuse and layer_id < self.rmv_fuse_id)
                            else:
                                assert not (self.enable_st_rmv_fuse and layer_id < self.st_rmv_fuse_id), \
                                    f"self.enable_st_rmv_fuse:{self.enable_st_rmv_fuse}, layer_id:{layer_id}, self.st_rmv_fuse_id:{self.st_rmv_fuse_id}"
                            rmv = data[mv_name.format(self.he_ids[he_id])]
                            rmv = resize(rmv, scale_factor=1/ratio, mode=self.rmv_downsample_mode)

                        if is_enable_lmv_res and (he_id == 0 or is_enable_his_lmv_res):
                            lmv_res, cur_offset = calculate_pyramid_flow(
                                "lmv_res", cur_offset, self.layer_flow_channel, local_last_lmv_res)
                            if rmv is not None:
                                lmv_res = fix_the_size_with(rmv, lmv_res)
                            if he_id == 0 and is_lmv_res_cat or he_id != 0 and is_his_lmv_res_cat:
                                # if not warp_last_dec:
                                concator.add(lmv_res)
                        else:
                            lmv_res = None

                        tmv = get_moflow(rmv,  lmv_res)
                        assert tmv is not None
                        if tmv is not None:
                            final_mv = tmv
                            if self.enable_tmv_skip_conn:
                                concator.add(final_mv)
                        else:
                            final_mv = rmv

                        if not self.enable_timing:
                            ret[f"pred_layer_{layer_id}_{lmv_pf}tmv_{he_id}"] = final_mv
                            
                        ''' is_rmv_fuse = self.enable_rmv_fuse and layer_id == self.rmv_fuse_id '''
                        if is_rmv_fuse: 
                            ''' when is_rmv_fuse, lmv_res is set by tmv, acting as a residual to compute tmv[layer_id-1] '''
                            _is_lmv_res = he_id == 0 and is_enable_lmv_res or he_id > 0 and is_enable_his_lmv_res
                            local_last_lmv_res[he_id] = final_mv
                            if not self.enable_timing:
                                assert _is_lmv_res
                                ret[f"pred_layer_{layer_id}_{lmv_pf}lmv_res_{he_id}"] = final_mv

                        local_last_tmv[he_id] = final_mv
                    if warp_last_dec:
                        return False, cur_offset

                    if is_skip_conn:
                        if ((is_enable_lmv_res) and is_enable_feature_warp) \
                                and final_mv is not None and not self.enable_debug_fake_feature_warp:
                            ''' last_dec only deal with he_id=0, no need to continue the for-loop(num_he), break here. '''
                            ''' if skip_conn == False, it means we dont need to warp the feature, break here. '''
                            if not (skip_conn):
                                return False, cur_offset
                            ''' skip-connection start here, concat the warped history encoding '''
                            mode = self.config['config'].get('feature_warp_mode', 'bilinear')
                            if self.enable_st_feature_warp:
                                if self.enable_single_encoding_split_st:
                                    half_c = sc_layers[he_id][layer_id-1].shape[1] // 2
                                    # warped_sc_layer = warp(sc_layers[he_id][layer_id-1][:,:half_c], final_mv, mode=mode,
                                    #                                     padding_mode="border", flow_type="mv")
                                    if lmv_pf == "":
                                        warped_sc_layer = warp(sc_layers[he_id][layer_id-1][:, :half_c], final_mv, mode=mode,
                                                               padding_mode="border", flow_type="mv")
                                    else:
                                        warped_sc_layer = warp(sc_layers[he_id][layer_id-1][:, half_c:], final_mv, mode=mode,
                                                               padding_mode="border", flow_type="mv")
                                else:
                                    warped_sc_layer = warp(st_sc_layers[he_id][layer_id-1], final_mv, mode=mode,
                                                           padding_mode="border", flow_type="mv")

                            else:
                                warped_sc_layer = warp(sc_layers[he_id][layer_id-1], final_mv, mode=mode,
                                                       padding_mode="border", flow_type="mv")
                            # warped_sc_layers[he_id][layer_id-1]
                            if not self.enable_timing:
                                ret[f'{he_pfs[he_id]}warped_sc_layers_{layer_id}'] = warped_sc_layer

                            concator.add(warped_sc_layer)
                        else:
                            concator.add(sc_layers[he_id][layer_id-1])
                    return True, cur_offset

                lmv_fused = self.enable_rmv_fuse and layer_id < self.rmv_fuse_id
                if not self.enable_timing:
                    ret[f'debug_layer_{layer_id}_lmv_fused'] = lmv_fused
                st_lmv_fused = self.enable_st_rmv_fuse and layer_id < self.st_rmv_fuse_id
                for he_id in range(num_he):
                    residual_from_last_layer = dec_id > 1 or (dec_id == 1 and lmv_fused)
                    # rmv = resize(data[self.he_mv_name.format(self.history_id[he_id])], scale_factor=1/ratio, mode=self.rmv_downsample_mode)
                    # ret[f"pred_layer_{layer_id}_tmv_{he_id}"] = rmv
                    # if warp_last_dec:
                    #     flag = False
                    # else:
                    #     concator.add(ret[f"{self.he_pfs[he_id]}sc_layers_{layer_id}"])
                    #     flag = True
                    # log.debug(dict_to_string(tmp_tensor))
                    flag, offset = calc_lmv_for_each_his_encoder(he_id, self.he_pfs,  offset, self.he_mv_name,
                                                                 residual_from_last_layer,
                                                                 last_lmv_res, last_tmv,
                                                                 lmv_pf="", split_offset=0,
                                                                 is_enable_his_lmv_res=self.enable_his_lmv_res,
                                                                 is_his_lmv_res_cat=self.enable_his_lmv_res_cat,
                                                                 is_enable_lmv_res=self.enable_lmv_res,
                                                                 is_lmv_res_cat=self.enable_lmv_res_cat,
                                                                 is_enable_feature_warp=self.enable_feature_warp and not self.enable_debug_fake_feature_warp,
                                                                 is_skip_conn=True,
                                                                 is_initial_by_rmv=not lmv_fused and self.enable_initial_by_rmv,
                                                                 is_rmv_fuse=self.enable_rmv_fuse and layer_id == self.rmv_fuse_id)
                    # log.debug(dict_to_string(concator.skip_conn_arr))
                    if self.enable_st_lmv_res:
                        residual_from_last_layer = dec_id > 1 or (dec_id == 1 and st_lmv_fused)
                        # rmv = resize(data[self.he_mv_name.format(self.history_id[he_id])], scale_factor=1/ratio, mode=self.rmv_downsample_mode)
                        # ret[f"pred_layer_{layer_id}_st_tmv_{he_id}"] = rmv
                        flag, st_offset = calc_lmv_for_each_his_encoder(he_id, self.he_st_pfs, st_offset, self.he_st_mv_name,
                                                                        residual_from_last_layer,
                                                                        last_st_lmv_res, last_st_tmv,
                                                                        lmv_pf="st_", split_offset=no_st_feature_c,
                                                                        is_enable_his_lmv_res=self.enable_his_st_lmv_res,
                                                                        is_his_lmv_res_cat=self.enable_his_st_lmv_res_cat,
                                                                        is_enable_lmv_res=self.enable_st_lmv_res,
                                                                        is_lmv_res_cat=self.enable_st_lmv_res_cat,
                                                                        is_enable_feature_warp=self.enable_st_feature_warp and not self.enable_debug_fake_feature_warp,
                                                                        is_skip_conn=self.enable_single_encoding_split_st,
                                                                        is_initial_by_rmv=(
                                                                            not st_lmv_fused) and self.enable_st_initial_by_rmv,
                                                                        is_rmv_fuse=self.enable_rmv_fuse and layer_id == self.rmv_fuse_id)
                        # log.debug(dict_to_string(concator.skip_conn_arr))
                    if not flag:
                        break
                # log.debug(dict_to_string(concator.skip_conn_arr))
                tmp = tmp_tensor[:, offset:no_st_feature_c]
                concator.add(tmp)
                st_tmp = tmp_tensor[:, no_st_feature_c+st_offset:]
                concator.add(st_tmp)
                ''' warp_last_dec: no skip-connection in last decoder, so there's no nead to warping the feature '''
                ''' warp_last_dec: last decoder only calculates lmv for his_id == 0 '''
                ''' warp_last_dec: finish concat here in advance, and break for-loop(num_dec) here.'''
                # log.debug(dict_to_string(tmp_tensor))
                # log.debug(dict_to_string(tmp_tensor, f"@dec_id={dec_id}, num_dec={num_dec}"))

                if dec_id == num_dec:
                    ''' TODO:  feature was not concated '''
                    # if self.enable_output_block:
                    #     tmp_tensor = tmp_tensor[:, offset:]
                    # log.debug(dict_to_string(concator.skip_conn_arr))
                    break
                
                # if not skip_conn:
                #     break
                ''' skip_conn '''
                concator.add(g_sc_layers[layer_id-1])
                # log.debug(dict_to_string(skip_conn_arr, mmm=True))
                # log.debug(dict_to_string(concator.skip_conn_arr))
                tmp_tensor = concator.get()
                # if dec_id > 0:
                #     ''' onnx return decx '''
                #     return {
                #         'output': tmp_tensor,
                #     }
            ''' end of if dec_id>0 '''

            ''' i == num_dec means we are at output_block, the decoding is done here. '''
            # if dec_id == num_dec:
            #     log.debug(dict_to_string(tmp_tensor))
            #     log.debug(dict_to_string(tmp_tensor, f"@dec_id={dec_id}"))
            #     break

            ''' upscale the feature with the decoder '''
            dec_input = tmp_tensor
            
            tmp_tensor, recurrent_feature = self.shade_decoder.decoders[dec_id](dec_input, None, None)
            if not self.enable_timing:
                ret[f'layer_{layer_id}_dec_input'] = dec_input * 1.0
                ret[f'layer_{layer_id}_decoding'] = tmp_tensor * 1.0
            if layer_id - 2 >= 0:
                ''' temporary fix for 1080p '''
                tmp_tensor = fix_the_size_with(sc_layers[0][layer_id-2], tmp_tensor)
            # log.debug(dict_to_string([sc_layers, layer_id, tmp_tensor]))
            # log.debug(dict_to_string([offset, layer_id, tmp_tensor]))
            ''' compress the upscaled feature into input for recurrent feature streaming in future frames '''
            if self.enable_recurrent_d2e and dec_id < num_dec - 1:
                ''' when i == num_dec - 1 we are at last decoder where no recurrent feature streaming '''
                # ret['d2e_sc_layers'].append(self.recurrent_encoders[dec_id](tmp_tensor))
                # log.debug(dict_to_string([ret, num_dec, dec_id, layer_id]))
                ''' decoder will upscale the feature, so "layer_id-1". idx start from 0(layer_id=2), so "layer_id-1-1" '''
                if not self.enable_timing:
                    recurrent_channel: int = self.recurrent_blocks[(layer_id-1)-1].out_channel  # type: ignore
                    no_st_feature_c = tmp_tensor.shape[1] // 2
                    if self.enable_recurrent_d2e_decoding_split:
                        ret[f'd2e_sc_layers_{layer_id-1}'] = torch.cat([tmp_tensor[:, :recurrent_channel//2],
                                                                       tmp_tensor[:, no_st_feature_c:no_st_feature_c+recurrent_channel//2]],
                                                                       dim=1)
                    else:
                        ret[f'd2e_sc_layers_{layer_id-1}'] = tmp_tensor[:, :recurrent_channel]
                    ''' recurrent_d2e_st should be disabled, just for ablation test '''

        ''' end of decoding loop '''
        ''' output_block at full-size '''
        # return self.return_output_by_type(ret)
        if self.shade_decoder.output_block is not None:
            tmp_tensor = self.shade_decoder.output_block(tmp_tensor)
        else:
            tmp_tensor = tmp_tensor[:, offset:]

        ''' re-arrange the output tensor into meaningful dict-like data '''
        # tmp_output = self.shade_decoder.split_output_to_dict(
        #     tmp_tensor, prefix=self.shade_decoder.output_prefix)
        if self.enable_output_block:
            offset = 0
        residual_output = tmp_tensor[:, offset:offset+3]
        if not self.enable_timing:
            ret['residual_output'] = residual_output
        offset += 3
        assert offset == tmp_tensor.shape[1], f'offset={offset}, tmp_tensor.shape={tmp_tensor.shape}'

        # ret.update(tmp_output)
        # for k in tmp_output.keys():
        #     ret[k] = tmp_output[k]
        # ret['tmp_output'] = tmp_tensor
        # return

        if self.enable_output_block:
            if self.enable_st:
                assert self.enable_st_lmv_res
        ''' end of decoding stage '''

        # decoding_stage()
        # return self.return_output_by_type(ret)

        if self.enable_timing:
            mv_ow1 = last_tmv[0]
            mv_ow2 = None
            mv_st_ow1 = last_st_tmv[0]
            mv_st_ow2 = None
        else:
            mv_ow1 = ret[f'pred_layer_{0}_tmv_{0}']
            mv_st_ow1 = ret[f'pred_layer_{0}_st_tmv_{0}'] if self.enable_st_lmv_res else None

        if self.enable_timing:
            ''' onnx return '''
            if self.enable_st:
                return {
                    'residual_output': residual_output,
                    'pred_scene_light_no_st': warp(data['history_scene_light_no_st_0'], mv_ow1, mode='nearest'),
                    'pred_st_color': warp(data['history_st_color_0'], mv_st_ow1, mode='nearest'),
                    'pred_st_alpha': warp(data['history_st_alpha_0'], mv_st_ow1, mode='nearest'),
                    'pred_sky_color': warp(data['history_sky_color_0'], mv_ow1, mode='nearest'),
                }
            else:
                return {
                    'residual_output': residual_output,
                    'pred_scene_light_no_st': warp(data['history_scene_light_no_st_0'], mv_ow1, mode='nearest'),
                }
            return self.return_output_by_type(ret)

        unwarped_residual_item = data[self.config["residual_item"]]
        if self.enable_demodulate:
            dmdl_color = fix_dmdl_color_zero_value(data['dmdl_color'])
            unwarped_residual_item = create_de_color(inv_tonemap_func(unwarped_residual_item, use_global_settings=True),
                                                        data['history_dmdl_color_0'], fix=True, max_luminance=True)
            unwarped_residual_item = tonemap_func(unwarped_residual_item, use_global_settings=True)

        mode = self.config['config'].get('output_warp_mode', 'bilinear')
        warped_residual_item = warp(
            unwarped_residual_item, mv_ow1, mode=mode, padding_mode="border", flow_type="mv")
        ret['pred_recurrent_tmv'] = mv_ow1

        ret['warped_residual_item_log'] = warped_residual_item

        if self.enable_demodulate:
            ret['pred_warped_scene_color_no_st'] = inv_tonemap_func(warped_residual_item, use_global_settings=True) * dmdl_color
        else:
            ret['pred_warped_scene_color_no_st'] = inv_tonemap_func(warped_residual_item, use_global_settings=True)
            
        ret['pred_warped_scene_color_no_st'] = ret['pred_warped_scene_color_no_st'] * \
            (1-data['skybox_mask']) + data['skybox_mask'] * inv_tonemap_func(data['sky_color'], use_global_settings=True)

        # return self.return_output_by_type(ret)
        if self.method == "residual":
            ''' FIXME: temporary solution for grad nan '''
            residual_item_max = warped_residual_item.max()
            assert residual_output.shape[1] == 3, "pred_residual's output channel size must be 3, it's {} now".format(
                residual_output.shape[1])
            pred_scene_light_no_st = residual_output + warped_residual_item
            pred_scene_light_no_st = torch.clamp(pred_scene_light_no_st, min=0)
            pred_scene_light_no_st = torch.clamp(pred_scene_light_no_st, max=residual_item_max)
            pred_scene_light_no_st_log = pred_scene_light_no_st * 1.0
            pred_scene_light_no_st = inv_tonemap_func(pred_scene_light_no_st, use_global_settings=True)
            ''' pred_scene_color_no_st_no_sky '''
            if self.enable_demodulate:
                pred_scene_color_no_st = pred_scene_light_no_st * dmdl_color
            else:
                pred_scene_color_no_st = pred_scene_light_no_st

        if not self.enable_timing:
            ret['pred_scene_light_no_st'] = pred_scene_light_no_st
            ret['pred_scene_light_no_st_log'] = pred_scene_light_no_st_log

        if self.enable_st:
            mode = self.config['config'].get('st_output_warp_mode', 'bilinear')
            pred_sky_color = torch.tensor(0)
            pred_st_color = torch.tensor(0)
            pred_st_alpha = torch.tensor(0)
            color_names = self.config['st_color_names']
            history_names = self.config['st_history_names']
            for ind, name in enumerate(color_names):
                warped_color = None
                if name.startswith("st_"):
                    warped_color = warp(
                        data[history_names[ind]], mv_st_ow1, mode=mode, padding_mode="border")
                    # log.debug(f"warp {name} by mv_ow2")
                    if not self.enable_timing:
                        ret['pred_recurrent_st_tmv'] = mv_st_ow1
                elif name.startswith("sky_"):
                    warped_color = warp(
                        data[history_names[ind]], mv_ow1, mode=mode, padding_mode="border")
                else:
                    assert False
                if not name.endswith('_color'):
                    warped_color = torch.clamp(warped_color, 0, 1)
                else:
                    ret[f'pred_{name}_log'] = warped_color
                    warped_color = inv_tonemap_func(warped_color, use_global_settings=True)
                ret[f'pred_{name}'] = warped_color
                # if name == 'st_color':
                #     pred_st_color = warped_color
                # elif name == 'sky_color':
                #     pred_sky_color = warped_color
                # elif name == 'st_alpha':
                #     pred_st_alpha = warped_color
                if not self.enable_timing:
                    ret[f'pred_{name}'] = warped_color

        ret['residual_item'] = inv_tonemap_func(warped_residual_item, use_global_settings=True)
        ret['pred_comp_color_before_sky_st'] = pred_scene_color_no_st * 1.0

        if self.enable_st:
            ret['pred_comp_color_sky'] = ret['pred_comp_color_before_sky_st'] * \
                (1 - data['skybox_mask']) + data['skybox_mask'] * ret['pred_sky_color']
            ret['pred_scene_color_no_st'] = ret['pred_comp_color_sky'] * 1.0
            ret['pred_comp_color_sky_st'] = ret['pred_comp_color_sky'] * \
                ret['pred_st_alpha'] + ret['pred_st_color']
            ret['pred_scene_color_st_blend'] = inv_tonemap_func(data['scene_color_no_st'], use_global_settings=True) * \
                ret['pred_st_alpha'] + ret['pred_st_color']

            ret['pred'] = ret['pred_comp_color_sky_st']
        else:
            ret['pred_comp_color_sky'] = ret['pred_comp_color_before_sky_st'] * \
                (1 - data['skybox_mask']) + data['skybox_mask'] * inv_tonemap_func(data['sky_color'], use_global_settings=True)
            ret['pred_scene_color_no_st'] = ret['pred_comp_color_sky'] * 1.0
            ret['pred'] = ret['pred_scene_color_no_st'] * 1.0

        ret['pred_log'] = tonemap_func(ret['pred'] * 1.0, use_global_settings=True)

        return self.return_output_by_type(ret)

    def get_inference_output(self, ret):
        ret_key_list = list(ret.keys())
        for k in ret_key_list:
            if k.startswith('debug_'):
                del ret[k]
        return ret

    def calc_loss(self, data, ret, trainer_config=None):
        assert trainer_config is not None, "trainer_config must be provided."
        num_he = int(self.num_history_encoder)  # type: ignore
        num_dec = int(self.num_shade_decoder_layer)  # type: ignore
        # if self.enable_st:
        #     ret['st_color_nonzero_mean'] = float(torch.where(torch.sum(data['st_color'], dim=1, keepdim=True) > 0,
        #                                        torch.ones_like(data['st_alpha']),
        #                                        torch.zeros_like(data['st_alpha'])).mean() + 1e-4)
        #     trainer_config['loss']['train_loss']['st_color_loss']['scale'] = min(2.0, 1.0 / ret['st_color_nonzero_mean'])
        #     trainer_config['loss']['train_loss']['st_alpha_loss']['scale'] = min(2.0, 1.0 / ret['st_color_nonzero_mean'])

        if self.config["loss_config"]["zero_flow_mask"]:
            mask = data['continuity_mask']
        else:
            mask = torch.ones_like(data['skybox_mask'])
        st_mask = torch.ones_like(data['skybox_mask'])

        ''' flow regularization loss '''
        def add_zero_loss(name, loss_name, loss_pf, loss_fn=zero_flow_l1_loss, scale=1.0, lmv_pf=""):
            cur_mask = mask
            if lmv_pf == "st_":
                cur_mask = st_mask
            name = 'pred_' + name
            if name in ret.keys():
                ret[f'{name}_{loss_pf}'] = loss_fn([ret[name] * resize(cur_mask, scale_factor=ret[name].shape[2]/mask.shape[2]),
                                                    get_zeros_map(ret[name])]).mean() * scale
                ret[loss_name] += ret[f'{name}_{loss_pf}']

        def add_c1_loss(name, loss_name, loss_pf, loss_fn=zero_flow_l1_loss, scale=1.0, layer=1):
            name = "pred_" + name
            if not name.format(layer) in ret.keys():
                return
            if layer == 3:
                return
                # add_zero_loss(name.format(layer), loss_name, loss_pf, loss_fn=loss_fn, scale=scale)
            elif layer == 2:
                last_layer = resize(ret[name.format(layer + 1)], 2.0)
                ret[f'{name.format(layer)}_{loss_pf}'] = loss_fn(
                    [last_layer, ret[name.format(layer)] - last_layer]).mean() * scale
                ret[loss_name] += ret[f'{name.format(layer)}_{loss_pf}']
            elif layer == 1 or layer == 0:
                last_layer_0 = resize(ret[name.format(layer + 1)], 2.0)
                last_layer_1 = resize(ret[name.format(layer + 2)], 4.0)
                ret[f'{name.format(layer)}_{loss_pf}'] = loss_fn(
                    [ret[name.format(layer)] - last_layer_0, last_layer_0 - last_layer_1]).mean() * scale
                ret[loss_name] += ret[f'{name.format(layer)}_{loss_pf}']
            else:
                return

        def add_c0_loss(name, loss_name, loss_pf, loss_fn=zero_flow_l1_loss, scale=1.0, layer=1):
            name = "pred_" + name
            if not name.format(layer) in ret.keys():
                return
            if layer == 3:
                return
                # add_zero_loss(name.format(layer), loss_name, loss_pf, loss_fn=loss_fn, scale=scale)
            else:
                last_layer = resize(ret[name.format(layer + 1)], 2.0)
                ret[f'{name.format(layer)}_{loss_pf}'] = loss_fn(
                    [last_layer, ret[name.format(layer)]]).mean() * scale
                ret[loss_name] += ret[f'{name.format(layer)}_{loss_pf}']
                return

        if self.enable_zero_flow_loss:
            zero_flow_ratio = self.config["loss_config"]["zero_flow_ratio"]
            ret['zero_flow_loss'] = torch.tensor(0.0, device=ret['pred'].device)
            scale = 1
            zfl_full_res_only = self.config["loss_config"].get("zfl_full_res_only", False)
            zfl_res_fuse_id = self.config["loss_config"].get("zfl_res_fuse_id", -1)
            for he_id in range(0, num_he):
                tot = 0
                tot += int(he_id == 0 and self.enable_lmv_res or he_id > 0 and self.enable_his_lmv_res)
                tot += int(he_id == 0 and self.enable_st_lmv_res or he_id > 0 and self.enable_his_st_lmv_res)
                for i in range(1, num_dec + 1):
                    layer_id = num_dec - i
                    scale = 2 ** (num_dec - layer_id - 1)

                    def add_zero_loss_he_layer(lmv_pf="", scale_div=1.0):
                        add_zero_loss(f'layer_{layer_id}_{lmv_pf}lmv_{he_id}_raw', "zero_flow_loss", "zf_ls",
                                      loss_fn=zero_flow_l1_loss, scale=zero_flow_ratio * scale / tot, lmv_pf=lmv_pf)
                        add_zero_loss(f'layer_{layer_id}_{lmv_pf}lmv_res_{he_id}_raw', "zero_flow_loss",
                                      "zf_ls", loss_fn=zero_flow_l1_loss, scale=zero_flow_ratio * scale / tot, lmv_pf=lmv_pf)

                    if self.enable_zero_flow_loss:
                        if not (zfl_full_res_only and layer_id != 0) and ((zfl_res_fuse_id > 0 and layer_id <= zfl_res_fuse_id) or zfl_res_fuse_id < 0):
                            add_zero_loss_he_layer()
                            add_zero_loss_he_layer("st_")

ones_map_cache = {}


def get_ones_map(data, device):
    b, c, h, w = data.shape
    k = f"{b}_{c}_{h}_{w}"
    if k not in ones_map_cache.keys():
        ones_map_cache[k] = torch.ones_like(data, device=data.device)
    return ones_map_cache[k]


zeros_map_cache = {}


def get_zeros_map(data):
    b, c, h, w = data.shape
    k = f"{b}_{c}_{h}_{w}"
    if k not in zeros_map_cache.keys():
        zeros_map_cache[k] = torch.zeros_like(data, device=data.device)
    return zeros_map_cache[k]


def get_occ_mask(data):
    warp_mode = 'bilinear'
    warp_padding_mode = 'border'
    warped_gt = warp(data[f'history_scene_color_0'], data['merged_motion_vector_0'],
                     mode=warp_mode, padding_mode=warp_padding_mode)
    gt = data['scene_color']
    return torch.where(torch.abs((warped_gt) - (gt)) > 1, torch.ones_like(gt), torch.zeros_like(gt))


def get_dmdl_occ_mask(data):
    warp_mode = 'bilinear'
    warp_padding_mode = 'border'
    warped_gt = warp(data[f'history_dmdl_color_0'], data['merged_motion_vector_0'],
                     mode=warp_mode, padding_mode=warp_padding_mode)
    gt = data['dmdl_color']
    return torch.where(torch.abs((warped_gt) - (gt)) > 0.05, torch.ones_like(gt), torch.zeros_like(gt))


class MFRRNetModel(ModelBaseEXT):
    def __init__(self, config):
        if isinstance(config, dict):
            config = DictToDataclassAdapter(config).to_task_config()
        config.model.unfreeze()
        super().__init__(config)

    def get_dummy_input(self, bs=1):
        input_2d_str = []
        input_2d_str += self.config['input_buffer']
        input_2d_str += self.config['gbuffer_encoder']['input_buffer']
        input_1d_str = []
        H, W = self.dummy_input_size_h, self.dummy_input_size_w
        ret = {}
        for item in input_2d_str:
            basic_element = parse_buffer_name(item)['basic_element']
            if item.startswith('d2_'):
                tmp_tensor = torch.zeros(bs, get_2d_dim(basic_element), H // 2, W // 2)
            else:
                tmp_tensor = torch.zeros(bs, get_2d_dim(basic_element), H, W)
            ret[item] = tmp_tensor
        for item in input_1d_str:
            tmp_tensor = torch.zeros(bs, get_1d_dim(item))
            ret[item] = tmp_tensor
        ret['cur_data_index'] = 0
        ret['metadata'] = {
            'dataset_name': ['dummy_scene_name'],
            'scene_name': ['scene_name'],
            'index': [0],
        }
        ret['dummy_input'] = True
        if self.get_net().enable_recurrent_d2e:
            ret['recurrent_d2e_he_id'] = -1
        return FlexDataStruct.from_dict(ret)

    def get_augment_data(self, data):
        return history_extend(data, self.trainer_config)

    def create_model(self):
        history_encoders_config = self.config['history_encoders']
        history_st_encoders_config = self.config['history_st_encoders']
        num = history_encoders_config['num']
        history_ids = history_encoders_config['history_id']

        def initialize_history_config(config):
            config['inputs'] = []
            config['output_prefixs'] = []
            for ind in range(num):
                inputs = []
                for name in config['input_template']:
                    inputs.append(name.format(history_ids[ind]))
                config['inputs'].append(inputs)
                config['output_prefixs'].append(
                    config['output_prefix_template'].format(ind))
        initialize_history_config(history_encoders_config)
        initialize_history_config(history_st_encoders_config)
        self.net = MRFFNet(self.config)

    def calc_loss(self, data, ret):
        self.get_net().calc_loss(data, ret, trainer_config=self.trainer_config)  # type: ignore

    def calc_preprocess_input(self, data):
        return data_to_input(data, self.config)

    def inference_for_timing(self, data):
        self.set_eval()
        self.get_net().enable_timing = True  # type: ignore
        with torch.no_grad():
            output = self.net(data)
        self.get_net().enable_timing = False  # type: ignore
        return output

    def update(self, data, mode: ForwardMode=ForwardMode.train):
        buffer_config = self.trainer_config['buffer_config']
        max_luminance = buffer_config.get('max_luminance', -1)
        mu = buffer_config.get('mu', 8.0)
        if max_luminance > 0:
            DatasetGlobalConfig.max_luminance = max_luminance
        DatasetGlobalConfig.log_tonemapper__mu = mu

        net: MRFFNet = self.get_net()

        net_input = self.calc_preprocess_input(data)
        output = self.forward(net_input)

        if not net.enable_timing:
            data['scene_color_no_st_reference'] = data['scene_color_no_st']
            dmdl_color = fix_dmdl_color_zero_value(data['dmdl_color'])
            data['scene_color_no_st'] = (data["scene_light_no_st"] * dmdl_color *
                                         (1 - data['skybox_mask']) + data['sky_color'] *
                                         data['skybox_mask'])

            if net.enable_st:
                data['scene_color_reference'] = data['scene_color']
                data['scene_color'] = data['scene_color_no_st'] * data['st_alpha'] + data['st_color']

            gt_alias_name = net.gt_alias_name
            output['gt'] = data[gt_alias_name]
            output['warped_residual_item'] = inv_tonemap_func(output['warped_residual_item_log'], use_global_settings=True)
            output['scene_color_no_st'] = data['scene_color_no_st']
            if net.enable_st:
                output['st_color'] = data['st_color']

            output['skybox_mask'] = data['skybox_mask']
            dtype = data['continuity_mask'].dtype
            device = data['continuity_mask'].device
            output['disc_mask'] = torch.where(data['continuity_mask'] > 0.5, torch.zeros_like(data['continuity_mask'], dtype=dtype, device=device),
                                              torch.ones_like(data['continuity_mask'], dtype=dtype, device=device))
        return output

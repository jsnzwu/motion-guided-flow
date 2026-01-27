from models.general.unet import UNetDecoder, UNetEncoder, dual_conv
from models.general.unet import single_conv
from wickit.utils.log import log
from wickit.utils.basic.string import dict_to_string
from models.general.common_structure import NetBase, create_act_func, create_norm_func
import torch.nn as nn
import torch
from .common import ConvGRUCellV6, ConvLSTMCellHiddenV6, ConvLSTMCellV6


class InputBlock(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, cfg, act_func_name, norm_func_name=None):
        super().__init__()
        self.seq = nn.Sequential()
        self.seq = dual_conv(in_channel, mid_channel, out_channel,
                             seq=self.seq, act_func_name=act_func_name, norm_func_name=norm_func_name, single_act_channel=cfg['is_single_act_channel'])

    def forward(self, input_layer):
        return self.seq(input_layer)


class EncoderBlock(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, cfg, act_func_name, norm_func_name=None):
        super().__init__()
        self.conv1 = single_conv(in_channel, mid_channel, stride=2, act_func_name=act_func_name,
                                 norm_func_name=norm_func_name, single_act_channel=cfg['is_single_act_channel'])
        self.conv2 = single_conv(mid_channel, out_channel, stride=1,
                                    act_func_name=act_func_name, norm_func_name=norm_func_name, single_act_channel=cfg['is_single_act_channel'])
        self.out_channel = out_channel

    def forward(self, data):
        input_layer = self.conv1(data)
        output = torch.tensor(0)
        output = self.conv2(input_layer)
        return output


class ShadeNetEncoder(UNetEncoder):
    name = "ShadeNetEncoder"
    cnt_instance = 0

    def __init__(self, config, *args, **kwargs):
        self.instance_names = []
        self.instance_names.append("{}{}".format(
            ShadeNetEncoder.name, ShadeNetEncoder.cnt_instance))
        ShadeNetEncoder.cnt_instance += 1
        super().__init__(config=config)

    def create_input_block(self, config):
        cfg = {'is_single_act_channel': self.config['single_act_channel']}
        return InputBlock(config['in_channel'], config['mid_channel'], config['out_channel'], cfg,
                          config['act_func_name'], config['norm_func_name'])

    def create_encoder_block(self, config):
        cfg = {'is_single_act_channel': self.config['single_act_channel']}
        return EncoderBlock(config['in_channel'], config['mid_channel'], config['out_channel'],
                            cfg, config['act_func_name'], config['norm_func_name'])

    def forward(self, encoding):
        # sc_layers = []
        # ret = {}
        cur_input = encoding
        if self.input_block is not None:
            cur_input = self.input_block(cur_input)
        sc_layers = [torch.tensor(0) for i in range(len(self.encoders)-1)]

        for i in range(len(self.encoders)):
            cur_input = self.encoders[i](cur_input)
            if i < self.n_encoder - 1 and 'skip_layer_split' in self.config.keys():
                skip_channel = self.config['skip_layer_split'][i]
            else:
                skip_channel = cur_input.shape[1]
            if self.config['skip-layer'] and i != len(self.encoders) - 1:
                sc_layer = cur_input[:,:skip_channel]
                sc_layers[i] = sc_layer
        return cur_input, sc_layers


class ResBlock(nn.Module):
    def __init__(self, in_channel, side_channel, bias=True, act_func_name=None, norm_func_name=None):
        super().__init__()
        if norm_func_name == 'batch_norm_2d':
            bias = False
        self.conv1 = single_conv(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=bias,
                                 norm_func_name=norm_func_name, act_func_name=act_func_name, single_act_channel=False)
        # self.side_conv = single_conv(side_channel, side_channel, kernel_size=3, stride=1, padding=1, bias=bias,
        #                          norm_func_name=norm_func_name, act_func_name=act_func_name, single_act_channel=False)
        self.final_conv = nn.Conv2d(in_channel, in_channel,
                               kernel_size=3, stride=1, padding=1, bias=bias)
        self.single_norm_act = nn.Sequential()
        if norm_func_name:
            self.single_norm_act.add_module("{}_1".format(norm_func_name),
                                            create_norm_func(norm_func_name)(in_channel))
        if act_func_name:
            self.single_norm_act.add_module("{}_1".format(act_func_name),
                                            create_act_func(act_func_name, out_channel=in_channel)())

    def forward(self, x, skip_add=None):
        out = (self.conv1(x))
        if skip_add is None:
            skip_add = x
        # out[:,:self.side_channel] = (self.side_conv(out[:,:self.side_channel]))
        out = self.single_norm_act(skip_add + self.final_conv(out))
        return out


class ConvLSTMCellHiddenV6Wrapper(nn.Module):
    def __init__(self, in_channel, gbuffer_channel, out_channel, ks, norm=False):
        super().__init__()
        self.conv = ConvLSTMCellHiddenV6(in_channel + gbuffer_channel, out_channel, kernel_size=ks)
        self.out_channel = self.conv.out_channel
        self.gbuffer_channel = gbuffer_channel
        self.enable_norm = norm
        self.layer_norm_c = None
        self.layer_norm_h = None

    def forward(self, input_tensor, cur_state):
        c_next, h_next = self.conv(input_tensor, cur_state)
        if self.enable_norm:
            if self.layer_norm_c is None:
                self.layer_norm_c = torch.nn.LayerNorm(c_next.shape[1:], device=c_next.device, dtype=c_next.dtype, elementwise_affine=False)
                self.layer_norm_h = torch.nn.LayerNorm(h_next.shape[1:], device=h_next.device, dtype=h_next.dtype, elementwise_affine=False)
            c_next = self.layer_norm_c(c_next)
            h_next = self.layer_norm_h(h_next)
        return c_next, h_next


class DecoderBlock(nn.Module):
    def __init__(self, last_c, cat_c, mid_c, out_c, cfg, act_func_name=None, norm_func_name=None):
        super().__init__()
        self.cfg = cfg

        self.cat_conv = single_conv(
            last_c + cat_c, mid_c, kernel_size=3, act_func_name=act_func_name, norm_func_name=norm_func_name, single_act_channel=False)
        # self.cat_conv = single_conv(
        #     last_c + cat_c, out_c, kernel_size=3, act_func_name=act_func_name, norm_func_name=norm_func_name, single_act_channel=False)
        
        # self.cat_conv = single_conv(
        #     last_c + cat_c, out_c, kernel_size=1, padding=0, act_func_name=act_func_name, norm_func_name=norm_func_name, single_act_channel=False)

        self.res_block = ResBlock(mid_c, mid_c//2, act_func_name=act_func_name, norm_func_name=norm_func_name)
        assert cfg['upscale_mode'] in ['deconv', 'conv_bilinear']
        
        if cfg['upscale_mode'] == 'deconv':
            self.upscale = nn.ConvTranspose2d(
                mid_c, out_c, kernel_size=4, stride=2, padding=1, bias=True)
        elif cfg['upscale_mode'] == 'conv_bilinear':
            self.upscale = nn.Sequential()
            # self.upscale.add_module("upscale_conv3x3s1", nn.Conv2d(mid_c, mid_c, 3,padding=1))
            self.upscale.add_module("upsample_bilinear",nn.UpsamplingBilinear2d(scale_factor=2.0))
            self.upscale.add_module("upscale_conv3x3s2", nn.Conv2d(mid_c, out_c, kernel_size=3, padding=1))
        self.mid_c = mid_c
        self.upscales = nn.ModuleList()
        self.enable_output_slicing = False
        self.block_id = 0
    

    ''' START: debug code for tensor slicing '''   
    
    # def is_slicing_available(self):
    #     return self.enable_output_slicing
    
    # def reset_iteration(self):
    #     self.block_id = 0
    
    # def get_next_slicing(self, tensor):
    #     assert self.block_id < len(self.upscales)
    #     ret = self.upscales[self.block_id](tensor)
    #     self.block_id += 1
    #     return ret 
    # def add_slicing_block(self, out_c):
    #     dtype = next(self.cat_conv.parameters()).dtype
    #     device = next(self.cat_conv.parameters()).device
    #     if self.cfg['upscale_mode'] == 'deconv':
    #         upscale = nn.ConvTranspose2d(
    #             self.mid_c, out_c, kernel_size=4, stride=2, padding=1, bias=True, 
    #             dtype=dtype, device=device)
    #         self.upscales.append(upscale)
    #     elif self.cfg['upscale_mode'] == 'conv_bilinear':
    #         upscale = nn.Sequential()
    #         # self.upscale.add_module("upscale_conv3x3s1", nn.Conv2d(mid_c, mid_c, 3,padding=1))
    #         upscale.add_module("upsample_bilinear",nn.UpsamplingBilinear2d(scale_factor=2.0))
    #         upscale.add_module("upscale_conv3x3s2", nn.Conv2d(self.mid_c, out_c, 3,padding=1, 
    #                                                           dtype=dtype, device=device))
    #         self.upscales.append(upscale)
        
    # def record_slice(self, out_c):
    #     self.add_slicing_block(out_c)
    #     log.debug(f"record {out_c}")
    
    # def end_slice(self, out_c):
    #     log.debug(f"end {out_c}")
    #     delattr(self, "upscale")
    #     self.add_slicing_block(out_c)
    #     self.enable_output_slicing = True
        
    ''' END: debug code for tensor slicing '''    
    def forward(self, data, hidden_feature=None, gbuffer_feature=None):
        cur_input = data
        cur_input = self.cat_conv(cur_input)
        cur_input = self.res_block(cur_input)
            
        if self.enable_output_slicing:
            ret = cur_input
        else:
            ret = self.upscale(cur_input)
        
        return ret, hidden_feature
        

class OutputBlock(nn.Module):
    def __init__(self, in_channel,  mid_channel, out_channel, act_func_name=None, norm_func_name=None):
        super(OutputBlock, self).__init__()
        if norm_func_name == 'batch_norm_2d':
            bias = False
        else:
            bias = True
        self.seq = nn.Sequential()
        self.conv1x1 = single_conv(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=bias,
                                   norm_func_name=norm_func_name, act_func_name=act_func_name, single_act_channel=False)
        self.output1x1 = nn.Conv2d(
            mid_channel, out_channel, kernel_size=1)

    def forward(self, data):
        ret = self.conv1x1(data)
        ret = self.output1x1(ret)
        return ret


class ShadeNetDecoder(UNetDecoder):
    name = "ShadeNetDecoder"
    cnt_instance = 0

    def __init__(self, config, decoder_block_class, *args, **kwargs):
        self.instance_names = []
        self.instance_names.append("{}{}".format(
            ShadeNetDecoder.name, ShadeNetDecoder.cnt_instance))
        self.enable_output_concat = False
        self.decoder_block_class = decoder_block_class
        ShadeNetDecoder.cnt_instance += 1
        super().__init__(config=config)

    def create_decoder_block(self, config) -> nn.Module:
        cfg = {
            'upscale_mode': self.config['upscale_mode'],
        }
        return self.decoder_block_class(config['in_channel'], config['concat_channel'], config['mid_channel'], config['out_channel'],
                            cfg, act_func_name=config['act_func_name'], norm_func_name=config['norm_func_name'])

    def create_output_block(self, config) -> nn.Module:
        return OutputBlock(config['in_channel'], config['mid_channel'], config['out_channel'],
                           config['act_func_name'], config['norm_func_name'])

    def forward(self, data):
        raise Exception("The decoder should be called in the MFRRNet.forward")

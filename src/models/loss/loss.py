from typing import Dict, List
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch
from utils.dataset_utils import DatasetGlobalConfig
from utils.utils import del_dict_item
from utils.loss_utils import dssim, lpips, msssim, normalized_psnr, psnr, psnr_mask
from wickit.losses.loss_ops import ssim
from utils.buffer_utils import aces_tonemapper, buffer_data_to_vis, log_tonemapper, inv_gamma, inv_log_tonemapper
from wickit.utils.basic.tensor import align_channel_buffer
from utils.log import get_local_rank, log
from wickit.utils.basic.string import dict_to_string
from utils.utils import create_dir, get_tensor_mean_min_max_str
import numpy as np

vgg_kernel = None


def zero_l1_loss(data: list, config=None, **kwargs):
    return torch.abs(data[0])


def binary_cross_entropy_loss(data: list, config=None, reduction='none', **kwargs):
    return F.binary_cross_entropy(data[0], data[1], reduction=reduction)

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=False):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        features:nn.Sequential = nn.Sequential(torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).features)[0]
        blocks.append(features[:4].eval()) # relu 1_2
        blocks.append(features[4:9].eval()) # relu 2_2
        blocks.append(features[9:16].eval()) # relu 3_3
        blocks.append(features[16:23].eval()) # relu 4_3
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target, layers=[1, 3]):
        ''' layers: 0-1: feature, 2-3: style.  '''
        if pred.shape[1] != 3:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            if i in layers:
                x = block(x)
                y = block(y)
                loss += torch.nn.functional.l1_loss(x, y)
            # if i in style_layers:
            #     act_x = x.reshape(x.shape[0], x.shape[1], -1)
            #     act_y = y.reshape(y.shape[0], y.shape[1], -1)
            #     gram_x = act_x @ act_x.permute(0, 2, 1)
            #     gram_y = act_y @ act_y.permute(0, 2, 1)
            #     loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss



def ssim_per_pixel(img1: torch.Tensor, img2: torch.Tensor, C1: float = 1e-6, C2: float = 9e-6) -> torch.Tensor:
    # 确保输入是浮点类型
    img1 = img1.float()
    img2 = img2.float()
    
    D3 = False
    if len(img1.shape) == 3:
        D3 = True
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    # 均值
    mu1 = F.conv2d(img1, torch.ones(1, 3, 3, 3).to(img1.device) / 9.0, padding=1)
    mu2 = F.conv2d(img2, torch.ones(1, 3, 3, 3).to(img2.device) / 9.0, padding=1)

    # 方差和协方差
    sigma1_sq = F.conv2d(img1**2, torch.ones(1, 3, 3, 3).to(img1.device) / 9.0, padding=1) - mu1**2
    sigma2_sq = F.conv2d(img2**2, torch.ones(1, 3, 3, 3).to(img2.device) / 9.0, padding=1) - mu2**2
    sigma12 = F.conv2d(img1 * img2, torch.ones(1, 3, 3, 3).to(img1.device) / 9.0, padding=1) - mu1 * mu2

    # SSIM公式
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_map = numerator / denominator
    if D3:
        ssim_map = ssim_map.squeeze(0)
    return ssim_map  # 移除批量维度



def vgg_loss(data: list, config=None, **kwargs):
    global vgg_kernel
    if vgg_kernel is None:
        vgg_kernel = VGGPerceptualLoss()
    if data[0].device != next(vgg_kernel.parameters()).device:
        vgg_kernel = vgg_kernel.to(data[0].device)
    layers = config.get('layers', False) if config is not None else [1,3]
    return vgg_kernel(data[0], data[1], layers=layers)


class Charbonnier_L1(nn.Module):
    def __init__(self):
        super(Charbonnier_L1, self).__init__()

    def forward(self, diff, mask=None):
        if mask is None:
            loss = ((diff ** 2 + 1e-6) ** 0.5).mean()
        else:
            loss = (((diff ** 2 + 1e-6) ** 0.5) * mask).mean() / (mask.mean() + 1e-9)
        return loss


def charbonnier_loss(data: list, config=None, **kwargs):
    return ((l1_loss(data) ** 2 + 1e-6) ** 0.5).mean()


def l1_loss(data: list, config=None, **kwargs):
    return F.l1_loss(data[0], data[1], reduction='none')


def rel_l1_loss(data: list, config=None, **kwargs):
    return F.l1_loss(data[0], data[1], reduction='none') / (torch.mean(data[1], dim=1, keepdim=True) + 1e-1)


def shadow_attention_mask(data: list, config=None, **kwargs):
    return (torch.abs(data[0] - data[1])) / (torch.mean(torch.min(data[0], data[1]), dim=1, keepdim=True) + 1e-2)


def rel_l1_loss_before(data: list, config=None, **kwargs):
    pass
#     return F.l1_loss(data[0], data[1], reduction='none') / (F.l1_loss(data[1].mean(), data[1], reduction='none') + 1e-1)

    # torch.abs(scene_color - warped_scene_color) /\
    #         torch.abs(scene_color.mean() - scene_color)


def rel_l2_loss(data: list, config=None, **kwargs):
    return F.mse_loss(data[0], data[1], reduction='none') / (F.mse_loss(data[1].mean(), data[1], reduction='none') + 1e-1)


def l2_loss(data, config=None, **kwargs):
    for i in range(len(data[0].shape)):
        if data[0].shape[i] != data[1].shape[i]:
            raise Exception("data.0 and data.1 dont have same shape: {} {}".format(
                data[0].shape, data[1].shape))
    return F.mse_loss(data[0], data[1], reduce=False)


class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.kernelX = torch.tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1],
        ]).float()
        self.kernelY = self.kernelX.clone().T
        self.kernelX = self.kernelX.unsqueeze(0).unsqueeze(0)
        self.kernelY = self.kernelY.unsqueeze(0).unsqueeze(0)

    def forward_x(self, data):
        # if self.kernelX.device != data.device:
        #     self.kernelX = self.kernelX.to(data)
        ret = F.conv2d(data, self.kernelX, padding=1)
        return ret

    def forward_y(self, data):
        # if self.kernelY.device != data.device:
        #     self.kernelY = self.kernelY.to(data)
        ret = F.conv2d(data, self.kernelY, padding=1)
        # log.debug(dict_to_string({
        #     'ret': ret,
        # }))
        return ret

    def forward(self, pred, gt):
        N, C, H, W = pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3]
        img_stack = torch.cat(
            [pred.reshape(N * C, 1, H, W), gt.reshape(N * C, 1, H, W)], 0)
        if self.kernelX.device != gt.device:
            self.kernelX = self.kernelX.to(gt.device)
            self.kernelY = self.kernelY.to(gt.device)
        sobel_stack_x = F.conv2d(img_stack, self.kernelX, padding=1)
        sobel_stack_y = F.conv2d(img_stack, self.kernelY, padding=1)
        pred_X, gt_X = sobel_stack_x[:N * C], sobel_stack_x[N * C:]
        pred_Y, gt_Y = sobel_stack_y[:N * C], sobel_stack_y[N * C:]

        L1X, L1Y = torch.abs(pred_X - gt_X), torch.abs(pred_Y - gt_Y)
        loss = (L1X + L1Y)
        return loss

sobel_cache_map = {}

def get_sobel(device, dtype) -> Sobel:
    global sobel 
    key = (device, dtype)
    if key not in sobel_cache_map:
        sobel = Sobel()
        sobel.kernelX = sobel.kernelX.to(device, dtype=dtype)
        sobel.kernelY = sobel.kernelY.to(device, dtype=dtype)
        sobel_cache_map[key] = sobel
        return sobel
    else:
        return sobel_cache_map[key]

def sobel(pred, gt):
    sobel_op = get_sobel(pred.device, pred.dtype)
    return sobel_op(pred, gt)

def sobel_x(data):
    sobel_op = get_sobel(data.device, data.dtype)
    # assert sobel_op.kernelX.device == data.device, dict_to_string([sobel_op.kernelX.device, data.device])
    return sobel_op.forward_x(data)

def sobel_y(data):
    sobel_op = get_sobel(data.device, data.dtype)
    return sobel_op.forward_y(data)

def l1_mask_loss(data, config=None, **kwargs):
    return F.l1_loss(data[0] * data[2], data[1] * data[2], reduce=False)


def l2_mask_loss(data, config=None, **kwargs):
    return F.mse_loss(data[0] * data[2], data[1] * data[2], reduce=False)


def l2_minus_mask_loss(data, config=None, **kwargs):
    return F.mse_loss(data[0] * (1 - data[2]), data[1] * (1 - data[2]), reduce=False)

# pred, gt, mask


def shadow_mask_loss(data, config=None, **kwargs):
    loss_l1 = l1_loss(data[0], data[1])
    mask = 1 - data[2]
    return loss_l1 * mask

# pred, gt, mask


def extranet_hole_loss(data, config=None, output=None, **kwargs):
    debug = config.get('debug', False) if config is not None else False
    mask = data[2]

    ret = l1_loss([data[0] * mask, data[1] * mask])
    if debug:
        assert output is not None
        output['hole_data_0'] = data[0] * mask
        output['hole_data_1'] = data[1] * mask
        output['hole_diff'] = torch.abs(data[0] * mask - data[1] * mask)
        output['hole_l1'] = F.l1_loss(data[0], data[1], reduction='none')
    return ret


def extranet_shadow_loss(data, config=None, output=None, **kwargs):
    ratio = config.get('ratio', 0.1) if config is not None else 0.1
    debug = config.get('debug', False) if config is not None else False
    # log.debug(dict_to_string(config))
    # log.debug(dict_to_string(output))
    # log.debug(dict_to_string(data))
    # log.debug("{} {}".format("extra_shadow_loss", debug))
    B, C, H, W = data[0].shape
    if len(data) == 3:
        tmp_data0 = data[0] * data[2]
        tmp_data1 = data[1] * data[2]
    elif len(data) == 2:
        tmp_data0 = data[0]
        tmp_data1 = data[1]
    else:
        raise
    val, ind = torch.topk(l1_loss([tmp_data0, tmp_data1]).view(B, C, -1), k=int(H * W * ratio))
    val *= ratio
    if debug:
        assert output is not None
        output['shadow_val'] = val
        output['shadow_ind'] = ind
        output['shadow_diff'] = torch.abs(tmp_data0 - tmp_data1)

    return val


def ssim_value(data, config=None, **kwargs):
    return ssim(data[0], data[1])

def msssim_value(data, config=None, **kwargs):
    return msssim(data[0], data[1])

def dssim_value(data, config=None, **kwargs):
    return dssim(data[0], data[1])

def dmsssim_value(data, config=None, **kwargs):
    return dmsssim_value(data[0], data[1])

def psnr_value(data, config=None, **kwargs):
    return psnr(data[0], data[1])

def psnr_value_mask(data, config=None, **kwargs):
    return psnr_mask(data[0], data[1], data[2])

def normalized_psnr_value(data, config=None, **kwargs):
    return normalized_psnr(data[0], data[1])

def lpips_value(data, config=None, **kwargs):
    return lpips(data[0], data[1])


def contrastive_loss(feature1, feature2, output1, output2, num_pair=10):
    f_shape = feature1.shape
    o_shape = output1.shape
    pair_index1 = (torch.rand(num_pair) *
                   f_shape[-2] * f_shape[-1]).type(torch.int).tolist()
    pair_index2 = (torch.rand(num_pair) *
                   o_shape[-2] * o_shape[-1]).type(torch.int).tolist()
    feature1 = feature1.reshape(f_shape[0], f_shape[1], -1)[..., pair_index1]
    feature2 = feature2.reshape(f_shape[0], f_shape[1], -1)[..., pair_index2]
    output1 = output1.reshape(o_shape[0], o_shape[1], -1)[..., pair_index1]
    output2 = output2.reshape(o_shape[0], o_shape[1], -1)[..., pair_index2]
    log.debug("{} {} {} {}".format(feature1.shape,
              feature2.shape, output1.shape, output2.shape))
    diff_feature = torch.sum((feature1 - feature2) ** 2, dim=1, keepdim=True)
    diff_output = torch.sum((output1 - output2) ** 2, dim=1, keepdim=True)
    return l2_loss(diff_feature, diff_output)

def apply_tonemapper(tensor, tonemap_name, name=""):
    if tonemap_name == "log":
        tensor = log_tonemapper(tensor)
    elif tonemap_name == "aces":
        tensor = aces_tonemapper(tensor)
    elif tonemap_name is not None:
        raise Exception(
            f'unsupported tonemap func given in LossFunction.forward_single("{name}"): "{tonemap_name}"')
    return tensor

def dmdl_mu_log_pApB_gAgBgC(data, config=None, output=None, **kwargs):
    ''' C = A*B '''
    assert len(data) == 5
    pA, pB, gA, gB, gC = data
    assert config is not None
    mode = config['mode']
    assert mode in ['l1', 'charbonnier_l1']
    loss_fn = LossFunction.single_ops[config['mode']]
    mu = DatasetGlobalConfig.log_tonemapper__mu
    # mu = 1.0
    diff = log_tonemapper(mu*gC+gB+gA, mu=mu) - log_tonemapper(pA, mu=mu) - log_tonemapper(pB, mu=mu)
    # diff = log_tonemapper(mu*gC+gB+gA, mu=mu) - log_tonemapper(pA, mu=mu) - log_tonemapper(pB, mu=mu)
    return loss_fn([diff, torch.zeros_like(diff, dtype=diff.dtype, device=diff.device)])
# lap_loss_ins = LapLoss()

def dmdl_pAgBgC(data, config=None, output=None, **kwargs):
    ''' C = A*B '''
    assert len(data) == 3
    pA, gB, gC = data
    assert config is not None
    mode = config['mode']
    assert mode in ['l1', 'charbonnier_l1']
    loss_fn = LossFunction.single_ops[config['mode']]
    # mu = DatasetGlobalConfig.log_tonemapper__mu
    modulated_color = apply_tonemapper(pA * gB, config.get("tonemap", None))
    gt_color = apply_tonemapper(gC, config.get("tonemap", None))
    
    return loss_fn([modulated_color, gt_color])


# def laplace_loss(data, config=None, **kwargs):
#     return lap_loss_ins(data[0], data[1])
def rgb2yuv(rgb):
    rgb_to_yuv_matrix = torch.tensor([
        [0.299, 0.587, 0.114],
        [-0.14713, -0.28886, 0.436],
        [0.615, -0.51499, -0.10001]
    ], dtype=rgb.dtype, device=rgb.device)
    yuv_images = torch.einsum('nchw,mc->nmhw', rgb, rgb_to_yuv_matrix)
    # log.debug(dict_to_string([rgb, yuv_images], mmm=True))
    return yuv_images


class LossFuncDataBase:
    def __init__(self, name, args, mode, enable):
        self.name = name
        self.args = args
        self.mode = mode
        self.enable = enable
        
class LossFuncData(LossFuncDataBase):
    def __init__(self, name, args, mode, scale, config, is_paired, enable):
        super().__init__(name, args, mode, enable)
        self.scale = scale
        self.config = config
        self.is_paired = is_paired
        
    def __str__(self) -> str:
        return f'{self.name}'
    
class LossFunction:
    single_ops = {
        "l1": l1_loss,
        "zero_l1": zero_l1_loss,
        "charbonnier_l1": charbonnier_loss,
        # "lap": laplace_loss,
        "l1_rel": rel_l1_loss,
        "l2_rel": rel_l2_loss,
        "l2": l2_loss,
        "l1_mask": l1_mask_loss,
        "l2_mask": l2_mask_loss,
        "l2_minus_mask": l2_minus_mask_loss,
        "shadow_mask": shadow_mask_loss,
        "extranet_hole": extranet_hole_loss,
        "extranet_shadow": extranet_shadow_loss,
        "psnr": psnr_value,
        "psnr_mask": psnr_value_mask,
        "normalized_psnr": normalized_psnr_value,
        "ssim": ssim_value,
        "msssim": msssim_value,
        "dssim": dssim_value,
        "dmsssim": dmsssim_value,
        "lpips": lpips_value,
        "binary_cross_entropy_loss": binary_cross_entropy_loss,
        "vgg": vgg_loss,
        "dmdl_mu_log_pApB_gAgBgC": dmdl_mu_log_pApB_gAgBgC,
        "dmdl_pAgBgC": dmdl_pAgBgC,
    }
    paired_ops = {
        "contrastive": contrastive_loss
    }

    def __init__(self, config):
        self.config = config
        self.loss_funcs: List[LossFuncData] = []
        self.loss_func_dict: Dict[str, LossFuncData] = {}
        self.debug_loss_funcs: List[LossFuncData] = []
        self.debug_loss_func_dict: Dict[str, LossFuncData] = {}
        self.involved_loss_mode = set()
        # self.sum_ratio = 0
        for loss_name in self.config["train_loss"].keys():
            mode = self.config["train_loss"][loss_name]["mode"]
            loss_func_data = LossFuncData(loss_name,
                                          self.config["train_loss"][loss_name]['args'],
                                          mode,
                                          self.config["train_loss"][loss_name].get("scale", 1.0),
                                          self.config["train_loss"][loss_name].get("config", {}),
                                          self.is_paired(mode),
                                          self.config["train_loss"][loss_name].get("enable", True))
            self.loss_func_dict[loss_name] = loss_func_data
            self.loss_funcs.append(loss_func_data)
            # self.sum_ratio += self.loss_func[-1]["ratio"]
            self.involved_loss_mode.add(self.config["train_loss"][loss_name]["mode"])
        # log.debug(self.loss_func)
        self.has_paired_loss = self.has_paired()
        for loss_name in self.config["debug_loss"].keys():
            mode = self.config["debug_loss"][loss_name]["mode"]
            loss_func_data = LossFuncData(loss_name,
                                          self.config['debug_loss'][loss_name]['args'],
                                          mode,
                                          self.config["debug_loss"][loss_name].get("scale", 1.0),
                                          self.config["debug_loss"][loss_name].get("config", {}),
                                          False,
                                          self.config["debug_loss"][loss_name].get("enable", True))
            self.debug_loss_func_dict[loss_name] = loss_func_data
            self.debug_loss_funcs.append(loss_func_data)
        log.debug("loss func details: {}".format(dict_to_string(self.loss_funcs)))
        log.debug("debug loss func details: {}".format(dict_to_string(self.debug_loss_funcs)))
        log.info("[LossFunction]: created. info: {}".format(self.__str__()))
        
    def update_loss_info(self):
        # self.sum_ratio = 0
        for loss_name in self.config["train_loss"].keys():
            cur_scale = self.config["train_loss"][loss_name].get("scale", 1.0)
            if cur_scale != self.loss_func_dict[loss_name].scale:
                # log.debug(f"loss_name: {loss_name}, last_scale:{self.loss_func_dict[loss_name].scale}, cur_scale: {cur_scale}")
                self.loss_func_dict[loss_name].scale = cur_scale
        # for loss_name in self.config["debug_loss"].keys():
        #     mode = self.config["debug_loss"][loss_name]["mode"]
        #     self.debug_loss_func_dict[loss_name].scale = self.config["debug_loss"][loss_name].get("scale", 1.0)
            
    def __str__(self):
        return "loss: {}, has_paired_loss: {}, debug_loss: {}".format(
            [item.name for item in self.loss_funcs],
            self.has_paired_loss,
            [item.name for item in self.debug_loss_funcs])

    def is_paired(self, name):
        if name == "contrastive":
            return True
        return False

    def has_paired(self):
        for mode in self.involved_loss_mode:
            if self.is_paired(mode):
                return True
        return False

    def check_data(self, data, non_local=None):
        for item in (self.loss_funcs + self.debug_loss_funcs):
            if not item.enable:
                continue
            if item.is_paired:
                pass
            else:
                input_names = [item.args[i]
                               for i in range(len(item.args))]
                for name in input_names:
                    if name not in data.keys():
                        log.warning('[Loss] {} in "{}" is not in data ({})'.format(
                            name, item.name, list(data.keys())))
                        assert False
                        item.enable = False

    def get_active_loss_func_names(self) -> list:
        names = []
        for item in (self.loss_funcs + self.debug_loss_funcs):
            if item.enable:
                names.append(item.name)
        return names

    def forward(self, data, non_local=None):
        loss = {}
        total_loss = 0.0
        for item in self.loss_funcs:
            if not item.enable:
                continue
            if item.is_paired:
                if non_local is None:
                    continue
                loss[item.name] = self.forward_paired(item.mode,
                                                         data[item.args[0]],
                                                         data[item.args[1]],
                                                         non_local[item.args[0]],
                                                         non_local[item.args[1]])
            else:
                loss[item.name] = self.forward_single(
                    item, data, output=data) * item.scale
            # data[item['name']] = loss[item['name']]
            # log.debug(dict_to_string(data))

        keys_list = list(data.keys())
        for name in keys_list:
            if name.endswith("_loss"):
                loss[name] = data[name]

        raw_loss = {}
        for item in loss:
            if len(loss[item].shape) > 0:
                raw_loss[item + '_ls'] = loss[item].clone()
                loss[item] = loss[item].mean()
            total_loss += loss[item]
            # loss[item] = loss[item].item()
        loss.update(raw_loss)

        for name in keys_list:
            if name.endswith("_loss"):
                data = del_dict_item(data, name)

        loss['loss'] = total_loss
        # log.debug(dict_to_string(loss))
        return loss

    def forward_debug(self, data, cpu=False, force_full_precision=False):
        if get_local_rank() != 0:
            return {}
        loss = {}
        for item in self.debug_loss_funcs:
            if not item.enable:
                continue
            with torch.no_grad():
                loss[item.name] = self.forward_single(item, data, cpu=cpu, force_full_precision=force_full_precision).detach().mean()
                # assert torch.isnan(loss[item["name"]]).any()
        return loss
    
    @staticmethod
    def calc_loss(data_input, mode, config, name="unnamed_loss", data_dict=None, output_dict=None, cpu=False, force_full_precision=False):
        if config.get("skybox_mask", False):
            assert data_dict is not None
            # log.debug("+++++ ENABLE SKYBOX_MASK: {:>10}, {:>16} +++++".format(item['mode'], item['name']))
            for i in range(len(data_input)):
                data_input[i] = data_input[i] * data_dict['skybox_mask']
        elif config.get("skybox_mask_out", False):
            assert data_dict is not None
            # log.debug("+++++ ENABLE SKYBOX_MASK_OUT: {:>10}, {:>16} +++++".format(item['mode'], item['name']))
            for i in range(len(data_input)):
                data_input[i] = data_input[i] * (1 - data_dict['skybox_mask'])

        # from torch import autocast
        # device = 'cuda' if str(data_input[0].device).startswith('cuda') else 'cpu'
        # with autocast(device, enabled=False):

            
        if mode not in ['dmdl_mu_log_pApB_gAgBgC', 'dmdl_pAgBgC']:
            for i in range(len(data_input)):
                data_input[i] = apply_tonemapper(data_input[i], config.get("tonemap", None))
        # for i in range(len(data_input)):
        #     if torch.isnan(data_input[i]).any():
        #         log.debug(dict_to_string(item['args'], mmm=True))
        #         log.debug(dict_to_string(data_input, mmm=True))
        #         assert False
        if ((color_fmt := config.get('color_format', None)) == 'yuv'):
            for i in range(len(data_input)):
                data_input[i] = rgb2yuv(data_input[i])
            
        elif color_fmt is not None:
            raise Exception(
                f'unsupported color_fmt given in LossFunction.forward_single("{name}"): "{color_fmt}"')
        # log.debug(dict_to_string({
        #     'tonemap': tonemap_name,
        #     'color_fmt': color_fmt,
        #     'data_input': data_input,
        #     'value':float(LossFunction.single_ops[mode]([data_input[0].cpu(), data_input[1].cpu()]).mean().item()),
        #     'value2':float(LossFunction.single_ops[mode]([data_input[0], data_input[1]]).mean().item()),
        #     'value_16':float(LossFunction.single_ops[mode]([data_input[0].type(torch.float16), data_input[1].type(torch.float16)]).mean().item()),
        #     'value3':float(LossFunction.single_ops[mode](data_input, config=config, output=output_dict))
        # }))
        for i in range(len(data_input)):
            if force_full_precision:
                data_input[i] = data_input[i].type(torch.float32)
            if cpu:
                data_input[i] = data_input[i].cpu()
        ret = LossFunction.single_ops[mode](data_input, config=config, output=output_dict)
        return ret

    def forward_single(self, item: LossFuncData, data, cpu=False, force_full_precision=False, output=None):
        assert not cpu, 'loss must be calculated on cuda'
        # log.debug(dict_to_string(data))
        mode = item.mode
        for arg in item.args:
            assert arg in data.keys(), f"item: {item.name}, {item.args}, {item.config}"
        data_input = [data[item.args[i]] for i in range(len(item.args))]
        config = item.config
        ret = LossFunction.calc_loss(data_input, mode, config, cpu=cpu, force_full_precision=force_full_precision, name=item.name, data_dict=data, output_dict=output)
        return ret
        

    def forward_paired(self, mode, local_f, local_o, non_local_f, non_local_o):
        if (non_local_f is None or non_local_o is None):
            raise ValueError("lcoal and non_local must not be None, with\n local={},\n non_local={}".format(
                dict_to_string(local_f, "local_f") + "\n" +
                dict_to_string(local_o, " local_o"),
                dict_to_string(non_local_f, "non_local_f") + "\n" + dict_to_string(non_local_o, "non_local_o")))
        ret = 0
        cnt = 0
        # ret += self.paired_ops[mode](local_f, non_local_f, local_o, non_local_o)
        # cnt += 1
        # log.debug(get_tensor_mean_min_max_str(local_f))
        # log.debug(get_tensor_mean_min_max_str(non_local_f))
        ret += self.paired_ops[mode](local_f,
                                     non_local_f, local_o, non_local_o)
        cnt += 1
        return ret / cnt

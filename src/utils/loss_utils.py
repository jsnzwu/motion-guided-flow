import torch
from wickit.losses.loss_ops import (
    dmsssim,
    dssim,
    lpips,
    msssim,
    psnr,
    psnr_hdr,
    psnr_mask,
    ssim,
)


def normalized_psnr(pred, gt, **kwargs):
    diff_map = torch.square(torch.clamp(pred, 0, 1) - torch.clamp(gt, 0, 1))
    se_map = diff_map.sum(dim=1, keepdim=True)
    se = se_map.sum()
    se_cnt = torch.where(se_map > 0, torch.ones_like(se_map), torch.zeros_like(se_map)).sum() + 1
    return 10 * torch.log10(1.0 / (se / se_cnt))


from wickit.models.model_abc import ModelABC
from wickit.utils.enums import ForwardMode
from wickit.utils.log import log
import torch


class BlendModelABC(ModelABC):
    def __init__(self, config):
        model_cfg = config["model"] if isinstance(config, dict) else config.model
        self.enable_blend = model_cfg.get('config', {}).get('enable_blend_mode', False)
        log.debug(f'BlendModelABC.enable_blend_mode: {self.enable_blend}')
        super().__init__(config)

    def update(self, data, mode: ForwardMode | None = None):
        if not self.enable_blend:
            return super().update(data, mode=mode)

        data = self.calc_preprocess_input(data)
        base_input = {
            'metadata': data['metadata'],
            'time': data['time'],
        }
        net_input0 = {
            'img0': data[self.config['st_alpha_0_alias']].repeat(1, 3, 1, 1),
            'img1': data[self.config['st_alpha_1_alias']].repeat(1, 3, 1, 1),
            'gt': data['st_alpha'].repeat(1, 3, 1, 1),
            **base_input,
        }
        res0 = self.forward(net_input0)
        pred_st_alpha = res0['pred'][:, :1].detach()
        torch.cuda.empty_cache()

        net_input1 = {
            'img0': data[self.config['st_color_0_alias']],
            'img1': data[self.config['st_color_1_alias']],
            'gt': data['st_color'],
            **base_input,
        }
        res1 = self.forward(net_input1)
        pred_st_color = res1['pred'].detach()
        mv_st = res1['motion_vector'].detach()
        torch.cuda.empty_cache()

        net_input2 = {
            'img0': data[self.config['scene_color_no_st_0_alias']],
            'img1': data[self.config['scene_color_no_st_1_alias']],
            'gt': data['scene_color_no_st'],
            **base_input,
        }
        res2 = self.forward(net_input2)
        pred_scene_color_no_st = res2['pred'].detach()
        mv = res2['motion_vector'].detach()
        torch.cuda.empty_cache()

        output = {}
        output['pred_scene_color_no_st'] = pred_scene_color_no_st
        output['pred_st_color'] = pred_st_color
        output['pred'] = pred_scene_color_no_st * pred_st_alpha + pred_st_color
        output['gt'] = data['scene_color']
        output['scene_color'] = data['scene_color']
        output['st_color'] = data['st_color']
        output['motion_vector'] = mv
        output['motion_vector_st'] = mv_st
        output['scene_color_no_st'] = data['scene_color_no_st']
        output['history_scene_color_no_st_0'] = data['history_scene_color_no_st_0']
        return output

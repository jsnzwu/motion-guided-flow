from copy import deepcopy
import copy
import datetime
import time
import torch
import argparse
import os
import sys
import includes.importer
from utils.utils import create_dir, remove_all_in_dir, write_text_to_file
from wickit.utils.ext.warp import warp
from wickit.utils.basic.tensor import data_to_device
from utils.config_enhancer import enhance_train_config, initialize_recipe
from wickit.utils.basic.tensor import tensor_as_type_str
from wickit.utils.basic.string import dict_to_string
from wickit.utils.log import add_prefix_to_log, log, shutdown_log
from wickit.models import MODELS
from config.moflow_config_utils import parse_config

def convert_onnx(model, patch_loader=None):
    # model.set_eval()
    model.set_eval()
    _ = model.net(model.dummy_net_input)
    model_input =  model.dummy_net_input
    ''' TODO: use real data '''
    
    # for k in list(model_input.keys()):
    #     if not isinstance(model_input[k], torch.Tensor):
    #         del model_input[k]
        
    # net = model
    # log.debug(net)
    # net.requires_grad_(False)
    model_input = data_to_device(model_input, device='cuda:0')
    model_output = model.dummy_output
    log.debug(dict_to_string(model_input))
    log.debug(dict_to_string(model_output))
    # _ = torch.jit.trace(model.net, (model_input,), strict=False)
    # net = _
    net = model.net
    export_path = f'../output/onnx/{model.model_name}/'
    create_dir(export_path)
    remove_all_in_dir(export_path)
    write_text_to_file(export_path+"info.log", str(net), mode="w")
    with torch.no_grad():
        torch.onnx.export(net,  # model being run
                        (model_input, {}),  # model input (or a tuple for multiple inputs)
                       f"{export_path}/{model.model_name}.onnx",  # where to save the model
                        export_params=True,  # store the trained parameter weights inside the model file
                        opset_version=17,  # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        # input_names=list(model_input.keys()),  # the model's input names
                        input_names=[],  # the model's input names
                        # output_names=list(model_output.keys()),  # the model's output names
                        output_names=[],  # the model's output names
                        # input_names=list(model_input.keys()),  # the model's input names
                        # output_names=model_output.keys(),  # the model's output names
                        verbose=False,
                        )
    torch.save(model_input, f"{export_path}/model_input.pt")
    torch.save(model_output, f"{export_path}/model_output.pt")
    from onnxsim import simplify
    import onnx
    onnx_model = onnx.load(f"{export_path}/{model.model_name}.onnx")  # load onnx model
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, f"{export_path}/{model.model_name}_sim.onnx")
    
def update_config(config):
    if hasattr(config, "unfreeze") and getattr(config, "_frozen", False):
        config.unfreeze()
    num_gpu = config.runner.num_gpu
    config.runtime.use_ddp = num_gpu > 1
    config.runtime.use_gpu = num_gpu > 0
    config.runtime.device = "cuda:0" if config.runtime.use_gpu else "cpu"
    assert config.train_parameter.batch_size % max(num_gpu, 1) == 0
    config.train_parameter.batch_size = config.train_parameter.batch_size // max(num_gpu, 1)
    assert config.dataset.train_num_worker_sum % max(num_gpu, 1) == 0
    config.dataset.train_num_worker = config.dataset.train_num_worker_sum // max(num_gpu, 1)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="trainer")
    parser.add_argument("--config", help="trainer config file path")
    args = parser.parse_args()
    
    config = parse_config(args.config, root_path="")
    update_config(config)
    enhance_train_config(config)
    
    config_onnx = deepcopy(config)
    config_onnx.model.export_onnx = True
    ''' important for export onnx in fp16 !! '''
    config_onnx.model.inference_precision = "fp16"
    # config_onnx["inital_inference"] = False
    model = MODELS.build(config_onnx.model.entry, config_onnx)
    # import cProfile
    # cProfile.run("model = MODELS.build(config_onnx.model.entry, config_onnx)", filename="profile.out", sort="cumulative")
    # model = TestNet().cuda()
    # model(model.dummy_input)
    convert_onnx(model)

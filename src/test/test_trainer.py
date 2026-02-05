from copy import deepcopy
import copy
import datetime
import gc
import time
import torch
import argparse
import sys
import os
from tqdm import tqdm
import includes.importer

from config.config_utils import load_yaml_with_replacements, parse_config as project_parse_config
from utils.utils import Accumulator, seconds_to_str, str_to_seconds
from wickit.utils.basic.string import dict_to_string
from utils.config_enhancer import enhance_buffer_config, enhance_train_config, update_config
from wickit.utils.log import add_prefix_to_log, get_local_rank, log, shutdown_log
import torch.distributed as dist
import torch.distributed
from models.mfrrnet.mfrrnet import MFRRNetModel
from trainers.mfrrnet_runner import MFRRNetRunner
from torch.profiler import profile, record_function, ProfilerActivity


def create_config(path: str) -> dict:
    return load_yaml_with_replacements(path)


def parse_config(path: str, root_path: str = ""):
    return project_parse_config(path, root_path=root_path)


def train(config_train):
    resume = config_train['args'].get('resume', False)
        
    model = eval(config_train.model.type)(
        config_train)
    trainer = eval(config_train.trainer.type)(
        config_train, model, resume=resume)
    trainer.train()
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #             # profile_memory=True, record_shapes=True) as prof:
    #             ) as prof:
    #     trainer.train()
    # prof.export_chrome_trace(f"trace_{get_local_rank()}.json")


def test(config_test):
    log.debug("start test")
    config_test.runtime.use_ddp = False
    test_only = config.args['test_only']
    resume = True
    if test_only:
        resume = False
    log.debug(dict_to_string([test_only, resume]))
    model = eval(config_test.model.type)(
        config_test)
    trainer = eval(config_test.trainer.type)(
        config_test, model, resume=resume)
    trainer.test()


def single_start(local_rank: int, config: dict) -> None:
    log.info("creating trainer, local_rank: {}".format(local_rank))
    log.debug("torch cuda gpu num: {}".format(torch.cuda.device_count()))

    if config.runtime.use_ddp and config.args['train']:
        os.environ['MASTER_ADDR'] = 'localhost'
        env_dict = {
            key: os.environ[key]
            for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
        }
        log.debug(
            f"[{os.getpid()}] Initializing process group with: {env_dict}")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl',
                                init_method="env://",
                                # timeout=datetime.timedelta(seconds=60)
                                )
        log.debug(f"torch.distributed.is_initialized():{torch.distributed.is_initialized()}")
        config.runtime.local_rank = local_rank
        config.runtime.device = str(torch.device("cuda", config.runtime.local_rank))
        log.debug("env_local_rank: {}, dist_local_rank:{}".format(
            local_rank, torch.distributed.get_rank()))
    else:
        config.runtime.local_rank = local_rank

    config_train = deepcopy(config)
    
    ''' wait for seconds to start training '''
    if (waiting_time:=str_to_seconds((time_str:=config.trainer.wait_to_start))) > 0:
        time_start = time.time()
        time_end = time.time()
        ''' create a custom tqdm bar here and update it manually every 5 seconds '''
        bar = tqdm(total=waiting_time, desc=f'wait-to-start: {waiting_time}s', disable=get_local_rank()!=0)
        while time_end - time_start < waiting_time:
            time.sleep(1)
            time_end = time.time()
            bar.set_description(f'wait-to-start: {seconds_to_str(int(time_end - time_start))}/{time_str}')
            bar.update(1)
        bar.close()
        
    if config.args['train']:
        log.debug(f'rank_{get_local_rank()}: start_training')
        train(config_train)
        log.debug(f'rank_{get_local_rank()}: end_training')
        if config_train.runtime.use_ddp:
            try:
                # 销毁进程组
                dist.destroy_process_group()
            except Exception as e:
                log.debug(f"Error occurred while destroying process group: {e}")
                # 同样，可以在这里添加其他的错误处理逻辑
            log.debug(f'rank_{get_local_rank()}: destroyed process')

    if config.args['test'] and local_rank <= 0:
        config_test = deepcopy(config)
        if config.args['train'] and 'time_string' in config_train.keys():
            config_test.time_string = config_train.time_string
        test(config_test)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description="trainer")
    parser.add_argument("--config", help="trainer config file path")
    parser.add_argument("--num_gpu", default=0, type=int, help="num_gpu")
    parser.add_argument('--wait-to-start', type=str, default="")
    # parser.add_argument("--port", default=23333, type=int, help="port")
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--test_only', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true', default=False)
    args = parser.parse_args()

    config = parse_config(args.config, root_path="")
    config.unfreeze()
    config.trainer.wait_to_start = args.wait_to_start
    input_config = copy.deepcopy(create_config(args.config))
    # log.debug(dict_to_string(input_config))
    config._input_config = input_config

    config.args = vars(args)
    if (args.num_gpu) > 0:
        config.trainer.num_gpu = args.num_gpu
    update_config(config)
    enhance_train_config(config)

    log.debug("{}:\n {}".format(
        config.job_name, dict_to_string(config.args, 'args')))

    if config.runtime.use_ddp and config.args['train']:
        config.runtime.world_size = config.trainer.num_gpu
        single_start(int(os.environ['LOCAL_RANK']), config)
        # import cProfile
        # cProfile.run("single_start(get_local_rank(), config)", filename=f"result_{get_local_rank()}.out", sort="cumulative")
        # MP.spawn(start_train, nprocs=config['num_gpu'], args=(config,))
    #     processes = []
    #     log.info("creating process, size: {}".format(config['gpu_num']))
    #     for i in range(config['gpu_num']):
    #         p = MP.Process(target=start_train, args=(config, i,))
    #         p.start()
    #         processes.append(p)
    #     for p in processes:
    #         p.join()
    else:
        # if config['args']['multi']:
        #     multi_start(config)
        # else:
        single_start(-1, config)
        # import cProfile
        # cProfile.run("single_start(-1, config)", filename=f"result.out", sort="cumulative")

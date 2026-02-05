import argparse
import torch
import numpy as np
from tqdm import tqdm
import os
import sys

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    import includes.importer
    from wickit.utils.basic.tensor import align_channel_buffer
    from dataloaders.asset_loader import AssetLoader
    from config.moflow_config_utils import parse_config
    from utils.dataset_utils import get_input_filter_list
    from utils.config_enhancer import enhance_buffer_config
    from wickit.dataloaders.metadata import MetaData
    from wickit.utils.basic.string import dict_to_string
    from utils.utils import get_tensor_mean_min_max_str
    from wickit.utils.log import log
    from utils.utils import remove_all_in_dir
    from utils.utils import create_dir

    parser = argparse.ArgumentParser(description="exporter")
    parser.add_argument('--config', type=str, default="config/export/export_st.yaml")
    args = parser.parse_args()
    # config = create_py_parser("config/config_extra_net_test.py")
    job_config = parse_config(args.config, root_path="")
    if 'dataset' in job_config.keys():
        scale_config = job_config.dataset.get('scale_config', {})
    else:
        scale_config = {}
    enhance_buffer_config(job_config.buffer_config, scale_config=scale_config)
    assert 'demodulation_mode' in job_config.buffer_config.keys()
    job_config.buffer_config['demodulation_mode'] = 'extranet'
    loader = AssetLoader(
        job_config.buffer_config['basic_part_enable_list'],
        buffer_config=job_config.buffer_config,
        job_config=job_config)
    loader.export_extend_patch()

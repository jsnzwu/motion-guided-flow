from utils.log import log
from wickit.datasets.metadata import MetaData, MetaDataWithPath
from wickit.utils.basic.string import dict_to_string
import time
import glob
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
import random


def range_task_by_metadata(task, metadatas, start_idx: int, end_idx: int):
    time.sleep(end_idx * 0.001)
    log.debug(f"start range_task[{start_idx}:{end_idx}]")
    for i in tqdm(range(start_idx, end_idx)):
        task(metadatas[i])


def dispatch_task_by_metadata(task, metadatas: list[MetaData], num_thread=0):
    ''' single thread '''
    if num_thread <= 0:
        range_task_by_metadata(task, metadatas, 0, len(metadatas))
        return
    ''' multi thread '''
    n_core = num_thread
    num = len(metadatas)
    pool = mp.Pool(processes=n_core)
    thread_part = max(num // n_core + 1, 1)
    try:
        log.debug("scene:{} n_core:{} thread_part:{}".format(
            metadatas[0].dataset_name, n_core, thread_part))
        _ = [pool.apply_async(range_task_by_metadata, (task, metadatas, i * thread_part,
                                                       min((i + 1) * thread_part, num), ),
                              callback=None)
             for i in range(n_core)]
        pool.close()
    except KeyboardInterrupt:
        pool.terminate()
    except Exception as e:
        log.debug(e)
        pool.terminate()
    finally:
        log.debug(f"joined threads: len={len(metadatas)}")
        pool.join()


def range_task_by_part_name(args):
    task, metadatas, part_name = args
    log.debug(f"start part: {part_name}")
    task(metadatas, part_name)


def dispatch_task_by_part_name(task, metadatas: list[MetaData], part_names, num_thread=0):
    if num_thread <= 0:
        for part_name in part_names:
            range_task_by_part_name((task, metadatas, part_name,))
        return
    n_core = num_thread
    pool = mp.Pool(processes=n_core)
    try:
        pool.imap_unordered(range_task_by_part_name, [(task, metadatas, part_name) for part_name in part_names],
                            chunksize=max(1, len(part_names) // n_core))
        pool.close()
    except KeyboardInterrupt:
        pool.terminate()
    except Exception as e:
        log.debug(e)
        pool.terminate()
    finally:
        log.debug(f"joined threads: len={len(part_names)}")
        pool.join()


def create_metadata_by_glob(path, scene, part_name):
    file_name_list = glob.glob(
        "{}/{}/{}/*.npz".format(path, scene, part_name))
    num = len(file_name_list)
    metadatas = []
    for i in range(0, num):
        metadatas.append(MetaData(scene, i))
    return metadatas


end_cutoff = 1


def _get_num_gpu(config) -> int:
    if isinstance(config, dict):
        num_gpu = config.get('num_gpu')
        if num_gpu is not None:
            return int(num_gpu)
        trainer_cfg = config.get('trainer', {})
        if isinstance(trainer_cfg, dict):
            return int(trainer_cfg.get('num_gpu', 0))
        return int(getattr(trainer_cfg, 'num_gpu', 0))
    return int(config.get('num_gpu', getattr(config.trainer, 'num_gpu', 0)))


def create_meta_data_list(config, start_cutoff=5):
    global end_cutoff
    dataset_cfg = config['dataset']
    shuffle = dataset_cfg['shuffle_metadata']
    train_list = []
    test_lists = []
    valid_list = []
    batch_size = config['train_parameter']['batch_size']
    num_gpu = _get_num_gpu(config)
    is_block = dataset_cfg['is_block']
    if is_block:
        is_block_part = dataset_cfg['is_block_part']
    else:
        is_block_part = False
    if is_block:
        block_size = dataset_cfg['block_size']
    else:
        block_size = 0

    if "sep" in dataset_cfg['mode']:
        train_scenes = list(dataset_cfg['train_scene'])
        assert len(train_scenes) > 0, f'config["dataset"]["train_scene"] must be no empty!,\n\
config["dataset"]:{dict_to_string(dataset_cfg)}'
        for item in train_scenes:
            dir_name = item['name']
            path = config['job_config']['dataset_path'][item['config'].get('path_alias', 'default')]
            res = glob.glob(f"{path}/{dir_name}/{config['buffer_config']['basic_part_enable_list'][0]}/[0-9]*.npz")
            assert len(res) > 0, f"{config['buffer_config']['basic_part_enable_list'][0]} in {path}/{dir_name} not found.\
\n(path_alias: {item['config'].get('path_alias', 'default')}) in config: {item['config']}"
            log.debug(f"{path}/{dir_name}/{config['buffer_config']['basic_part_enable_list'][0]}/[0-9]*.npz")
            log.debug(path)
            log.debug(dir_name)
            log.debug(len(res))
            num = len(res) - start_cutoff - end_cutoff
            index = np.arange(start_cutoff, start_cutoff + num)
            sep_rule = item['config'].get('indice', [])
            if len(sep_rule) == 1:
                num = min(sep_rule[0], num)
                index = np.arange(start_cutoff, start_cutoff + num)
            elif len(sep_rule) == 2:
                start = sep_rule[0]
                end = min(num, sep_rule[1])
                num = end - start
                index = np.arange(start_cutoff + start, start_cutoff + end)

            if is_block:
                index = index[:-block_size - 1:block_size]
                num = len(index)

            train_list += [MetaDataWithPath(dir_name, int(index[i]), item['config'].get('path_alias', 'default'))
                           for i in range(num)]

            log.info("train_scene: {}, path: {} len: {}".format(
                dir_name,
                path,
                num))

        test_scenes = list(dataset_cfg['test_scene'])
        for item in test_scenes:
            dir_name = item['name']
            path = config['job_config']['dataset_path'][item['config'].get('path_alias', 'default')]
            res = glob.glob(f"{path}/{dir_name}/{config['buffer_config']['basic_part_enable_list'][0]}/[0-9]*.npz")
            assert len(res) > 0, f"{config['buffer_config']['basic_part_enable_list'][0]} in {path}/{dir_name} not found.\nconfig: {item}"
            num = len(res) - start_cutoff - end_cutoff
            index = np.arange(start_cutoff, start_cutoff + num)
            sep_rule = item['config'].get('indice', [])
            if len(sep_rule) == 1:
                end = sep_rule[0]
                index = index[:end]
            elif len(sep_rule) == 2:
                start = sep_rule[0]
                end = sep_rule[1]
                index = index[start:end]
            if is_block:
                if not is_block_part:
                    index = index[:-block_size+1:block_size]
            num = len(index)
            test_lists.append([MetaDataWithPath(dir_name, int(index[i]), item['config'].get('path_alias', 'default'))
                               for i in range(num)])
            log.info("test_scene: {}, path: {} len: {}".format(
                dir_name,
                path,
                num))
    else:
        raise NotImplementedError(
            f"create dataset with {dataset_cfg['mode']} mode, but only 'seq' mode supported for dataset!")

    is_initial_shuffle_metadata = True
    if is_initial_shuffle_metadata:
        random.seed(2025)
        random.shuffle(train_list)

    if shuffle:
        random.seed(time.time())
        random.shuffle(train_list)

    train_scale = dataset_cfg.get("train_scale", 1)

    if train_scale != 1:
        log.debug(f"train_scale={train_scale}, scaling train_list(len={len(train_list)})")
        np.random.seed(2025)
        train_ind = np.random.choice(np.arange(len(train_list), dtype=int), int(len(train_list) * train_scale), replace=False)
        train_list = list(np.array(train_list)[train_ind])
        log.debug(f"scaled train_list(len={len(train_list)})")

    if is_block:
        minimum_total_size = num_gpu * batch_size
        while len(train_list) % (minimum_total_size) != 0:
            train_list += train_list[:minimum_total_size - len(train_list) % minimum_total_size]

        if is_block_part:
            def generate_block_metadata(block_list: list[MetaDataWithPath], _batch_size, _num_gpu, _block_size):
                part_size = dataset_cfg['part_size']
                assert _block_size % part_size == 0
                _minimum_total_size = _num_gpu * _batch_size
                assert len(block_list) % _minimum_total_size == 0
                expand_list = []
                for md in block_list:
                    expand_list.append(md)
                    for block_id in range(part_size, _block_size, part_size):
                        expand_list.append(md.get_offset(block_id))
                len_expand_list = len(expand_list)
                num_part_per_block = _block_size // part_size
                assert len_expand_list == len(block_list) * num_part_per_block
                ret_list = []
                len_batched_seq = _batch_size * num_part_per_block
                for seq_id in range(len_expand_list // len_batched_seq):
                    cut_list = expand_list[seq_id * len_batched_seq: (seq_id + 1) * len_batched_seq]
                    for block_id in range(num_part_per_block):
                        for batch_id in range(0, _batch_size):
                            ret_list.append(cut_list[batch_id * num_part_per_block + block_id])
                return ret_list

            train_list = generate_block_metadata(train_list, batch_size, num_gpu, block_size)
    else:
        minimum_total_size = batch_size * num_gpu
        while len(train_list) % (minimum_total_size) != 0:
            train_list += train_list[:minimum_total_size - len(train_list) % minimum_total_size]
    log.debug("train: {} ... {} len={}".format(str(train_list[:3]), str(train_list[-3:]), len(train_list)))
    log.debug("test: {} ... {} len={}".format(str(test_lists[0][:3]), str(test_lists[0][-3:]), len(test_lists)))
    log.info("complete creating metadata.")
    return train_list, valid_list, test_lists


import copy
import gc
import json
import math
import multiprocessing as mp
import os
import re
import time
import zipfile
from glob import glob
from typing import Optional
from zipfile import BadZipfile

import numpy as np
import torch
from tqdm import tqdm
from utils.buffer_utils import aces_tonemapper
from utils.dataset_utils import create_warped_buffer, write_npz
# from .patch_cropper import crop
from utils.utils import (create_dir, del_dict_item, get_file_component,
                         write_text_to_file)
from wickit.dataloaders.asset_loader_abc import AssetLoaderABC
from wickit.dataloaders.metadata import MetaData
from wickit.utils.basic.string import dict_to_string
from wickit.utils.basic.tensor import data_as_type, data_to_device
from wickit.utils.log import log

from .moflow_metadata_task_utils import create_metadata_by_glob
from .raw_data_importer import (DatasetFormat, UE4RawDataLoader,
                                compress_buffer, get_augmented_buffer,
                                get_extend_buffer, parse_buffer_name,
                                split_buffer)

g_augmented_data_output = None
def history_extend(data, config):
    # assert isinstance(data, dict), dict_to_string(data)
    global g_augmented_data_output
    # buffer_config['max_luminance'] = 128
    buffer_config = config['buffer_config']
    history_data_list = data['history_data_list']
    history_config = config['dataset']['history_config']
    future_data_list = data['future_data_list']
    future_config = config['dataset']['future_config']
    require_list = config['dataset']['require_list']
    # ignore_list = config['dataset']['ignore_list']
    
    assert len(history_data_list) == history_config['num']
    for ind, history_data in enumerate(history_data_list):
        # if len(list(history_data.keys())) <= 0:
        #     continue
        get_augmented_buffer(
            history_config['augmented_data'][ind],
            buffer_config,
            history_data,
            last_data=[],
            allow_skip=history_config['allow_skip'],
            with_batch=True)
        for item in history_config['augmented_data'][ind]:
            assert item in history_data.keys()
        
    if future_config is not None:
        assert len(future_data_list) == future_config['num']
        for ind, future_data in enumerate(future_data_list):
            # if len(list(future_data.keys())) <= 0:
            #     continue
            get_augmented_buffer(
                future_config['augmented_data'][ind],
                buffer_config,
                future_data,
                last_data=[],
                allow_skip=future_config['allow_skip'],
                with_batch=True)
        
    if len(require_list) > 0 and g_augmented_data_output is None:
        g_augmented_data_output = []
        ''' if data_config isnt empty, we continue generating. '''
        if buffer_config is not None:
            # log.debug(dict_to_string(self.buffer_config))
            # log.debug(self.require_filter)
            # log.debug(dict_to_string(data))
            for item in require_list:
                if item not in data.keys():
                    res = parse_buffer_name(item)
                    pref = res['sample']
                    buffer_name = res['buffer_name']
                    frame_id = res['frame_id']
                    postf = res['postf']
                    if "history_masked_warped_scene_light_0_extranet" in item:
                        continue
                    if item in ['masked_scene_light_no_st', 'masked_scene_color_no_st', 'render_mask', 
                                'shading_age', 'shading_confidence', 'history_pixel_render_mask_0',
                                'history_forward_warped_pixel_render_mask_0', 'history_pixel_render_mask_final_0',
                                'history_pred_render_mask_0', 'history_pred_flow_0', 'history_pred_0']:
                        continue
                    # log.debug("'{}' '{}' '{}' '{}'".format(item, buffer_name, his_id, pf))
                    if pref + buffer_name + postf in buffer_config['augmented_data_recipe'].keys():
                        g_augmented_data_output.append(item)
                    else:
                        raise NotImplementedError("{}({}) is required but not in augmented_data_recipe.keys ({})\nrequire_list:{}".format(
                            pref + buffer_name + postf, item, buffer_config['augmented_data_recipe'].keys(), require_list))

    # log.debug(dict_to_string([data, g_augmented_data_output]))
    get_augmented_buffer(
        g_augmented_data_output,
        buffer_config,
        data,
        last_data=history_data_list,
        next_data=future_data_list,
        allow_skip=False,
        with_batch=True)
    
    # assert False, dict_to_string({
    #     'require_list': require_list,
    #     'data': data,
    # })
    # if len(require_list) > 0:
    ''' FIXME: temporary disable to allow all data to be loaded '''
    if len(require_list) > 0 and False:
        ret = {}
        for k in data.keys():
            if k in ['history_data_list', 'future_data_list']:
                continue
            if k in require_list or not isinstance(data[k], torch.Tensor):
                ret[k] = data[k]
        data.clear()
        # torch.cuda.empty_cache()
        data = ret
    
    return data


retry_num = 6
sleep_time = 10


class AssetLoader(AssetLoaderABC):
    # use MetaData to load a series of pass of a frame, return tersor data
    cache_dict = {}
    last_cached_key = []

    def __init__(self, data_part, buffer_config={}, cache_size = 4,
                 job_config={}, require_list=[], augmented_data_output=None, with_augment=True):
        self.job_config = job_config
        self.dataset_format = DatasetFormat.get_by_str(job_config['dataset_format'])
        self.export_path = self.job_config['export_path']
        self.buffer_config = buffer_config
        self.augmented_data_output = augmented_data_output
        self.data_part = data_part
        self.last_data = {}
        self.require_list = require_list
        self.__debug_cache_hit = 0 
        self.__debug_load_total = 0
        self.__debug_cache_miss = 0
        self.__debug_cache_part_miss = 0
        self.with_augment = with_augment
        self.cache_size = cache_size
        ''' always False'''
        # self.gpu = True
        self.gpu = False
        
        self.enable_cache = True
        # self.enable_cache = False
        log.debug(f"require_list: {self.require_list}")
        
    def load_patch(self, metadata, part_name="metadata", path=None):
        ret = None
        self.__debug_load_total += 1
        if self.enable_cache:
            cache_key = f"{metadata.__str__()}"
            if cache_key in AssetLoader.cache_dict.keys():
                if part_name in AssetLoader.cache_dict[cache_key].keys():
                    ret = copy.deepcopy(AssetLoader.cache_dict[cache_key][part_name])
                    self.__debug_cache_hit += 1
                else:
                    self.__debug_cache_part_miss += 1
                    # log.debug(dict_to_string({
                    #     'query_part_name':  f"{cache_key}.{part_name}",
                    #     'cached_part_name': list(AssetLoader.cache_dict[cache_key].keys())
                    # }))
            else:
                self.__debug_cache_miss += 1
            # log.debug(dict_to_string({
            #     'cache_hit_ratio': self.__debug_cache_hit / self.__debug_load_total,
            #     'cache_miss_ratio': self.__debug_cache_miss / self.__debug_load_total,
            #     'cache_part_miss_ratio': self.__debug_cache_part_miss / self.__debug_load_total
            # }))
        else: cache_key = ""
        
        if ret is not None:
            return ret
        
        if part_name in ['metadata'] or self.dataset_format == DatasetFormat.NPZ:
            tmp_data = self.load_npz(metadata, path=path, part_name=part_name)
        elif self.dataset_format == DatasetFormat.HDF5:
            tmp_data = self.load_hdf5(metadata, path=path, part_name=part_name)
        else:
            assert False, dict_to_string(self.job_config)
        
        ret = tmp_data
        if self.enable_cache:
            if cache_key not in AssetLoader.cache_dict.keys():
                AssetLoader.cache_dict[cache_key] = {}
                AssetLoader.last_cached_key.append(cache_key)
                if len(AssetLoader.last_cached_key) > self.cache_size:
                    # del_key = list(AssetLoader.cache_dict.keys())[-1]
                    del_key = AssetLoader.last_cached_key[0]
                    # log.debug(f"delete key:{del_key}")
                    del AssetLoader.cache_dict[del_key]
                    del AssetLoader.last_cached_key[0]
                    gc.collect()
            AssetLoader.cache_dict[cache_key][part_name] = ret
            # AssetLoader.cache_dict[cache_key][part_name] = copy.deepcopy(ret)

        return ret

    def load_npz(self, metadata, part_name="metadata", path=None):
        file_path = self.get_file_path_npz(
            metadata, part_name, path=path)
        data = None
        for i in range(retry_num):
            try:
                data = np.load(file_path, allow_pickle=True)
                break
            except FileNotFoundError:
                log.debug("not found in:{}, retry_time: {}".format(file_path, i + 1))
                time.sleep(sleep_time)
            except zipfile.BadZipFile:
                log.debug("bad zip found in:{}, retry_time: {}".format(file_path, i + 1))
                time.sleep(sleep_time)
        if data is None:
            log.debug("ERROR. file_path:{}".format(file_path))
            data = np.load(file_path, allow_pickle=False)
        assert data is not None
        ret = []
        for name in data.files:
            if name == 'allow_pickle':
                continue
            else:
                for i in range(retry_num):
                    try:
                        item = data[name].item()
                        break
                    except Exception as e:
                        assert False, "error: {}, file_path:{}, retry_time: {}".format(e, file_path, i + 1)
                    time.sleep(sleep_time)
                ret.append(item)
                break
        ret = ret[0]
        ''' FIXME: new version of pytorch will preserve 'cuda:0' device in the loaded data, so we need to move it back to cpu '''
        ret = data_to_device(ret, 'cpu')
        ''' FIXME: cant work with multi-core dataloader '''
        return ret
    
    def load_hdf5(self, metadata, part_name="metadata", path=None):
        file_path = self.get_file_path_hdf5(
            metadata, part_name, path=path)
        data = read_hdf5(file_path, part_name, metadata.index)
        # log.debug(dict_to_string(part_name))
        # log.debug(dict_to_string(self.buffer_config['part'][part_name]['buffer_name']))
        # log.debug(dict_to_string(data))
        ret = split_buffer(data, self.buffer_config['part'][part_name]['buffer_name'])
        ''' cant work with multi-core dataloader '''
        return ret

    def load_data(self, metadata: MetaData, path=None, data_part=None, data_type:Optional[torch.dtype]=None) -> dict:
        data = {}
        if data_part is None:
            data_part = self.data_part
        assert isinstance(data_part, list), f"data_part: {data_part}"
        if len(data_part) <= 0:
            raise ValueError(f"can't load data without data_part ({data_part}) set.")
        for part in data_part:
            tmp_data = self.load_patch(metadata, path=path, part_name=part)
            # log.debug(dict_to_string(tmp_data, part))
            data.update(tmp_data)
            
        prefs = ['']
        for scale_name in self.buffer_config['scale_regex']:
            scale_config = self.buffer_config['scale_regex'][scale_name]
            if scale_config['enable']:
                prefs.append(scale_config['target'].format(int(scale_config['value'])) + "_")
                
        if data_type is not None:
            data = data_as_type(data, data_type)
        
        for k in data.keys():
            if isinstance(data[k], np.ndarray):
                data[k] = torch.from_numpy(data[k])
            if 'scene_color_no_st' in k or 'st_color' in k or 'sky_color' in k:
                data[k] = torch.clamp(data[k], min=0)
            if 'motion_vector' in k:
                data[k] = torch.clamp(data[k], -1.0, 1.0)
            if 'st_alpha' in k:
                data[k] = torch.clamp(data[k], 0.0, 1.0)
        # if 'scene_light' in data.keys():
        #     if self.buffer_config['brdf_demodualte']:
        #         get_augmented_buffer(['brdf_color'], self.buffer_config, data)
        #         self.scene_light
        ''' temporary solution for small number in brdf_color '''
        # for pref in prefs:
        #     new_key = pref + 'brdf_color'
        #     if new_key in data.keys():
        #         data[new_key] = torch.clamp(data[new_key], min=0.01)

        # if 'a64_scene_color' in data.keys():
        #     data['scene_color'] = data['a64_scene_color']
        # if 'a64_brdf_color' in data.keys():
        #     data['brdf_color'] = data['a64_brdf_color']

        # if 'scene_color' in data.keys():
        #     data['scene_color'] = aces_tonemapper(data['scene_color'])
        # if 'sky_color' in data.keys():
        #     data['sky_color'] = aces_tonemapper(data['sky_color'])
        # if 'st_color' in data.keys():
        #     data['sky_color'] = aces_tonemapper(data['st_color'])

        ''' re-composition '''
        ''' get exact sc_no_st for raw'''
        # data['brdf_color'] =
        # comp_sc_no_st = create_scene_color_no_st(data['scene_color'], data['st_color'], data['st_alpha'])
        # comp_sc_no_st = torch.clamp(data['comp_sc_no_st'], min=0, max=500.0)

        # # ''' get sl_no_st for GT '''
        # comp_sl_no_st = create_de_color_by_brdf(data['comp_sc_no_st'], data['brdf_color'])
        # comp_sl_no_st = torch.clamp(data['comp_sl_no_st'], min=0.0, max=500.0)

        # ''' regenerate sc for GT '''
        # comp_sc = (data['comp_sl_no_st'] * data['brdf_color'] * (1 - data['skybox_mask'])
        #                 + data['sky_color'] * data['skybox_mask']) * data['st_alpha'] + data['st_color']

        # ''' regenerate sc_no_st for GT '''
        # comp_sc_no_st = data['comp_sl_no_st'] * data['brdf_color']
        # data['scene_color'] = comp_sc
        # data['scene_color_no_st'] = comp_sc_no_st
        if self.gpu:
            data = data_to_device(data, "cuda")
        assert isinstance(data, dict), dict_to_string(data)
        return data

    def load(self, metadata: MetaData, path=None, history_config=None, future_config=None, allow_skip=False, data_type:Optional[torch.dtype]=None) -> dict:
        '''
        history_config = {
            'num':1,
            'part':['fp16', 'fp32']
            'augmented_data':[]
        }
        '''
        # log.debug(metadata.__str__())
        # log.debug(self.job_config['dataset_path'])
        if hasattr(metadata, "path_alias"):
            path = self.job_config['dataset_path'][getattr(metadata, "path_alias")]
            # log.debug(f'load metadata: {metadata.__str__()}, from {path}')
        data = self.load_data(metadata, path=path, data_type=data_type)
        if history_config is not None:
            if history_config['generate_from_data']:
                num = history_config['num']
                history_data_list = [{} for _ in range(num)]
                for k in data.keys():
                    res = parse_buffer_name(k)
                    pref = res['sample']
                    buffer_name = res['buffer_name']
                    his_id = res['his_id']
                    postf = res['postf']
                    if buffer_name.startswith("history_") and "warped" not in buffer_name \
                            and his_id is not None and his_id < num:
                        history_data_list[his_id][pref + buffer_name.replace("history_", "") + postf] = data[k]
            else:
                inds = history_config.get("index", range(history_config["num"]))
                history_data_list = list(reversed([self.load_data(
                    metadata.get_offset(-i - 1), path=path, data_part=history_config['part'][i], data_type=data_type)
                    for i in reversed(inds)]))

            ''' augmenting history data '''
            if self.with_augment:
                for ind, history_data in enumerate(history_data_list):
                    # if len(list(history_data.keys())) <= 0:
                    #     continue
                    get_augmented_buffer(
                        history_config['augmented_data'][ind],
                        self.buffer_config,
                        history_data,
                        last_data=[],
                        allow_skip=history_config['allow_skip'])
        else:
            history_data_list = []

        if future_config is not None:
            inds = future_config.get("index", range(future_config["num"]))
            future_data_list = list([self.load_data(
                metadata.get_offset(i + 1), path=path, data_part=future_config['part'][i], data_type=data_type)
                for i in inds])
            ''' augmenting history data '''
            if self.with_augment:
                for ind, future_data in enumerate(future_data_list):
                    # log.debug(dict_to_string(future_data))
                    # if len(list(future_data.keys())) <= 0:
                    #     continue
                    get_augmented_buffer(
                        future_config['augmented_data'][ind],
                        self.buffer_config,
                        future_data,
                        last_data=[],
                        allow_skip=future_config['allow_skip'])
        else:
            future_data_list = []
            
            
        # log.debug(dict_to_string(data))
        ''' if augmented_data_output list is empty, then generate it. '''
        if self.with_augment:
            if len(self.require_list) > 0 and self.augmented_data_output is None:
                self.augmented_data_output = []
                ''' if data_config isnt empty, we continue generating. '''
                if self.buffer_config is not None:
                    # log.debug(dict_to_string(self.buffer_config))
                    # log.debug(self.require_filter)
                    # log.debug(dict_to_string(data))
                    for item in self.require_list:
                        if item not in data.keys():
                            res = parse_buffer_name(item)
                            pref = res['sample']
                            buffer_name = res['buffer_name']
                            his_id = res['frame_id']
                            postf = res['postf']
                            ''' temporay exception for history_masked_warped_scene_color_extranet '''
                            if item == "history_masked_warped_scene_color_0_extranet":
                                buffer_name = item
                            elif "extra" in item:
                                continue
                            # log.debug("'{}' '{}' '{}' '{}'".format(item, buffer_name, his_id, pf))
                            if pref + buffer_name + postf in self.buffer_config['augmented_data_recipe'].keys():
                                self.augmented_data_output.append(item)
                            else:
                                raise NotImplementedError("{}({}) is required but not in augmented_data_recipe.keys ({})".format(
                                    pref + buffer_name + postf, item, self.buffer_config['augmented_data_recipe'].keys()))

            get_augmented_buffer(
                self.augmented_data_output,
                self.buffer_config,
                data,
                last_data=history_data_list,
                next_data=future_data_list,
                allow_skip=allow_skip)

            if len(self.require_list) > 0:
                ret = {}
                for k in data.keys():
                    if k in self.require_list:
                        ret[k] = data[k]
                del data
                data = ret
        else:
            data['history_data_list'] = history_data_list
            data['future_data_list'] = future_data_list
            # log.debug(dict_to_string(ret, "after filter"))
        # log.debug(f"{os.getpid()} cached ratio: {self._debug_hit / self._debug_total}")
        if self.gpu:
            data_cpu = data_to_device(data, "cpu")
            del data
            torch.cuda.empty_cache()
            gc.collect()
        else:
            data_cpu = data
        # for k in data.keys():
        #     if isinstance(data[k], torch.Tensor):
        #         data[k] = data[k].type(torch.float32)
        return data_cpu

    def get_file_path_npz(self, metadata: MetaData, dir_name: str, postfix="", path=None):
        if path is None:
            path = self.export_path
        return "{}/{}/{}/{}.npz".format(
            path, metadata.dataset_name, dir_name, "{}{}".format(metadata.index, postfix))

    def get_buffered_last_data(self, metadata: MetaData, buffer_config, num: int = 1, augmented_list=[]):
        ret = []
        # if len(augmented_list) == 0:
        #     return ret
        for i in range(num):
            cur_key = str(metadata.get_offset(-i - 1))
            if cur_key not in self.last_data.keys():
                data = self.load_data(metadata.get_offset(-i - 1), data_type=torch.float32)
                # log.debug(dict_to_string(data))
                # log.debug(augmented_output)
                if len(augmented_list) > 0:
                    get_augmented_buffer(augmented_list,
                                         buffer_config,
                                         data,
                                         [],
                                         allow_skip=False)
                log.debug("{} not found, regenerated.".format(cur_key))
                self.last_data[cur_key] = data
            else:
                data = self.last_data[cur_key]
                if len(augmented_list) > 0:
                    get_augmented_buffer(augmented_list,
                                         buffer_config,
                                         data,
                                         [],
                                         allow_skip=False)
                self.last_data[cur_key] = data
            ret.append(self.last_data[cur_key])
        return ret


    def extend(self, metadata: MetaData, start_cutoff=5):
        if metadata.index < start_cutoff:
            return
        
        overwrite = self.job_config.get('overwrite', False)
        ''' overwrite handler '''
        if not overwrite:
            assert self.dataset_format == DatasetFormat.NPZ
            todo_part_list = []
            for part_name in self.buffer_config['addition_part_enable_list']:
                path = self.get_file_path_npz(metadata, part_name)
                if os.path.exists(path):
                    log.debug("{} exists.".format(path))
                else:
                    todo_part_list.append(part_name)
        else:
            todo_part_list = self.buffer_config['addition_part_enable_list']
            
        if len(todo_part_list) <= 0:
            return

        origin_data = self.load_data(metadata, data_type=torch.float32)
        self.last_data[str(metadata)] = origin_data
        if metadata.index>start_cutoff and str(metadata.get_offset(-start_cutoff-1)) in self.last_data:
            del self.last_data[str(metadata.get_offset(-start_cutoff-1))]
        extended_patch = {}
        ''' extend process '''
        for part_name in todo_part_list:
            part_config = self.buffer_config['part'][part_name]
            log.debug(f'extending {metadata} {part_name}')
            last_datas = self.get_buffered_last_data(metadata, self.buffer_config,
                                                     num=part_config['num_history'],
                                                     augmented_list=part_config['history_augmented_data'])
            extended_patch[part_name] = get_extend_buffer(origin_data, part_name,
                                                          self.buffer_config,
                                                          start_cutoff=0,
                                                          last_datas=last_datas)
        if metadata.index == 5:
            for k in extended_patch.keys():
                log.info(dict_to_string(extended_patch[k], k))
        for k in extended_patch.keys():
            if k in self.buffer_config['basic_part_enable_list']:
                continue
            path = self.get_file_path_npz(metadata, k)
            log.debug(dict_to_string(extended_patch[k], k, mmm=True))
            create_dir(get_file_component(path)['path'])
            scene_path = os.path.join(get_file_component(path)['path'], "..")
            write_npz(path, extended_patch[k])
            if metadata.index >= 5 and metadata.index <= 10:
                write_text_to_file(scene_path + "/" + "{}.txt".format(k),
                                   json.dumps(list(extended_patch[k].keys()), indent=4).replace(
                    "true", "True").replace("false", "False").replace("null", "None"), "w")
            log.debug(f"write npz: {path}")

    def export_extend_patch_range(self, metadatas, start, end):
        log.debug("start: {}, end: {}".format(start, end))
        for i in tqdm(range(start, end)):
            self.extend(metadatas[i])
            # torch.cuda.empty_cache()
            # gc.collect()

    def export_extend_patch(self):
        for s in self.job_config['scene']:
            file_name_list = glob(
                "{}/{}/{}/*.npz".format(self.export_path, s, self.buffer_config["metadata_part"]))
            num = len(file_name_list)
            log.debug("{}/{}/{}/*.npz, num: {}".format(self.export_path, s, self.buffer_config["metadata_part"], num))
            metadatas = []
            for i in range(0, num):
                metadatas.append(MetaData(s, i))

            ''' single thread '''
            if self.job_config.get('num_thread', 0) <= 0:
                self.export_extend_patch_range(metadatas, 0, len(metadatas))
                continue

            ''' multi thread '''
            n_core = self.job_config['num_thread']
            pool = mp.Pool(processes=n_core)
            thread_part = max(num // n_core + 1, 1)
            try:
                log.debug("scene:{} n_core:{} thread_part:{}".format(
                    s, n_core, thread_part))

                _ = [pool.apply_async(self.export_extend_patch_range,
                                      (metadatas, i * thread_part,
                                       min((i + 1) * thread_part, num)),
                                      callback=None)

                     for i in range(n_core)]
                pool.close()
            except KeyboardInterrupt:
                pool.terminate()
            finally:
                pool.join()

    def release(self) -> None:
        return
    
    

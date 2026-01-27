import os
import re
import shutil
import torch

cached_cat_shape = {}
class TensorConcator:
    def __init__(self, name, use_cache=False):         
        self.cat_shape_name = name
        self.use_cached_shape = False
        self.cur_channel_idx = 0
        self._already_got = False
        self.use_cache = False
        # self.use_cache = use_cache
        if use_cache and self.cat_shape_name in cached_cat_shape.keys():
            self.cat_tensor = cached_cat_shape[self.cat_shape_name].detach()
            self.max_channel = self.cat_tensor.shape[1]
            self.use_cached_shape = True
            # log.debug(cached_cat_shape.keys())
            # log.debug(f"TensorConcator: use_cache, name:{self.cat_shape_name}, shape:{self.cat_tensor.shape}")
        else:
            self.skip_conn_arr = []
        
    def add(self,x):
        assert len(x.shape) == 4
        if self.use_cached_shape:
            cat_channel = x.shape[1]
            assert self.cur_channel_idx + cat_channel <= self.max_channel
            self.cat_tensor[:,self.cur_channel_idx:self.cur_channel_idx + cat_channel] = x
            self.cur_channel_idx += cat_channel
        else:
            if self.skip_conn_arr:
                assert x.shape[2] == self.skip_conn_arr[0].shape[2], f"{x.shape[2]}, {self.skip_conn_arr[0].shape[2]}"
                assert x.shape[3] == self.skip_conn_arr[0].shape[3], f"{x.shape[3]}, {self.skip_conn_arr[0].shape[3]}"
            self.skip_conn_arr.append(x)
    
    def get(self):
        assert not self._already_got
        if self.use_cached_shape:
            assert self.cur_channel_idx == self.max_channel, 'concat is not complete'
            return self.cat_tensor
        else:
            assert self.cat_shape_name not in cached_cat_shape.keys(), 'the tensor name was already taken'
            ret = torch.cat(self.skip_conn_arr, dim=1)
            if self.use_cache:
                cached_cat_shape[self.cat_shape_name] = torch.zeros_like(ret, dtype=ret.dtype, device=ret.device)
            self._already_got = True
            return ret 
        

cached_split_list = {}
class TensorSplitor:
    def __init__(self, layer_id):
        self.available = layer_id in cached_split_list.keys()
        self.layer_id = layer_id
        self.split_idx = 0
        if self.available:
            self.slice_idx_list = cached_split_list[layer_id]
        else:
            self.slice_idx_list = []
        
    def is_available(self):
        return self.available
        
    def get_next(self, tensor):
        assert self.available
        assert self.split_idx <= len(self.slice_idx_list)
        if self.split_idx == len(self.slice_idx_list):
            slice_idx = self.slice_idx_list[self.split_idx-1]
            self.split_idx += 1
            return tensor[:, slice_idx[1]:]
        slice_idx = self.slice_idx_list[self.split_idx]
        ret = tensor[:, slice_idx[0]: slice_idx[1]]
        self.split_idx += 1
        return ret
        
    def record(self, slice_idx):
        self.slice_idx_list.append(slice_idx)
        
    def end(self):
        if not self.available:
            cached_split_list[self.layer_id] = self.slice_idx_list
            self.available = True


def add_metaname(ins, base):
    base_class = base
    if getattr(ins, "full_name", None) is not None:
        ins.full_name = '{}_{}{}'.format(ins.full_name, base_class.class_name, base_class.cnt_instance)
    else:
        setattr(ins,"full_name",'{}{}'.format(base_class.class_name, base_class.cnt_instance))
        setattr(ins,"name",'{}{}'.format(base_class.class_name, base_class.cnt_instance))
    base_class.cnt_instance += 1


def del_dict_item(data: dict, k: str) -> dict:
    del data[k]
    return data


def del_data(data):
    if isinstance(data, dict):
        key_list = list(data.keys())
        for k in key_list:
            data = del_dict_item(data, k)
        return data
    elif isinstance(data, list):
        for i in range(len(data)):
            # FIXME: may not work
            data[i] = del_data(data[i])
        return data
    elif isinstance(data, torch.Tensor):
        del data
    else:
        del data


class Accumulator:

    def __init__(self):
        self.data = None
        self.last_data = None
        self.cnt = 0

    def add(self, data):
        if torch.isinf(torch.tensor(data)) or torch.isnan(torch.tensor(data)):
            return
        if self.cnt == 0:
            self.data = data
        else:
            self.data += data
        self.last_data = data
        self.cnt += 1

    def get(self):
        if self.cnt == 0:
            return "no data."
        return self.data / self.cnt

    def reset(self):
        self.data = None
        self.cnt = 0

def format_time_str_from_colon_to_labels(time_str: str):
    if time_str == '-':
        return time_str
    parts = time_str.split(':')
    parts = [str(int(p)) for p in parts]
    unit_mapping = {
        4: ['d', 'h', 'm', 's'],
        3: ['h', 'm', 's'],
        2: ['m', 's'],
        1: ['s']
    }
    assert (num_parts:=len(parts)) in unit_mapping.keys()
    units = unit_mapping[num_parts]
    return ''.join(f"{part}{unit}" for part, unit in zip(parts, units))

def seconds_to_str(seconds: int, unit=True):
    if seconds is None:
        return '-'
    seconds = int(seconds)
    d, remainder = divmod(seconds, 3600 * 24)
    h, remainder = divmod(remainder, 3600)
    m, s = divmod(remainder, 60)

    if d:
        ret = '{:02d}:{:02d}:{:02d}:{:02d}'.format(d, h, m, s)
    elif h:
        ret = '{:02d}:{:02d}:{:02d}'.format(h, m, s)
    elif m:
        ret = '{:02d}:{:02d}'.format(m, s)
    else:
        ret = '{:02d}'.format(s)
    if unit:
        ret = format_time_str_from_colon_to_labels(ret)
    return ret

def str_to_seconds(time_str: str) -> int:
    if time_str == '':
        return 0
    parts = time_str.split(':')
    multipliers = [24 * 3600, 3600, 60, 1]
    assert (length:=len(parts)) <= 4
    return sum(int(part) * mult for part, mult in zip(parts, multipliers[-length:]))
    
def create_dir(path):
    if not (os.path.exists(path)):
        os.makedirs(path)
        return True
    return False

def write_text_to_file(path, output, mode="w", mkdir=False):
    if mkdir:
        create_dir(get_file_component(path)['path'])
    f = open(path, mode)
    f.write(output)
    f.close()


def remove_all_in_dir(path, file_name=None):
    if not (os.path.exists(path)):
        return
    if file_name is not None:
        total_path = os.path.join(path, file_name)
        if os.path.isfile(total_path):
            os.remove(total_path)  # 删除特定文件
            return True
        else:
            return False
    content = os.listdir(path)
    if len(content) > 0:
        for item in content:
            total_path = path + "/" + item
            if os.path.isdir(total_path):
                shutil.rmtree(total_path, ignore_errors=True)
            else:
                os.remove(total_path)
        return True
    return False


def get_file_component(file_path):
    file_path = file_path.replace("\\", "/")
    result = re.match("(.*)/(.*)[.](.*)", file_path).groups()
    if len(result) != 3:
        raise RuntimeError(
            "cant get correct component of {}, result:{}".format(file_path, result))
    return {
        'path': result[0],
        'filename': result[1],
        'suffix': result[2],
    }


def is_item_all_in_another(test_arr, another):
    '''check if items of test_arr is all in another

    Args:
        test_arr: List
        another: List
    Returns:
        (1): bool, if items of test_arr is all in another
        (2): <type of item>, first invalid item
    '''
    for a_item in test_arr:
        if a_item not in another:
            return False, a_item
    return True, None


def get_tensor_mean_min_max(t):
    # fp16 will always output nan when using nanmean
    if t.numel() == 0:
        return 0, 0, 0
    if t.dtype == torch.float16:
        return t.float().mean().half(), t.min(), t.max()
    else:
        return t.mean(), t.min(), t.max()


def get_tensor_mean_min_max_str(t, name="", mode="f"):
    return "{{}}: {{:.3{}}} {{:.3{}}} {{:.3{}}}".format(mode, mode, mode).format(name, *get_tensor_mean_min_max(t))

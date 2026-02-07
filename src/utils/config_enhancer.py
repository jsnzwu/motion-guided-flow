import copy
import json
from utils.dataset_utils import get_input_filter_list
from utils.parser_utils import parse_buffer_name
from wickit.utils.basic.string import dict_to_string
from utils.utils import del_dict_item
from wickit.utils.log import log

default_scale_config = {"ds_scale": 2}


def _ensure_unfrozen(config) -> None:
    if hasattr(config, "unfreeze") and getattr(config, "_frozen", False):
        config.unfreeze()


def initialize_recipe(recipe: dict, history_num: int = -1, history_list: list[int] = [], 
                      future_num: int = -1, future_list: list[int] = [],
                      scale_config={}):
    ''' demodulation color buffer '''
    albedo_name = "dmdl_color"
    for buffer_name in recipe['augmented_data_recipe__demodulate_template']:
        target = buffer_name.replace("scene_color", "scene_light")
        recipe['augmented_data_recipe'][target] = {
            "dep": [buffer_name, albedo_name],
        }

    def multi_frame_recipe_dupilication(_frame_type, _num, _frame_list):
        for buffer_name in recipe[f"augmented_data_recipe__{_frame_type}_template"]:
            deps = []
            for i in range(_num):
                if i in _frame_list:
                    deps.append([buffer_name])
                else:
                    deps.append([])
                recipe['augmented_data_recipe'][f'{_frame_type}_{buffer_name}_{i}'] = {
                    'dep': [],
                    f'num_{_frame_type}': _num,
                    f'dep_{_frame_type}': [[buffer_name] if _ == i else [] for _ in range(_num)]
                }
            recipe['augmented_data_recipe'][f'{_frame_type}_{buffer_name}'] = {
                "dep": [],
                f'num_{_frame_type}': _num,
                f'dep_{_frame_type}': deps
            }
            
    ''' history dupilication '''
    if history_num >= 0:
        assert len(history_list) == history_num, f"len(history_list): {len(history_list)}, history_num: {history_num}"
    history_num = history_num if history_num >= 0 else recipe.get("augmented_data_recipe__history_num", 0)
    multi_frame_recipe_dupilication('history', history_num, history_list)
    
    ''' future dupilication '''
    if future_num >= 0:
        assert len(future_list) == future_num, f'future_num: {future_num}, future_list: {future_list}'
    future_num = future_num if future_num >= 0 else recipe.get("augmented_data_recipe__future_num", 0)
    multi_frame_recipe_dupilication('future', future_num, future_list)
        
    ''' merged_motion_vector '''
    recipe['augmented_data_recipe']['merged_motion_vector']['num_history'] = history_num

    ''' history warping '''
    for buffer_name in recipe["augmented_data_recipe__history_warped_template"]:
        dep_history = []
        dep = []
        for i in range(history_num):
            if i in history_list:
                dep_history.append([buffer_name])
                dep.append(f"merged_motion_vector_{i}")
            else:
                dep_history.append([])
        recipe['augmented_data_recipe']["history_warped_" + buffer_name] = {
            "dep": dep,
            "num_history": history_num,
            "dep_history": dep_history
        }

    ''' ssaa prefix dupilication'''
    log.debug(dict_to_string(scale_config))
    
    # attr = recipe['augmented_data_attribute']
    from utils.model_utils import dim2d_dict, dim1d_dict
    from wickit.utils import model as wickit_model
    
    for key,item in recipe['data_attribute'].items():
        if item['type'] == 'image':
            dim2d_dict[key] = item['channel']
            wickit_model.dim2d_dict[key] = item['channel']
        else:
            dim1d_dict[key] = item['channel']
            wickit_model.dim1d_dict[key] = item['channel']
    old_recipe = copy.deepcopy(recipe['augmented_data_recipe'])
    additional_recipe = {}
    for key_scale, config in scale_config.items():
        if not config['enable']:
            continue
        if config['pattern'] not in [r'%ds', r'%aa']:
            continue
        prefix = config["target"].format(config["value"])
        if key_scale == 'aa_scale':
            prefix = 'aa'
        for buffer_name in old_recipe.keys():
            if parse_buffer_name(buffer_name)['basic_element'] in dim2d_dict.keys():
                new_item = copy.deepcopy(
                    recipe['augmented_data_recipe'][buffer_name])
                for index, value in enumerate(new_item["dep"]):
                    if parse_buffer_name(value)['basic_element'] in dim2d_dict.keys():
                        new_item['dep'][index] = f'{prefix}_{value}'
                for i, dep in enumerate(new_item.get('dep_history', [])):
                    for j, value in enumerate(dep):
                        if parse_buffer_name(value)['basic_element'] in dim2d_dict.keys():
                            new_item['dep_history'][i][j] = f'{prefix}_{value}'
                additional_recipe[f'{prefix}_{buffer_name}'] = new_item
    recipe['augmented_data_recipe'].update(additional_recipe)


def enhance_buffer_config(buffer_config, history_num=3, history_list=[], scale_config=None):
    _ensure_unfrozen(buffer_config)
    flag = False
    if scale_config is None:
        scale_config = default_scale_config
    # log.debug(dict_to_string(scale_config))
    for name, value in scale_config.items():
        buffer_config[name] = value
        buffer_config["scale_regex"][name]['value'] = value
        buffer_config["scale_regex"][name]['enable'] = True
        flag = True

    if flag:
        ''' format the scale regex '''
        json_str = json.dumps(buffer_config)
        for scale_name, config in buffer_config["scale_regex"].items():
            if scale_name not in scale_config.keys():
                continue
            json_str = json_str.replace(config['pattern'], config['target'].format(config['value']))
        new_dict = json.loads(json_str)
        new_dict = del_dict_item(new_dict, "scale_regex")
        buffer_config.update(new_dict)

    augmented_data_recipe = copy.deepcopy(buffer_config['augmented_data_recipe'])
    if len(history_list) <= 0:
        history_list = [i for i in range(history_num)]

    initialize_recipe(augmented_data_recipe,
                    history_num=history_num,
                    history_list=history_list,
                    scale_config=buffer_config['scale_regex'])
    # log.debug(dict_to_string(augmented_data_recipe['augmented_data_recipe']))
    buffer_config.update({'augmented_data_recipe': augmented_data_recipe['augmented_data_recipe']})
    # log.debug(dict_to_string(buffer_config['augmented_data_recipe']))


def enhance_train_config(config):
    _ensure_unfrozen(config)
    config.dataset.augment_loader = config.dataset.get("augment_loader", True)
    input_buffer = config.model.get("require_data", {})
    config.dataset.require_list = get_input_filter_list({
        "input_config": config,
        "input_buffer": input_buffer,
    })
    history_config = config.dataset["history_config"]
    history_num = history_config["num"]
    history_list = history_config.get("index", [i for i in range(history_num)])
    history_config["index"] = history_list
    if "demodulation_mode" in config.dataset.keys():
        config.buffer_config["demodulation_mode"] = config.dataset.demodulation_mode
    # log.debug(dict_to_string([history_num, history_list]))
    enhance_buffer_config(
        config.buffer_config,
        history_num=history_num,
        history_list=history_list,
        scale_config=config.dataset.get("scale_config", {}),
    )
    
    ''' set history_config to buffer_config '''
    config.buffer_config["history_config"] = config.dataset.get("history_config", None)
    
    ''' add export_path overwrite'''
    config.dataset.path = config.job_config["export_path"]
    
    
def update_config(config):
    _ensure_unfrozen(config)
    num_gpu = config.runner.num_gpu
    config.runtime.use_ddp = num_gpu > 1
    config.runtime.use_gpu = num_gpu > 0
    config.runtime.device = "cuda:0" if config.runtime.use_gpu else "cpu"
    assert config.train_parameter.batch_size % max(num_gpu, 1) == 0
    config.train_parameter.batch_size = config.train_parameter.batch_size // max(num_gpu, 1)
    assert config.dataset.train_num_worker_sum % max(num_gpu, 1) == 0
    config.dataset.train_num_worker = config.dataset.train_num_worker_sum // max(num_gpu, 1)

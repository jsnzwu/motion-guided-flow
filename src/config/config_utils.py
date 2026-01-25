
from utils.utils import create_dir, get_file_component
from yacs.config import CfgNode
from wickit.utils.basic.string import dict_to_string
from utils.log import get_local_rank, log
            
def merge_from_another(a: CfgNode, b: CfgNode, allow_new=False):
    '''
    merge b into a
    '''
    last_allow_new = a.is_new_allowed()
    if allow_new and not last_allow_new:
        a.set_new_allowed(allow_new)
    a.merge_from_other_cfg(b)

    if allow_new and not last_allow_new:
        a.set_new_allowed(last_allow_new)


def convert_to_dict(cfg_node):
    if not isinstance(cfg_node, CfgNode):
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v)
        return cfg_dict


def create_config(path: str) -> CfgNode:
    try:
        path = path.replace('\\', '/')
        config = CfgNode.load_cfg(open(path, "r"))
        # log.debug(f"CfgNode loaded: {path}")
    except Exception as e:
        log.debug(f"error when creating config {path}, {e}")
    return config


def parse_config(path: str, root_path="") -> CfgNode:
    config = create_config(root_path + path)
    # log.debug(dict_to_string(convert_to_dict(config.get("dataset", None)), root_path + path))
    
    # if 'CV' in path:
    #     log.debug(dict_to_string(convert_to_dict(config.get("dataset", None))))
    # log.debug("{} config: ".format(path)+str(config.get('cuda_visible_devices', None)))
    # log.debug(type(config.get('cuda_visible_devices', None)))
    root_path = config.get("config_root_path", root_path)

    ''' add config as a dict's value.
        the key can be defined with \"include_name\".
    '''
    if includes := config.get('include', []):
        log.debug(f"include path {config.get('include', [])} in {root_path+path}")
    for tmp_path in includes:
        if get_local_rank() == 0:
            log.debug('==== include "{}" >>> "{}"'.format(root_path+tmp_path, path))
        tmp_cfg = parse_config(tmp_path, root_path=root_path)
        file_name = get_file_component(tmp_path)['filename']
        include_name = tmp_cfg.get('include_name', file_name)
        config[include_name] = tmp_cfg

    ''' add feature to current config.
        always with subclass, overwrite the base config.
    '''
    initial_pipeline = CfgNode()
    for tmp_path in config.get('pipeline', []):
        if get_local_rank() == 0:
            log.debug('=== pipeline "{}" >>> "{}":'.format(root_path+tmp_path, path))
        tmp_cfg = parse_config(tmp_path, root_path=root_path)
        # log.debug("pip: {} config: ".format(path)+str(config.get('cuda_visible_devices', None)))
        # log.debug(type(config.get('cuda_visible_devices', None)))
        # log.debug("pip: {} tmp_cfg: ".format(tmp_path)+str(tmp_cfg.get('cuda_visible_devices', None)))
        # log.debug(type(tmp_cfg.get('cuda_visible_devices', None)))
        merge_from_another(initial_pipeline, tmp_cfg, allow_new=True)
        # log.debug("{} pipline {} config: ".format(path, tmp_path)+str(config.get('cuda_visible_devices', None)))
        # log.debug(type(config.get('cuda_visible_devices', None)))

    ''' import baseline config.
        only one base file allowed.
    '''
    # log.debug(dict_to_string(convert_to_dict(config.get("dataset", None))))
    merge_from_another(initial_pipeline, config, allow_new=True)
    config = initial_pipeline
    base_path = config.get('base', None)
    if base_path is not None:
        if get_local_rank() == 0:
            log.debug('=== base "{}" >>> "{}":'.format(root_path+base_path, path))
        # log.debug(">===base {} into {}:".format(tmp_path, path))
        base_cfg = parse_config(base_path, root_path=root_path)
        # log.debug("{} config: ".format(path)+str(config.get('cuda_visible_devices', None)))
        # log.debug(type(config.get('cuda_visible_devices', None)))
        # log.debug("{} tmp_cfg: ".format(tmp_path)+str(tmp_cfg.get('cuda_visible_devices', None)))
        # log.debug(type(tmp_cfg.get('cuda_visible_devices', None)))
        # log.debug(dict_to_string(base_cfg['trainer']['wait_to_start']))
        # log.debug(dict_to_string(config['trainer'].get('wait_to_start', None)))
        merge_from_another(base_cfg, config, allow_new=True)
        config = base_cfg

    return config


def load_config(path: str, root_path: str = "") -> dict:
    config = parse_config(path, root_path=root_path)
    return convert_to_dict(config)


def load_task_config(path: str, root_path: str = ""):
    from utils.config_adapter import dict_to_config

    config = load_config(path, root_path=root_path)
    return dict_to_config(config)

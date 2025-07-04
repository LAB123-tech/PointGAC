import yaml
from easydict import EasyDict


def merge_new_config(config, new_config):
    """
    Merge config files
    :param config:
    :param new_config:
    :return:
    """
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == 'base':
                with open(new_config['base'], 'r') as f:
                    val = yaml.load(f, Loader=yaml.FullLoader)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config


def cfg_from_yaml_file(cfg_file):
    """
    Read config file
    :param cfg_file:
    :return:
    """
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        new_config = yaml.load(f, Loader=yaml.FullLoader)
    merge_new_config(config=config, new_config=new_config)
    return config

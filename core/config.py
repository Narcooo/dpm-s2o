import os
import yaml
from collections import OrderedDict
from datetime import datetime

# class Config():
def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')

def from_yaml(filepath):
    if os.path.isabs(filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            cfg = yaml.safe_load(file.read())
    else:
        filepath = os.path.abspath(os.path.expanduser(filepath))
        with open(filepath, 'r', encoding='utf-8') as file:
            cfg = yaml.safe_load(file.read())
    dict_cfgs = cfg
    return Attr_Dict(cfg), dict_cfgs

class Attr_Dict(object):
    def __init__(self, data):
        for name, value in data.items():
            setattr(self, name, self._wrap(value))
    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)):
            return type(value)([self._wrap(v) for v in value])
        else:
            return Attr_Dict(value) if isinstance(value, dict) else value

def convert_to_config(opt, config):
    opt = vars(opt)
    for key in opt:
        if key not in config:
            config[key] = opt[key]
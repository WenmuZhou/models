# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved. 
#   
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#   
#     http://www.apache.org/licenses/LICENSE-2.0    
#   
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.

import os
import numpy as np
import math
import paddle
import collections
from collections import defaultdict
from collections.abc import Sequence, Mapping
import yaml
import copy
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from ppcv.utils.logger import setup_logger
import ppcv
from ppcv.ops import *

logger = setup_logger('config')

__all__ = ['Config']


class ArgsParser(ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__(
            formatter_class=RawDescriptionHelpFormatter)
        self.add_argument(
            "-o", "--opt", nargs='*', help="set configuration options")

    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        assert args.config is not None, \
            "Please specify --config=configure_file_path."
        args.opt = self._parse_opt(args.opt)
        return args

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split('=', 1)
            if '.' not in k:
                config[k] = yaml.load(v, Loader=yaml.Loader)
            else:
                keys = k.split('.')
                if keys[0] not in config:
                    config[keys[0]] = {}
                cur = config[keys[0]]
                for idx, key in enumerate(keys[1:]):
                    if idx == len(keys) - 2:
                        cur[key] = yaml.load(v, Loader=yaml.Loader)
                    else:
                        cur[key] = {}
                        cur = cur[key]
        return config


class AttrDict(dict):
    """Single level attribute dict, NOT recursive"""

    def __init__(self, **kwargs):
        super(AttrDict, self).__init__()
        super(AttrDict, self).update(kwargs)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError("object has no attribute '{}'".format(key))


def _merge_dict(config, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.

    Args:
        config: dict onto which the merge is executed
        merge_dct: dct merged into config

    Returns: dct
    """
    # for k, v in merge_dct.items():
    #     if (k in dct and isinstance(dct[k], dict) and
    #             isinstance(v, Mapping)):
    #         _merge_dict(dct[k], v)
    #     else:
    #         dct[k] = v
    for key, value in merge_dct.items():
        sub_keys = key.split('.')
        key = sub_keys[0]
        if key in config and len(sub_keys) > 1:
            _merge_dict(config[key], {'.'.join(sub_keys[1:]): value})
        elif key in config and isinstance(config[key], dict) and isinstance(
                value, Mapping):
            _merge_dict(config[key], value)
        else:
            config[key] = value
    return config


def print_dict(cfg, print_func=print, delimiter=0):
    """
    Recursively visualize a dict and
    indenting acrrording by the relationship of keys.
    """
    for k, v in sorted(cfg.items()):
        if isinstance(v, dict):
            print_func("{}{} : ".format(delimiter * " ", str(k)))
            print_dict(v, print_func, delimiter + 4)
        elif isinstance(v, list) and len(v) >= 1 and isinstance(v[0], dict):
            print_func("{}{} : ".format(delimiter * " ", str(k)))
            for value in v:
                print_dict(value, print_func, delimiter + 4)
        else:
            print_func("{}{} : {}".format(delimiter * " ", k, v))


class Config(object):
    def __init__(self, config_path, BASE_KEY='_BASE_'):
        self.BASE_KEY = BASE_KEY
        self.cfg = self._load_config_with_base(config_path)

    def _load_config_with_base(self, file_path):
        """
               Load config from file.

               Args:
                   file_path (str): Path of the config file to be loaded.

               Returns: global config
               """
        _, ext = os.path.splitext(file_path)
        assert ext in ['.yml', '.yaml'], "only support yaml files for now"

        with open(file_path) as f:
            file_cfg = yaml.load(f, Loader=yaml.Loader)

        # NOTE: cfgs outside have higher priority than cfgs in _BASE_
        if self.BASE_KEY in file_cfg:
            all_base_cfg = AttrDict()
            base_ymls = list(file_cfg[self.BASE_KEY])
            for base_yml in base_ymls:
                if base_yml.startswith("~"):
                    base_yml = os.path.expanduser(base_yml)
                if not base_yml.startswith('/'):
                    base_yml = os.path.join(
                        os.path.dirname(file_path), base_yml)

                with open(base_yml) as f:
                    base_cfg = self._load_config_with_base(base_yml)
                    all_base_cfg = _merge_dict(all_base_cfg, base_cfg)

            del file_cfg[self.BASE_KEY]
            file_cfg = _merge_dict(all_base_cfg, file_cfg)
        file_cfg['filename'] = os.path.splitext(os.path.split(file_path)[-1])[
            0]
        return file_cfg

    def merge_dict(self, args):
        self.cfg = _merge_dict(self.cfg, args)

    def print_cfg(self, print_func=print):
        """
        Recursively visualize a dict and
        indenting acrrording by the relationship of keys.
        """
        print_func('----------- Config -----------')
        print_dict(self.cfg, print_func)
        print_func('---------------------------------------------')

    def save(self, p):
        with open(p, 'w') as f:
            yaml.dump(
                dict(self.cfg), f, default_flow_style=False, sort_keys=False)


class ConfigParser(object):
    def __init__(self, args):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        print('args: ', args)
        self.model_cfg, self.env_cfg = self.merge_cfg(args, cfg)
        self.check_cfg()

    def merge_cfg(self, args, cfg):
        env_cfg = cfg['ENV']
        model_cfg = cfg['MODEL']

        def merge(cfg, arg):
            merge_cfg = copy.deepcopy(cfg)
            for k, v in cfg.items():
                if k in arg:
                    merge_cfg[k] = arg[k]
                else:
                    if isinstance(v, dict):
                        merge_cfg[k] = merge(v, arg)
            return merge_cfg

        def merge_opt(cfg, arg):
            for k, v in arg.items():
                if isinstance(cfg, Sequence):
                    k = eval(k)
                    cfg[k] = merge_opt(cfg[k], v)
                else:
                    if (k in cfg and (isinstance(cfg[k], Sequence) or
                                      isinstance(cfg[k], Mapping)) and
                            isinstance(arg[k], Mapping)):
                        merge_opt(cfg[k], arg[k])
                    else:
                        cfg[k] = arg[k]
            return cfg

        args_dict = vars(args)
        for k, v in args_dict.items():
            if k not in env_cfg:
                env_cfg[k] = v
        env_cfg = merge(env_cfg, args_dict)
        print('debug env_cfg: ', env_cfg)
        if 'opt' in args_dict.keys() and args_dict['opt']:
            opt_dict = args_dict['opt']
            if opt_dict.get('ENV', None):
                env_cfg = merge_opt(env_cfg, opt_dict['ENV'])
            if opt_dict.get('MODEL', None):
                model_cfg = merge_opt(model_cfg, opt_dict['MODEL'])

        return model_cfg, env_cfg

    def check_cfg(self):
        unique_name = set()
        unique_name.add('input')
        op_list = ppcv.ops.__all__
        for model in self.model_cfg:
            model_name = list(model.keys())[0]
            model_dict = list(model.values())[0]
            # check the name and last_ops is legal
            if 'name' not in model_dict:
                raise ValueError(
                    'Missing name field in {} model config'.format(model_name))
            inputs = model_dict['Inputs']
            for input in inputs:
                input_str = input.split('.')
                assert len(
                    input_str
                ) > 1, 'The Inputs name should be in format of {last_op_name}.{last_op_output_name}, but receive {} in {} model config'.format(
                    input, model_name)
                last_op = input.split('.')[0]
                assert last_op in unique_name, 'The last_op {} in {} model config is not exist.'.format(
                    last_op, model_name)
            unique_name.add(model_dict['name'])

        device = self.env_cfg.get("device", "CPU")
        assert device.upper() in ['CPU', 'GPU', 'XPU'
                                  ], "device should be CPU, GPU or XPU"

    def parse(self):
        return self.model_cfg, self.env_cfg

    def print_cfg(self):
        print('----------- Environment Arguments -----------')
        buffer = yaml.dump(self.env_cfg)
        print(buffer)
        print('------------- Model Arguments ---------------')
        buffer = yaml.dump(self.model_cfg)
        print(buffer)
        print('---------------------------------------------')

# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import copy
import inspect
from collections.abc import Sequence, Mapping

import warnings


class Register:
    """
    Implement a Register to manager the module.
    The module can be added as either class or function type.

    Args:
        name (str): The name of Register.

    Returns:
        A callable object of Register.

    Examples 1:

        from paddlecv import Register

        register = Register()

        class AlexNet: ...
        class ResNet: ...

        register.register(AlexNet)
        register.register(ResNet)

        # Or pass a sequence alliteratively:
        register.register([AlexNet, ResNet])
        print(register.modules_dict)
        # {'AlexNet': <class '__main__.AlexNet'>, 'ResNet': <class '__main__.ResNet'>}

    Examples 2:

        # Or an easier way, using it as a Python decorator, while just add it above the class declaration.
        from paddlecv import Register

        register = Register()

        @register.register
        class AlexNet: ...

        @register.register
        class ResNet: ...

        print(register.modules_dict)
        # {'AlexNet': <class '__main__.AlexNet'>, 'ResNet': <class '__main__.ResNet'>}
    """

    def __init__(self, name=None, build_func=None):
        self._modules_dict = dict()
        self._name = name
        self._build_func = build_func

    def __len__(self):
        return len(self._modules_dict)

    def __repr__(self):
        name_str = self._name if self._name else self.__class__.__name__
        return "{}:{}".format(name_str, self._modules_dict)

    def __getitem__(self, item):
        if item not in self._modules_dict.keys():
            raise KeyError("{} does not exist in {}".format(item, self))
        return self._modules_dict[item]

    @property
    def modules_dict(self):
        return self._modules_dict

    @property
    def name(self):
        return self._name

    @property
    def build_func(self):
        return self._build_func

    def _add_single_module(self, module):
        """
        Add a single module into the corresponding manager.

        Args:
            module (function|class): A new module.

        Raises:
            TypeError: When `module` is neither class nor function.
            KeyError: When `module` was added already.
        """

        # Currently only support class or function type
        if not (inspect.isclass(module) or inspect.isfunction(module)):
            raise TypeError("Expect class/function type, but received {}".
                            format(type(module)))

        # Obtain the internal name of the module
        module_name = module.__name__

        # Check whether the module was added already
        if module_name in self._modules_dict.keys():
            warnings.warn("{} exists already! It is now updated to {} !!!".
                          format(module_name, module))
            self._modules_dict[module_name] = module

        else:
            # Take the internal name of the module as its key
            self._modules_dict[module_name] = module

    def register(self, modules):
        """
        Add module(s) into the corresponding manager.

        Args:
            modules (function|class|list|tuple): Support four types of modules.

        Returns:
            modules (function|class|list|tuple): Same with input modules.
        """

        # Check whether the type is a sequence
        if isinstance(modules, Sequence):
            for module in modules:
                self._add_single_module(module)
        else:
            self._add_single_module(modules)

        return modules

    def build(self, cfg, **kwargs):
        """
        Build a  module from init dict.

        Args:
            _cfg (dict): Support four types of modules.

        Returns:
            module (function|class): Same with input modules.
        """
        # update config
        _cfg = copy.deepcopy(cfg)
        for k, v in kwargs.items():
            if k not in _cfg:
                _cfg[k] = v
        name = _cfg.pop('name')
        if name not in self._modules_dict:
            raise Exception('{} is not register in '.format(name, self))
        if self._build_func is not None:
            return self.build_func(_cfg, op_cls=self._modules_dict[name])
        return self._modules_dict[name](**_cfg)

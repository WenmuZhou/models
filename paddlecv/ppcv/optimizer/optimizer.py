# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.nn as nn

import paddle.optimizer as optimizer
import paddle.regularizer as regularizer

from paddlecv.ppcv.register import OPTIMIZER

from .adamwdl import build_adamwdl

__all__ = ['OptimizerBuilder']

from ppcv.utils.logger import setup_logger
logger = setup_logger(__name__)


@OPTIMIZER.register()
class OptimizerBuilder:
    """
    Build optimizer handles
    Args:
        regularizer (object): an `Regularizer` instance
        optimizer (object): an `Optimizer` instance
    """
    __category__ = 'optim'

    def __init__(self,
                 clip_grad_by_norm=None,
                 regularizer={'type': 'L2',
                              'factor': .0001},
                 optimizer={'type': 'Momentum',
                            'momentum': .9}):
        self.clip_grad_by_norm = clip_grad_by_norm
        self.regularizer = regularizer
        self.optimizer = optimizer

    def __call__(self, learning_rate, model, *args, **kwargs):

        if self.clip_grad_by_norm is not None:
            grad_clip = nn.ClipGradByGlobalNorm(
                clip_norm=self.clip_grad_by_norm)
        else:
            grad_clip = None
        if self.regularizer and self.regularizer != 'None':
            regularizer_name = self.regularizer['name']
            if 'Decay' not in regularizer_name:
                regularizer_name = regularizer_name + 'Decay'
            reg_factor = self.regularizer['factor']
            regularization = getattr(regularizer, regularizer_name)(reg_factor)
        else:
            regularization = None

        optim_args = self.optimizer.copy()
        optim_name = optim_args.pop('name')

        if optim_name != 'AdamW':
            optim_args['weight_decay'] = regularization

        op = getattr(optimizer, optim_name)

        parameters = self.set_params(model, *args, **kwargs)

        return op(learning_rate=learning_rate,
                  parameters=parameters,
                  grad_clip=grad_clip,
                  **optim_args)

    def set_params(self, model, *args, **kwargs):
        parameters = [
            param for param in model.parameters() if param.trainable is True
        ]
        return parameters

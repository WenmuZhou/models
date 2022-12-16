# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from paddle.optimizer import lr
from paddle.optimizer.lr import LRScheduler

from paddlecv.ppcv.register import LRSCHEDULER
from ppcv.utils.logger import setup_logger
logger = setup_logger(__name__)


@LRSCHEDULER.register
class Polynomial(object):
    """
    Polynomial learning rate decay
    Args:
        learning_rate (float): The initial learning rate. It is a python float number.
        epochs(int): The decay epoch size. It determines the decay cycle, when by_epoch is set to true, it will change to epochs=epochs*step_each_epoch.
        step_each_epoch: all steps in each epoch.
        end_lr(float, optional): The minimum final learning rate. Default: 0.0001.
        power(float, optional): Power of polynomial. Default: 1.0.
        warmup_epoch(int): The epoch numbers for LinearWarmup. Default: 0, , when by_epoch is set to true, it will change to warmup_epoch=warmup_epoch*step_each_epoch.
        warmup_start_lr(float): Initial learning rate of warm up. Default: 0.0.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
        by_epoch: Whether the set parameter is based on epoch or iter, when set to true,, epochs and warmup_epoch will be automatically multiplied by step_each_epoch. Default: True
    """

    def __init__(self,
                 learning_rate,
                 epochs,
                 step_each_epoch,
                 end_lr=0.0,
                 power=1.0,
                 warmup_epoch=0,
                 warmup_start_lr=0.0,
                 last_epoch=-1,
                 by_epoch=True,
                 **kwargs):
        super().__init__()
        if warmup_epoch >= epochs:
            msg = f"When using warm up, the value of \"epochs\" must be greater than value of \"Optimizer.lr.warmup_epoch\". The value of \"Optimizer.lr.warmup_epoch\" has been set to {epochs}."
            logger.warning(msg)
            warmup_epoch = epochs
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.end_lr = end_lr
        self.power = power
        self.last_epoch = last_epoch
        self.warmup_epoch = warmup_epoch
        self.warmup_start_lr = warmup_start_lr

        if by_epoch:
            self.epochs *= step_each_epoch
            self.warmup_epoch = int(self.warmup_epoch * step_each_epoch())

    def __call__(self):
        learning_rate = lr.PolynomialDecay(
            learning_rate=self.learning_rate,
            decay_steps=self.epochs,
            end_lr=self.end_lr,
            power=self.power,
            last_epoch=self.
            last_epoch) if self.steps > 0 else self.learning_rate
        if self.warmup_steps > 0:
            learning_rate = lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=self.warmup_steps,
                start_lr=self.warmup_start_lr,
                end_lr=self.learning_rate,
                last_epoch=self.last_epoch)
        return learning_rate


@LRSCHEDULER.register
class Constant(LRScheduler):
    """
    Constant learning rate
    Args:
        lr (float): The initial learning rate. It is a python float number.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    """

    def __init__(self, learning_rate, last_epoch=-1, **kwargs):
        self.learning_rate = learning_rate
        self.last_epoch = last_epoch
        super().__init__()

    def get_lr(self):
        return self.learning_rate


@LRSCHEDULER.register
class CosineAnnealing(object):
    """
    CosineAnnealing learning rate decay
    lr = 0.05 * (math.cos(epoch * (math.pi / epochs)) + 1)
    Args:
        lr(float): initial learning rate
        step_each_epoch(int): steps each epoch
        epochs(int): total training epochs, when by_epoch is set to true, it will change to epochs=epochs*step_each_epoch.
        eta_min(float): Minimum learning rate. Default: 0.0.
        warmup_epoch(int): The epoch numbers for LinearWarmup. Default: 0, when by_epoch is set to true, it will change to warmup_epoch=warmup_epoch*step_each_epoch.
        warmup_start_lr(float): Initial learning rate of warm up. Default: 0.0.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
        by_epoch: Whether the set parameter is based on epoch or iter, when set to true,, epochs and warmup_epoch will be automatically multiplied by step_each_epoch. Default: True
    """

    def __init__(self,
                 learning_rate,
                 step_each_epoch,
                 epochs,
                 eta_min=0.0,
                 warmup_epoch=0,
                 warmup_start_lr=0.0,
                 last_epoch=-1,
                 by_epoch=True,
                 **kwargs):
        super().__init__()
        if warmup_epoch >= epochs:
            msg = f"When using warm up, the value of \"epochs\" must be greater than value of \"Optimizer.lr.warmup_epoch\". The value of \"Optimizer.lr.warmup_epoch\" has been set to {epochs}."
            logger.warning(msg)
            warmup_epoch = epochs
        self.learning_rate = learning_rate
        self.T_max = (epochs - warmup_epoch)
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.warmup_epoch = warmup_epoch
        self.warmup_start_lr = warmup_start_lr

        if by_epoch:
            self.T_max *= step_each_epoch
            self.warmup_epoch = int(self.warmup_epoch * step_each_epoch())

    def __call__(self):
        learning_rate = lr.CosineAnnealingDecay(
            learning_rate=self.learning_rate,
            T_max=self.T_max,
            eta_min=self.eta_min,
            last_epoch=self.
            last_epoch) if self.T_max > 0 else self.learning_rate
        if self.warmup_steps > 0:
            learning_rate = lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=self.warmup_steps,
                start_lr=self.warmup_start_lr,
                end_lr=self.learning_rate,
                last_epoch=self.last_epoch)
        return learning_rate


@LRSCHEDULER.register
class Step(object):
    """
    Piecewise learning rate decay
    Args:
        step_each_epoch(int): steps each epoch
        learning_rate (float): The initial learning rate. It is a python float number.
        step_size (int): the interval to update.
        gamma (float, optional): The Ratio that the learning rate will be reduced. ``new_lr = origin_lr * gamma`` .
            It should be less than 1.0. Default: 0.1.
        warmup_epoch(int): The epoch numbers for LinearWarmup. Default: 0.
        warmup_start_lr(float): Initial learning rate of warm up. Default: 0.0.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
        by_epoch: Whether the set parameter is based on epoch or iter, when set to true,, step_size and warmup_epoch will be automatically multiplied by step_each_epoch. Default: True
    """

    def __init__(self,
                 learning_rate,
                 step_size,
                 step_each_epoch,
                 epochs,
                 gamma,
                 warmup_epoch=0,
                 warmup_start_lr=0.0,
                 last_epoch=-1,
                 by_epoch=True,
                 **kwargs):
        super().__init__()
        if warmup_epoch >= epochs:
            msg = f"When using warm up, the value of \"epochs\" must be greater than value of \"Optimizer.lr.warmup_epoch\". The value of \"Optimizer.lr.warmup_epoch\" has been set to {epochs}."
            logger.warning(msg)
            warmup_epoch = epochs
        self.step_size = step_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.warmup_epoch = warmup_epoch
        self.warmup_start_lr = warmup_start_lr

        if by_epoch:
            self.step_size *= step_each_epoch
            self.warmup_epoch = int(self.warmup_epoch * step_each_epoch())

    def __call__(self):
        learning_rate = lr.StepDecay(
            learning_rate=self.learning_rate,
            step_size=self.step_size,
            gamma=self.gamma,
            last_epoch=self.last_epoch)
        if self.warmup_steps > 0:
            learning_rate = lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=self.warmup_steps,
                start_lr=self.warmup_start_lr,
                end_lr=self.learning_rate,
                last_epoch=self.last_epoch)
        return learning_rate


@LRSCHEDULER.register
class Piecewise(object):
    """
    Piecewise learning rate decay
    Args:
        boundaries(list): A list of steps numbers. The type of element in the list is python int.
        values(list): A list of learning rate values that will be picked during different epoch boundaries.
            The type of element in the list is python float.
        warmup_epoch(int): The epoch numbers for LinearWarmup. Default: 0.
        warmup_start_lr(float): Initial learning rate of warm up. Default: 0.0.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
        by_epoch: Whether the set parameter is based on epoch or iter, when set to true,, boundaries and warmup_epoch will be automatically multiplied by step_each_epoch. Default: True
    """

    def __init__(self,
                 step_each_epoch,
                 decay_epochs,
                 values,
                 epochs,
                 warmup_epoch=0,
                 warmup_start_lr=0.0,
                 by_epoch=True,
                 last_epoch=-1,
                 **kwargs):
        super().__init__()
        if warmup_epoch >= epochs:
            msg = f"When using warm up, the value of \"Global.epochs\" must be greater than value of \"Optimizer.lr.warmup_epoch\". The value of \"Optimizer.lr.warmup_epoch\" has been set to {epochs}."
            logger.warning(msg)
            warmup_epoch = epochs
        self.boundaries = decay_epochs
        self.values = values
        self.last_epoch = last_epoch
        self.warmup_epoch = warmup_epoch
        self.warmup_start_lr = warmup_start_lr

        if by_epoch:
            self.boundaries *= step_each_epoch
            self.warmup_epoch = int(self.warmup_epoch * step_each_epoch())

    def __call__(self):
        learning_rate = lr.PiecewiseDecay(
            boundaries=self.boundaries,
            values=self.values,
            last_epoch=self.last_epoch)
        if self.warmup_steps > 0:
            learning_rate = lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=self.warmup_epoch,
                start_lr=self.warmup_start_lr,
                end_lr=self.values[0],
                last_epoch=self.last_epoch)
        return learning_rate


@LRSCHEDULER.register
class MultiStepDecay(object):
    """
    Piecewise learning rate decay
    Args:
        learning_rate (float): The initial learning rate. It is a python float number.
        milestones (tuple|list): List or tuple of each boundaries. Must be increasing.
        gamma (float, optional): The Ratio that the learning rate will be reduced. ``new_lr = origin_lr * gamma`` .
            It should be less than 1.0. Default: 0.1.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``False`` .
        step_each_epoch(int): steps each epoch
        warmup_epoch(int): The epoch numbers for LinearWarmup. Default: 0.
        warmup_start_lr(float): Initial learning rate of warm up. Default: 0.0.
        by_epoch: Whether the set parameter is based on epoch or iter, when set to true,, milestones and warmup_epoch will be automatically multiplied by step_each_epoch. Default: True
    """

    def __init__(self,
                 learning_rate,
                 milestones,
                 step_each_epoch,
                 epochs,
                 gamma,
                 warmup_epoch=0,
                 warmup_start_lr=0.0,
                 last_epoch=-1,
                 by_epoch=True,
                 **kwargs):
        super().__init__()
        if warmup_epoch >= epochs:
            msg = f"When using warm up, the value of \"epochs\" must be greater than value of \"Optimizer.lr.warmup_epoch\". The value of \"Optimizer.lr.warmup_epoch\" has been set to {epochs}."
            logger.warning(msg)
            warmup_epoch = epochs
        self.milestones = milestones
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.warmup_epoch = warmup_epoch
        self.warmup_start_lr = warmup_start_lr

        if by_epoch:
            self.milestones = [x * step_each_epoch for x in milestones]
            self.warmup_epoch = int(self.warmup_epoch * step_each_epoch())

    def __call__(self):
        learning_rate = lr.MultiStepDecay(
            learning_rate=self.learning_rate,
            milestones=self.milestones,
            gamma=self.gamma,
            last_epoch=self.last_epoch)
        if self.warmup_steps > 0:
            learning_rate = lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=self.warmup_steps,
                start_lr=self.warmup_start_lr,
                end_lr=self.learning_rate,
                last_epoch=self.last_epoch)
        return learning_rate

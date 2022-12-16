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

import math
import numpy as np

import paddle

__all__ = [
    'init_weights', 'normal', 'kaiming_normal', 'kaiming_uniform',
    'xavier_normal', 'xavier_uniform'
]


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = len(tensor.shape)
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if len(tensor.shape) > 2:
        receptive_field_size = paddle.numel(tensor[0][0])
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(
            mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def _calculate_gain(nonlinearity, param=None):
    """Return the recommended gain value for the given nonlinearity function.
    The values are as follows:
    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    ================= ====================================================
    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function
    """
    linear_fns = [
        'linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d',
        'conv_transpose2d', 'conv_transpose3d'
    ]
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(
                param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(
                param))
        return math.sqrt(2.0 / (1 + negative_slope**2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


@paddle.no_grad()
def _constant(x, value):
    temp_value = paddle.full(x.shape, value, x.dtype)
    x.set_value(temp_value)
    return x


@paddle.no_grad()
def _normal(x, mean=0., std=1.):
    temp_value = paddle.normal(mean, std, shape=x.shape)
    x.set_value(temp_value)
    return x


@paddle.no_grad()
def _uniform(x, a=-1., b=1.):
    temp_value = paddle.uniform(min=a, max=b, shape=x.shape)
    x.set_value(temp_value)
    return x


@paddle.no_grad()
def _xavier_uniform(x, gain=1.):
    """Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where
    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}
    Also known as Glorot initialization.
    Args:
        x: an n-dimensional `paddle.Tensor`
        gain: an optional scaling factor
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(x)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(
        3.0) * std  # Calculate uniform bounds from standard deviation

    return _uniform(x, -a, a)


@paddle.no_grad()
def _xavier_normal(x, gain=1.):
    """
    Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a normal
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where
    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}
    Also known as Glorot initialization.
    Args:
        tensor: an n-dimensional `paddle.Tensor`
        gain: an optional scaling factor
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(x)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

    return _normal(x, 0., std)


@paddle.no_grad()
def _kaiming_uniform(x, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    """Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where
    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}
    Also known as He initialization.
    Args:
        x: an n-dimensional `paddle.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).
    """
    fan = _calculate_correct_fan(x, mode)
    gain = _calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(
        3.0) * std  # Calculate uniform bounds from standard deviation

    temp_value = paddle.uniform(x.shape, min=-bound, max=bound)
    x.set_value(temp_value)

    return x


@paddle.no_grad()
def _kaiming_normal(x, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    """Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    normal distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where
    .. math::
        \text{std} = \frac{\text{gain}}{\sqrt{\text{fan\_mode}}}
    Also known as He initialization.
    Args:
        x: an n-dimensional `paddle.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).
    """
    fan = _calculate_correct_fan(x, mode)
    gain = _calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)

    temp_value = paddle.normal(0, std, shape=x.shape)
    x.set_value(temp_value)
    return x


def init_weights(net,
                 init_type='kaiming_normal',
                 init_gain=0.02,
                 init_func=None):
    """Initialize network weights.
    Args:
        net (nn.Layer): network to be initialized
        init_type (str): the name of an initialization method: normal | kaiming_normal | kaiming_uniform | xavier_normal | xavier_uniform
        init_gain (float): scaling factor for normal, xavier_uniform and xavier_normal.
        init_func: the func use to init layer in net
    """
    init_dict = {
        kaiming_normal.__name__: kaiming_normal,
        kaiming_uniform.__name__: kaiming_uniform,
        xavier_uniform.__name__: xavier_uniform,
        xavier_normal.__name__: xavier_normal,
        normal.__name__: normal,
    }
    if init_func is None and init_type not in init_dict.keys():
        raise Exception(
            "{} is not in support init type: {} or init_func is not None".
            format(init_type, init_dict.keys()))
    init_dict[init_type](net, init_gain=init_gain)


def kaiming_normal(net, **kwargs):
    """Initialize network weights with kaiming_normal.
    Args:
        net (nn.Layer): network to be initialized
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or
                                     classname.find('Linear') != -1):
            _kaiming_normal(m.weight, a=0, mode='fan_in')
            if hasattr(m, 'bias') and m.bias is not None:
                _constant(m.bias, 0.0)
        elif classname.find(
                'BatchNorm'
        ) != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            _normal(m.weight, 1.0)
            _constant(m.bias, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


def kaiming_uniform(net, **kwargs):
    """Initialize network weights with kaiming_uniform.
    Args:
        net (nn.Layer): network to be initialized
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or
                                     classname.find('Linear') != -1):
            _kaiming_uniform(m.weight, a=0, mode='fan_in')
            if hasattr(m, 'bias') and m.bias is not None:
                _constant(m.bias, 0.0)
        elif classname.find(
                'BatchNorm'
        ) != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            _normal(m.weight, 1.0)
            _constant(m.bias, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


def xavier_normal(net, init_gain=0.02, **kwargs):
    """Initialize network weights with xavier_normal.
    Args:
        net (nn.Layer): network to be initialized
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or
                                     classname.find('Linear') != -1):
            _xavier_normal(m.weight, gain=init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                _constant(m.bias, 0.0)
        elif classname.find(
                'BatchNorm'
        ) != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            _normal(m.weight, 1.0)
            _constant(m.bias, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


def xavier_uniform(net, init_gain=0.02, **kwargs):
    """Initialize network weights with xavier_uniform.
    Args:
        net (nn.Layer): network to be initialized
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or
                                     classname.find('Linear') != -1):
            _xavier_uniform(m.weight, gain=init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                _constant(m.bias, 0.0)
        elif classname.find(
                'BatchNorm'
        ) != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            _normal(m.weight, 1.0)
            _constant(m.bias, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


def normal(net, init_gain=0.02, **kwargs):
    """Initialize network weights with normal.
    Args:
        net (nn.Layer): network to be initialized
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or
                                     classname.find('Linear') != -1):
            _normal(m.weight, 0.0, init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                _constant(m.bias, 0.0)
        elif classname.find(
                'BatchNorm'
        ) != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            _normal(m.weight, 1.0)
            _constant(m.bias, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>

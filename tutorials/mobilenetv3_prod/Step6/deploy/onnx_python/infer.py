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

import os
import sys
sys.path.insert(0, ".")
import argparse
import numpy as np
from PIL import Image

import paddle
import paddle.nn as nn
from onnxruntime import InferenceSession

# 从模型代码中导入模型
from paddlevision.models import mobilenet_v3_small
from presets import ClassificationPresetEval


def infer():
    # Step1：初始化ONNXRuntime库并配置相应参数, 并进行预测
    # 加载ONNX模型
    sess = InferenceSession(FLAGS.onnx_file)

    # define transforms
    input_shape = sess.get_inputs()[0].shape[2:]
    eval_transforms = ClassificationPresetEval(
        crop_size=input_shape, resize_size=FLAGS.crop_size)
    # 准备输入
    with open(FLAGS.img_path, 'rb') as f:
        img = Image.open(f).convert('RGB')

    img = eval_transforms(img)
    img = np.expand_dims(img, axis=0)

    # 模型预测
    ort_outs = sess.run(output_names=None,
                        input_feed={sess.get_inputs()[0].name: img})

    output = ort_outs[0]
    class_id = output.argmax()
    prob = output[0][class_id]
    print("ONNXRuntime predict: ")
    print(f"class_id: {class_id}, prob: {prob}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--onnx_file',
        type=str,
        default="model.onnx",
        help="onnx model filename")
    parser.add_argument(
        '--img_path', type=str, default="image.jpg", help="image filename")
    parser.add_argument('--crop_size', default=256, help='crop_szie')

    FLAGS = parser.parse_args()

    infer()

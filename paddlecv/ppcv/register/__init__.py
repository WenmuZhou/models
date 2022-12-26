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
from .register import Register

__all__ = [
    'Register', 'MODEL', 'TRANSFORM', 'BACKBONE', 'NECK', 'HEAD', 'DATASET',
    'OPERATOR', 'LOSS', 'METRIC', 'LRSCHEDULER', 'OPTIMIZER', 'POSTPROCESS',
    'VISUALIZER', 'CALLBACK', 'COLLATEFN', 'DATALOADER'
]


def build_vis(cfg, op_cls):
    try:
        return op_cls(model_cfg={'Inputs': []}, env_cfg=cfg)
    except Exception as e:
        return op_cls(**cfg)


# model
MODEL = Register("model")
TRANSFORM = Register('transform')
BACKBONE = Register("backbone")
NECK = Register("neck")
HEAD = Register("head")

# dataset, preprocess op, collate_fn and DATALOADER
DATASET = Register("dataset")
OPERATOR = Register("operators")
COLLATEFN = Register("collate_fn")
DATALOADER = Register("dataLoader")

# loss, metric , lr and optimizer
LOSS = Register("loss")
METRIC = Register("metric")
LRSCHEDULER = Register("lrscheduler")
OPTIMIZER = Register("optimizer")

# post-process
POSTPROCESS = Register("postprocess")

# Visualizer
VISUALIZER = Register("visualizer", build_vis)

# Callbacks
CALLBACK = Register("callback")

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import copy
import random
import time

import numpy as np
import typing

import paddle
from paddle.io import DataLoader
import paddle.distributed as dist
from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients
from paddlecv.ppcv.register import MODEL, DATASET, LOSS, METRIC, LRSCHEDULER, OPTIMIZER, POSTPROCESS, VISUALIZER, CALLBACK, COLLATEFN, DATALOADER

from ppcv.optimizer import ModelEMA

from ppcv.utils.checkpoint import load_weight, load_pretrain_weight
import ppcv.utils.stats as stats
from ppcv.utils import profiler

from ppcv.engine.callbacks import LogPrinter, Checkpointer

from ppcv.utils.logger import setup_logger

__all__ = ['Trainer']


class Trainer(object):
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg.cfg
        self.cfg['output_dir'] = self.cfg.get('output_dir', 'output')
        self._global_cfg = {}
        self.by_epoch = self.cfg.get('by_epoch', True)
        self.set_random_seed(self.cfg.get('seed', 48))
        self.set_device(self.cfg['device'])
        self._nranks = dist.get_world_size()
        self._local_rank = dist.get_rank()
        if self._nranks > 1:
            dist.init_parallel_env()

        assert mode.lower() in ['train_eval', 'train', 'eval', 'test'], \
            "mode should be 'train', 'eval' or 'test'"
        self.mode = mode.lower()

        os.makedirs(self.cfg['output_dir'], exist_ok=True)
        self.logger = setup_logger(
            'paddlecv.trainer',
            output=os.path.join(self.cfg['output_dir'], ' train.log'))
        cfg.save(os.path.join(self.cfg['output_dir'], ' config.yml'))

        self.eval_interval = [0, 1] if self.by_epoch else [0, 1000]
        if isinstance(self.cfg['eval_interval'],
                      typing.Sequence) and len(self.cfg['eval_interval']) >= 2:
            self.eval_interval = self.cfg['eval_interval'][:2]

        # build data loader
        self.train_dataloader = self.build_dataloader(self.cfg[
            'Train']) if 'train' in mode else None
        self.eval_dataloader = self.build_dataloader(self.cfg[
            'Eval']) if 'eval' in mode else None
        self.test_dataloader = self.build_dataloader(self.cfg[
            'Test']) if 'test' in mode else None

        # build loss
        self.loss = LOSS.build(self.cfg[
            'Loss']) if 'Loss' in self.cfg else None

        # build PostProcess
        self.postprocess = POSTPROCESS.build(
            self.cfg['PostProcess'],
            **self.global_cfg) if 'PostProcess' in self.cfg else None

        # build Visualizer
        if 'Visualizer' in self.cfg and self.test_dataloader is not None:
            self.visualizer = VISUALIZER.build(
                self.cfg['Visualizer'], mode='train', **self.global_cfg)
        else:
            self.visualizer = None

        # build Metric
        self.metric = METRIC.build(self.cfg[
            'Metric']) if 'Metric' in self.cfg else None

        # build LRScheduler
        if 'LRScheduler' in self.cfg and self.train_dataloader is not None:
            self.lr_scheduler = LRSCHEDULER.build(
                self.cfg['LRScheduler'],
                step_each_epoch=len(self.train_dataloader),
                **self.global_cfg)()
        else:
            self.lr_scheduler = None

        # build model
        self.model = MODEL.build(self.cfg['Model'])
        if self.cfg.get('sync_bn', False) and self._nranks > 1:
            self.model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(
                self.model)

        self.load_weights()

        # build Optimizer
        if 'Optimizer' in self.cfg and self.train_dataloader is not None:
            optimizer = OPTIMIZER.build(self.cfg['Optimizer'])
            self.optimizer = optimizer(self.lr_scheduler, self.model,
                                       **self.cfg['Optimizer'])
        else:
            self.optimizer = None

        # build Callbacks
        callbacks_cfg = copy.deepcopy(self.cfg['Callbacks'])
        callbacks_cfg = {'name': 'ComposeCallback', 'callbacks': callbacks_cfg}
        self.compose_callback = CALLBACK.build(
            callbacks_cfg,
            trainer=self,
            main_indicator=self.metric.
            main_indicator) if 'Callbacks' in self.cfg else None

        self._check_callbacks()

        # init amp
        self.amp = self.cfg.get('Amp', None)
        if self.amp and self.optimizer is not None:
            self.amp['scaler'] = paddle.amp.GradScaler(
                init_loss_scaling=self.amp.get("scale_loss", 1.0))
            self.model, self.optimizer = paddle.amp.decorate(
                models=self.model,
                optimizers=self.optimizer,
                level=self.amp.get('amp_level', 'O2'))

        # init ema
        ema_cfg = self.cfg.get('Ema', None)
        self.ema = ModelEMA(self.model,
                            **ema_cfg) if ema_cfg is not None else None

        # distributed train
        if self._nranks > 1:
            self.model = paddle.DataParallel(self.model)

        # train status
        self.start_epoch = 0
        self.status = {'epoch_id': self.start_epoch, 'global_step': 0}
        self.end_epoch = 0 if 'epochs' not in self.cfg else self.cfg['epochs']

        cfg.print_cfg(self.logger.info)
        self.logger.info('run model {} with paddle {} and device {}'.format(
            self.mode, paddle.__version__, self.cfg['device']))

    @property
    def global_cfg(self):
        if len(self._global_cfg) == 0:
            for k, v in self.cfg.items():
                if not isinstance(v, typing.Mapping):
                    self._global_cfg[k] = v

        return self._global_cfg

    def set_random_seed(self, seed):
        paddle.seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def set_device(self, device):
        if device == 'gpu' and paddle.is_compiled_with_cuda():
            place = 'gpu'
        elif device == 'xpu' and paddle.is_compiled_with_xpu():
            place = 'xpu'
        elif device == 'npu' and paddle.is_compiled_with_npu():
            place = 'npu'
        elif device == 'mlu' and paddle.is_compiled_with_mlu():
            place = 'mlu'
        else:
            place = 'cpu'
        self.cfg['device'] = place
        paddle.set_device(place)

    def _check_callbacks(self):
        def check(compose_callback, types):
            for c in compose_callback._callbacks:
                for _type in types:
                    if isinstance(c, _type):
                        return True
            return False

        _pass = check(self.compose_callback, [LogPrinter])
        if not _pass:
            raise Exception('LogPrinter callback must be set')
        if 'train' in self.mode:
            _pass = check(self.compose_callback, [Checkpointer])
            if not _pass:
                raise Exception(
                    'when in train mode, Checkpointer callback must be set')

    def load_weights(self):
        if self.cfg['resume_weights'] is not None:
            load_weight(self.model, self.cfg['resume_weights'], self.optimizer,
                        self.ema)
            self.logger.info("resume from {}".format(self.cfg[
                'resume_weights']))
        elif self.cfg['pretrain_weights'] is not None:
            load_pretrain_weight(self.model, self.cfg['pretrain_weights'])
            self.logger.info("load pretrain weights from {}".format(self.cfg[
                'pretrain_weights']))

    def train(self):
        assert 'train' in self.mode, "Mode not in ['train', 'train_eval']"
        self.status.update({
            'batch_step_id': 0,
            'steps_per_epoch': len(self.train_dataloader),
            'epochs': self.cfg['epochs']
        })

        self.status['batch_time'] = stats.SmoothedValue(
            self.cfg['log_iter'], fmt='{avg:.4f}')
        self.status['data_time'] = stats.SmoothedValue(
            self.cfg['log_iter'], fmt='{avg:.4f}')
        self.status['training_staus'] = stats.TrainingStats(self.cfg[
            'log_iter'])

        profiler_options = self.cfg.get('profiler_options', None)

        self.compose_callback.on_train_begin(self.status)

        for epoch_id in range(self.start_epoch, self.cfg['epochs']):
            self.status['mode'] = 'train'
            self.status['epoch_id'] = epoch_id
            self.compose_callback.on_epoch_begin(self.status)
            self.model.train()
            iter_tic = time.time()
            for batch_step_id, data in enumerate(self.train_dataloader):
                self.status['data_time'].update(time.time() - iter_tic)
                self.status['batch_size'] = data['image'].shape[0]
                self.status['batch_step_id'] = batch_step_id
                self.status['global_step'] += 1

                profiler.add_profiler_step(profiler_options)
                self.compose_callback.on_step_begin(self.status)
                data['epoch_id'] = epoch_id

                # forward
                if self.amp:
                    if self._nranks > 1 and self.cfg.get(
                            'use_fused_allreduce_gradients', False):
                        with self.model.no_sync():
                            with paddle.amp.auto_cast(
                                    enable=self.cfg.get('device', 'gpu') ==
                                    'gpu',
                                    custom_white_list=self.amp.get(
                                        'custom_white_list', []),
                                    custom_black_list=self.amp.get(
                                        'custom_black_list', []),
                                    level=self.amp.get('level', 'O2')):
                                # model forward
                                outputs = self.model(data)
                                if self.loss is not None:
                                    outputs = self.loss(outputs, data)
                                loss = outputs['loss']
                            # model backward
                            scaled_loss = self.amp['scaler'].scale(loss)
                            scaled_loss.backward()
                        fused_allreduce_gradients(
                            list(self.model.parameters()), None)
                    else:
                        with paddle.amp.auto_cast(
                                enable=self.cfg.get('device', 'gpu') == 'gpu',
                                custom_white_list=self.amp.get(
                                    'custom_white_list', []),
                                custom_black_list=self.amp.get(
                                    'custom_black_list', []),
                                level=self.amp.get('level', [])):
                            # model forward
                            outputs = self.model(data)
                            if self.loss is not None:
                                outputs = self.loss(outputs, data)
                            loss = outputs['loss']
                        # model backward
                        scaled_loss = self.amp['scaler'].scale(loss)
                        scaled_loss.backward()
                    # in dygraph mode, optimizer.minimize is equal to optimizer.step
                    self.amp['scaler'].minimize(self.optimizer, scaled_loss)
                else:
                    if self._nranks > 1 and self.cfg.get(
                            'use_fused_allreduce_gradients', False):
                        with self.model.no_sync():
                            # model forward
                            outputs = self.model(data)
                            if self.loss is not None:
                                outputs = self.loss(outputs, data)
                            loss = outputs['loss']
                            # model backward
                            loss.backward()
                        fused_allreduce_gradients(
                            list(self.model.parameters()), None)
                    else:
                        # model forward
                        outputs = self.model(data)
                        if self.loss is not None:
                            outputs = self.loss(outputs, data)
                        loss = outputs['loss']
                        # model backward
                        loss.backward()
                    self.optimizer.step()
                self.optimizer.clear_grad()
                self.lr_scheduler.step()

                curr_lr = self.optimizer.get_lr()
                self.status['learning_rate'] = curr_lr
                if self._local_rank == 0:
                    self.status['training_staus'].update(outputs)

                self.status['batch_time'].update(time.time() - iter_tic)
                self.compose_callback.on_step_end(self.status)
                # ema
                if self.ema is not None:
                    self.ema.update()

                # eval in train
                if self.cfg.get(
                        'eval_during_train', False
                ) and self.metric is not None and self.postprocess is not None:
                    post_result = self.postprocess(
                        outputs,
                        data) if self.postprocess is not None else outputs
                    self.metric.reset()
                    self.metric.update(post_result, data)
                    metric = self.metric.accumulate()
                    self.status.update(metric)
                if not self.by_epoch and 'eval' in self.mode:
                    if self.status['global_step'] > self.cfg['eval_interval'][0] and \
                        (self.status['global_step'] - self.cfg['eval_interval'][0]) % self.cfg['eval_interval'][1] == 0 \
                            and self._local_rank == 0:
                        self.evaluate()
                iter_tic = time.time()
            if self.by_epoch and 'eval' in self.mode:
                if epoch_id > self.cfg['eval_interval'][0] and \
                        (epoch_id - self.cfg['eval_interval'][0]) % self.cfg['eval_interval'][1] == 0 \
                            and self._local_rank == 0:
                    self.evaluate()
            self.compose_callback.on_epoch_end(self.status)
        self.compose_callback.on_train_end(self.status)

    def evaluate(self):
        self.metric.reset()
        with paddle.no_grad():
            self.status['mode'] = 'eval'
            self.status['steps_per_epoch'] = len(self.eval_dataloader)
            sample_num = 0
            tic = time.time()
            self.compose_callback.on_epoch_begin(self.status)
            self.model.eval()
            for batch_step_id, data in enumerate(self.eval_dataloader):
                sample_num += data['image'].shape[0]
                self.status['batch_step_id'] = batch_step_id
                self.compose_callback.on_step_begin(self.status)
                # forward
                if self.amp:
                    with paddle.amp.auto_cast(
                            enable=self.cfg.get('device', 'gpu') == 'gpu',
                            custom_white_list=self.amp.get('custom_white_list',
                                                           []),
                            custom_black_list=self.amp.get('custom_black_list',
                                                           []),
                            level=self.amp.get('level', [])):
                        outs = self.model(data)
                else:
                    outs = self.model(data)
                post_result = self.postprocess(
                    outs, data) if self.postprocess is not None else outs
                # update metrics
                self.metric.update(post_result, data)
                self.compose_callback.on_step_end(self.status)

            self.status['sample_num'] = sample_num
            self.status['cost_time'] = time.time() - tic

            # accumulate metric to log out
            self.status['metric'] = self.metric.accumulate()
            self.compose_callback.on_epoch_end(self.status)
            # reset status
            self.model.train()
            self.status['steps_per_epoch'] = len(self.eval_dataloader)

    def test(self, output_dir='output', visualize=True):
        os.makedirs(output_dir, exist_ok=True)

        with paddle.no_grad():
            self.status['mode'] = 'test'
            self.status['steps_per_epoch'] = len(self.test_dataloader)
            self.compose_callback.on_epoch_begin(self.status)
            self.model.eval()
            for batch_step_id, data in enumerate(self.test_dataloader):
                self.status['batch_step_id'] = batch_step_id
                self.compose_callback.on_step_begin(self.status)
                # forward
                if self.amp:
                    with paddle.amp.auto_cast(
                            enable=self.cfg.get('device', 'gpu') == 'gpu',
                            custom_white_list=self.amp.get('custom_white_list',
                                                           []),
                            custom_black_list=self.amp.get('custom_black_list',
                                                           []),
                            level=self.amp.get('level', [])):
                        outs = self.model(data)
                else:
                    outs = self.model(data)
                post_result = self.postprocess(
                    outs, data) if self.postprocess is not None else outs

                if visualize and self.visualizer is not None:
                    self.visualizer(
                        post_result,
                        data,
                        test_dataloader=self.test_dataloader)
                self.compose_callback.on_step_end(self.status)

    def build_dataloader(self, cfg):
        # build dataset
        _cfg = copy.deepcopy(cfg)
        dataset = DATASET.build(_cfg['dataset'], **self.global_cfg)

        # build loader
        loader_cfg = _cfg['loader']

        collate_fn = None
        if 'collate_fn' in loader_cfg:
            collate_fn = COLLATEFN.build(loader_cfg['collate_fn'],
                                         **self.global_cfg)

        loader_cfg['dataset'] = dataset
        loader_cfg['collate_fn'] = collate_fn
        loader_cfg.update(self.global_cfg)
        # update config
        for k, v in self.global_cfg.items():
            if k not in loader_cfg:
                loader_cfg[k] = v
        if 'name' not in loader_cfg:
            if self._nranks > 1 and 'train' in self.mode:
                batch_sampler = paddle.io.DistributedBatchSampler(
                    dataset,
                    batch_size=loader_cfg['batch_size'],
                    shuffle=loader_cfg.get('shuffle', True),
                    drop_last=loader_cfg.get('drop_last', True))
            else:
                batch_sampler = paddle.io.BatchSampler(
                    dataset,
                    batch_size=loader_cfg['batch_size'],
                    shuffle=loader_cfg.get('shuffle', True),
                    drop_last=loader_cfg.get('drop_last', True))
            loader = DataLoader(
                dataset=dataset,
                batch_sampler=batch_sampler,
                num_workers=loader_cfg.get('num_workers', 1),
                return_list=True,
                use_shared_memory=loader_cfg.get('use_shared_memory', True),
                collate_fn=collate_fn)
        else:
            loader = DATALOADER.build(loader_cfg)
        return loader

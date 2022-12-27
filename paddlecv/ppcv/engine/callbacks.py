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

import numbers
import os
import datetime
import six
import copy

from ppcv.utils.checkpoint import save_model
from paddlecv.ppcv.register import CALLBACK

__all__ = [
    'Callback', 'ComposeCallback', 'LogPrinter', 'Checkpointer',
    'VisualDLWriter'
]


@CALLBACK.register()
class Callback(object):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_begin(self, status):
        pass

    def on_step_end(self, status):
        pass

    def on_epoch_begin(self, status):
        pass

    def on_epoch_end(self, status):
        pass

    def on_train_begin(self, status):
        pass

    def on_train_end(self, status):
        pass


@CALLBACK.register()
class ComposeCallback(object):
    def __init__(self, callbacks, **kwargs):
        self._callbacks = []
        for c in callbacks:
            if isinstance(c, dict):
                c = CALLBACK.build(c, **kwargs)
            assert isinstance(
                c, Callback), "callback should be subclass of Callback"
            self._callbacks.append(c)

    def on_step_begin(self, status):
        for c in self._callbacks:
            c.on_step_begin(status)

    def on_step_end(self, status):
        for c in self._callbacks:
            c.on_step_end(status)

    def on_epoch_begin(self, status):
        for c in self._callbacks:
            c.on_epoch_begin(status)

    def on_epoch_end(self, status):
        for c in self._callbacks:
            c.on_epoch_end(status)

    def on_train_begin(self, status):
        for c in self._callbacks:
            c.on_train_begin(status)

    def on_train_end(self, status):
        for c in self._callbacks:
            c.on_train_end(status)


@CALLBACK.register()
class LogPrinter(Callback):
    def __init__(self, trainer, **kwargs):
        super(LogPrinter, self).__init__(trainer)
        self.logger = trainer.logger

    def on_train_begin(self, status):
        if self.trainer._local_rank != 0:
            return
        if self.trainer.train_dataloader is not None:
            self.logger.info('train dataloader has {} iters'.format(
                len(self.trainer.train_dataloader)))
        if self.trainer.eval_dataloader is not None:
            self.logger.info('eval dataloader has {} iters'.format(
                len(self.trainer.eval_dataloader)))
        if self.trainer.test_dataloader is not None:
            self.logger.info('test dataloader has {} iters'.format(
                len(self.trainer.test_dataloader)))
        if self.trainer.train_dataloader is not None and self.trainer.eval_dataloader is not None:
            self.logger.info(
                "During the training process, after the {}th iteration, " \
                "an evaluation is run every {} iterations".
                format(self.trainer.eval_interval[0], self.trainer.eval_interval[1]))

    def on_step_end(self, status):
        if self.trainer._local_rank != 0:
            return
        mode = status['mode']
        if mode == 'train':
            epoches = status['epochs']
            epoch_id = status['epoch_id']
            batch_step_id = status['batch_step_id']
            global_step = status['global_step']
            steps_per_epoch = status['steps_per_epoch']
            batch_size = status['batch_size']
            training_staus = status['training_staus']
            batch_time = status['batch_time']
            data_time = status['data_time']

            logs = training_staus.log()
            if global_step % self.trainer.cfg.log_iter == 0:
                eta_steps = (epoches - epoch_id
                             ) * steps_per_epoch - global_step
                eta_sec = eta_steps * batch_time.global_avg
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                ips = float(batch_size) / batch_time.avg
                fmt = ', '.join([
                    'epoch: [{}/{}]',
                    'iter: [{}/{}]',
                    'global_step: {global_step}',
                    'learning_rate: {lr:.6f}',
                    '{meters}',
                    'eta: {eta}',
                    'reader_cost: {dtime}',
                    'batch_cost: {btime}',
                    'ips: {ips:.4f} samples/sec',
                ])
                fmt = fmt.format(
                    epoch_id,
                    epoches,
                    batch_step_id,
                    steps_per_epoch,
                    global_step=global_step,
                    lr=status['learning_rate'],
                    meters=logs,
                    eta=eta_str,
                    btime=str(batch_time),
                    dtime=str(data_time),
                    ips=ips)
                self.logger.info(fmt)
        if mode == 'eval':
            batch_step_id = status['batch_step_id']
            if batch_step_id % 100 == 0 or batch_step_id == status[
                    'steps_per_epoch'] - 1:
                self.logger.info("Eval iter: {}/{}".format(
                    batch_step_id, status['steps_per_epoch']))

    def on_epoch_end(self, status):
        if self.trainer._local_rank != 0:
            return
        mode = status['mode']
        if mode == 'eval':
            metric = status['metric']
            for k, v in metric.items():
                self.logger.info('{}: {}'.format(k, v))
            sample_num = status['sample_num']
            cost_time = status['cost_time']
            self.logger.info('Total sample number: {}, averge FPS: {}'.format(
                sample_num, sample_num / cost_time))


@CALLBACK.register()
class Checkpointer(Callback):
    def __init__(self, trainer, main_indicator, **kwargs):
        super(Checkpointer, self).__init__(trainer)
        self.save_dir = self.trainer.cfg.output_dir
        self.model = self.trainer.model
        self.train = 'train' in self.trainer.mode

        if isinstance(main_indicator, str):
            main_indicator = [main_indicator]
        self.main_indicator = main_indicator
        self.best_metric = {k: {} for k in self.main_indicator}

    def on_step_end(self, status):
        self.on_epoch_end(status)

    def on_epoch_end(self, status):
        if self.trainer._local_rank != 0:
            return
        # Checkpointer only performed during training
        mode = status['mode']
        epoch_id = status['epoch_id']
        global_step = status['global_step']
        steps_per_epoch = status['steps_per_epoch']
        metric = status.get('metric', {})
        weight = None
        save_name = None

        if mode == 'train':
            end_epoch = self.trainer.cfg['epochs']
            save_epoch = self.trainer.by_epoch and (
                epoch_id + 1) % self.trainer.cfg[
                    'save_interval'] == 0 or epoch_id == end_epoch - 1
            save_iter = not self.trainer.by_epoch and (
                global_step + 1
            ) % self.trainer.cfg[
                'save_interval'] == 0 or global_step == end_epoch * steps_per_epoch - 1

            if save_epoch or save_iter:
                save_name = str(
                    epoch_id) if epoch_id != end_epoch - 1 else "model_final"
                weight = copy.deepcopy(self.model.state_dict())
                self.save_model(weight, save_name, epoch_id)
                save_name = 'latest'
                self.save_model(weight, save_name, epoch_id)
        elif mode == 'eval':
            if len(metric) != 0:
                for k in self.main_indicator:
                    if metric[k] >= self.best_metric[k].get(k, -1):
                        self.best_metric[k] = metric
                        save_name = 'best_{}'.format(k)
                        weight = copy.deepcopy(self.model.state_dict())
                        self.save_model(weight, save_name, epoch_id)

    def save_model(self, weight, save_name, epoch_id):
        if self.train and weight:
            if self.trainer.ema is not None:
                self.trainer.model.set_dict(self.trainer.ema.apply())
                save_model(weight, self.trainer.optimizer, self.save_dir,
                           save_name, epoch_id + 1, self.model.state_dict())
                # reset original weight
                self.trainer.model.set_dict(weight)
            else:
                save_model(weight, self.trainer.optimizer, self.save_dir,
                           save_name, epoch_id + 1)


@CALLBACK.register()
class VisualDLWriter(Callback):
    """
    Use VisualDL to log data or image
    """

    def __init__(self, trainer, **kwargs):
        super(VisualDLWriter, self).__init__(trainer)

        assert six.PY3, "VisualDL requires Python >= 3.5"
        try:
            from visualdl import LogWriter
        except Exception as e:
            self.trainer.logger.error(
                'visualdl not found, plaese install visualdl. '
                'for example: `pip install visualdl`.')
            raise e
        self.vdl_writer = LogWriter(self.trainer.cfg.get('output_dir'))
        self.vdl_loss_step = 0
        self.vdl_mAP_step = 0
        self.vdl_image_step = 0
        self.vdl_image_frame = 0
        self.vdl_metric_step = 0

    def on_step_end(self, status):
        mode = status['mode']
        if self.trainer._local_rank != 0:
            return
        if mode == 'train':
            training_staus = status['training_staus']
            for loss_name, loss_value in training_staus.get().items():
                self.vdl_writer.add_scalar('{}/{}'.format(mode, loss_name),
                                           loss_value, self.vdl_loss_step)
            self.vdl_writer.add_scalar('{}/learning_rate'.format(mode),
                                       status['learning_rate'],
                                       self.vdl_loss_step)
            self.vdl_loss_step += 1
        elif 'mode' == 'eval':
            self.on_epoch_end(status)

    def on_epoch_end(self, status):
        mode = status['mode']
        if self.trainer._local_rank != 0:
            return
        if mode == 'eval':
            metric = status.get('metric', {})
            for k, v in metric.items():
                if isinstance(v, numbers.Number):
                    self.vdl_writer.add_scalar('{}/{}'.format(mode, k), v,
                                               self.vdl_metric_step)
            self.vdl_metric_step += 1

    def on_train_end(self, status):
        self.vdl_writer.close()


@CALLBACK.register()
class WandbCallback(Callback):
    def __init__(self, trainer, **kwargs):
        super(WandbCallback, self).__init__(trainer)

        try:
            import wandb
            self.wandb = wandb
        except Exception as e:
            self.trainer.logger.error('wandb not found, please install wandb. '
                                      'Use: `pip install wandb`.')
            raise e

        self.wandb_params = kwargs
        self.save_dir = self.trainer.cfg.get('output_dir')
        self._run = None
        if self.trainer._local_rank == 0:
            _ = self.run
            self.run.config.update(self.trainer.cfg)
            self.run.define_metric("epoch")
            self.run.define_metric("eval/*", step_metric="epoch")

        self.best_ap = 0

    @property
    def run(self):
        if self._run is None:
            if self.wandb.run is not None:
                self.trainer.logger.info(
                    "There is an ongoing wandb run which will be used"
                    "for logging. Please use `wandb.finish()` to end that"
                    "if the behaviour is not intended")
                self._run = self.wandb.run
            else:
                self._run = self.wandb.init(**self.wandb_params)
        return self._run

    def save_model(self,
                   save_name,
                   last_epoch,
                   ema_model=None,
                   metric={},
                   tags=None):
        if self.trainer._local_rank != 0:
            return
        model_path = os.path.join(self.save_dir, save_name)
        metadata = {}
        metadata["last_epoch"] = last_epoch
        metadata.update(metric)
        if ema_model is not None:
            ema_artifact = self.wandb.Artifact(
                name="ema_model-{}".format(self.run.id),
                type="model",
                metadata=metadata)
            model_artifact = self.wandb.Artifact(
                name="model-{}".format(self.run.id),
                type="model",
                metadata=metadata)

            ema_artifact.add_file(model_path + ".pdema", name="model_ema")
            model_artifact.add_file(model_path + ".pdparams", name="model")

            self.run.log_artifact(ema_artifact, aliases=tags)
            self.run.log_artfact(model_artifact, aliases=tags)
        else:
            model_artifact = self.wandb.Artifact(
                name="model-{}".format(self.run.id),
                type="model",
                metadata=metadata)
            model_artifact.add_file(model_path + ".pdparams", name="model")
            self.run.log_artifact(model_artifact, aliases=tags)

    def on_step_end(self, status):
        mode = status['mode']
        if self.trainer._local_rank != 0:
            return
        if mode == 'train':
            training_status = status['training_staus'].get()
            for k, v in training_status.items():
                training_status[k] = float(v)
            metrics = {
                "{}/{}".format(mode, k): v
                for k, v in training_status.items()
            }
            self.run.log(metrics)

    def on_epoch_end(self, status):
        mode = status['mode']
        epoch_id = status['epoch_id']
        save_name = None
        if self.trainer._local_rank != 0:
            return
        if mode == 'train':
            end_epoch = self.trainer.cfg['epochs']
            if (
                    epoch_id + 1
            ) % self.trainer.cfg.save_interval == 0 or epoch_id == end_epoch - 1:
                save_name = 'latest'
                tags = [save_name, "epoch_{}".format(epoch_id)]
                self.save_model(
                    self.save_dir,
                    save_name,
                    epoch_id + 1,
                    self.trainer.ema is not None,
                    tags=tags)
        if mode == 'eval':
            metric = status.get('metric', {})
            metrics = {
                "{}/{}".format(mode, k): float(v)
                for k, v in metric.items()
            }
            self.run.log(metrics)

            save_name = 'best'
            tags = ["best", "epoch_{}".format(epoch_id)]
            self.save_model(
                self.save_dir,
                save_name,
                epoch_id + 1,
                self.trainer.ema is not None,
                metric=metric,
                tags=tags)

    def on_train_end(self, status):
        self.run.finish()

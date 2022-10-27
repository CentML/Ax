#!/usr/bin/env python3
# Copyright (c) CentML Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect

from logging import Logger
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Set

from ax.core import Trial
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.runner import Runner
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import not_none

logger: Logger = get_logger(__name__)


class MetricBase:

    def __init__(self):
        pass

    def on_train_iter_end(self, epoch, iter, pred, label):
        pass

    def on_eval_iter_end(self, epoch, iter, pred, label):
        pass

    def on_train_epoch_end(self, epoch):
        pass

    def on_eval_epoch_end(self, epoch):
        pass

    def on_train_iter_end_fused(self, *args, **kwargs):
        metrics = []
        for pred_single, label_single in zip(pred, label):
            metrics.append(on_train_iter_end_fused(*args, **kwargs))
        return metrics

    def on_eval_iter_end_fused(self, *args, **kwargs):
        metrics = []
        for pred_single, label_single in zip(pred, label):
            metrics.append(on_eval_iter_end_fused(*args, **kwargs))
        return metrics

    def on_train_epoch_end_fused(self, *args, **kwargs):
        metrics = []
        for pred_single, label_single in zip(pred, label):
            metrics.append(on_train_epoch_end_fused(*args, **kwargs))
        return metrics

    def on_eval_epoch_end_fused(self, *args, **kwargs):
        metrics = []
        for pred_single, label_single in zip(pred, label):
            metrics.append(on_eval_epoch_end_fused(*args, **kwargs))
        return metrics
 


class HFTATrainer():

    def __init__(self, 
                 model_cls, model_args, model_kwargs, 
                 optim_cls, optim_args, optim_kwargs
                 train_loader, eval_loader,
                 loss_fn, metric_fn,
                 epochs, device):

        self.model_cls = [model_cls]
        self.model_args = [model_args]
        self.model_kwargs = [model_kwargs]

        self.optim_cls = [optim_cls]
        self.optim_args = [optim_args]
        self.optim_kwargs = [optim_kwargs]

        self.train_loader = [train_loader]
        self.eval_loader = [eval_loader]

        self.loss_fn = [loss_fn]
        self.metric_fn = [metric_fn]

        self.epochs = epochs
        self.device = device

        self.is_concrete = False

    def train(self):
        pass

    def eval(self):
        pass

    def join(self, other):
        pass

    def start(self, other):
        pass

    def get_metrics(self, other)

class AutoHFTARunner(Runner):
    """
    An implementation of ``ax.core.runner.Runner`` that delegates job submission to
    HFTARunner. This works hand to hand with HFTATrainer (a simple, proof-of-concept
    trainer) to provide nessisary information to the 

    """

    def __init__(
        self,
        tracker_base: str,
        component: Callable[..., AppDef],
        component_const_params: Optional[Dict[str, Any]] = None,
        scheduler: str = "local",
        cfg: Optional[Mapping[str, CfgVal]] = None,
    ) -> None:
        self._component: Callable[..., HFTATrainer] = component
        self._scheduler: str = scheduler
        self._cfg: Optional[Mapping[str, CfgVal]] = cfg
        # need to use the same runner in case it has state
        # e.g. torchx's local_scheduler has state hence need to poll status
        # on the same scheduler instance
        self._torchx_runner: torchx_Runner = get_runner()
        self._tracker_base = tracker_base
        self._component_const_params: Dict[str, Any] = component_const_params or {}

    def run(self, trial: BaseTrial) -> Dict[str, Any]:
        """
        Submits the trial (which maps to an AppDef) as a job
        onto the scheduler using ``torchx.runner``.

        ..  note:: only supports `Trial` (not `BatchTrial`).
        """

        if not isinstance(trial, Trial):
            raise ValueError(
                f"{type(trial)} is not supported. Check your experiment setup"
            )

        parameters = dict(self._component_const_params)
        parameters.update(not_none(trial.arm).parameters)
        component_args = inspect.getfullargspec(self._component).args
        if "trial_idx" in component_args:
            parameters["trial_idx"] = trial.index

        if "experiment_name" in component_args:
            parameters["experiment_name"] = trial.experiment.name

        if "tracker_base" in component_args:
            parameters["tracker_base"] = self._tracker_base

        appdef = self._component(**parameters)
        app_handle = self._torchx_runner.run(appdef, self._scheduler, self._cfg)
        return {
            TORCHX_APP_HANDLE: app_handle,
            TORCHX_RUNNER: self._torchx_runner,
            TORCHX_TRACKER_BASE: self._tracker_base,
        }

    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> Dict[TrialStatus, Set[int]]:
        trial_statuses: Dict[TrialStatus, Set[int]] = {}

        for trial in trials:
            app_handle: str = trial.run_metadata[TORCHX_APP_HANDLE]
            torchx_runner = trial.run_metadata[TORCHX_RUNNER]
            app_status: AppStatus = torchx_runner.status(app_handle)
            trial_status = APP_STATE_TO_TRIAL_STATUS[app_status.state]

            indices = trial_statuses.setdefault(trial_status, set())
            indices.add(trial.index)

        return trial_statuses

    def stop(
        self, trial: BaseTrial, reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Kill the given trial."""
        app_handle: str = trial.run_metadata[TORCHX_APP_HANDLE]
        self._torchx_runner.stop(app_handle)
        return {"reason": reason} if reason else {}

import copy
import torch
import torch.optim as torch_optim
import torch.optim.lr_scheduler as torch_lr_scheduler

import pytorch_lightning as pl

from ..parsers import get_class
from ..models.layers import init_weights
from .assembler import get_model, get_preprocessors, get_losses
from . import schedulers as custom_lr_scheduler


class Default(pl.LightningModule):
    def __init__(self, cfg, logdir, n_epochs):
        super().__init__()
        self.cfg = cfg
        self.logdir = logdir

        # -- Load data processing pipeline
        transforms, augmentations = get_preprocessors(cfg)

        # -- Make Model
        model = get_model(cfg)
        init_weights(model, cfg)

        losses, metrics, omlosses = get_losses(cfg)

        self.model = model
        self.transforms = transforms
        self.augmentations = augmentations
        self.losses = torch.nn.ModuleList(losses)
        self.metrics = torch.nn.ModuleList(metrics)
        self.omlosses = omlosses

        self.optimizer_cfg = cfg["optimizer"]
        self.lr_scheduler_cfg = cfg["lr_scheduler"]

        self.n_epochs = n_epochs

        from collections import defaultdict
        self.phases = defaultdict(int)

        # Explicitely save those hyperparameters
        self.save_hyperparameters(
            "cfg", "logdir", "n_epochs"
        )

    def get_n_epochs(self):
        return self.n_epochs

    def forward(self, nx):
        return self.model(nx)

    def compute_loss(self, outputs, outputs_true, phase:str, batch_idx=-1):
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        for loss_object in self.losses:
            apply_to = loss_object._apply_to
            loss = loss_object(outputs[apply_to], outputs_true[apply_to])
            self.log(f"{phase}/loss/{loss_object._name}", loss, on_epoch=True)
            total_loss = total_loss + loss

        for omloss in self.omlosses:
            raw_batch = dict(prediction=outputs, truth=outputs_true,
                             epoch=self.current_epoch, step=batch_idx)
            lobj = omloss(
                raw_batch=raw_batch, device=self.device, dtype=self.dtype,
                logger=lambda name, value, **kwargs: self.log(f"{phase}/omloss/{name}", value, on_epoch=True, **kwargs)
            )
            loss = lobj.loss()
            lobj._write_logs()  # FIXME: move this inside ouptput_math in some way

            if phase != "train" or self.global_step%50==0:
                pa = self.logdir + f"/{phase}_{self.current_epoch}_{self.phases[phase]}"
                lobj._plt(pa, save_all=("phase"=="test"))
            self.phases[phase] += 1

            total_loss = total_loss + loss

        self.log(f"{phase}/loss", total_loss, on_epoch=True)
        return total_loss

    @torch.no_grad()
    def compute_metrics(self, outputs, outputs_true, phase:str):
        for metric_object in self.metrics:
            apply_to = metric_object._apply_to
            val =  metric_object(outputs[apply_to], outputs_true[apply_to])
            self.log(f"{phase}/metric/{metric_object._name}", val, on_epoch=True, prog_bar=True)

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            batch = self.transforms(batch)
            batch = self.augmentations(batch)
        outputs = self(batch)

        loss = self.compute_loss(outputs, batch, phase="train", batch_idx=batch_idx)
        self.compute_metrics(outputs, batch, phase="train")
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            batch = self.transforms(batch)
        outputs = self(batch)

        self.compute_loss(outputs, batch, phase="val")
        self.compute_metrics(outputs, batch, phase="val")

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            batch = self.transforms(batch)
        outputs = self(batch)

        self.compute_loss(outputs, batch, phase="test")
        self.compute_metrics(outputs, batch, phase="test")

    def configure_optimizers(self):
        # Filter model parameters w/o gradients
        model_params = list(filter(
            lambda npa: npa[1].requires_grad, self.model.named_parameters()
        ))
        # No weight-decay on bias
        opti_params = {**copy.deepcopy(self.optimizer_cfg)}
        wd = opti_params.get("weight_decay", 0)
        if "weight_decay" in opti_params:
            del opti_params["weight_decay"]

        model_params = [
            {"params": list(
                p for n, p in model_params if (not n.endswith(".weight")) or (".batchnorm." in n)
            ), "weight_decay": 0
            }, {"params": list(
                p for n, p in model_params if n.endswith(".weight") and (".batchnorm." not in n)
            ), "weight_decay": wd
            }
        ]
        optimizer = get_class(self.optimizer_cfg, torch_optim, params=model_params)

        # Learning-rate scheduler
        lr_scheduler = get_class(
            self.lr_scheduler_cfg, [torch_lr_scheduler, custom_lr_scheduler], optimizer=optimizer,
            n_epochs=self.n_epochs, steps_per_epoch=self.training_steps_per_epoch
        )
        if "onecycle" in self.lr_scheduler_cfg["type"].lower():
            lr_scheduler = {
                "scheduler": lr_scheduler,
                "interval": "step",
            }

        return [optimizer], [lr_scheduler]

    def optimizers(self):  # Fix annoying pt-lightning behaviour
        optimizers = super().optimizers()
        if type(optimizers) != list:
            optimizers = [optimizers]
        return optimizers

    def lr_schedulers(self):  # Fix annoying pt-lightning behaviour
        lr_schedulers = super().lr_schedulers()
        if type(lr_schedulers) != list:
            lr_schedulers = [lr_schedulers]
        return lr_schedulers

    @property
    def test_models(self):
        return ["best"]

    @property
    def training_steps_per_epoch(self) -> int:
        """Total training steps inferred from datamodule and devices.
        Feature still missing in pt-lightning so little hack from:
        https://github.com/PyTorchLightning/pytorch-lightning/issues/5449#issuecomment-774265729
        """
        if hasattr(self, "_training_steps_per_epoch"):
            return self._training_steps_per_epoch

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.trainer._data_connector._train_dataloader_source.dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        self._training_steps_per_epoch = (batches // effective_accum)
        return self._training_steps_per_epoch


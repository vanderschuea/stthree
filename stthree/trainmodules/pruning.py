import numpy as np
import torch
import copy
import warnings
import torch.optim as torch_optim
import torch.optim.lr_scheduler as torch_lr_scheduler
import pytorch_lightning.callbacks as pl_callbacks

from . import schedulers as custom_lr_scheduler
from ..parsers import get_class
from .default import Default
from .utils import HybridLRScheduler, HybridOptim
from ..models.layers.model_config import load_pretrained_weights
from ..models.utils import count as count_ops


class RatioIncrease(Default):
    def __init__(
        self, cfg, logdir, n_epochs, *, start_epoch, end_epoch, power=1, fix_from=None, k_models=1,
        ratio_granularity="step", update_frequency=1, reload_best=False, n_ft_epochs=10
    ):
        super().__init__(cfg, logdir, n_epochs)
        assert ratio_granularity in {"epoch", "step"}

        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.power = power
        self.fix_from = fix_from or n_epochs
        self.fixed = False  # TODO: change every fix to frozen!
        self.k_models = k_models

        self.update_frequency = update_frequency
        self.ratio_granularity = ratio_granularity
        self.reload_best = reload_best
        self.n_ft_epochs = n_ft_epochs

        self.save_hyperparameters(
            "start_epoch", "end_epoch", "power", "fix_from", "k_models", "ratio_granularity", "reload_best", "n_ft_epochs"
        )

    @property
    def n_steps(self):
        return ((self.end_epoch-self.start_epoch)*super().training_steps_per_epoch) // self.update_frequency

    def get_n_epochs(self):
        return super().get_n_epochs() + self.n_ft_epochs

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        if self.reload_best and self.current_epoch == self.n_epochs and self.n_ft_epochs > 0:
            # Find model checkpoint in callbacks to reload weights
            for callback in self.trainer.callbacks:
                if isinstance(callback, pl_callbacks.ModelCheckpoint):
                    print("Reloading weights from callback.best_model_path") # reload optimizer states??
                    load_pretrained_weights(self.model, callback.best_model_path)
                    if hasattr(self, "fix_layers"):
                        self.fix_layers()
                    break

    def training_step(self, batch, batch_idx):
        if self.current_epoch >= self.n_epochs or self.k_models == 1:
            # BN tuning
            if batch_idx % self.k_models != 0:
                return None
            return super().training_step(batch, batch_idx)

        else:
            if batch_idx % self.k_models == 0:
                with torch.no_grad():
                    batch = self.transforms(batch)
                    batch = self.augmentations(batch)
                self._current_batch = batch
            else:
                batch = self._current_batch
            # Gradient accumulation for k_models to avg gradients
            outputs = self(batch)
            loss = self.compute_loss(outputs, batch, phase="train", batch_idx=batch_idx)
            loss = loss/self.k_models
            self.compute_metrics(outputs, batch, phase="train")
            return loss


    def on_train_batch_end(self, outputs, batch, batch_idx):
        super().on_train_batch_end(outputs, batch, batch_idx)

        current_epoch = self.current_epoch
        if self.fixed:
            return

        # Check if we can update the pruning mask this step
        steps_per_epoch = 1 if self.ratio_granularity =="epoch" else self.training_steps_per_epoch
        batch_steps = 0 if self.ratio_granularity =="epoch" else (batch_idx//self.k_models + 1)
        id_step = (current_epoch-self.start_epoch)*steps_per_epoch + batch_steps

        if id_step % self.update_frequency != 0: # FIXME
            return

        # Compute the value of the pruning-ratio
        if current_epoch < self.start_epoch:
            ratio = 0.0  # Safer than returning in case of pretrained w
        elif current_epoch < self.end_epoch and self.power!=0:
            max_steps = (self.end_epoch-self.start_epoch) if self.ratio_granularity =="epoch" else self.n_steps
            ratio = 1 - (1 - (id_step//self.update_frequency)/max_steps)**self.power
        else:
            ratio = 1.0

        fix_network = False
        if current_epoch == self.fix_from:
            if ratio != 1.0:
                import warnings
                warnings.warn("You are trying to freeze a network without having reached your max prune rate!\n Max prune rate will be applied anyways")
                ratio = 1.0
            fix_network = True

        # Temperature scaling is for ProbMask only
        eps = 0.03
        temperature = 1 / ((1-eps) * (1-current_epoch/self.n_epochs) + eps)

        # Update model
        if (batch_idx+1) % self.k_models == 0 and not self.fixed:
            self._change_model(ratio, temperature, fix_model=fix_network)
            self.fixed = fix_network

    def configure_optimizers(self):
        # Filter model parameters w/o gradients
        all_model_params = list(filter(
            lambda npa: npa[1].requires_grad, self.model.named_parameters()
        ))
        proba_params = [p for n, p in all_model_params if n.endswith(".probas")]

        # In case only one optimizer is set
        if type(self.optimizer_cfg) != list or len(self.optimizer_cfg) < 2:
            if type(self.optimizer_cfg) == list:
                self.optimizer_cfg = self.optimizer_cfg[0]
            # Warning in case of found probaparams
            if len(proba_params) > 0:
                warnings.warn("Using probabilities but only one optimizer was specified, this could decrease performance")
            return super().configure_optimizers()

        # No weight-decay on bias
        opti_params = {**copy.deepcopy(self.optimizer_cfg[0])}
        wd = opti_params.get("weight_decay", 0)
        if "weight_decay" in opti_params:
            del opti_params["weight_decay"]

        model_params = [
            {"params": list(
                p for n, p in all_model_params if n.endswith(".bias") or (".batchnorm." in n)
            ), "weight_decay": 0
            }, {"params": list(
                p for n, p in all_model_params if n.endswith(".weight") and (".batchnorm." not in n)
            ), "weight_decay": wd
            }
        ]

        optimizers = [
            get_class(self.optimizer_cfg[0], torch_optim, params=model_params),
            get_class(self.optimizer_cfg[1], torch_optim, params=proba_params)
        ]
        hybrid_optimizer = HybridOptim(optimizers)

        # Learning-rate scheduler
        lr_schedulers = []
        for iopt, optimizer in enumerate(optimizers):
            lr_scheduler = get_class(
                self.lr_scheduler_cfg, [torch_lr_scheduler, custom_lr_scheduler], optimizer=optimizer,
                n_epochs=self.n_epochs, steps_per_epoch=self.training_steps_per_epoch
            )
            if "onecycle" in self.lr_scheduler_cfg["type"].lower():
                lr_scheduler = {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            lr_scheduler = HybridLRScheduler(hybrid_optimizer, iopt, lr_scheduler)
            lr_schedulers.append(lr_scheduler)

        return [hybrid_optimizer], lr_schedulers

    def _change_model(self, ratio, temperature, fix_model):
        pass


class PruningUtilities():
    def get_prunable_weights(self):
        for lname, layer in self.get_prunable_modules():
            yield lname + ".weight", layer.weight

    def get_prunable_probas(self):
        for lname, layer in self.get_prunable_modules():
            yield lname + ".probas", layer.probas

    def get_prunable_modules(self):
        for lname, layer in self.model.named_modules():
            if not hasattr(layer, "set_prune"):
                continue
            yield lname, layer

    @staticmethod
    def gpu_quantile(x, ratio):
        sorted_x, _ = torch.sort(x)
        idx = (len(sorted_x)-1)*ratio
        idx1 = int(np.floor(idx))
        idx2 = int(np.ceil(idx))

        e1, e2 = sorted_x[idx1], sorted_x[idx2]
        perc = e1 + (e2-e1)*(idx-idx1)
        return perc.item()

    @staticmethod
    def cpu_quantile(x, ratio):
        return np.quantile(x, ratio)

    @staticmethod
    def gpu_cumsum_quantile_ops(x, ratio, n_ops, ops_per_layer):
        k = n_ops*(1-ratio)
        a, b = -1, 1

        def f(v):
            s = 0
            for lname, probas in x.items():
                s += (probas - v).clamp(0, 1).sum()/probas.numel()*ops_per_layer[lname]
            return (s - k)/n_ops
        itr = 0
        while True:
            itr += 1
            v = (a + b) / 2
            obj = f(v)
            if abs(obj) < 1e-7 or itr > 40:
                break
            if obj < 0:
                b = v
            else:
                a = v
        return v

    @staticmethod
    def gpu_cumsum_wquantile_ops(x, ratio, n_ops, ops_per_layer):
        k = n_ops*(1-ratio)
        a, b = 0, 0
        for absw in x.values():
            b = max(b, absw.max().item())
        v = 0.7*ratio*b
        def f(v):
            s = 0
            for lname, absw in x.items():
                s += (absw >= v).int().sum()/absw.numel()*ops_per_layer[lname]
            return (s - k)/n_ops
        itr = 0
        while True:
            itr += 1
            obj = f(v)
            if abs(obj) < 1e-7 or itr > 40:
                break
            if obj < 0:
                b = v
            else:
                a = v
            v = (a + b) / 2
        return v

    @staticmethod
    def gpu_cumsum_quantile(x, ratio):
        k = (len(x))*(1-ratio)
        a, b = -1, 1

        def f(v):
            s = (x - v).clamp(0, 1).sum()
            return s - k
        itr = 0
        while (1):
            itr += 1
            v = (a + b) / 2
            obj = f(v)
            if abs(obj) < 1e-7 or itr > 40:
                break
            if obj < 0:
                b = v
            else:
                a = v
        return v

    # Code used for l1-thresholding for ST-3
    @torch.no_grad()
    def l1(self, prune_ratio, use_gpu, use_std=False):
        # Make all weights into a single vector and compute global l1 th
        if prune_ratio == 0:
            th = -1
        elif use_gpu:
            flat_model_weights = []
            for n, data in self.get_prunable_weights():
                layer_weights = data.flatten()
                if use_std:
                    std = 1/np.sqrt(np.sum(data.shape))
                    layer_weights = layer_weights/std
                flat_model_weights.append(layer_weights)
            flat_model_weights = torch.cat(flat_model_weights)

            th = PruningUtilities.gpu_quantile(flat_model_weights.abs_(), prune_ratio)

        else:
            flat_model_weights = np.array([])
            for n, data in self.get_prunable_weights():
                layer_weights = data.cpu().numpy().flatten()
                if use_std:
                    std = 1/np.sqrt(np.sum(data.shape))
                    layer_weights = layer_weights/std
                flat_model_weights = np.concatenate((flat_model_weights, layer_weights))

            th = PruningUtilities.cpu_quantile(np.abs(flat_model_weights), prune_ratio)
        self.log("pruning/offset", th, prog_bar=True, on_step=self.on_step_logging, on_epoch=(not self.on_step_logging))
        threshold_per_layer = {}
        for n, data in self.get_prunable_weights():
            layer_th = th
            if use_std:
                std = 1/np.sqrt(np.sum(data.shape))
                layer_th = layer_th*std
            threshold_per_layer[n] = layer_th
        return threshold_per_layer

    # Code used for l1-thresholding for ST-3-sigma
    @torch.no_grad()
    def l1std(self, prune_ratio, use_gpu):
        return self.l1(prune_ratio, use_gpu, use_std=True)

    @torch.no_grad()
    def proba(self, prune_ratio, use_gpu):
        # Make all weights into a single vector and compute global probability offset
        if use_gpu:
            flat_model_weights = []
            for n, data in self.get_prunable_probas():
                layer_weights = data.flatten()
                flat_model_weights.append(layer_weights)
            flat_model_weights = torch.cat(flat_model_weights)

            th = PruningUtilities.gpu_cumsum_quantile(flat_model_weights, prune_ratio)
            th = max(th, 0)
            self.log("pruning/offset", th, prog_bar=True, on_step=self.on_step_logging, on_epoch=(not self.on_step_logging))
        else:
            raise NotImplementedError()

        threshold_per_layer = {}
        for n, _ in self.get_prunable_weights():
            threshold_per_layer[n] = th
        return threshold_per_layer

    @torch.no_grad()
    def apply_thresholds_to_layers(self, threshold_per_layer, temperature):
        for lname, layer in self.get_prunable_modules():
            name = lname + ".weight"
            layer.set_prune(threshold_per_layer[name])
            if hasattr(layer, "set_temperature"):
                layer.set_temperature(temperature)

    @torch.no_grad()
    def fix_layers(self):
        for lname, layer in self.get_prunable_modules():
            if hasattr(layer, "fix"):
                layer.fix()

    @property
    def sparsity(self):
        """ Computes and returns the current 'exact' sparsity of the model
        """
        with torch.no_grad():
            pruned, tot = 0,0
            for lname, layer in self.get_prunable_modules():
                lpruned = (layer.mask==0).int().sum()
                pruned += lpruned
                tot += layer.mask.numel()
                self.log("pruning/sparsity/"+lname, lpruned/layer.mask.numel(), prog_bar=False, on_step=True, on_epoch=False)
            if tot == 0:
                return 0.0
            return pruned/tot

    @property
    def proba_sparsity(self):
        """ Computes and returns the current 'probabilistic' sparsity of the model
        """
        with torch.no_grad():
            pruned, tot = 0,0
            for lname, layer in self.get_prunable_modules():
                lpruned = (1.0-layer.probas).sum()
                pruned += lpruned
                tot += layer.probas.numel()
                self.log("pruning/sparsity/"+lname, lpruned/layer.probas.numel(), prog_bar=False, on_step=True, on_epoch=False)
            if tot == 0:
                return 0.0
            return pruned/tot

    @property
    def sparsity_ops(self):
        with torch.no_grad():
            pruned, tot = 0,0
            for lname, layer in self.get_prunable_modules():
                lpruned = (layer.mask==0).int().sum()/layer.mask.numel()*self.ops_per_layer[lname]
                pruned += lpruned
                tot += self.ops_per_layer[lname]
            if tot == 0:
                return 0.0
            return pruned/tot

    @property
    def proba_sparsity_ops(self):
        with torch.no_grad():
            pruned, tot = 0,0
            for lname, layer in self.get_prunable_modules():
                lpruned = (1.0-layer.probas).sum()/layer.probas.numel()*self.ops_per_layer[lname]
                pruned += lpruned
                tot += self.ops_per_layer[lname]
            if tot == 0:
                return 0.0
            return pruned/tot

    def save_ops(self, model, cfg, device):
        model.eval()
        # FIXME: kinda dirty hack for now
        size = 32 if "Cifar" in cfg["dataset"]["type"] else 224
        inputs = dict(image=torch.zeros((1,3,size,size), device=device))
        _, self.n_ops, _, self.ops_per_layer = count_ops(model, inputs)


class PruningIncrease(RatioIncrease, PruningUtilities):
    def __init__(self, cfg, logdir, n_epochs, *, prune_max, pruning_criterion, use_gpu=True, normalize_ops=False, **kwargs):
        super().__init__(cfg, logdir, n_epochs, **kwargs)
        self.prune_max = prune_max
        self.pruning_criterion = pruning_criterion
        self.use_gpu = use_gpu
        self.normalize_ops = normalize_ops
        self.on_step_logging = True
        self.save_hyperparameters(
            "prune_max", "pruning_criterion", "use_gpu"
        )

    @torch.no_grad()
    def _change_model(self, ratio, temperature, fix_model):
        # search for method with name 'pruning_criterion'
        suffix = "_ops" if self.normalize_ops else ""
        threshold_per_layer = getattr(self, self.pruning_criterion+suffix)(self.prune_max*ratio, self.use_gpu)
        self.apply_thresholds_to_layers(threshold_per_layer, temperature)
        if fix_model:
            self.fix_layers()

        if "proba" in self.pruning_criterion and not self.fixed:
            self.log("pruning/sparsity", self.proba_sparsity, prog_bar=True, on_step=True, on_epoch=False)
            self.log("pruning/ops", self.proba_sparsity_ops, prog_bar=True, on_step=True, on_epoch=False)
        else:
            self.log("pruning/sparsity", self.sparsity, prog_bar=True, on_step=True, on_epoch=False)
            self.log("pruning/ops", self.sparsity_ops, prog_bar=True, on_step=True, on_epoch=False)

    @torch.no_grad()
    def on_pretrain_routine_start(self) -> None:
        super().on_pretrain_routine_start()
        self.save_ops(self.model, self.cfg, self.device)



class RewindUtilities():
    def _get_saving_callback(self):
        for callback in self.trainer.callbacks:
            if isinstance(callback, pl_callbacks.ModelCheckpoint):
                return callback
        else:
            warnings.warn(f"Couldn't find model checkpoint !")
        return None

    def rewind(self):
        # Find model checkpoint in callbacks to reload weights
        callback = self._get_saving_callback()
        if callback:
            print(f"Reloading weights from {callback.best_model_path}")
            load_pretrained_weights(self.model, callback.best_model_path)
            callback.rewind_best_models.append(callback.best_model_path)  # save path to run test set on it
            for k, v in callback.rewind_state.items():
                setattr(callback, k, copy.deepcopy(v))

        # Reset to initial lr
        for optimizer in self.optimizers():
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["rewind_state"]

        # Reset state of lr_scheduler
        for lr_scheduler in self.lr_schedulers():
            lr_scheduler.load_state_dict(lr_scheduler.rewind_state)

    def save_initial_optimizers(self):
        for optimizer in self.optimizers():
            for param_group in optimizer.param_groups:
                param_group["rewind_state"] = param_group["lr"]

        for lr_scheduler in self.lr_schedulers():
            lr_scheduler.rewind_state = lr_scheduler.state_dict()

        callback = self._get_saving_callback()
        if callback:
            callback.rewind_state = {
                "current_score": copy.deepcopy(callback.current_score),
                "best_k_models": copy.deepcopy(callback.best_k_models),
                "kth_best_model_path": copy.deepcopy(callback.kth_best_model_path),
                "best_model_score": copy.deepcopy(callback.best_model_score),
                "best_model_path": copy.deepcopy(callback.best_model_path),
                "last_model_path": copy.deepcopy(callback.last_model_path),
            }
            callback.rewind_best_models = []
            callback.rewind_last_models = [] # TODO: support at some point?


class RewindPruningIncrease(PruningIncrease, RewindUtilities):
    def get_n_epochs(self):
        return super().get_n_epochs() + self.n_epochs

    def on_train_start(self) -> None:
        super().on_train_start()
        self.save_initial_optimizers()

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        if self.current_epoch == super().get_n_epochs():
            self.rewind()
            self.fix_layers()

    def _change_model(self, ratio, temperature, fix_model):
        if self.current_epoch < super().get_n_epochs():
            super()._change_model(ratio, temperature, fix_model)

    @property
    def test_models(self):
        return self._get_saving_callback().rewind_best_models + super().test_models


class RecursiveRewindPruning(Default, PruningUtilities, RewindUtilities):
    def __init__(self, cfg, logdir, n_epochs, *, n_steps, pruning_criterion, prune_step=0.2, use_gpu=True, normalize_ops=False, **kwargs):
        super().__init__(cfg, logdir, n_epochs, **kwargs)
        self.n_steps = n_steps
        self.pruning_criterion = pruning_criterion
        self.prune_step = prune_step
        self.use_gpu = use_gpu
        self.normalize_ops = normalize_ops
        self.on_step_logging = False
        self.save_hyperparameters(
            "n_steps", "pruning_criterion", "prune_step", "use_gpu"
        )

    def get_n_epochs(self):
        return super().get_n_epochs()*self.n_steps

    def on_train_start(self) -> None:
        super().on_train_start()
        self.save_initial_optimizers()

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        if self.current_epoch>0 and self.current_epoch % super().get_n_epochs() == 0:
            self.rewind()

            # Compute which pruning perc to apply
            step_idx = self.current_epoch // super().get_n_epochs()
            prune_ratio = 1 - (1.0 - self.prune_step)**step_idx
            suffix = "_ops" if self.normalize_ops else ""
            threshold_per_layer = getattr(self, self.pruning_criterion+suffix)(prune_ratio, self.use_gpu)

            self.apply_thresholds_to_layers(threshold_per_layer, 0)

    def training_step(self, batch, batch_idx):
        self.log("pruning/ops", self.sparsity_ops, prog_bar=True, on_step=True, on_epoch=False)
        self.log("pruning/sparsity", self.sparsity, prog_bar=True, on_step=True, on_epoch=False)
        return super().training_step(batch, batch_idx)

    @property
    def test_models(self):
        return self._get_saving_callback().rewind_best_models + super().test_models

    @torch.no_grad()
    def on_pretrain_routine_start(self) -> None:
        super().on_pretrain_routine_start()
        self.save_ops(self.model, self.cfg, self.device)







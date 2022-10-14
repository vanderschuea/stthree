import math
import warnings
from torch.optim.lr_scheduler import _LRScheduler

class CosineOneCycleLR(_LRScheduler):
    def __init__(self, optimizer, n_epochs, steps_per_epoch=1, warmup_epoch=0, lr_div=25, beta_max=0.95, beta_min=0.85, update_beta=False):
        """
        params:
            - optimizer: pytorch optimizer
            - lr: maximum learning rate
            - n_epochs: total #epochs
            - steps_per_epoch: length of an epoch (if =1 only updates every epoch, NOT RECOMMENDED)
            - warmup_epoch: #epochs during which the lr increases
            - lr_div: lr_start = lr/lr_div (default: 25)
            - beta_max: maximal momentum value (default: 0.95)
            - beta_min: minimal momentum value (default: 0.85)
        """
        self.optimizer = optimizer
        # TODO: support different starting LRs at some point
        for param_group in self.optimizer.param_groups:
            self.lr_max = param_group["lr"] # Supposes this was set through lr-tuning
            self.is_adam = "betas" in param_group
        self.lr_start = self.lr_max/lr_div
        self.lr_min = self.lr_max/100

        self.n_steps = n_epochs*steps_per_epoch
        self.mid_step = warmup_epoch*steps_per_epoch
        self.current_step = 0

        self.beta_min = beta_min
        self.beta_max = beta_max
        self.update_beta = update_beta

        self.warned = False
        self.step()

    def step(self):
        if self.current_step == 0 or self.n_steps == 0:
            lr = self.lr_min
            beta = self.beta_min
        elif self.current_step < self.mid_step:
            cosine_factor = 1 + math.cos(math.pi*(1-self.current_step/self.mid_step))
            lr = self.lr_start + cosine_factor * (self.lr_max-self.lr_start) / 2
            beta = self.beta_max + cosine_factor * (self.beta_min-self.beta_max) / 2
        elif self.current_step <= self.n_steps:
            cosine_factor = 1 + math.cos(math.pi*(1-(self.current_step-self.mid_step)/(self.n_steps-self.mid_step)))
            lr = self.lr_max + cosine_factor * (self.lr_min-self.lr_max) / 2
            beta = self.beta_min + cosine_factor * (self.beta_max-self.beta_min) / 2
        else:
            if not self.warned:
                warnings.warn(f"Had a maximum of {self.n_steps}, but got up to {self.current_step}, lr won't be modified")
            self.warned = True
            return

        self.current_step += 1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            if self.update_beta:
                if self.is_adam:
                    betas = param_group["betas"]
                    param_group["betas"] = (beta, betas[1])
                else:
                    param_group["momentum"] = beta


class CosineLR(_LRScheduler):
    def __init__(self, optimizer, n_epochs, warmup_epoch):
        self.optimizer = optimizer
        for param_group in self.optimizer.param_groups:
            self.min_lr = param_group['lr']
        self.epoch = 0
        self.warmup_length = warmup_epoch
        self.n_epochs = n_epochs
        self.step()

    def step(self):
        if self.epoch < self.warmup_length:
            lr = self.min_lr * (self.epoch + 1) / self.warmup_length
        else:
            e = self.epoch - self.warmup_length
            es = self.n_epochs - self.warmup_length
            lr = 0.5 * (1 + math.cos(math.pi * e / es)) * self.min_lr
        self.epoch += 1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
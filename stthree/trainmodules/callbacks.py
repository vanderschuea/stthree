import pytorch_lightning.callbacks as pl_callbacks


class DelayedModelCheckpoint(pl_callbacks.ModelCheckpoint):
    def __init__(self, start_saving_epoch=-1, **kwargs):
        super().__init__(**kwargs)
        self.start_saving_epoch = start_saving_epoch

    def check_monitor_top_k(self, trainer, current = None):
        if trainer.current_epoch < self.start_saving_epoch:
            return False
        return super().check_monitor_top_k(trainer, current)
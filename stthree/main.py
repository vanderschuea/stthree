import os
from datetime import datetime
from filelock import FileLock
import time
import qtoml as toml

import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks

from .parsers import get_config, get_class
from . import trainmodules
from . import datasets
from .trainmodules import callbacks as custom_callbacks


def get_trainer_callbacks(cfg):
    callbacks = []
    for callback_params in cfg["trainer"].get("callbacks"):
        callback = get_class(callback_params, [pl_callbacks, custom_callbacks])
        callbacks.append(callback)
    return callbacks


def get_dataset(cfg):
    return get_class(cfg["dataset"], datasets)


def execute_strategy(cfg, logdir=None):
    # -- Load data
    dataset = get_dataset(cfg)


    # -- Load trainingmodule (and restart from checkpoint if necessary)
    train_module = get_class(
        cfg["trainmodule"], trainmodules, cfg=cfg, logdir=logdir
    )
    load_ckpt = ("resume" in cfg["trainer"]) and (cfg["trainer"]["resume"]["path"] != "")
    if load_ckpt:
        ckpt = cfg["trainer"]["resume"]["path"]
        train_module = train_module.load_from_checkpoint(
            checkpoint_path=ckpt, map_location="cpu"
        )

    # -- Start training
    pl.seed_everything(cfg["network"]["weight_init"]["seed"], workers=True)
    callbacks = get_trainer_callbacks(cfg)
    loggers = [
        pl.loggers.CSVLogger(logdir, name="."),
        pl.loggers.TensorBoardLogger(logdir, name=".", default_hp_metric=False)
    ]

    trainer = pl.Trainer(**{
        "default_root_dir": logdir,
        "callbacks": callbacks,
        "max_epochs": train_module.get_n_epochs(),
        "logger": loggers,
        **cfg["trainer"]["parameters"],
    })
    if "tuner" in cfg["trainer"] and not load_ckpt:
        print("tuning")
        lr_finder = trainer.tuner.lr_find(train_module, dataset, **cfg["trainer"]["tuner"])

        # Plot with
        fig = lr_finder.plot(suggest=True, show=False)
        import matplotlib.pyplot as plt
        fig.savefig(os.path.join(logdir, "lr-finder.png"))
        plt.close(fig)

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()
        if new_lr is not None:
            print(f"Going to use New LR {new_lr}")
            cfg["optimizer"]["lr"] = new_lr
            train_module.optimizer_cfg["lr"] = new_lr
        print("Exiting program due to pl-lightning memory error")
        exit(-1)

    # -- Save config (save only here in cas LR-tuner was set)
    if logdir is not None:
        with open(os.path.join(logdir, "config.toml"), 'w') as cfg_fp:
            cfg_fp.write(toml.dumps(cfg))

    trainer.fit(train_module, dataset, ckpt_path=(ckpt if load_ckpt else None))

    # Test on last weights and then reload best weights and test
    trainer.test(train_module, dataset, ckpt_path=None)
    for ckpt in train_module.test_models:
        trainer.test(train_module, dataset, ckpt_path=ckpt)


def main():
    cfg = get_config()
    import matplotlib
    matplotlib.use(cfg.get("mpl_backend", "agg"))
    logdir_suffix = "-" + cfg.get("name", "")

    # Create main log directory
    main_logspath = os.environ.get("STTHREE_LOGSPATH", "logs")
    if not os.path.exists(main_logspath):
        os.mkdir(main_logspath)
    lock = FileLock(f"{main_logspath}/mk_logdir.lock")
    with lock:
        while True:
            logdir = f"{main_logspath}/" + "{:%Y-%m-%dT%H%M%S}".format(datetime.now()) +\
                     logdir_suffix
            if not os.path.exists(logdir):
                break
            time.sleep(1)
        os.mkdir(logdir)

    # Assemble all modules initialized in the config file
    execute_strategy(cfg, logdir=logdir)




if __name__ == "__main__":
    main()
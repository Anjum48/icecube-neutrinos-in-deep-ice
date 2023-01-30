import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.strategies import DDPStrategy

from src.datasets import IceCubeDataModule
from src.models import IceCubeModel
from src.utils import LogSummaryCallback, prepare_loggers_and_callbacks, resume_helper

torch.set_float32_matmul_precision("high")


def get_num_steps(cfg, dm):
    if isinstance(cfg.trainer.devices, ListConfig):
        n_devices = len(cfg.trainer.devices)
    else:
        n_devices = cfg.trainer.devices

    n_steps = (cfg.trainer.max_epochs) * dm.train_steps / n_devices
    n_steps *= cfg.trainer.get("limit_train_batches", 1.0)
    return int(n_steps)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def run_fold(cfg: DictConfig):
    pl.seed_everything(cfg.run.seed + cfg.run.fold, workers=True)
    resume, run_id = resume_helper(cfg)

    monitor_list = [("loss/valid", "min", "loss"), ("metric", "min", "metric")]

    loggers, callbacks = prepare_loggers_and_callbacks(
        cfg.run.timestamp,
        cfg.model.model_name,
        cfg.run.fold,
        monitors=monitor_list,
        tensorboard=cfg.run.logging,
        wandb=cfg.run.logging,
        patience=None,
        run_id=run_id,
        save_weights_only=True,
    )

    callbacks["metric_summary"] = LogSummaryCallback("loss/valid", "min")

    dm = IceCubeDataModule(**cfg.model)
    dm.setup("fit", cfg.run.fold)

    # swa = StochasticWeightAveraging(swa_epoch_start=0.5, annealing_epochs=2)
    # callbacks["swa"] = swa

    model = IceCubeModel(
        T_max=get_num_steps(cfg, dm), **cfg.model, **cfg.trainer, **cfg.run
    )

    trainer = pl.Trainer(
        logger=list(loggers.values()),
        callbacks=list(callbacks.values()),
        resume_from_checkpoint=resume,
        # plugins=DDPStrategy(find_unused_parameters=False),
        # fast_dev_run=True,
        # auto_lr_find=True,
        # overfit_batches=10,
        **cfg.trainer,
    )

    # trainer.tune(model, datamodule=dm)  # Use with auto_lr_find
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    run_fold()

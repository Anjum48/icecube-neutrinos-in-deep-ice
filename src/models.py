import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from graphnet.models.task.reconstruction import (
    AzimuthReconstructionWithKappa,
    ZenithReconstruction,
)
from graphnet.training.loss_functions import VonMisesFisher2DLoss
from transformers import get_cosine_schedule_with_warmup

from src.losses import angular_dist_score, CosineLoss

from src.modules import DynEdge
from src.utils import add_weight_decay


class IceCubeModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "DynEdge",
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        warmup: float = 0.1,
        T_max: int = 1000,
        nb_inputs: int = 8,
        nearest_neighbours: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.loss_fn_azi = VonMisesFisher2DLoss()
        self.loss_fn_zen = nn.L1Loss()
        self.loss_fn_cos = CosineLoss()

        self.model = DynEdge(
            nb_inputs=nb_inputs,
            nb_neighbours=nearest_neighbours,
            global_pooling_schemes=["min", "max", "mean", "sum"],
            features_subset=slice(0, 4),  # NN search using xyzt
        )
        # self.head = nn.Linear(self.model.nb_outputs, 2)
        self.azimuth_task = AzimuthReconstructionWithKappa(
            hidden_size=self.model.nb_outputs,
            loss_function=self.loss_fn_azi,
            target_labels=["azimuth", "kappa"],
        )

        self.zenith_task = ZenithReconstruction(
            hidden_size=self.model.nb_outputs,
            loss_function=self.loss_fn_zen,
            target_labels=["zenith"],
        )

    def forward(self, x):
        emb = self.model(x)
        azi_out = self.azimuth_task(emb)
        zen_out = self.zenith_task(emb)

        return azi_out, zen_out

    def training_step(self, batch, batch_idx):
        pred_azi, pred_zen = self.forward(batch)

        target = batch.y.reshape(-1, 2)

        # weight = 1 - np.exp(-self.global_step / (self.hparams.T_max))
        loss_azi = self.loss_fn_azi(pred_azi, target)
        loss_zen = self.loss_fn_zen(pred_zen, target[:, -1].unsqueeze(-1))
        loss = loss_azi + loss_zen

        pred_angles = torch.stack([pred_azi[:, 0], pred_zen[:, 0]], dim=1)
        # metric = angular_dist_score(pred_angles, target)

        loss_cos = self.loss_fn_cos(pred_angles, target)

        if self.current_epoch > 0:
            loss += loss_cos

        self.log_dict({"loss/train_step": loss})
        return {"loss": loss}

    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack([x["loss"] for x in training_step_outputs]).mean()
        self.log("loss/train", avg_loss, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        pred_azi, pred_zen = self.forward(batch)

        target = batch.y.reshape(-1, 2)

        # weight = 1 - np.exp(-self.global_step / (self.hparams.T_max))
        loss_azi = self.loss_fn_azi(pred_azi, target)
        loss_zen = self.loss_fn_zen(pred_zen, target[:, -1].unsqueeze(-1))
        loss = loss_azi + loss_zen

        pred_angles = torch.stack([pred_azi[:, 0], pred_zen[:, 0]], dim=1)
        metric = angular_dist_score(pred_angles, target)

        loss_cos = self.loss_fn_cos(pred_angles, target)

        if self.current_epoch > 0:
            loss += loss_cos

        output = {
            "val_loss": loss,
            "metric": metric,
            "val_loss_azi": loss_azi,
            "val_loss_zen": loss_zen,
            "val_loss_cos": loss_cos,
        }

        return output

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x["val_loss"] for x in outputs]).mean()
        loss_val_azi = torch.stack([x["val_loss_azi"] for x in outputs]).mean()
        loss_val_zen = torch.stack([x["val_loss_zen"] for x in outputs]).mean()
        val_loss_cos = torch.stack([x["val_loss_cos"] for x in outputs]).mean()
        metric = torch.stack([x["metric"] for x in outputs]).mean()

        self.log_dict(
            {"loss/valid": loss_val, "metric": metric},
            prog_bar=True,
            sync_dist=True,
        )
        self.log_dict(
            {
                "loss/valid_azi": loss_val_azi,
                "loss/valid_zen": loss_val_zen,
                "loss/valid_cos": val_loss_cos,
            },
            prog_bar=False,
            sync_dist=True,
        )

    def configure_optimizers(self):
        parameters = add_weight_decay(
            self,
            self.hparams.weight_decay,
            skip_list=["bias", "LayerNorm.bias"],  # , "LayerNorm.weight"],
        )

        opt = torch.optim.AdamW(parameters, lr=self.hparams.learning_rate)

        sch = get_cosine_schedule_with_warmup(
            opt,
            # num_warmup_steps=int(0.1 * self.hparams.T_max),
            num_warmup_steps=int(0 * self.hparams.T_max),
            num_training_steps=self.hparams.T_max,
            num_cycles=0.5,  # 1,
            last_epoch=-1,
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "interval": "step"},
        }

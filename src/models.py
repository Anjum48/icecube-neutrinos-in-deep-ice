import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from graphnet.models.gnn import DynEdge
from transformers import AdamW, get_cosine_schedule_with_warmup

from src.losses import angular_dist_loss
from src.utils import add_weight_decay


class IceCubeModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "DynEdge",
        lr: float = 0.001,
        weight_decay: float = 0.01,
        warmup: float = 0.1,
        T_max: int = 1000,
        nb_inputs: int = 7,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = DynEdge(
            nb_inputs=nb_inputs,
            global_pooling_schemes=["min", "max", "mean", "sum"],
        )
        self.head = nn.Linear(128, 2)

        self.loss_fn = angular_dist_loss

    def forward(self, x):
        x_trfm = self.pre_transform(x)
        emb = self.model(x_trfm)
        y = self.head(emb)

        y[:, 0] = (1 + torch.tanh(y[:, 0])) * np.pi  # Azimuth can range from 0 to 2pi
        y[:, 1] = torch.sigmoid(y[:, 1]) * np.pi  # Zenith can range from 0 to pi
        return y

    def training_step(self, batch, batch_idx):
        pred = self.forward(batch)
        loss = self.loss_fn(pred, batch.y)

        self.log_dict({"loss/train_step": loss})
        return {"loss": loss}

    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack([x["loss"] for x in training_step_outputs]).mean()
        self.log("loss/train", avg_loss, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch)
        loss = self.loss_fn(pred, batch.y)

        output = {"val_loss": loss}

        return output

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x["val_loss"] for x in outputs]).mean()

        self.log_dict(
            {"loss/valid": loss_val},
            prog_bar=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        parameters = add_weight_decay(
            self,
            self.hparams.weight_decay,
            skip_list=["bias", "LayerNorm.bias"],  # , "LayerNorm.weight"],
        )

        opt = AdamW(parameters, lr=self.hparams.lr)

        sch = get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=int(0.1 * self.hparams.T_max),
            num_training_steps=self.hparams.T_max,
            num_cycles=0.5,  # 1,
            last_epoch=-1,
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "interval": "step"},
        }

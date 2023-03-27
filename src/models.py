import gc

import pytorch_lightning as pl
import torch
import torch.nn as nn
from graphnet.models.task.reconstruction import (
    AzimuthReconstructionWithKappa,
    DirectionReconstructionWithKappa,
    ZenithReconstruction,
)
from graphnet.training.loss_functions import VonMisesFisher2DLoss, VonMisesFisher3DLoss
from torch_geometric.data import Batch
from transformers import get_cosine_schedule_with_warmup

from src.losses import angular_dist_score
from src.modules import GPS, DynEdge, GraphAttentionNetwork, GravNet
from src.utils import add_weight_decay


class IceCubeModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "DynEdge",
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        eps: float = 1e-8,
        warmup: float = 0.0,
        T_max: int = 1000,
        nb_inputs: int = 10,
        nearest_neighbours: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.loss_fn_vmf = VonMisesFisher3DLoss()
        self.loss_fn_cos = nn.CosineSimilarity()

        if model_name == "DynEdge":
            self.model = DynEdge(
                nb_inputs=nb_inputs,
                nb_neighbours=nearest_neighbours,
                global_pooling_schemes=["min", "max", "mean", "sum"],
                features_subset=slice(0, 4),  # NN search using xyzt
            )
        elif model_name == "GPS":
            self.model = GPS(channels=128, num_layers=7, dropout=0.5, heads=4)
        elif model_name == "GAT":
            self.model = GraphAttentionNetwork()
        elif model_name == "GravNet":
            self.model = GravNet()

        self.task = DirectionReconstructionWithKappa(
            hidden_size=self.model.nb_outputs,
            target_labels=["x", "y", "z"],
            loss_function=VonMisesFisher3DLoss(),
        )

    def forward(self, x):
        emb = self.model(x)
        out = self.task(emb)

        return out

    def xyz_to_angles(self, xyz):
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]
        r = torch.sqrt(x**2 + y**2 + z**2)

        zen = torch.arccos(z / r)
        azi = torch.arctan2(y, x)

        return torch.stack([azi, zen], dim=1)

    def angles_to_xyz(self, angles):
        azimuth, zenith = angles[:, 0], angles[:, 1]
        x = torch.cos(azimuth) * torch.sin(zenith)
        y = torch.sin(azimuth) * torch.sin(zenith)
        z = torch.cos(zenith)
        return torch.stack([x, y, z], dim=1)

    def training_step(self, batch, batch_idx):
        pred_xyzk = self.forward(batch)
        # pred_angles = self.xyz_to_angles(pred_xyzk)

        target_angles = batch.y.reshape(-1, 2)
        target_xyz = self.angles_to_xyz(target_angles)

        loss_vmf = self.loss_fn_vmf(pred_xyzk, target_xyz)
        loss_cos = 1 - self.loss_fn_cos(pred_xyzk[:, :3], target_xyz).mean()
        loss = loss_vmf + loss_cos

        self.log(
            "loss/train",
            loss,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
            batch_size=self.hparams.batch_size,
        )
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        pred_xyzk = self.forward(batch)
        pred_angles = self.xyz_to_angles(pred_xyzk)

        target_angles = batch.y.reshape(-1, 2)
        target_xyz = self.angles_to_xyz(target_angles)

        loss_vmf = self.loss_fn_vmf(pred_xyzk, target_xyz)
        loss_cos = 1 - self.loss_fn_cos(pred_xyzk[:, :3], target_xyz).mean()
        loss = loss_vmf + loss_cos

        metric = angular_dist_score(pred_angles, target_angles)

        self.log(
            "metric",
            metric,
            prog_bar=True,
            sync_dist=True,
            on_epoch=True,
            batch_size=self.hparams.batch_size,
        )
        self.log_dict(
            {
                "loss/valid": loss,
                "loss/valid_cos": loss_cos,
                "loss/valid_vmf": loss_vmf,
            },
            sync_dist=True,
            on_epoch=True,
            batch_size=self.hparams.batch_size,
        )

    def configure_optimizers(self):
        parameters = add_weight_decay(
            self,
            self.hparams.weight_decay,
            skip_list=["bias", "LayerNorm.bias"],  # , "LayerNorm.weight"],
        )

        opt = torch.optim.AdamW(
            parameters, lr=self.hparams.learning_rate, eps=self.hparams.eps
        )
        # opt = Lion(parameters, lr=self.hparams.learning_rate)

        sch = get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=int(self.hparams.warmup * self.hparams.T_max),
            num_training_steps=self.hparams.T_max,
            num_cycles=0.5,  # 1,
            last_epoch=-1,
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "interval": "step"},
        }

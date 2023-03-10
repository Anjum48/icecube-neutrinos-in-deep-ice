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

    def training_epoch_end(self, outputs):
        gc.collect()

    def validation_epoch_end(self, outputs):
        gc.collect()

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


class IceCubeContastiveModel(pl.LightningModule):
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
        self.loss_fn_dist = nn.CosineEmbeddingLoss()

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
        # self.norm = nn.BatchNorm1d(self.model.nb_outputs)

    def forward(self, x, return_emb=False):
        emb = self.model(x)
        # emb = self.norm(emb)
        azi_out = self.azimuth_task(emb)
        zen_out = self.zenith_task(emb)

        if return_emb:
            return azi_out, zen_out, emb
        else:
            return azi_out, zen_out

    def shuffle_batch(self, batch):
        bs = batch.num_graphs

        idx = torch.arange(bs)
        idx_neg = torch.randperm(bs // 2)
        idx_pos = idx[bs // 2 :]
        idx = torch.cat([idx_neg, idx_pos])

        target = torch.where(idx != torch.arange(bs), -1, 1).to(self.device)

        data_list = batch.index_select(idx)
        batch = Batch.from_data_list(data_list)

        return batch, target

    def training_step(self, batch, batch_idx):
        batch_a, batch_b = batch

        batch_b, target_contrastive = self.shuffle_batch(batch_b)
        target = batch_a.y.reshape(-1, 2)

        pred_azi, pred_zen, emb_a = self.forward(batch_a, True)
        _, _, emb_b = self.forward(batch_b, True)

        # weight = 1 - np.exp(-self.global_step / (self.hparams.T_max))
        loss_azi = self.loss_fn_azi(pred_azi, target)
        loss_zen = self.loss_fn_zen(pred_zen, target[:, -1].unsqueeze(-1))
        loss = loss_azi + loss_zen

        pred_angles = torch.stack([pred_azi[:, 0], pred_zen[:, 0]], dim=1)

        loss_cos = self.loss_fn_cos(pred_angles, target)

        if self.current_epoch > 0:
            loss += loss_cos

        loss += self.loss_fn_dist(emb_a, emb_b, target_contrastive)

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

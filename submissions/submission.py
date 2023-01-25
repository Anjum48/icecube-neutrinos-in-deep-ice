import getpass
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

KERNEL = False if getpass.getuser() == "anjum" else True
COMP_NAME = "icecube-neutrinos-in-deep-ice"

if not KERNEL:
    INPUT_PATH = Path(f"/mnt/storage_dimm2/kaggle_data/{COMP_NAME}")
    OUTPUT_PATH = Path(f"/mnt/storage_dimm2/kaggle_output/{COMP_NAME}")
    MODEL_CACHE = Path("/mnt/storage/model_cache/torch")
else:
    INPUT_PATH = Path(f"/kaggle/input/{COMP_NAME}")
    MODEL_CACHE = None

    # Install packages
    import subprocess

    whls = [
        "/kaggle/input/pytorchgeometric/torch_cluster-1.6.0-cp37-cp37m-linux_x86_64.whl",
        "/kaggle/input/pytorchgeometric/torch_scatter-2.1.0-cp37-cp37m-linux_x86_64.whl",
        "/kaggle/input/pytorchgeometric/torch_sparse-0.6.16-cp37-cp37m-linux_x86_64.whl",
        "/kaggle/input/pytorchgeometric/torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl",
        "/kaggle/input/pytorchgeometric/torch_geometric-2.2.0-py3-none-any.whl",
        "/kaggle/input/pytorchgeometric/ruamel.yaml-0.17.21-py3-none-any.whl",
    ]

    for w in whls:
        print("Installing", w)
        subprocess.call(["pip", "install", w, "--no-deps", "--upgrade"])

    import sys

    sys.path.append("/kaggle/input/graphnet/graphnet-main/src")

from graphnet.models.gnn import DynEdge
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import (
    AzimuthReconstructionWithKappa,
    ZenithReconstruction,
)
from graphnet.training.loss_functions import VonMisesFisher2DLoss
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

# import polars as pls

_dtype = {
    "batch_id": "int16",
    "event_id": "int64",
}


# models.py
class IceCubeModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "DynEdge",
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        warmup: float = 0.1,
        T_max: int = 1000,
        nb_inputs: int = 7,
        nearest_neighbours: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.loss_fn_azi = VonMisesFisher2DLoss()
        self.loss_fn_zen = nn.MSELoss()

        self.model = DynEdge(
            nb_inputs=nb_inputs,
            nb_neighbours=nearest_neighbours,
            global_pooling_schemes=["min", "max", "mean", "sum"],
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

        loss_azi = self.loss_fn_azi(pred_azi, target)
        loss_zen = self.loss_fn_zen(pred_zen, target[:, -1].unsqueeze(-1))
        loss = loss_azi + loss_zen

        self.log_dict({"loss/train_step": loss})
        return {"loss": loss}

    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack([x["loss"] for x in training_step_outputs]).mean()
        self.log("loss/train", avg_loss, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        pred_azi, pred_zen = self.forward(batch)

        target = batch.y.reshape(-1, 2)

        loss_azi = self.loss_fn_azi(pred_azi, target)
        loss_zen = self.loss_fn_zen(pred_zen, target[:, -1].unsqueeze(-1))
        loss = loss_azi + loss_zen

        pred_angles = torch.stack([pred_azi[:, 0], pred_zen[:, 0]], dim=1)
        metric = angular_dist_score(pred_angles, target)

        output = {
            "val_loss": loss,
            "metric": metric,
            "val_loss_azi": loss_azi,
            "val_loss_zen": loss_zen,
        }

        return output

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x["val_loss"] for x in outputs]).mean()
        loss_val_azi = torch.stack([x["val_loss_azi"] for x in outputs]).mean()
        loss_val_zen = torch.stack([x["val_loss_zen"] for x in outputs]).mean()
        metric = torch.stack([x["metric"] for x in outputs]).mean()

        self.log_dict(
            {"loss/valid": loss_val, "metric": metric},
            prog_bar=True,
            sync_dist=True,
        )
        self.log_dict(
            {"loss/valid_azi": loss_val_azi, "loss/valid_zen": loss_val_zen},
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


# utils.py
def add_weight_decay(
    model,
    weight_decay=1e-5,
    skip_list=("bias", "bn", "LayerNorm.bias", "LayerNorm.weight"),
):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


# losses.py
def angular_dist_score(y_pred, y_true):
    """
    calculate the MAE of the angular distance between two directions.
    The two vectors are first converted to cartesian unit vectors,
    and then their scalar product is computed, which is equal to
    the cosine of the angle between the two vectors. The inverse
    cosine (arccos) thereof is then the angle between the two itorchut vectors

    # https://www.kaggle.com/code/sohier/mean-angular-error

    Parameters:
    -----------

    y_pred : float (torch.Tensor)
        Prediction array of [N, 2], where the second dim is azimuth & zenith
    y_true : float (torch.Tensor)
        Ground truth array of [N, 2], where the second dim is azimuth & zenith

    Returns:
    --------

    dist : float (torch.Tensor)
        mean over the angular distance(s) in radian
    """

    az_true = y_true[:, 0]
    zen_true = y_true[:, 1]

    az_pred = y_pred[:, 0]
    zen_pred = y_pred[:, 1]

    # pre-compute all sine and cosine values
    sa1 = torch.sin(az_true)
    ca1 = torch.cos(az_true)
    sz1 = torch.sin(zen_true)
    cz1 = torch.cos(zen_true)

    sa2 = torch.sin(az_pred)
    ca2 = torch.cos(az_pred)
    sz2 = torch.sin(zen_pred)
    cz2 = torch.cos(zen_pred)

    # scalar product of the two cartesian vectors (x = sz*ca, y = sz*sa, z = cz)
    scalar_prod = sz1 * sz2 * (ca1 * ca2 + sa1 * sa2) + (cz1 * cz2)

    # scalar product of two unit vectors is always between -1 and 1, this is against nummerical instability
    # that might otherwise occure from the finite precision of the sine and cosine functions
    scalar_prod = torch.clamp(scalar_prod, -1, 1)

    # convert back to an angle (in radian)
    return torch.mean(torch.abs(torch.arccos(scalar_prod)))


# datasets.py
class IceCubeSubmissionDataset(Dataset):
    def __init__(
        self,
        batch_id,
        event_ids,
        sensor_df,
        mode="test",
        pulse_limit=300,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        super().__init__(transform, pre_transform, pre_filter)
        self.event_ids = event_ids
        self.batch_df = pd.read_parquet(INPUT_PATH / mode / f"batch_{batch_id}.parquet")
        self.sensor_df = sensor_df
        self.pulse_limit = pulse_limit

        self.batch_df["time"] = (self.batch_df["time"] - 1.0e04) / 3.0e4
        self.batch_df["charge"] = np.log10(self.batch_df["charge"]) / 3.0
        self.batch_df["auxiliary"] = self.batch_df["auxiliary"].astype(int) - 0.5

    def len(self):
        return len(self.event_ids)

    def get(self, idx):
        event_id = self.event_ids[idx]
        event = self.batch_df.loc[event_id]

        event = pd.merge(event, self.sensor_df, on="sensor_id")

        x = event[["x", "y", "z", "time", "charge", "qe", "auxiliary"]].values
        x = torch.tensor(x, dtype=torch.float32)
        data = Data(x=x, n_pulses=torch.tensor(x.shape[0], dtype=torch.int32))

        # Only use aux = False
        mask = data.x[:, -1] < 0
        data.x = data.x[mask]
        data.n_pulses = torch.tensor(data.x.shape[0], dtype=torch.int32)

        # Downsample the large events
        if data.n_pulses > self.pulse_limit:
            data.x = data.x[np.random.choice(data.n_pulses, self.pulse_limit)]
            data.n_pulses = torch.tensor(self.pulse_limit, dtype=torch.int32)

        return data


# class IceCubeSubmissionDatasetV2(Dataset):
#     def __init__(
#         self,
#         batch_id,
#         event_ids,
#         sensor_df,
#         mode="test",
#         pulse_limit=300,
#         transform=None,
#         pre_transform=None,
#         pre_filter=None,
#     ):
#         super().__init__(transform, pre_transform, pre_filter)
#         self.event_ids = event_ids
#         self.batch_df = pls.read_parquet(
#             INPUT_PATH / mode / f"batch_{batch_id}.parquet"
#         )
#         self.sensor_df = sensor_df
#         self.pulse_limit = pulse_limit

#         self.batch_df = self.batch_df.with_columns(
#             [
#                 (pls.col("time") - 1.0e04) / 3.0e4,
#                 pls.col("charge").log() / 3.0,
#                 pls.col("auxiliary").cast(int) - 0.5,
#             ]
#         )

#     def len(self):
#         return len(self.event_ids)

#     def get(self, idx):
#         event_id = self.event_ids[idx]

#         event = self.batch_df.filter(pls.col("event_id") == 24)
#         event = event.join(self.sensor_df, left_on="sensor_id", right_on="sensor_id")

#         x = event[["x", "y", "z", "time", "charge", "qe", "auxiliary"]].to_numpy()
#         x = torch.tensor(x, dtype=torch.float32)
#         data = Data(x=x, n_pulses=torch.tensor(x.shape[0], dtype=torch.int32))

#         # Only use aux = False
#         mask = data.x[:, -1] < 0
#         data.x = data.x[mask]
#         data.n_pulses = torch.tensor(data.x.shape[0], dtype=torch.int32)

#         # Downsample the large events
#         if data.n_pulses > self.pulse_limit:
#             data.x = data.x[np.random.choice(data.n_pulses, self.pulse_limit)]
#             data.n_pulses = torch.tensor(self.pulse_limit, dtype=torch.int32)

#         return data


# preprocessing.py
def prepare_sensors():
    sensors = pd.read_csv(INPUT_PATH / "sensor_geometry.csv")
    sensors["string"] = 0
    sensors["qe"] = 1

    for i in range(len(sensors) // 60):
        start, end = i * 60, (i * 60) + 60
        sensors.loc[start:end, "string"] = i

        # High Quantum Efficiency in the lower 50 DOMs - https://arxiv.org/pdf/2209.03042.pdf (Figure 1)
        if i in range(78, 86):
            start_veto, end_veto = i * 60, (i * 60) + 10
            start_core, end_core = end_veto + 1, (i * 60) + 60
            sensors.loc[start_core:end_core, "qe"] = 1.35

    # https://github.com/graphnet-team/graphnet/blob/b2bad25528652587ab0cdb7cf2335ee254cfa2db/src/graphnet/models/detector/icecube.py#L33-L41
    # Assume that "rde" (relative dom efficiency) is equivalent to QE
    sensors["x"] /= 500
    sensors["y"] /= 500
    sensors["z"] /= 500
    sensors["qe"] -= 1.25
    sensors["qe"] /= 0.25

    return sensors


def infer(model, dataset, batch_size=32, device="cuda"):
    model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=2)

    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred_azi, pred_zen = model(batch)
            pred_angles = torch.stack([pred_azi[:, 0], pred_zen[:, 0]], dim=1)
            predictions.append(pred_angles.cpu())

    return torch.cat(predictions, 0)


def make_predictions(dataset_paths, device="cuda", suffix="metric", mode="test"):
    mpaths = []
    for p in dataset_paths:
        mpaths.append(sorted(list(p.rglob(f"*{suffix}.ckpt"))))

    num_models = len([item for sublist in mpaths for item in sublist])
    print(f"{num_models} models found.")

    sensors = prepare_sensors()
    # sensors["sensor_id"] = sensors["sensor_id"].astype(np.int16)
    # sensors = pls.from_pandas(sensors)

    meta = pd.read_parquet(
        INPUT_PATH / f"{mode}_meta.parquet", columns=["batch_id", "event_id"]
    ).astype(_dtype)
    batch_ids = meta["batch_id"].unique()
    output = 0

    if mode == "train":
        batch_ids = batch_ids[:6]

    # for i, group in enumerate(mpaths):
    #     for j, p in enumerate(group):

    p = mpaths[0][0]
    model = IceCubeModel.load_from_checkpoint(p, strict=False)
    pre_transform = KNNGraphBuilder(nb_nearest_neighbours=8)

    batch_preds = []
    for b in batch_ids:
        event_ids = meta[meta["batch_id"] == b]["event_id"].tolist()
        dataset = IceCubeSubmissionDataset(
            b, event_ids, sensors, mode=mode, pre_transform=pre_transform
        )
        batch_preds.append(infer(model, dataset, device=device, batch_size=1024))
        print("Finished batch", b)

    output += torch.cat(batch_preds, 0)

    # After looping through folds
    output /= num_models

    event_id_labels = []
    for b in batch_ids:
        event_id_labels.extend(meta[meta["batch_id"] == b]["event_id"].tolist())

    sub = {
        "event_id": event_id_labels,
        "azimuth": output[:, 0],
        "zenith": output[:, 1],
    }

    sub = pd.DataFrame(sub)
    sub.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    pl.seed_everything(48, workers=True)

    model_folders = [
        "20230125-152628",
    ]

    if KERNEL:
        dataset_paths = [Path(f"../input/icecube-{f}") for f in model_folders]
    else:
        dataset_paths = [OUTPUT_PATH / f for f in model_folders]

    predictions = make_predictions(dataset_paths, mode="test")

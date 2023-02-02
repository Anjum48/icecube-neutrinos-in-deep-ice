import getpass
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from graphnet.models.gnn.gnn import GNN
from graphnet.models.utils import calculate_xyzt_homophily
from graphnet.utilities.config import save_model_config
from scipy.interpolate import interp1d
from sklearn.preprocessing import RobustScaler
from torch import LongTensor, Tensor
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from torch_geometric.nn import EdgeConv
from torch_geometric.nn.pool import knn_graph
from torch_geometric.typing import Adj
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_sum
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

KERNEL = False if getpass.getuser() == "anjum" else True
COMP_NAME = "icecube-neutrinos-in-deep-ice"

GLOBAL_POOLINGS = {
    "min": scatter_min,
    "max": scatter_max,
    "sum": scatter_sum,
    "mean": scatter_mean,
}

if not KERNEL:
    INPUT_PATH = Path(f"/mnt/storage_dimm2/kaggle_data/{COMP_NAME}")
    OUTPUT_PATH = Path(f"/mnt/storage_dimm2/kaggle_output/{COMP_NAME}")
    MODEL_CACHE = Path("/mnt/storage/model_cache/torch")
    TRANSPARENCY_PATH = INPUT_PATH / "ice_transparency.txt"
else:
    INPUT_PATH = Path(f"/kaggle/input/{COMP_NAME}")
    MODEL_CACHE = None
    TRANSPARENCY_PATH = INPUT_PATH / "ice_transparency.txt"

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

from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import (
    AzimuthReconstructionWithKappa,
    ZenithReconstruction,
)
from graphnet.training.loss_functions import VonMisesFisher2DLoss
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

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
        nb_inputs: int = 8,
        nearest_neighbours: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.loss_fn_azi = VonMisesFisher2DLoss()
        self.loss_fn_zen = nn.L1Loss()
        # self.loss_fn_cos = CosineLoss()

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

    def forward(self, x):
        emb = self.model(x)
        # emb = self.norm(emb)
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


# modules.py
class DynEdgeConv(EdgeConv, pl.LightningModule):
    """Dynamical edge convolution layer."""

    def __init__(
        self,
        nn: Callable,
        aggr: str = "max",
        nb_neighbors: int = 8,
        features_subset: Optional[Union[Sequence[int], slice]] = None,
        **kwargs: Any,
    ):
        """Construct `DynEdgeConv`.
        Args:
            nn: The MLP/torch.Module to be used within the `EdgeConv`.
            aggr: Aggregation method to be used with `EdgeConv`.
            nb_neighbors: Number of neighbours to be clustered after the
                `EdgeConv` operation.
            features_subset: Subset of features in `Data.x` that should be used
                when dynamically performing the new graph clustering after the
                `EdgeConv` operation. Defaults to all features.
            **kwargs: Additional features to be passed to `EdgeConv`.
        """
        # Check(s)
        if features_subset is None:
            features_subset = slice(None)  # Use all features
        assert isinstance(features_subset, (list, slice))

        # Base class constructor
        super().__init__(nn=nn, aggr=aggr, **kwargs)

        # Additional member variables
        self.nb_neighbors = nb_neighbors
        self.features_subset = features_subset

    def forward(
        self, x: Tensor, edge_index: Adj, batch: Optional[Tensor] = None
    ) -> Tensor:

        """Forward pass."""
        # Standard EdgeConv forward pass
        x = super().forward(x, edge_index)

        # Recompute adjacency
        edge_index = knn_graph(
            x=x[:, self.features_subset],
            k=self.nb_neighbors,
            batch=batch,
        ).to(self.device)

        return x, edge_index


class DynEdge(GNN):
    """DynEdge (dynamical edge convolutional) model."""

    @save_model_config
    def __init__(
        self,
        nb_inputs: int,
        *,
        nb_neighbours: int = 8,
        features_subset: Optional[Union[List[int], slice]] = None,
        dynedge_layer_sizes: Optional[List[Tuple[int, ...]]] = None,
        post_processing_layer_sizes: Optional[List[int]] = None,
        readout_layer_sizes: Optional[List[int]] = None,
        global_pooling_schemes: Optional[Union[str, List[str]]] = None,
        add_global_variables_after_pooling: bool = False,
    ):
        """Construct `DynEdge`.
        Args:
            nb_inputs: Number of input features on each node.
            nb_neighbours: Number of neighbours to used in the k-nearest
                neighbour clustering which is performed after each (dynamical)
                edge convolution.
            features_subset: The subset of latent features on each node that
                are used as metric dimensions when performing the k-nearest
                neighbours clustering. Defaults to [0,1,2].
            dynedge_layer_sizes: The layer sizes, or latent feature dimenions,
                used in the `DynEdgeConv` layer. Each entry in
                `dynedge_layer_sizes` corresponds to a single `DynEdgeConv`
                layer; the integers in the corresponding tuple corresponds to
                the layer sizes in the multi-layer perceptron (MLP) that is
                applied within each `DynEdgeConv` layer. That is, a list of
                size-two tuples means that all `DynEdgeConv` layers contain a
                two-layer MLP.
                Defaults to [(128, 256), (336, 256), (336, 256), (336, 256)].
            post_processing_layer_sizes: Hidden layer sizes in the MLP
                following the skip-concatenation of the outputs of each
                `DynEdgeConv` layer. Defaults to [336, 256].
            readout_layer_sizes: Hidden layer sizes in the MLP following the
                post-processing _and_ optional global pooling. As this is the
                last layer(s) in the model, the last layer in the read-out
                yields the output of the `DynEdge` model. Defaults to [128,].
            global_pooling_schemes: The list global pooling schemes to use.
                Options are: "min", "max", "mean", and "sum".
            add_global_variables_after_pooling: Whether to add global variables
                after global pooling. The alternative is to  added (distribute)
                them to the individual nodes before any convolutional
                operations.
        """
        # Latent feature subset for computing nearest neighbours in DynEdge.
        if features_subset is None:
            features_subset = slice(0, 3)

        # DynEdge layer sizes
        if dynedge_layer_sizes is None:
            dynedge_layer_sizes = [
                (
                    128,
                    256,
                ),
                (
                    336,
                    256,
                ),
                (
                    336,
                    256,
                ),
                (
                    336,
                    256,
                ),
            ]

        assert isinstance(dynedge_layer_sizes, list)
        assert len(dynedge_layer_sizes)
        assert all(isinstance(sizes, tuple) for sizes in dynedge_layer_sizes)
        assert all(len(sizes) > 0 for sizes in dynedge_layer_sizes)
        assert all(all(size > 0 for size in sizes) for sizes in dynedge_layer_sizes)

        self._dynedge_layer_sizes = dynedge_layer_sizes

        # Post-processing layer sizes
        if post_processing_layer_sizes is None:
            post_processing_layer_sizes = [
                336,
                256,
            ]

        assert isinstance(post_processing_layer_sizes, list)
        assert len(post_processing_layer_sizes)
        assert all(size > 0 for size in post_processing_layer_sizes)

        self._post_processing_layer_sizes = post_processing_layer_sizes

        # Read-out layer sizes
        if readout_layer_sizes is None:
            readout_layer_sizes = [
                128,
            ]

        assert isinstance(readout_layer_sizes, list)
        assert len(readout_layer_sizes)
        assert all(size > 0 for size in readout_layer_sizes)

        self._readout_layer_sizes = readout_layer_sizes

        # Global pooling scheme(s)
        if isinstance(global_pooling_schemes, str):
            global_pooling_schemes = [global_pooling_schemes]

        if isinstance(global_pooling_schemes, list):
            for pooling_scheme in global_pooling_schemes:
                assert (
                    pooling_scheme in GLOBAL_POOLINGS
                ), f"Global pooling scheme {pooling_scheme} not supported."
        else:
            assert global_pooling_schemes is None

        self._global_pooling_schemes = global_pooling_schemes

        if add_global_variables_after_pooling:
            assert self._global_pooling_schemes, (
                "No global pooling schemes were request, so cannot add global"
                " variables after pooling."
            )
        self._add_global_variables_after_pooling = add_global_variables_after_pooling

        # Base class constructor
        super().__init__(nb_inputs, self._readout_layer_sizes[-1])

        # Remaining member variables()
        self._activation = torch.nn.GELU()
        self._nb_inputs = nb_inputs
        self._nb_global_variables = 5 + nb_inputs
        self._nb_neighbours = nb_neighbours
        self._features_subset = features_subset

        self._construct_layers()

    def _construct_layers(self) -> None:
        """Construct layers (torch.nn.Modules)."""
        # Convolutional operations
        nb_input_features = self._nb_inputs
        if not self._add_global_variables_after_pooling:
            nb_input_features += self._nb_global_variables

        self._conv_layers = torch.nn.ModuleList()
        nb_latent_features = nb_input_features
        for sizes in self._dynedge_layer_sizes:
            layers = []
            layer_sizes = [nb_latent_features] + list(sizes)
            for ix, (nb_in, nb_out) in enumerate(
                zip(layer_sizes[:-1], layer_sizes[1:])
            ):
                if ix == 0:
                    nb_in *= 2
                layers.append(torch.nn.Linear(nb_in, nb_out))
                layers.append(nn.BatchNorm1d(nb_out))
                layers.append(self._activation)

            conv_layer = DynEdgeConv(
                torch.nn.Sequential(*layers),
                aggr="add",
                nb_neighbors=self._nb_neighbours,
                features_subset=self._features_subset,
            )
            self._conv_layers.append(conv_layer)

            nb_latent_features = nb_out

        # Post-processing operations
        nb_latent_features = (
            sum(sizes[-1] for sizes in self._dynedge_layer_sizes) + nb_input_features
        )

        post_processing_layers = []
        layer_sizes = [nb_latent_features] + list(self._post_processing_layer_sizes)
        for nb_in, nb_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            post_processing_layers.append(torch.nn.Linear(nb_in, nb_out))
            post_processing_layers.append(nn.BatchNorm1d(nb_out))
            post_processing_layers.append(self._activation)

        self._post_processing = torch.nn.Sequential(*post_processing_layers)

        # Read-out operations
        nb_poolings = (
            len(self._global_pooling_schemes) if self._global_pooling_schemes else 1
        )
        nb_latent_features = nb_out * nb_poolings
        if self._add_global_variables_after_pooling:
            nb_latent_features += self._nb_global_variables

        readout_layers = []
        layer_sizes = [nb_latent_features] + list(self._readout_layer_sizes)
        for nb_in, nb_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            readout_layers.append(torch.nn.Linear(nb_in, nb_out))
            readout_layers.append(nn.BatchNorm1d(nb_out))
            readout_layers.append(self._activation)

        self._readout = torch.nn.Sequential(*readout_layers)

    def _global_pooling(self, x: Tensor, batch: LongTensor) -> Tensor:
        """Perform global pooling."""
        assert self._global_pooling_schemes
        pooled = []
        for pooling_scheme in self._global_pooling_schemes:
            pooling_fn = GLOBAL_POOLINGS[pooling_scheme]
            pooled_x = pooling_fn(x, index=batch, dim=0)
            if isinstance(pooled_x, tuple) and len(pooled_x) == 2:
                # `scatter_{min,max}`, which return also an argument, vs.
                # `scatter_{mean,sum}`
                pooled_x, _ = pooled_x
            pooled.append(pooled_x)

        return torch.cat(pooled, dim=1)

    def _calculate_global_variables(
        self,
        x: Tensor,
        edge_index: LongTensor,
        batch: LongTensor,
        *additional_attributes: Tensor,
    ) -> Tensor:
        """Calculate global variables."""
        # Calculate homophily (scalar variables)
        h_x, h_y, h_z, h_t = calculate_xyzt_homophily(x, edge_index, batch)

        # Calculate mean features
        global_means = scatter_mean(x, batch, dim=0)

        # Add global variables
        global_variables = torch.cat(
            [
                global_means,
                h_x,
                h_y,
                h_z,
                h_t,
            ]
            + [attr.unsqueeze(dim=1) for attr in additional_attributes],
            dim=1,
        )

        return global_variables

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass."""
        # Convenience variables
        x, edge_index, batch = data.x, data.edge_index, data.batch

        global_variables = self._calculate_global_variables(
            x,
            edge_index,
            batch,
            torch.log10(data.n_pulses),
        )

        # Distribute global variables out to each node
        if not self._add_global_variables_after_pooling:
            distribute = (
                batch.unsqueeze(dim=1) == torch.unique(batch).unsqueeze(dim=0)
            ).type(torch.float)

            global_variables_distributed = torch.sum(
                distribute.unsqueeze(dim=2) * global_variables.unsqueeze(dim=0),
                dim=1,
            )

            x = torch.cat((x, global_variables_distributed), dim=1)

        # DynEdge-convolutions
        skip_connections = [x]
        for conv_layer in self._conv_layers:
            x, edge_index = conv_layer(x, edge_index, batch)
            skip_connections.append(x)

        # Skip-cat
        x = torch.cat(skip_connections, dim=1)

        # Post-processing
        x = self._post_processing(x)

        # (Optional) Global pooling
        if self._global_pooling_schemes:
            x = self._global_pooling(x, batch=batch)
            if self._add_global_variables_after_pooling:
                x = torch.cat(
                    [
                        x,
                        global_variables,
                    ],
                    dim=1,
                )

        # Read-out
        x = self._readout(x)

        return x


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
def ice_transparency(data_path, datum=1950):
    # Data from page 31 of https://arxiv.org/pdf/1301.5361.pdf
    # Datum is from footnote 8 of page 29
    df = pd.read_csv(data_path, delim_whitespace=True)
    df["z"] = df["depth"] - datum
    df["z_norm"] = df["z"] / 500
    df[["scattering_len_norm", "absorption_len_norm"]] = RobustScaler().fit_transform(
        df[["scattering_len", "absorption_len"]]
    )

    # These are both roughly equivalent after scaling
    f_scattering = interp1d(df["z_norm"], df["scattering_len_norm"])
    f_absorption = interp1d(df["z_norm"], df["absorption_len_norm"])
    return f_scattering, f_absorption


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
        self.f_scattering, self.f_absorption = ice_transparency(TRANSPARENCY_PATH)

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

        # Add ice transparency data
        z = data.x[:, 2].numpy()
        scattering = torch.tensor(self.f_scattering(z), dtype=torch.float32).view(-1, 1)
        # absorption = torch.tensor(self.f_absorption(z), dtype=torch.float32).view(-1, 1)

        data.x = torch.cat([data.x, scattering], dim=1)

        # Downsample the large events
        if data.n_pulses > self.pulse_limit:
            data.x = data.x[np.random.choice(data.n_pulses, self.pulse_limit)]
            data.n_pulses = torch.tensor(self.pulse_limit, dtype=torch.int32)

        return data


# preprocessing.py
def prepare_sensors():
    sensors = pd.read_csv(INPUT_PATH / "sensor_geometry.csv").astype(
        {
            "sensor_id": np.int16,
            "x": np.float32,
            "y": np.float32,
            "z": np.float32,
        }
    )
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


# tta.py
class TTAWrapper(nn.Module):
    def __init__(
        self,
        model,
        device,
        angles=[0, 180],
    ):
        super().__init__()
        self.model = model
        self.device = device
        self.angles = [a * np.pi / 180 for a in angles]
        self.rmats = [self.rotz(a) for a in self.angles]

    def rotz(self, theta):
        # Counter clockwise rotation
        return (
            torch.tensor(
                [
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1],
                ],
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .to(self.device)
        )

    def forward(self, data):
        azi_out_sin, azi_out_cos, zen_out = 0, 0, 0
        data_rot = data

        for a, mat in zip(self.angles, self.rmats):
            data_rot.x[:, :3] = torch.matmul(data.x[:, :3], mat)
            a_out, z_out = self.model(data_rot)

            # Remove rotation from the azimuth prediction
            azi_out_sin += torch.sin(a_out + a)
            azi_out_cos += torch.cos(a_out + a)
            zen_out += z_out

        # https://en.wikipedia.org/wiki/Circular_mean
        azi_out = torch.atan2(azi_out_sin, azi_out_cos)
        zen_out /= len(self.angles)

        return azi_out, zen_out


def infer(model, dataset, batch_size=32, device="cuda"):
    model.to(device)
    model.eval()
    model = TTAWrapper(model, device)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=2)

    predictions = []
    with torch.no_grad():
        for batch in tqdm(loader):
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

        if mode == "train" and b == 6:
            break

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
        "20230131-084311",
    ]

    if KERNEL:
        dataset_paths = [Path(f"../input/icecube-{f}") for f in model_folders]
    else:
        dataset_paths = [OUTPUT_PATH / f for f in model_folders]

    predictions = make_predictions(dataset_paths, mode="test")

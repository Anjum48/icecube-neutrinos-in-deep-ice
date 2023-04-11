import getpass
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import interp1d
from sklearn.preprocessing import RobustScaler
from torch import LongTensor, Tensor
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import pairwise_euclidean_distance
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

KERNEL = False if getpass.getuser() == "anjum" else True
COMP_NAME = "icecube-neutrinos-in-deep-ice"

if not KERNEL:
    INPUT_PATH = Path(f"/mnt/storage_dimm2/kaggle_data/{COMP_NAME}")
    OUTPUT_PATH = Path(f"/mnt/storage_dimm2/kaggle_output/{COMP_NAME}")
    MODEL_CACHE = Path("/mnt/storage/model_cache/torch")
    TRANSPARENCY_PATH = INPUT_PATH / "ice_transparency.txt"
else:
    INPUT_PATH = Path(f"/kaggle/input/{COMP_NAME}")
    MODEL_CACHE = None
    TRANSPARENCY_PATH = "/kaggle/input/icecubetransparency/ice_transparency.txt"

    # Install packages
    import subprocess

    whls = [
        "/kaggle/input/pytorchgeometric/torch_cluster-1.6.0-cp37-cp37m-linux_x86_64.whl",
        "/kaggle/input/pytorchgeometric/torch_scatter-2.1.0-cp37-cp37m-linux_x86_64.whl",
        "/kaggle/input/pytorchgeometric/torch_sparse-0.6.16-cp37-cp37m-linux_x86_64.whl",
        "/kaggle/input/pytorchgeometric/torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl",
        # "/kaggle/input/pytorchgeometric/torch_geometric-2.2.0-py3-none-any.whl",
        "/kaggle/input/pytorchgeometric/pyg_nightly-2.3.0.dev20230302-py3-none-any.whl",
        "/kaggle/input/pytorchgeometric/ruamel.yaml-0.17.21-py3-none-any.whl",
    ]

    for w in whls:
        print("Installing", w)
        subprocess.call(["pip", "install", w, "--no-deps", "--upgrade"])

    import sys

    sys.path.append("/kaggle/input/graphnet/graphnet-main/src")

import torch_geometric.transforms as T
from graphnet.models.gnn.gnn import GNN
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import DirectionReconstructionWithKappa
from graphnet.models.utils import calculate_xyzt_homophily
from graphnet.training.loss_functions import VonMisesFisher3DLoss
from graphnet.utilities.config import save_model_config
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import EdgeConv, GINEConv, GPSConv, aggr
from torch_geometric.nn.pool import knn_graph
from torch_geometric.typing import Adj
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_sum

GLOBAL_POOLINGS = {
    "min": scatter_min,
    "max": scatter_max,
    "sum": scatter_sum,
    "mean": scatter_mean,
}

MAX_PULSES = 256

C_VACUUM = 299792458  # m/s
N_ICE = 1.319  # At 400nm. From 2.3 of https://arxiv.org/pdf/2203.02303.pdf
C_ICE = (C_VACUUM / N_ICE) * 1e-9  # m/ns
T_DELAY = 0  # ns  DOM Error = 3 ns Section 2.4 of https://arxiv.org/pdf/2203.02303.pdf

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
        eps: float = 1e-8,
        warmup: float = 0.0,
        T_max: int = 1000,
        nb_inputs: int = 11,
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
            self.model = GPS(
                nb_inputs=nb_inputs,
                channels=128,
                num_layers=7,
                dropout=0.5,
                heads=4,
            )
        # elif model_name == "GAT":
        #     self.model = GraphAttentionNetwork(nb_inputs=nb_inputs)
        # elif model_name == "GravNet":
        #     self.model = GravNet(nb_inputs=nb_inputs)

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


class GPS(torch.nn.Module):
    def __init__(
        self,
        nb_inputs: int = 8,
        nb_outputs: int = 128,
        channels: int = 64,
        num_layers: int = 8,
        walk_length: int = 20,
        heads: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs

        self.node_emb = nn.Linear(self.nb_inputs, channels, bias=False)
        self.edge_emb = nn.Linear(2, channels, bias=False)  # Edge distance & delta_t
        self.pe_lin = nn.Linear(walk_length, channels, bias=False)

        self.pe_transform = T.AddRandomWalkPE(walk_length=walk_length, attr_name="pe")

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            net = nn.Sequential(
                nn.Linear(channels, channels, bias=False),
                nn.GELU(),
                nn.Linear(channels, channels, bias=False),
            )
            conv = GPSConv(
                channels,
                GINEConv(net),
                heads=heads,
                attn_dropout=dropout,
                act="gelu",
            )
            self.convs.append(conv)

        self.global_pooling = aggr.MultiAggregation(
            [
                "min",
                "max",
                "mean",
                "sum",
            ],
        )

        self.head = nn.Sequential(
            nn.GELU(),
            nn.Linear(channels * len(self.global_pooling.aggrs), self.nb_outputs),
        )

    def forward(self, data):
        data = self.pe_transform(data)
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.node_emb(x.squeeze(-1)) + self.pe_lin(data.pe)
        edge_attr = self.edge_emb(data.edge_attr)

        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)

        x = self.global_pooling(x, batch)
        return self.head(x)


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

    # From RobustScaler(). See ice_transparency.ipynb
    center = np.array([32.4, 111.8])
    scale = np.array([27.175, 89.325])
    features = ["scattering_len", "absorption_len"]

    df[features] = (df[features] - center) / scale

    # These are both roughly equivalent after scaling
    f_scattering = interp1d(df["z_norm"], df["scattering_len"])
    f_absorption = interp1d(df["z_norm"], df["absorption_len"])
    return f_scattering, f_absorption


def calculate_edge_attributes(d):
    dist = (d.x[d.edge_index[0], :3] - d.x[d.edge_index[1], :3]).sum(-1).pow(2)
    delta_t = (d.x[d.edge_index[0], 3] - d.x[d.edge_index[1], 3]).abs()
    d.edge_attr = torch.stack([dist, delta_t], dim=1)
    return d


class IceCubeSubmissionDataset(Dataset):
    def __init__(
        self,
        batch_id,
        event_ids,
        sensor_df,
        mode="test",
        pulse_limit=MAX_PULSES,
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

        self.origin = torch.tensor([46.29, -34.88]) / 500  # String 35

    def len(self):
        return len(self.event_ids)

    def get(self, idx):
        event_id = self.event_ids[idx]
        event = self.batch_df.loc[event_id]

        event = pd.merge(event, self.sensor_df, on="sensor_id")

        x = event[["x", "y", "z", "time", "charge", "qe", "auxiliary"]].values
        x = torch.tensor(x, dtype=torch.float32)
        data = Data(x=x, n_pulses=torch.tensor(x.shape[0], dtype=torch.int32))

        # Downsample the large events
        if data.n_pulses > self.pulse_limit:
            perm = torch.randperm(data.x.size(0))
            idx = perm[: self.pulse_limit]
            data.x = data.x[idx]

            data.n_pulses = torch.tensor(self.pulse_limit, dtype=torch.int32)

        # Add ice transparency data
        z = data.x[:, 2].numpy()
        scattering = torch.tensor(self.f_scattering(z), dtype=torch.float32).view(-1, 1)
        # absorption = torch.tensor(self.f_absorption(z), dtype=torch.float32).view(-1, 1)

        # Center on string 35
        data.x[:, :2] = data.x[:, :2] - self.origin

        # Data objects do not preserve order, so need to sort by time
        t, indices = torch.sort(data.x[:, 3])
        data.x = data.x[indices]

        # Calculate the scattering flag
        q_max_idx = torch.argmax(data.x[:, 4])
        xyz = data.x[:, :3]
        dists = (xyz - xyz[q_max_idx]).pow(2).sum(-1).pow(0.5) * 500
        delta_t = (torch.abs(t - t[q_max_idx])) * 3e4
        scatter_flag = dists / C_ICE >= delta_t + T_DELAY

        scatter_flag = scatter_flag.to(torch.float32).view(-1, 1) - 0.5

        # Rescale time
        data.x[:, 3] -= 0.06
        data.x[:, 3] *= 4

        # Distance from nearest previous pulse
        mat = pairwise_euclidean_distance(data.x[:, :3])
        mat = mat + torch.triu(torch.ones_like(mat)) * 1000

        dists, idx = mat.min(1)
        dists = (dists - 0.5) / 0.5
        t_delta = (t - t[idx] - 0.1) / 0.1

        prev = torch.stack([dists, t_delta], dim=-1)
        prev[0] = 0

        data.x = torch.cat([data.x, scattering, prev, scatter_flag], dim=1)
        # data.x = torch.cat([data.x, scattering, prev], dim=1)

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
        weight = 1 / len(self.angles)
        # data_rot = data

        x_data = torch.clone(data.x)

        for a, mat in zip(self.angles, self.rmats):
            data.x[:, :3] = torch.matmul(x_data[:, :3], mat)

            pred_xyzk = self.model(data)

            if isinstance(self.model, nn.DataParallel):
                pred_angles = self.model.module.xyz_to_angles(pred_xyzk)
            else:
                pred_angles = self.model.xyz_to_angles(pred_xyzk)

            a_out, z_out = pred_angles[:, 0], pred_angles[:, 1]

            # Remove rotation from the azimuth prediction by adding a
            a_out += a

            # https://en.wikipedia.org/wiki/Circular_mean
            azi_out_sin += weight * torch.sin(a_out)
            azi_out_cos += weight * torch.cos(a_out)
            zen_out += weight * z_out

        azi_out = torch.atan2(azi_out_sin, azi_out_cos)

        return azi_out, zen_out


def infer(model, dataset, batch_size=32, device="cuda"):
    model.to(device)
    model.eval()
    model = TTAWrapper(model, device, angles=[0, 60, 120, 180, 240, 300])
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=2)

    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred_azi, pred_zen = model(batch)
            pred_angles = torch.stack([pred_azi, pred_zen], dim=1)
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

    azi_out_sin, azi_out_cos, zen_out = 0, 0, 0

    if mode == "train":
        batch_ids = batch_ids[:6]

    for i, group in enumerate(mpaths):
        for j, p in enumerate(group):

            model = IceCubeModel.load_from_checkpoint(p, strict=False)

            pre_transform = T.Compose(
                [
                    KNNGraphBuilder(nb_nearest_neighbours=8, columns=[0, 1, 2, 3]),
                    # RadialGraphBuilder(radius=160 / 500, columns=[0, 1, 2, 3]),
                    calculate_edge_attributes,
                ]
            )

            batch_preds = []
            for b in batch_ids:
                event_ids = meta[meta["batch_id"] == b]["event_id"].tolist()
                dataset = IceCubeSubmissionDataset(
                    b, event_ids, sensors, mode=mode, pre_transform=pre_transform
                )
                batch_preds.append(
                    infer(model, dataset, device=device, batch_size=1024)
                )
                print("Finished batch", b, model.hparams.model_name)

                if mode == "train" and b == 6:
                    break

            model_output = torch.cat(batch_preds, 0)
            azi_out_sin += torch.sin(model_output[:, 0])
            azi_out_cos += torch.cos(model_output[:, 0])
            zen_out += model_output[:, 1]

    azi_out = torch.atan2(azi_out_sin, azi_out_cos)
    zen_out /= num_models
    output = torch.stack([azi_out, zen_out], dim=1)

    event_id_labels = []
    for b in batch_ids:
        event_id_labels.extend(meta[meta["batch_id"] == b]["event_id"].tolist())

    sub = {
        "event_id": event_id_labels,
        "azimuth": output[:, 0],
        "zenith": output[:, 1],
    }

    sub = pd.DataFrame(sub).sort_values(by="event_id")
    sub.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    pl.seed_everything(48, workers=True)

    model_folders = [
        # "20230223-160821",  # 0.99089 DynEdge (6 epoch). LB: 0.988
        # "20230227-083426",  # 0.99082 GPS (6 epoch). LB: ???
        # "20230303-224857",  # 0.98867 DynEdge (nearest pulse). LB: 0.988
        # "20230323-102724",
        "20230409-080525",  # DynEdge with Aug, 6x = 0.98701
        "20230405-063040",  # GPS with Aug. 2x = 0.98994, 6x = 0.98945
    ]

    if KERNEL:
        dataset_paths = [Path(f"../input/icecube-{f}") for f in model_folders]
    else:
        dataset_paths = [OUTPUT_PATH / f for f in model_folders]

    predictions = make_predictions(dataset_paths, mode="test")

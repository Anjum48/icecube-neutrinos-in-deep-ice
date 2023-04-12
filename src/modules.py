from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from graphnet.models.gnn.gnn import GNN
from graphnet.models.utils import calculate_xyzt_homophily
from graphnet.utilities.config import save_model_config
from pytorch_lightning import LightningModule
from torch import LongTensor, Tensor
from torch_geometric.data import Data
from torch_geometric.nn import (
    EdgeConv,
    GATConv,
    GATv2Conv,
    GINEConv,
    GPSConv,
    GravNetConv,
    aggr,
    SuperGATConv,
)
from torch_geometric.nn.pool import knn_graph, radius_graph
from torch_geometric.typing import Adj
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_sum

GLOBAL_POOLINGS = {
    "min": scatter_min,
    "max": scatter_max,
    "sum": scatter_sum,
    "mean": scatter_mean,
}


# https://stackoverflow.com/questions/71464582/how-to-use-pytorchs-nn-multiheadattention
class MHAttnLayer(nn.Module):
    def __init__(self, in_features, embed_dim=64, num_heads=2, dropout=0.0):
        super(MHAttnLayer, self).__init__()
        self.q_trfm = nn.Linear(in_features, embed_dim, bias=False)
        self.k_trfm = nn.Linear(in_features, embed_dim, bias=False)
        self.v_trfm = nn.Linear(in_features, embed_dim, bias=False)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout
        )

    def forward(self, x):
        q = self.q_trfm(x)  # .unsqueeze(0)
        k = self.k_trfm(x)  # .unsqueeze(0)
        v = self.v_trfm(x)  # .unsqueeze(0)
        attn_output, attn_output_weights = self.multihead_attn(q, k, v)
        return attn_output_weights @ x


class DynEdgeConv(EdgeConv, LightningModule):
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


# https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial3/Tutorial3.ipynb#scrollTo=YAiLrvGcz-6l
class GraphAttentionNetwork(torch.nn.Module):
    def __init__(
        self,
        nb_inputs=8,
        nb_outputs=128,
        num_layers: int = 10,
        heads=4,
        hidden=64,
        dropout=0.5,
    ):
        super(GraphAttentionNetwork, self).__init__()
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs
        self.hid = hidden
        self.heads = heads
        self.dropout = dropout

        self.convs = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                inputs = nb_inputs
                outputs = self.hid
            elif i == num_layers - 1:
                inputs = self.hid * self.heads
                outputs = self.hid
            else:
                inputs = self.hid * self.heads
                outputs = self.hid

            conv = GATConv(inputs, outputs, heads=self.heads, dropout=self.dropout)
            # conv = SuperGATConv(
            #     inputs,
            #     outputs,
            #     heads=self.heads,
            #     dropout=self.dropout,
            #     attention_type="MX",
            #     edge_sample_ratio=0.8,
            #     is_undirected=True,
            # )

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
            nn.Linear(
                self.hid * self.heads * len(self.global_pooling.aggrs),
                self.nb_outputs,
            ),
        )

    def forward(self, data):
        x, batch = data.x, data.batch

        # TODO: Skip connections
        # skip_connections = [x]
        for i, conv in enumerate(self.convs):

            if i > 0:
                x = F.gelu(x)

            x = F.dropout(x, p=self.dropout)
            x = conv(x, edge_index=data.edge_index, edge_attr=data.edge_attr)
            # x = conv(x, edge_index=data.edge_index)
            # skip_connections.append(x)

        # Skip-cat
        # x = torch.cat(skip_connections, dim=1)

        x = self.global_pooling(x, batch)
        x = self.head(x)
        return x


# # https://github.com/pyg-team/pytorch_geometric/blob/master/examples/super_gat.py
# class GraphAttentionNetwork(torch.nn.Module):
#     def __init__(self, nb_inputs=8, nb_outputs=128):
#         super().__init__()

#         self.nb_inputs = nb_inputs
#         self.nb_outputs = nb_outputs

#         self.conv1 = SuperGATConv(
#             nb_inputs,
#             8,
#             heads=8,
#             dropout=0.6,
#             attention_type="MX",
#             edge_sample_ratio=0.8,
#             is_undirected=True,
#         )
#         self.conv2 = SuperGATConv(
#             8 * 8,
#             nb_outputs,
#             heads=8,
#             concat=False,
#             dropout=0.6,
#             attention_type="MX",
#             edge_sample_ratio=0.8,
#             is_undirected=True,
#         )

#         self.global_pooling = aggr.MultiAggregation(
#             [
#                 "min",
#                 "max",
#                 "mean",
#                 "sum",
#                 "std",
#                 "median",
#             ],
#         )

#         self.head = nn.Sequential(
#             nn.GELU(),
#             nn.Linear(self.nb_outputs * 6, self.nb_outputs),
#         )

#     def forward(self, data):
#         x, edge_index, edge_attr, batch = (
#             data.x,
#             data.edge_index,
#             data.edge_attr,
#             data.batch,
#         )

#         x = F.dropout(x, p=0.6, training=self.training)
#         x = self.conv1(x, edge_index=edge_index, edge_attr=edge_attr)
#         x = F.elu(x)

#         x = F.dropout(x, p=0.6, training=self.training)
#         x = self.conv2(x, edge_index=edge_index, edge_attr=edge_attr)
#         x = F.elu(x)

#         x = self.global_pooling(x, batch)
#         x = self.head(x)
#         return x


# Recipe for a General, Powerful, Scalable Graph Transformer
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_gps.py
# https://arxiv.org/pdf/2205.12454.pdf
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
                # "std",
                # "median",
                # aggr.PowerMeanAggregation(learn=True),
                # aggr.LSTMAggregation(channels, channels),
                # aggr.SoftmaxAggregation(t=1, learn=True),
            ],
            # mode="attn",
            # mode_kwargs={
            #     "in_channels": channels,
            #     "out_channels": channels * 2,
            #     "num_heads": 4,
            # },
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


class GravNetBlock(nn.Module):
    def __init__(self, in_features, out_features, s, flr, k) -> None:
        super().__init__()
        self.conv = GravNetConv(in_features, out_features, s, flr, k)
        self.act = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, batch):
        x = self.conv(x, batch)
        # x = self.bn(x)
        x = self.act(x)
        # x = self.dropout(x)
        return x


class GravNet(torch.nn.Module):
    def __init__(
        self,
        nb_inputs: int = 8,
        nb_outputs: int = 128,
        hidden_channels: int = 256,
        num_blocks: int = 8,
        s: int = 4,
        flr: int = 64,
        k: int = 20,
    ):
        super(GravNet, self).__init__()
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs

        self.blocks = nn.ModuleList()

        total_hidden = nb_inputs
        step = 3

        for n in range(num_blocks):
            multiplier = n // step

            if n == 0:
                in_features = nb_inputs
                out_features = hidden_channels
            else:
                in_features = hidden_channels * 2**multiplier
                out_features = in_features

                if n % step == step - 1:
                    out_features *= 2

            total_hidden += out_features

            self.blocks.append(GravNetBlock(in_features, out_features, s, flr, k))

        self.global_pooling = aggr.MultiAggregation(
            [
                "min",
                "max",
                "mean",
                "sum",
                "std",
                "median",
            ],
        )

        self.mid = nn.Linear(total_hidden, hidden_channels * 2)
        self.head = nn.Linear(
            hidden_channels * 2 * len(self.global_pooling.aggrs), nb_outputs
        )

    def forward(self, data):
        x, batch = data.x, data.batch

        final_cat = [x]

        for block in self.blocks:
            x = block(x, batch)
            final_cat.append(x)

        x = torch.cat(final_cat, dim=1)
        x = self.mid(x)
        x = self.global_pooling(x, batch)
        return self.head(x)

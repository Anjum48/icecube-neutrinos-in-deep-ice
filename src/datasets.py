import copy
import gc

import numpy as np
import pandas as pd
import polars as pls
import pytorch_lightning as pl
import torch
import torch_geometric.transforms as T
from graphnet.models.graph_builders import KNNGraphBuilder, RadialGraphBuilder
from scipy.interpolate import interp1d
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torchmetrics.functional import pairwise_euclidean_distance

from src.config import INPUT_PATH, INPUT_PATH_ALT

MAX_PULSES = 512


def create_folds(data, n_splits=5, random_state=48):
    data["n_pulses"] = data["last_pulse_index"] - data["first_pulse_index"]
    data["bins"] = pd.qcut(np.log10(data["n_pulses"]), 10, labels=False)
    data["fold"] = -1

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for f, (t_, v_) in enumerate(kf.split(X=data, y=data["bins"])):
        data.loc[v_, "fold"] = f

    return data


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


def rotation_transform(data):
    theta = 2 * np.pi * np.random.rand()

    rotz = torch.tensor(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)

    data.x[:, :3] = torch.matmul(data.x[:, :3], rotz)

    azi_rot = data.y[0] - theta
    azi_rot = torch.where(azi_rot > 2 * np.pi, azi_rot - 2 * np.pi, azi_rot)
    azi_rot = torch.where(azi_rot < 0, azi_rot + 2 * np.pi, azi_rot)
    data.y[0] = azi_rot

    return data


def calculate_edge_attributes(d):
    dist = (d.x[d.edge_index[0], :3] - d.x[d.edge_index[1], :3]).sum(-1).pow(2)
    delta_t = (d.x[d.edge_index[0], 3] - d.x[d.edge_index[1], 3]).abs()
    d.edge_attr = torch.stack([dist, delta_t], dim=1)
    return d


class IceCubeDataset(Dataset):
    def __init__(
        self,
        df,
        pulse_limit=MAX_PULSES,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        super().__init__(transform, pre_transform, pre_filter)
        self.df = df
        self.pulse_limit = pulse_limit

    def len(self):
        return len(self.df)

    def get(self, idx):
        bid, eid = self.df[idx]

        # Batches 501-600 are on /mnt/storage due to the inode limitation
        # on /mnt/storage_dimm2
        if bid in range(501, 601):
            file_path = (
                INPUT_PATH_ALT / "train_events" / f"batch_{bid}" / f"event_{eid}.pt"
            )
        # The rest are on /mnt/storage_dimm2
        else:
            file_path = INPUT_PATH / "train_events" / f"batch_{bid}" / f"event_{eid}.pt"

        data = torch.load(file_path)

        # Drop the absorption data as its essentially the same as scattering
        data.x = data.x[:, :-1]

        # Rescale time
        data.x[:, 3] *= 10

        # Add cumulative features
        # t, indices = torch.sort(data.x[:, 3])  # Data objects no not preserve order
        # data.x = data.x[indices]
        # cum_charge = torch.cumsum(10 ** (3 * data.x[:, 4]), 0).view(-1, 1)
        # c_min, c_max = cum_charge.min(), cum_charge.max()
        # charge_norm = 2 * (cum_charge - c_min) / (c_max - c_min) - 1
        # t_norm = (2 * (t - t.min()) / (t.max() - t.min()) - 1).view(-1, 1)

        # Downsample the large events
        if data.n_pulses > self.pulse_limit:
            perm = torch.randperm(data.x.size(0))
            idx = perm[: self.pulse_limit]
            data.x = data.x[idx]

            data.n_pulses = torch.tensor(self.pulse_limit, dtype=torch.int32)

        # Distance from nearest previous pulse
        # Data objects no not preserve order, so need to sort by time
        t, indices = torch.sort(data.x[:, 3])
        data.x = data.x[indices]

        mat = pairwise_euclidean_distance(data.x[:, :3])
        mat = mat + torch.eye(data.n_pulses) * 1000
        prev = []

        for i in range(data.n_pulses):
            if i == 0:
                prev.append([0])
                # prev.append([0, 0])
            else:
                # prev.append([mat[: i + 1, i].min()])
                prev.append([(mat[: i + 1, i].min() - 0.5 / 0.5)])

                # masked_mat = mat[: i + 1, i]
                # idx = torch.argmin(masked_mat)
                # d = (masked_mat[idx] - 0.5) / 0.5
                # t_delta = (t[i] - t[idx] - 0.1) / 0.1
                # prev.append([d, t_delta])

        prev = torch.tensor(prev, dtype=torch.float32)
        data.x = torch.cat([data.x, prev], dim=1)

        return data


class IceCubeContrastiveDataset(Dataset):
    def __init__(
        self,
        df,
        pulse_limit=MAX_PULSES,
        drop_node_rate=0.2,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        super().__init__(transform, pre_transform, pre_filter)
        self.df = df  # DataFrame containing batch_id & event_id
        self.pulse_limit = pulse_limit
        self.drop_node_rate = drop_node_rate
        self.pre_transform = KNNGraphBuilder(nb_nearest_neighbours=8)
        self.f_scattering, self.f_absorption = ice_transparency(
            INPUT_PATH / "ice_transparency.txt"
        )

    def len(self):
        return len(self.df)

    def drop_nodes(self, data):
        prob = np.random.uniform(size=(data.n_pulses))
        mask = torch.tensor(prob > self.drop_node_rate)
        data.x = data.x[mask]
        data.n_pulses = torch.tensor(data.x.shape[0], dtype=torch.int32)
        return data

    def get(self, idx):
        bid, eid = self.df[idx, ["batch_id", "event_id"]]
        bid, eid = bid[0], eid[0]

        # Batches 501-600 are on /mnt/storage due to the inode limitation
        # on /mnt/storage_dimm2
        if bid in range(501, 601):
            file_path = (
                INPUT_PATH_ALT / "train_events" / f"batch_{bid}" / f"event_{eid}.pt"
            )
        # The rest are on /mnt/storage_dimm2
        else:
            file_path = INPUT_PATH / "train_events" / f"batch_{bid}" / f"event_{eid}.pt"

        data = torch.load(file_path)
        # data.batch_id = bid
        # data.event_id = eid

        # Add ice transparency data
        z = data.x[:, 2].numpy()
        scattering = torch.tensor(self.f_scattering(z), dtype=torch.float32).view(-1, 1)
        # absorption = torch.tensor(self.f_absorption(z), dtype=torch.float32).view(-1, 1)

        data.x = torch.cat([data.x, scattering], dim=1)

        # Create the augmenented graph
        data2 = copy.copy(data)
        data2 = self.drop_nodes(data2)
        data2 = rotation_transform(data2)

        # Downsample the large events
        if data.n_pulses > self.pulse_limit:
            data.x = data.x[np.random.choice(data.n_pulses, self.pulse_limit)]
            data.n_pulses = torch.tensor(self.pulse_limit, dtype=torch.int32)

        if data2.n_pulses > self.pulse_limit:
            data2.x = data2.x[np.random.choice(data2.n_pulses, self.pulse_limit)]
            data2.n_pulses = torch.tensor(self.pulse_limit, dtype=torch.int32)

        data = self.pre_transform(data)
        data2 = self.pre_transform(data2)

        return data, data2


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
        self.f_scattering, self.f_absorption = ice_transparency(
            INPUT_PATH / "ice_transparency.txt"
        )

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

        # Distance from nearest previous pulse
        # Data objects no not preserve order, so need to sort by time
        t, indices = torch.sort(data.x[:, 3])
        data.x = data.x[indices]

        mat = pairwise_euclidean_distance(data.x[:, :3])
        mat = mat + torch.eye(data.n_pulses) * 1000
        prev = []

        for i in range(data.n_pulses):
            if i == 0:
                prev.append([0])
                # prev.append([0, 0])
            else:
                prev.append([mat[: i + 1, i].min()])

                # masked_mat = mat[: i + 1, i]
                # idx = torch.argmin(masked_mat)
                # d = (masked_mat[idx] - 0.5) / 0.5
                # t_delta = (t[i] - t[idx] - 0.1) / 0.1
                # prev.append([d, t_delta])

        prev = torch.tensor(prev, dtype=torch.float32)
        data.x = torch.cat([data.x, scattering, prev], dim=1)
        return data


# Use polars
class IceCubeSubmissionDatasetV2(Dataset):
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
        self.batch_df = pls.read_parquet(
            INPUT_PATH / mode / f"batch_{batch_id}.parquet"
        )
        self.sensor_df = pls.from_pandas(sensor_df)
        self.pulse_limit = pulse_limit
        self.f_scattering, self.f_absorption = ice_transparency(
            INPUT_PATH / "ice_transparency.txt"
        )

        self.batch_df = self.batch_df.with_columns(
            [
                (pls.col("time") - 1.0e04) / 3.0e4,
                pls.col("charge").log() / 3.0,
                pls.col("auxiliary").cast(int) - 0.5,
            ]
        )

    def len(self):
        return len(self.event_ids)

    def get(self, idx):
        event_id = self.event_ids[idx]

        event = self.batch_df.filter(pls.col("event_id") == event_id)
        event = event.join(self.sensor_df, left_on="sensor_id", right_on="sensor_id")

        x = event[["x", "y", "z", "time", "charge", "qe", "auxiliary"]].to_numpy()

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


class IceCubeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_file: str = "folds_10.parquet",
        batch_size: int = 32,
        max_len: int = 2048,
        seed: int = 48,
        folds: int = 5,
        nearest_neighbours: int = 8,
        radius: float = 160,
        num_workers: int = 14,
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.max_len = max_len
        self.num_workers = num_workers
        self.train_file = train_file
        self.train_steps = 0
        self.pre_transform = T.Compose(
            [
                KNNGraphBuilder(
                    nb_nearest_neighbours=nearest_neighbours, columns=[0, 1, 2, 3]
                ),
                # RadialGraphBuilder(radius=radius / 500),
                calculate_edge_attributes,
            ]
        )

    def setup(self, stage=None, fold_n: int = 0):

        if stage == "fit":
            df = pls.read_parquet(
                INPUT_PATH / self.train_file, columns=["fold", "batch_id", "event_id"]
            )

            trn_df = (
                df.filter(pls.col("fold") != fold_n)
                .select(["batch_id", "event_id"])
                .to_numpy()
            )
            val_df = (
                df.filter(pls.col("fold") == fold_n)
                .select(["batch_id", "event_id"])
                .to_numpy()
            )

            self.clr_train = IceCubeDataset(trn_df, pre_transform=self.pre_transform)
            self.clr_valid = IceCubeDataset(val_df, pre_transform=self.pre_transform)

            self.train_steps = len(self.clr_train) / self.batch_size
            print(
                len(self.clr_train), "train and", len(self.clr_valid), "valid samples"
            )

            del df
            del trn_df
            del val_df

    def train_dataloader(self):
        return DataLoader(
            self.clr_train,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            # persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.clr_valid,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            # persistent_workers=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.clr_valid,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# if __name__ == "__main__":
#     import sys

#     COMP_NAME = "icecube-neutrinos-in-deep-ice"
#     sys.path.append(f"/home/anjum/kaggle/{COMP_NAME}/")

#     meta = pls.read_parquet(INPUT_PATH / "train_meta.parquet")
#     # ds = IceCubeDataset(meta)
#     ds = IceCubeDataset(meta, pre_transform=KNNGraphBuilder(nb_nearest_neighbours=8))
#     # dl = DataLoader(ds, batch_size=4)

#     # for d in dl:
#     #     print(d)
#     #     break

#     # d = ds[0]
#     # d = calculate_edge_attributes(d)
#     # print(d)
#     # print(d.edge_attr[0].mean(), d.edge_attr[0].min(), d.edge_attr[0].max())
#     # print(d.edge_attr[1].mean(), d.edge_attr[1].min(), d.edge_attr[1].max())

#     # for batch in dl:
#     #     print(batch)
#     #     a, b = batch
#     #     print(a.index_select([0]))
#     #     print(b.index_select([1]))
#     #     break

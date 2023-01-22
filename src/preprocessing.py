import sys

COMP_NAME = "icecube-neutrinos-in-deep-ice"
sys.path.append(f"/home/anjum/kaggle/{COMP_NAME}/")

import mlcrate as mlc
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from src.config import INPUT_PATH

_dtype = {
    "batch_id": "int16",
    "event_id": "int64",
    "first_pulse_index": "int32",
    "last_pulse_index": "int32",
    "azimuth": "float32",
    "zenith": "float32",
}

sensors = pd.read_csv(INPUT_PATH / "sensor_geometry_v2.csv", index_col="sensor_id")
meta = pd.read_parquet(INPUT_PATH / "train_meta.parquet").astype(_dtype)

sensors["x"] /= 500
sensors["y"] /= 500
sensors["z"] /= 500


def process_event(event, batch, event_id, targets):
    event = pd.merge(event, sensors, on="sensor_id")
    target = targets.loc[event_id, ["azimuth", "zenith"]]

    x = event[["x", "y", "z", "time", "charge", "qe", "auxiliary"]].values
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(target.values, dtype=torch.float32)
    data = Data(x=x, y=y, n_pulses=torch.tensor(x.shape[0], dtype=torch.int32))
    torch.save(data, INPUT_PATH / "train_events" / batch / f"event_{event_id}.pt")


def process_file(file_path):
    batch = file_path.stem
    batch_id = int(batch.split("_")[1])

    (INPUT_PATH / "train_events" / batch).mkdir(exist_ok=True)
    df = pd.read_parquet(INPUT_PATH / "train" / f"{batch}.parquet")

    df["time"] = (df["time"] - 1.0e04) / 3.0e4
    df["charge"] = np.log10(df["charge"]) / 3.0
    df["auxiliary"] = df["auxiliary"].astype(int)

    targets = meta.query(f"batch_id == {batch_id}").set_index("event_id", drop=True)

    for event_id in df.index.unique():
        process_event(df.loc[event_id], batch, event_id, targets)


if __name__ == "__main__":

    file_paths = [INPUT_PATH / "train" / f"batch_{i+1}.parquet" for i in range(660)]

    pool = mlc.SuperPool(16)
    pool.map(process_file, file_paths)

    # process_file(file_paths[0])

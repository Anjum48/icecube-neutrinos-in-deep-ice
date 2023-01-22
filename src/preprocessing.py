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

sensors = pd.read_csv(INPUT_PATH / "sensor_geometry_v2.csv")
meta = pd.read_parquet(INPUT_PATH / "train_meta.parquet")


def process_event(event, batch, event_id):
    batch_id = int(batch.split("_")[1])
    event = pd.merge(event, sensors, on="sensor_id")

    event["x"] /= 500
    event["y"] /= 500
    event["z"] /= 500
    event["time"] = (event["time"] - 1.0e04) / 3.0e4
    event["charge"] = np.log10(event["charge"]) / 3.0
    event["auxiliary"] = event["auxiliary"].astype(int)

    target = meta.query(f"batch_id == {batch_id} & event_id == {event_id}")
    y = torch.tensor(target[["azimuth", "zenith"]].values, dtype=torch.float32)

    x = event[["x", "y", "z", "time", "charge", "qe", "auxiliary"]].values
    x = torch.tensor(x, dtype=torch.float32)
    data = Data(x=x, y=y)
    data.n_pulses = torch.tensor(x.shape[0], dtype=torch.int32)
    torch.save(data, INPUT_PATH / "train_events" / batch / f"event_{event_id}.pt")


def process_file(file_path):
    batch = file_path.stem
    (INPUT_PATH / "train_events" / batch).mkdir(exist_ok=True)
    df = pd.read_parquet(INPUT_PATH / "train" / f"{batch}.parquet")

    for event_id in df.index.unique():
        process_event(df.loc[event_id], batch, event_id)


if __name__ == "__main__":

    file_paths = [INPUT_PATH / "train" / f"batch_{i+1}.parquet" for i in range(660)]

    pool = mlc.SuperPool(8)
    pool.map(process_file, file_paths)

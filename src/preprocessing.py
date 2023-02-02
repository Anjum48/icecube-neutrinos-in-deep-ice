import sys

COMP_NAME = "icecube-neutrinos-in-deep-ice"
sys.path.append(f"/home/anjum/kaggle/{COMP_NAME}/")

import contextlib

import joblib
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from torch_geometric.data import Data
from tqdm import tqdm

from src.config import INPUT_PATH


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


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


def make_meta_parts():
    (INPUT_PATH / "train_meta_parts").mkdir(exist_ok=True)
    meta = pd.read_parquet(INPUT_PATH / "train_meta.parquet")
    for i in tqdm(range(660)):
        part = meta[meta["batch_id"] == (i + 1)]
        part.to_parquet(INPUT_PATH / "train_meta_parts" / f"batch_{i+1}.parquet")


def process_event(event, batch, event_id, target):
    event = pd.merge(event, sensors, on="sensor_id")

    x = event[["x", "y", "z", "time", "charge", "qe", "auxiliary"]].values
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(target, dtype=torch.float32)
    data = Data(x=x, y=y, n_pulses=torch.tensor(x.shape[0], dtype=torch.int32))
    torch.save(data, INPUT_PATH / "train_events" / batch / f"event_{event_id}.pt")


def process_file(batch_id):
    batch = f"batch_{batch_id}"

    (INPUT_PATH / "train_events" / batch).mkdir(exist_ok=True)
    df = pd.read_parquet(INPUT_PATH / "train" / f"{batch}.parquet")

    df["time"] = (df["time"] - 1.0e04) / 3.0e4
    df["charge"] = np.log10(df["charge"]) / 3.0
    df["auxiliary"] = df["auxiliary"].astype(int) - 0.5

    targets = (
        pd.read_parquet(INPUT_PATH / "train_meta_parts" / f"{batch}.parquet")
        .set_index("event_id", drop=True)[["azimuth", "zenith"]]
        .to_dict(orient="index")
    )

    for event_id in df.index.unique():
        target = list(targets[event_id].values())
        process_event(df.loc[event_id], batch, event_id, target)


if __name__ == "__main__":
    sensors = prepare_sensors()
    batch_ids = [i + 1 for i in range(660)]

    with tqdm_joblib(tqdm(desc="Preprocessing", total=len(batch_ids))) as progress_bar:
        Parallel(n_jobs=32)(delayed(process_file)(b) for b in batch_ids)

    # process_file(batch_ids[0])

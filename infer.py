from argparse import ArgumentParser

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm.rich import tqdm

from src.config import OUTPUT_PATH
from src.datasets import IceCubeDataModule
from src.losses import angular_dist_score
from src.models import IceCubeModel
from src.tta import TTAWrapper


def infer(model, loader, device="cuda"):
    model.to(device)
    model.eval()

    # model = TTAWrapper(model, device, angles=[0])
    # model = TTAWrapper(model, device, angles=[0, 180])
    model = TTAWrapper(model, device, angles=[0, 60, 120, 180, 240, 300])

    # model = nn.DataParallel(model)

    predictions, target = [], []
    with torch.no_grad():
        for batch_n, batch in enumerate(tqdm(loader)):
            batch = batch.to(device)

            # pred_xyzk = model(batch)
            # pred_angles = model.xyz_to_angles(pred_xyzk)

            pred_azi, pred_zen = model(batch)
            pred_angles = torch.stack([pred_azi, pred_zen], dim=1)

            predictions.append(pred_angles.cpu())
            target.append(batch.y.reshape(-1, 2).cpu())

            if batch_n > 0.10 * len(loader):  # Roughly the size of test
                break

    return torch.cat(predictions, 0), torch.cat(target, 0)


def circular_mean(preds):
    azi_out_sin, azi_out_cos, zen_out = 0, 0, 0
    weight = 1 / len(preds)

    for p in preds:
        a_out, z_out = p[:, 0], p[:, 1]
        azi_out_sin += weight * torch.sin(a_out)
        azi_out_cos += weight * torch.cos(a_out)
        zen_out += weight * z_out

    # https://en.wikipedia.org/wiki/Circular_mean
    azi_out = torch.atan2(azi_out_sin, azi_out_cos)

    return torch.stack([azi_out, zen_out], dim=1)


def make_predictions(folders, suffix="metric", device="cuda"):

    pl.seed_everything(48, workers=True)

    mpaths = []
    for f in folders:
        mpaths.extend((OUTPUT_PATH / f).rglob(f"*{suffix}.ckpt"))
    print(f"{len(mpaths)} models found.")

    final_preds = []

    for i, p in enumerate(mpaths):
        cfg = OmegaConf.load(p.parent / ".hydra/config.yaml")
        dm = IceCubeDataModule(**cfg.model)
        dm.setup("predict", fold_n=0)
        loader = dm.predict_dataloader()

        model = IceCubeModel.load_from_checkpoint(p, strict=False)

        preds, target = infer(model, loader, device)
        mae = angular_dist_score(preds, target)
        final_preds.append(preds)

        print(f"Model {i} MAE: {mae:0.5f}")

        # Save preds
        oof = pd.DataFrame(
            {
                "azimuth": preds[:, 0],
                "zenith": preds[:, 1],
                "azimuth_gt": target[:, 0],
                "zenith_gt": target[:, 1],
            }
        )
        oof.to_parquet(p.parent / f"oofs.parquet")

    final_preds = circular_mean(final_preds)
    mae = angular_dist_score(final_preds, target)
    print(f"Ensemble MAE: {mae:0.5f}")


if __name__ == "__main__":

    # default_checkpoint = "20230130-085318"

    parser = ArgumentParser()

    # parser.add_argument(
    #     "--timestamp",
    #     action="store",
    #     dest="timestamp",
    #     help="Timestamp for versioning",
    #     default=default_checkpoint,
    #     type=str,
    # )

    parser.add_argument(
        "--gpu",
        action="store",
        dest="gpu",
        help="GPU index to use",
        default="0",
        type=str,
    )

    folders = [
        # "20230223-160821",  # 0.99089 DynEdge (6 epoch). LB: 0.988
        # "20230227-083426",  # 0.99082 GPS (6 epoch). LB:
        # "20230303-224857",  # 0.98867 DynEdge (nearest pulse). LB: 0.988
        # "20230315-112434",  # 0.99068
        # "20230313-213901",  # 0.98947
        # "20230319-112145",
        # "20230323-102724",
        # "20230402-083325",
        "20230409-080525",  # DynEdge with Aug, 6x = 0.98701
        "20230405-063040",  # GPS with Aug. 2x = 0.98994, 6x = 0.98940
        "20230411-220932",  # GravNet
    ]
    # Ensemble: 0.98513

    args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    predictions = make_predictions(folders, device=f"cuda:{str(args.gpu)}")

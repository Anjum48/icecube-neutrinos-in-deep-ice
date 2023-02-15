import os
from argparse import ArgumentParser

import torch
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

    model = TTAWrapper(model, device)

    predictions, target = [], []
    with torch.no_grad():
        for batch_n, batch in enumerate(tqdm(loader)):
            batch = batch.to(device)
            pred_azi, pred_zen = model(batch)
            pred_angles = torch.stack([pred_azi, pred_zen], dim=1)
            predictions.append(pred_angles.cpu())
            target.append(batch.y.reshape(-1, 2).cpu())

            if batch_n > 0.05 * len(loader):
                break

    return torch.cat(predictions, 0), torch.cat(target, 0)


def circular_mean(preds):
    azi_out_sin, azi_out_cos, zen_out = 0, 0, 0

    for p in preds:
        a_out, z_out = p[:, 0], p[:, 1]
        azi_out_sin += torch.sin(a_out)
        azi_out_cos += torch.cos(a_out)
        zen_out += z_out

    # https://en.wikipedia.org/wiki/Circular_mean
    azi_out = torch.atan2(azi_out_sin, azi_out_cos)
    zen_out /= len(preds)

    return torch.stack([azi_out, zen_out], dim=1)


def make_predictions(folders, suffix="metric", device="cuda"):

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
        # "20230207-203834",  # 0.99915
        # "20230210-081543",  # 0.99981
        "20230214-103416",
    ]
    # 0.99458

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    predictions = make_predictions(folders, device="cuda")

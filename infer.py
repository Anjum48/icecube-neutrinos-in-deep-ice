import os
from argparse import ArgumentParser

import torch
from omegaconf import OmegaConf
from tqdm.rich import tqdm

from src.config import OUTPUT_PATH
from src.datasets import IceCubeDataModule
from src.losses import angular_dist_score
from src.models import IceCubeModel


def infer(model, loader, device="cuda"):
    model.to(device)
    model.eval()

    predictions, target = [], []
    with torch.no_grad():
        for batch_n, batch in enumerate(tqdm(loader)):
            batch = batch.to(device)
            pred_azi, pred_zen = model(batch)
            pred_angles = torch.stack([pred_azi[:, 0], pred_zen[:, 0]], dim=1)
            predictions.append(pred_angles.cpu())
            target.append(batch.y.reshape(-1, 2).cpu())

            if batch_n > 0.05 * len(loader):
                break

    return torch.cat(predictions, 0), torch.cat(target, 0)


def make_predictions(folder_name, suffix="metric", device="cuda"):

    mpaths = list((OUTPUT_PATH / folder_name).rglob(f"*{suffix}.ckpt"))
    print(f"{len(mpaths)} models found.")

    for fold, p in enumerate(mpaths):
        cfg = OmegaConf.load(p.parent / ".hydra/config.yaml")
        dm = IceCubeDataModule(**cfg.model)
        dm.setup("predict", fold_n=fold)
        loader = dm.predict_dataloader()
        dataset = loader.dataset

        model = IceCubeModel.load_from_checkpoint(p, strict=False)

        preds, target = infer(model, loader, device)
        mae = angular_dist_score(preds, target)

        print(f"Fold {fold} MAE: {mae:0.5f}")


if __name__ == "__main__":

    default_checkpoint = "20230130-085318"

    parser = ArgumentParser()

    parser.add_argument(
        "--timestamp",
        action="store",
        dest="timestamp",
        help="Timestamp for versioning",
        default=default_checkpoint,
        type=str,
    )

    parser.add_argument(
        "--gpu",
        action="store",
        dest="gpu",
        help="GPU index to use",
        default="1",
        type=str,
    )

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    predictions = make_predictions(args.timestamp, device="cuda")

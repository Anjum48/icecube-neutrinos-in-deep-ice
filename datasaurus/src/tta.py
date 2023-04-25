import numpy as np
import torch
import torch.nn as nn


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

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
        data_rot = data

        for a, mat in zip(self.angles, self.rmats):
            data_rot.x[:, :3] = torch.matmul(data.x[:, :3], mat)

            pred_xyzk = self.model(data_rot)
            pred_angles = self.model.xyz_to_angles(pred_xyzk)
            a_out, z_out = pred_angles[:, 0], pred_angles[:, 1]

            # Remove rotation from the azimuth prediction
            azi_out_sin += torch.sin(a_out + a)
            azi_out_cos += torch.cos(a_out + a)
            zen_out += z_out

        # https://en.wikipedia.org/wiki/Circular_mean
        azi_out = torch.atan2(azi_out_sin, azi_out_cos)
        zen_out /= len(self.angles)

        return azi_out, zen_out

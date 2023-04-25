import numpy as np
import torch
import torch.nn as nn


def angular_dist_score_numpy(az_true, zen_true, az_pred, zen_pred):
    """
    calculate the MAE of the angular distance between two directions.
    The two vectors are first converted to cartesian unit vectors,
    and then their scalar product is computed, which is equal to
    the cosine of the angle between the two vectors. The inverse
    cosine (arccos) thereof is then the angle between the two input vectors

    Parameters:
    -----------

    az_true : float (or array thereof)
        true azimuth value(s) in radian
    zen_true : float (or array thereof)
        true zenith value(s) in radian
    az_pred : float (or array thereof)
        predicted azimuth value(s) in radian
    zen_pred : float (or array thereof)
        predicted zenith value(s) in radian

    Returns:
    --------

    dist : float
        mean over the angular distance(s) in radian
    """

    if not (
        np.all(np.isfinite(az_true))
        and np.all(np.isfinite(zen_true))
        and np.all(np.isfinite(az_pred))
        and np.all(np.isfinite(zen_pred))
    ):
        raise ValueError("All arguments must be finite")

    # pre-compute all sine and cosine values
    sa1 = np.sin(az_true)
    ca1 = np.cos(az_true)
    sz1 = np.sin(zen_true)
    cz1 = np.cos(zen_true)

    sa2 = np.sin(az_pred)
    ca2 = np.cos(az_pred)
    sz2 = np.sin(zen_pred)
    cz2 = np.cos(zen_pred)

    # scalar product of the two cartesian vectors (x = sz*ca, y = sz*sa, z = cz)
    scalar_prod = sz1 * sz2 * (ca1 * ca2 + sa1 * sa2) + (cz1 * cz2)

    # scalar product of two unit vectors is always between -1 and 1, this is against nummerical instability
    # that might otherwise occure from the finite precision of the sine and cosine functions
    scalar_prod = np.clip(scalar_prod, -1, 1)

    # convert back to an angle (in radian)
    return np.average(np.abs(np.arccos(scalar_prod)))


def angular_dist_score(y_pred, y_true):
    """
    calculate the MAE of the angular distance between two directions.
    The two vectors are first converted to cartesian unit vectors,
    and then their scalar product is computed, which is equal to
    the cosine of the angle between the two vectors. The inverse
    cosine (arccos) thereof is then the angle between the two input vectors

    # https://www.kaggle.com/code/sohier/mean-angular-error

    Parameters:
    -----------

    y_pred : float (torch.Tensor)
        Prediction array of [N, 2], where the second dim is azimuth & zenith
    y_true : float (torch.Tensor)
        Ground truth array of [N, 2], where the second dim is azimuth & zenith

    Returns:
    --------

    dist : float (torch.Tensor)
        mean over the angular distance(s) in radian
    """

    az_true = y_true[:, 0]
    zen_true = y_true[:, 1]

    az_pred = y_pred[:, 0]
    zen_pred = y_pred[:, 1]

    # pre-compute all sine and cosine values
    sa1 = torch.sin(az_true)
    ca1 = torch.cos(az_true)
    sz1 = torch.sin(zen_true)
    cz1 = torch.cos(zen_true)

    sa2 = torch.sin(az_pred)
    ca2 = torch.cos(az_pred)
    sz2 = torch.sin(zen_pred)
    cz2 = torch.cos(zen_pred)

    # scalar product of the two cartesian vectors (x = sz*ca, y = sz*sa, z = cz)
    scalar_prod = sz1 * sz2 * (ca1 * ca2 + sa1 * sa2) + (cz1 * cz2)

    # scalar product of two unit vectors is always between -1 and 1, this is against nummerical instability
    # that might otherwise occure from the finite precision of the sine and cosine functions
    scalar_prod = torch.clamp(scalar_prod, -1, 1)

    # convert back to an angle (in radian)
    return torch.mean(torch.abs(torch.arccos(scalar_prod)))


# https://stats.stackexchange.com/questions/218407/encoding-angle-data-for-neural-network
def sincos_encoded_loss(y_pred, y_true):
    zen_pred = np.pi * torch.sigmoid(y_pred[:, 2])
    az_pred_encoded = torch.tanh(y_pred[:, :2])
    az_true_encoded = torch.stack(
        [torch.sin(y_true[:, 0]), torch.cos(y_true[:, 0])],
        dim=1,
    )

    azimuth_loss = nn.functional.l1_loss(az_pred_encoded, az_true_encoded)
    zenith_loss = nn.functional.l1_loss(
        zen_pred.reshape(-1, 1), y_true[:, -1].reshape(-1, 1)
    )

    return azimuth_loss + zenith_loss


class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()
        self.cos = nn.CosineSimilarity()

    def forward(self, y_pred, y_true):
        xp = torch.cos(y_pred[:, 0]) * torch.sin(y_pred[:, 1])
        yp = torch.sin(y_pred[:, 0]) * torch.sin(y_pred[:, 1])
        zp = torch.cos(y_pred[:, 1])

        xt = torch.cos(y_true[:, 0]) * torch.sin(y_true[:, 1])
        yt = torch.sin(y_true[:, 0]) * torch.sin(y_true[:, 1])
        zt = torch.cos(y_true[:, 1])

        y_pred_cart = torch.stack([xp, yp, zp], dim=1)
        y_true_cart = torch.stack([xt, yt, zt], dim=1)

        cos_sim = self.cos(y_pred_cart, y_true_cart).mean()
        return 1 - cos_sim


# y_pred = np.random.normal(size=(5, 2))
# y_true = np.random.normal(size=(5, 2))

# score_numpy = angular_dist_score(y_true[:, 0], y_true[:, 1], y_pred[:, 0], y_pred[:, 1])
# print(score_numpy)

# y_pred_t = torch.tensor(y_pred, dtype=torch.float32)
# y_true_t = torch.tensor(y_true, dtype=torch.float32)

# score_torch = angular_dist_loss(y_pred_t, y_true_t)
# print(score_torch)

# shape = (5, 2)
# mu, sigma = torch.zeros(shape), torch.ones(shape)
# y_pred = torch.normal(mu, sigma)
# y_true = torch.normal(mu, sigma)

# score_torch = sincos_encoded_loss(y_pred, y_true)
# print(score_torch)

# cos = CosineLoss()
# print(cos(y_pred, y_true))
# print(cos(y_true, y_true))

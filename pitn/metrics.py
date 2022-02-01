# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
import monai
import skimage
import einops


@torch.no_grad()
def _minmax_scale_by_y(x, y, feature_range) -> tuple:

    spatial_dims = tuple(y.shape[2:])
    y_min = y.min(dim=spatial_dims)
    y_max = y.max(dim=spatial_dims)
    f_min = feature_range[0]
    f_max = feature_range[1]
    scale = (f_max - f_min) / (y_max - y_min)

    x_scaled = scale * x + f_min - y_min * scale
    y_scaled = scale * y + f_min - y_min * scale

    return x_scaled, y_scaled


@torch.no_grad()
def range_scaled_rmse(
    x: torch.Tensor,
    y: torch.Tensor,
    feature_range: tuple = (0.0, 1.0),
    range_based_on: str = "y",
    **mse_kwargs,
) -> torch.Tensor:
    if range_based_on.casefold() == "x":
        x, y = y, x

    x_scaled, y_scaled = _minmax_scale_by_y(x, y, feature_range)
    return torch.sqrt(F.mse_loss(x_scaled, y_scaled, **mse_kwargs))


@torch.no_grad()
def range_scaled_psnr(
    x: torch.Tensor,
    y: torch.Tensor,
    feature_range: tuple = (0.0, 1.0),
    range_based_on: str = "y",
    **psnr_kwargs,
) -> torch.Tensor:

    if range_based_on.casefold() == "x":
        x, y = y, x

    x_scaled, y_scaled = _minmax_scale_by_y(x, y, feature_range)
    data_range = abs(feature_range[1] - feature_range[0])
    psnr_metric = monai.metrics.PSNRMetric(max_val=data_range, **psnr_kwargs)

    return psnr_metric(x_scaled, y_scaled)


@torch.no_grad()
def range_scaled_ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    feature_range: tuple = (0.0, 1.0),
    range_based_on: str = "y",
    reduction="mean",
    **ssim_kwargs,
) -> torch.Tensor:

    if reduction is None:
        reduction = "none"
    elif not isinstance(reduction, str):
        reduction = str(reduction)

    if range_based_on.casefold() == "x":
        x, y = y, x

    x_scaled, y_scaled = _minmax_scale_by_y(x, y, feature_range)
    x_scaled = x_scaled.detach().cpu().numpy()
    y_scaled = y_scaled.detach().cpu().numpy()
    # Move channel dimensions to a channel-last format.
    x_scaled = einops.rearrange(x_scaled, "b c ... -> b ... c")
    y_scaled = einops.rearrange(y_scaled, "b c ... -> b ... c")

    data_range = abs(feature_range[1] - feature_range[0])

    batch_scores = list()
    batch_size = y.shape[0]
    for i in range(batch_size):
        score = skimage.metrics.structural_similarity(
            x_scaled[i],
            y_scaled[i],
            multichannel=True,
            data_range=data_range,
            **ssim_kwargs,
        )
        batch_scores.append(score)

    scores = torch.from_numpy(np.stack(batch_scores)).to(y)

    if reduction.casefold() == "mean":
        scores = torch.mean(scores)
    elif reduction.casefold() == "sum":
        scores = torch.sum(scores)
    elif reduction.casefold() == "none":
        scores = scores
    else:
        raise ValueError(f"ERROR: Invalid reduction {reduction}")

    return scores


# self.ssim_metric = functools.partial(
#     skimage.metrics.structural_similarity,
#     win_size=7,
#     K1=0.01,
#     K2=0.03,
#     use_sample_covariance=True,
#     data_range=psnr_max,
#     multichannel=True,
# )

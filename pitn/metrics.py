# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
import skimage
import einops

from pitn._lazy_loader import LazyLoader

# Make pytorch_msssim an optional import.
pytorch_msssim = LazyLoader("pytorch_msssim", globals(), "pytorch_msssim")


@torch.no_grad()
def _minmax_scale_by_y(x: torch.Tensor, y: torch.Tensor, feature_range) -> tuple:
    y = y.contiguous()
    batch = y.shape[0]
    channel = y.shape[1]
    n_spatial_dims = y.ndim - 2
    flat_spatial_dims = (1,) * n_spatial_dims
    y_min = y.view(batch, channel, -1).min(-1).values
    y_min = y_min.view(batch, channel, *flat_spatial_dims)
    y_max = y.view(batch, channel, -1).max(-1).values
    y_max = y_max.view(batch, channel, *flat_spatial_dims)
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


# @torch.no_grad()
# def minmax_norm_rmse(
#     x: torch.Tensor, y: torch.Tensor, range_based_on: str = "y", **mse_kwargs
# ) -> torch.Tensor:

#     if range_based_on.casefold() == "x":
#         x, y = y, x

#     rmse = torch.sqrt(F.mse_loss(x, y, **mse_kwargs))


#     data_range = y.max() - y.min()
#     nrmse = rmse / data_range

#     return nrmse


@torch.no_grad()
def range_scaled_psnr(
    x: torch.Tensor,
    y: torch.Tensor,
    feature_range: tuple = (0.0, 1.0),
    range_based_on: str = "y",
    **mse_kwargs,
) -> torch.Tensor:

    if range_based_on.casefold() == "x":
        x, y = y, x

    x_scaled, y_scaled = _minmax_scale_by_y(x, y, feature_range)
    data_range = abs(feature_range[1] - feature_range[0])
    psnr = 20 * torch.log10(
        torch.as_tensor(data_range).to(y_scaled)
    ) - 10 * torch.log10(F.mse_loss(x_scaled, y_scaled, **mse_kwargs))

    return psnr


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


@torch.no_grad()
def range_scaled_ms_ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    feature_range: tuple = (0.0, 1.0),
    range_based_on: str = "y",
    reduction: str = "mean",
    **ms_ssim_kwargs,
) -> torch.Tensor:

    if range_based_on.casefold() == "x":
        x, y = y, x
    x_scaled, y_scaled = _minmax_scale_by_y(x, y, feature_range)
    data_range = abs(feature_range[1] - feature_range[0])

    scores = pytorch_msssim.ms_ssim(
        x_scaled, y_scaled, data_range=data_range, size_average=False, **ms_ssim_kwargs
    )

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

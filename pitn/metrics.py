# -*- coding: utf-8 -*-
import itertools

import numpy as np
import torch
import torch.nn.functional as F
import skimage
import einops
import monai

from pitn._lazy_loader import LazyLoader
import pitn

# Make pytorch_msssim an optional import.
pytorch_msssim = LazyLoader("pytorch_msssim", globals(), "pytorch_msssim")


@torch.no_grad()
def psnr_batch_channel_regularized(
    x, y, range_min: torch.Tensor, range_max: torch.Tensor
):

    # Calculate the per-batch-per-channel range.
    range_min = range_min.to(y)
    range_max = range_max.to(y)
    y_range = range_max - range_min

    # Use a small epsilon to avoid 0s in the MSE.
    epsilon = 1e-10
    # Calculate squared error then regularize by the per-batch-per-channel range.
    regular_mse = (y_range ** 2) / torch.clamp_min(
        F.mse_loss(x, y, reduction="none"), epsilon
    )
    # Calculate the mean over channels and spatial dims.
    regular_mse = einops.reduce(regular_mse, "b ... -> b", "mean")
    # Find PSNR for each batch.
    psnr = 10 * torch.log10(regular_mse)
    # Mean of PSNR values for the whole batch.
    psnr = psnr.mean()

    return psnr


def _bc_y_range(y):

    # Calculate the per-batch-per-channel range.
    n_spatial_dims = len(y.shape[2:])
    reduce_str = "b c ... -> b c " + " ".join(itertools.repeat("1", n_spatial_dims))
    y_min = einops.reduce(y, reduce_str, "min")
    y_max = einops.reduce(y, reduce_str, "max")
    return y_max - y_min


@torch.no_grad()
def psnr_y_range(x, y):

    # Calculate the per-batch-per-channel range.
    y_range = _bc_y_range(y)

    # Calculate squared error then regularize by the per-batch-per-channel range.
    regular_mse = (y_range ** 2) / F.mse_loss(x, y, reduction="none")
    # Calculate the mean over channels and spatial dims.
    regular_mse = einops.reduce(regular_mse, "b .... -> b", "mean")
    # Find PSNR for each batch.
    psnr = 10 * torch.log10(regular_mse)
    # Mean of PSNR values for the whole batch.
    psnr = psnr.mean()

    return psnr


@torch.no_grad()
def ssim_y_range(
    x: torch.Tensor,
    y: torch.Tensor,
    **ssim_kwargs,
) -> torch.Tensor:

    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    y_range = _bc_y_range(y)
    y_range = y_range.detach().cpu().numpy()

    batch_scores = list()
    batch_size = y.shape[0]
    n_channels = y.shape[1]
    for i in range(batch_size):
        channel_scores = list()
        for j in range(n_channels):
            score = skimage.metrics.structural_similarity(
                x_np[i, j],
                y_np[i, j],
                multichannel=False,
                data_range=y_range[i, j],
                **ssim_kwargs,
            )
            channel_scores.append(score)
        channel_mean = np.mean(np.asarray(channel_scores))
        batch_scores.append(channel_mean)

    scores = torch.from_numpy(np.stack(batch_scores)).to(y)
    # Find average SSIM over the entire batch.
    scores = scores.mean()

    return scores


@torch.no_grad()
def ms_ssim_y_range(
    x: torch.Tensor,
    y: torch.Tensor,
    **ms_ssim_kwargs,
) -> torch.Tensor:

    y_range = _bc_y_range(y)

    # Pad to have a sufficient volume size given a window size.
    window_size = ms_ssim_kwargs.get("win_size", 11)
    min_side_len = (window_size - 1) * 2 ** 4
    if min(y.shape[2:]) <= min_side_len:
        # Monai transformer only takes C x [...] tensors, with no batch size.
        batch_size = y.shape[0]
        n_channels = y.shape[1]
        x = einops.rearrange(x, "b c ... -> (b c) ...")
        y = einops.rearrange(y, "b c ... -> (b c) ...")
        new_shape = torch.maximum(
            torch.as_tensor(
                [
                    min_side_len + 1,
                ]
            ),
            torch.as_tensor(y.shape[2:]),
        )
        padder = monai.transforms.SpatialPad(
            new_shape, method="symmetric", mode="reflect"
        )
        x = padder(x)
        y = padder(y)

        x = einops.rearrange(x, "(b c) ... -> b c ...", b=batch_size, c=n_channels)
        y = einops.rearrange(y, "(b c) ... -> b c ...", b=batch_size, c=n_channels)

    batch_size = y.shape[0]
    n_channels = y.shape[1]
    all_scores = list()
    # Calculate each batch and channel entry MS-SSIM, as the data_range only takes a
    # single scalar.
    for i in range(batch_size):
        c_scores = list()
        for j in range(n_channels):
            c_score = pytorch_msssim.ms_ssim(
                x[i, j][None, None],
                y[i, j][None, None],
                data_range=y_range[i, j],
                size_average=True,
                **ms_ssim_kwargs,
            )
            c_scores.append(c_score)
        all_scores.append(torch.stack(c_scores, dim=0))

    scores = torch.stack(all_scores, dim=0)
    scores = scores.mean()

    return scores


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


@torch.no_grad()
def minmax_normalized_rmse(
    x: torch.Tensor,
    y: torch.Tensor,
    range_based_on: str = "y",
    reduction="mean",
) -> torch.Tensor:
    if range_based_on.casefold() == "x":
        x, y = y, x

    if reduction is None:
        reduction = "none"
    elif not isinstance(reduction, str):
        reduction = str(reduction)

    n_spatial_dims = len(y.shape[2:])
    reduce_str = "b c ... -> b c " + " ".join(itertools.repeat("1", n_spatial_dims))
    y_min = einops.reduce(y, reduce_str, "min")
    y_max = einops.reduce(y, reduce_str, "max")
    y_range = y_max - y_min

    norm_square_err = F.mse_loss(x, y, reduction="none") / (y_range ** 2)

    if reduction.casefold() == "mean":
        nrmse = torch.sqrt(torch.mean(norm_square_err))
    elif reduction.casefold() == "sum":
        raise NotImplementedError("ERROR: sum reduction not implemented.")
    elif reduction.casefold() == "none":
        nrmse = torch.sqrt(norm_square_err)
    else:
        raise ValueError(f"ERROR: Invalid reduction {reduction}")

    return nrmse


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

    window_size = ms_ssim_kwargs.get("win_size", 11)

    min_side_len = (window_size - 1) * 2 ** 4
    if min(y.shape[2:]) <= min_side_len:
        # Monai transformer only takes C x [...] tensors, with no batch size.
        batch_size = x_scaled.shape[0]
        n_channels = x_scaled.shape[1]
        x_scaled = einops.rearrange(x_scaled, "b c ... -> (b c) ...")
        y_scaled = einops.rearrange(y_scaled, "b c ... -> (b c) ...")
        new_shape = torch.maximum(
            torch.as_tensor(
                [
                    min_side_len + 1,
                ]
            ),
            torch.as_tensor(y.shape[2:]),
        )
        padder = monai.transforms.SpatialPad(
            new_shape, method="symmetric", mode="reflect"
        )
        x_scaled = padder(x_scaled)
        y_scaled = padder(y_scaled)

        x_scaled = einops.rearrange(
            x_scaled, "(b c) ... -> b c ...", b=batch_size, c=n_channels
        )
        y_scaled = einops.rearrange(
            y_scaled, "(b c) ... -> b c ...", b=batch_size, c=n_channels
        )

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


@torch.no_grad()
def fast_fa(dti_lower_triangle, foreground_mask=None):
    batch_size = dti_lower_triangle.shape[0]
    spatial_dims = tuple(dti_lower_triangle.shape[2:])

    tri_dti = pitn.eig.tril_vec2sym_mat(dti_lower_triangle, tril_dim=1)
    eigvals = pitn.eig.eigvalsh_workaround(tri_dti, "L")
    # Give background voxels a value of 1 to avoid numerical errors.
    if foreground_mask is None:
        background_mask = (dti_lower_triangle == 0).all(1)
    else:
        background_mask = ~(foreground_mask.bool().to(eigvals.device))
        # Remove channel dimension from the mask.
        background_mask = background_mask.view(batch_size, *spatial_dims)

    eigvals[background_mask] = 0
    eigvals = torch.clamp_min_(eigvals, min=0)

    # Move eigenvalues dimension to the front of the tensor.
    eigvals = einops.rearrange(eigvals, "... e -> e ...", e=3)
    # Even if a voxel isn't in the background, it may still have a value close to
    # 0, so add those to avoid numerical errors.
    zeros_mask = background_mask | (
        torch.isclose(eigvals, eigvals.new_tensor(0), atol=1e-9)
    ).all(0)

    ev1, ev2, ev3 = eigvals
    fa_num = (ev1 - ev2) ** 2 + (ev2 - ev3) ** 2 + (ev3 - ev1) ** 2
    fa_denom = (eigvals * eigvals).sum(0) + zeros_mask
    fa = torch.sqrt(0.5 * fa_num / fa_denom)

    fa = fa.view(batch_size, 1, *spatial_dims)
    return fa


# self.ssim_metric = functools.partial(
#     skimage.metrics.structural_similarity,
#     win_size=7,
#     K1=0.01,
#     K2=0.03,
#     use_sample_covariance=True,
#     data_range=psnr_max,
#     multichannel=True,
# )

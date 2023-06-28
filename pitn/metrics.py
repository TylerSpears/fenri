# -*- coding: utf-8 -*-
import itertools

import einops
import monai
import numpy as np
import skimage
import torch
import torch.nn.functional as F

import pitn
from pitn._lazy_loader import LazyLoader

# Make pytorch_msssim an optional import.
pytorch_msssim = LazyLoader("pytorch_msssim", globals(), "pytorch_msssim")


@torch.no_grad()
def sphere_jensen_shannon_distance(
    input_odf_coeffs, target_odf_coeffs, mask, theta, phi, sh_order=8
):
    epsilon = 1e-5
    batch_size = input_odf_coeffs.shape[0]
    if batch_size != 1:
        raise NotImplementedError("ERROR: Batch size != 1 not implemented or tested")
    input_sphere_samples = pitn.odf.sample_sphere_coords(
        input_odf_coeffs * mask,
        theta=theta,
        phi=phi,
        sh_order=sh_order,
        sh_order_dim=1,
        mask=mask,
        force_nonnegative=True,
    )
    n_sphere_samples = input_sphere_samples.shape[1]
    sphere_mask = mask.expand_as(input_sphere_samples)
    # Mask and reshape to (n_vox x batch_size) x n_prob_samples
    input_sphere_samples = einops.rearrange(
        input_sphere_samples[sphere_mask],
        "(b s v) -> (b v) s",
        b=batch_size,
        s=n_sphere_samples,
    )
    # Normalize to sum to 1.0, as a probability density.
    input_sphere_samples /= torch.maximum(
        torch.sum(input_sphere_samples, dim=1, keepdim=True),
        input_odf_coeffs.new_zeros(1) + epsilon,
    )
    target_sphere_samples = pitn.odf.sample_sphere_coords(
        target_odf_coeffs * mask,
        theta=theta,
        phi=phi,
        sh_order=8,
        sh_order_dim=1,
        mask=mask,
    )
    target_sphere_samples = einops.rearrange(
        target_sphere_samples[sphere_mask],
        "(b s v) -> (b v) s",
        b=batch_size,
        s=n_sphere_samples,
    )
    # Normalize to sum to 1.0, as a probability density.
    target_sphere_samples /= torch.maximum(
        torch.sum(target_sphere_samples, dim=1, keepdim=True),
        target_odf_coeffs.new_zeros(1) + epsilon,
    )

    Q_log_in = torch.log(input_sphere_samples.to(torch.float64))
    P_log_target = torch.log(target_sphere_samples.to(torch.float64))
    M_log = torch.log(
        (input_sphere_samples + target_sphere_samples).to(torch.float64) / 2
    )
    del input_sphere_samples, target_sphere_samples
    d_P_M = F.kl_div(M_log, P_log_target, reduction="none", log_target=True)
    # Implement batchmean per-voxel.
    # nan values from the kl divergence occur when the expected density is 0.0 and the
    # log is -inf. The 'contribution' of that element is 0 as the limit approaches 0,
    # so just adding the non-nan values should be valid.
    d_P_M = d_P_M.nansum(1, keepdim=True) / d_P_M.shape[1]

    d_Q_M = F.kl_div(M_log, Q_log_in, reduction="none", log_target=True)
    d_Q_M = d_Q_M.nansum(1, keepdim=True) / d_Q_M.shape[1]

    js_div = d_P_M / 2 + d_Q_M / 2
    js_div = einops.rearrange(js_div, "(b v) s -> (b s v)", b=batch_size, s=1)
    js_dist = torch.zeros_like(mask).to(input_odf_coeffs)
    js_dist.masked_scatter_(mask, torch.sqrt(js_div).to(torch.float32)).to(
        input_odf_coeffs
    )

    return js_dist


class NormRMSEMetric(torch.nn.Module):
    def __init__(self, reduction="mean", normalization="min-max"):
        super().__init__()
        self.reduction = reduction
        self.normalization = normalization.lower()
        assert self.normalization == "min-max"

    def forward(self, y_pred, y):
        rmse_unreduced = F.mse_loss(y_pred, y, reduction="none")
        rmse = torch.sqrt(einops.reduce(rmse_unreduced, "b c ... -> b c", "mean"))
        max_y = einops.reduce(y, "b c ... -> b c", "max")
        min_y = einops.reduce(y, "b c ... -> b c", "min")
        minmax = max_y - min_y
        nrmse = rmse / minmax

        nrmse, notnans = monai.metrics.utils.do_metric_reduction(nrmse, self.reduction)
        return nrmse


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
    regular_mse = (y_range**2) / torch.clamp_min(
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

    norm_square_err = F.mse_loss(x, y, reduction="none") / (y_range**2)

    if reduction.casefold() == "mean":
        nrmse = torch.sqrt(torch.mean(norm_square_err))
    elif reduction.casefold() == "sum":
        raise NotImplementedError("ERROR: sum reduction not implemented.")
    elif reduction.casefold() == "none":
        nrmse = torch.sqrt(norm_square_err)
    else:
        raise ValueError(f"ERROR: Invalid reduction {reduction}")

    return nrmse


@torch.no_grad()
def minmax_normalized_dti_root_vec_fro_norm(
    x: torch.Tensor,
    y: torch.Tensor,
    mask=None,
    scale_off_diags=True,
    range_based_on: str = "y",
    reduction="mean",
) -> torch.Tensor:
    """Calculates normalized root frobenius norm on lower-triangular elements of matrix.

    This is similar to NRMSE with min-max normalization
    (<https://scikit-image.org/docs/0.18.x/api/skimage.metrics.html#normalized-root-mse>),
    but error is compounded between tensor components rather than averaged.

    By default, this metric
    1. Scales off-diagonals by sqrt(2)
    2. Finds the squared error of the entire input without any reduction.
    3. Sums the error of each diffusion tensor component ("sum over the channels dim")
    4. Calculates the mean squared error for each element in the batch, over all spatial
       dimensions; if a mask is given, the sum squared-error is divided by the number of
       elements in the mask.
       dimensions.
    5. Takes the square root of that per-batch MSE
    6. Normalizes this per-batch RMSE by the range in `y` calculated as:
        sum_{over tensor components}(max_{per-batch}(y)) - sum_{over tensor components}(min_{per-batch}(y))

        a.k.a., the sum of tensor component maxes, minux sum of tensor component mins.
    7. Takes the mean NRMSE (so, really, MNRMSE).

    Parameters
    ----------
    x : torch.Tensor
        Prediction tensor with dimensions `batch x 6 x sp_dim_1 [x sp_dim_2 ...]`

        The 6 component dims are assumed to be the lower triangular elements of a full
        3 x 3 symmetric diffusion tensor.
    y : torch.Tensor
        Ground truth tensor with dimensions `batch x 6 x sp_dim_1 [x sp_dim_2 ...]`

        The 6 component dims are assumed to be the lower triangular elements of a full
        3 x 3 symmetric diffusion tensor.
    mask : torch.Tensor, optional
        Boolean Tensor with dimensions `batch x 1 x sp_dim_1 [x sp_dim_2 ...]`.

        Indicates which elements in `input` and `target` are to be considered by the
        mean.
    scale_off_diags : bool, optional
        Indicates whether or not to scale off-diagonals by sqrt of 2, by default True

        This assumes that both `input` and `target` are *lower triangular* elements of
        the full 3x3 diffusion tensor.
    range_based_on : str, optional
        _description_, by default "y"
    reduction : str, optional
        Specifies reduction pattern of output, by default "mean"

        One of {None, 'none', 'mean', 'mean_dti'}

    Returns
    -------
    torch.Tensor

    Raises
    ------
    ValueError
    """

    if range_based_on.casefold() == "x":
        x, y = y, x

    reduce = str(reduction).casefold()

    n_spatial_dims = len(y.shape[2:])
    # Calculate y range for each item in the batch.
    reduce_str = "b c ... -> b c " + " ".join(itertools.repeat("1", n_spatial_dims))
    # Sum the tensor component dim, or "channels" dim. This
    y_min = einops.reduce(y, reduce_str, "min")
    y_min = y_min.sum(1, keepdim=True)
    y_max = einops.reduce(y, reduce_str, "max")
    y_max = y_max.sum(1, keepdim=True)
    y_range = y_max - y_min

    tensor_se = pitn.nn.loss.dti_vec_fro_norm_loss(
        x, y, scale_off_diags=scale_off_diags, mask=mask, reduction="none"
    )

    if reduce == "none":
        result = torch.sqrt(tensor_se) / y_range
    else:
        y_range = y_range.flatten()

        # Take sum of all tensor errors for each element in the batch.
        fro_bsse = einops.reduce(tensor_se, "b ... -> b", "sum")
        # If a mask is present, average out the sum of tensor errors by the amount of
        # selected elements in the mask.
        if mask is not None:
            fro_bmse = fro_bsse / mask.reshape(mask.shape[0], -1).sum(-1)
        # Otherwise, average by the number of "spatial" elements (non-batch dims).
        else:
            fro_bmse = fro_bsse / tensor_se[0].numel()
        # Find the per-volume NRMSE, with the mean properly calculated according to the
        # mask.
        fro_bmse = torch.sqrt(fro_bmse) / y_range

        # Take the mean across batches, the "MNRMSE"
        if reduce == "mean":
            result = fro_bmse.mean()
        elif reduce == "mean_dti":
            result = fro_bmse
        else:
            raise ValueError(f"ERROR: Invalid reduction {reduction}")

    return result


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

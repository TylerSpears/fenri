# -*- coding: utf-8 -*-
import einops
import torch
import torch.nn.functional as F


def jenson_shannon_diverge(
    log_input_dist: torch.Tensor,
    log_target_dist: torch.Tensor,
    reduction=None,
):
    kl = torch.nn.KLDivLoss(reduction=reduction, log_target=True)
    log_P = log_target_dist
    log_Q = log_input_dist

    log_M = (
        log_P
        + torch.log1p(torch.exp(log_Q - log_P))
        - torch.log(torch.tensor([0.5]).to(log_Q))
    )

    jsd = (
        torch.exp(
            torch.log(kl(log_M, log_P))
            + torch.log1p(torch.exp(kl(log_M, log_Q) - kl(log_M, log_P)))
        )
        / 2
    )

    return jsd


def dti_vec_fro_norm_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor = None,
    scale_off_diags=True,
    reduction="mean",
) -> torch.Tensor:
    """Calculate Frobenius norm of difference of vectorized DTIs.

    Analogous to mean-squared error for DTIs.

    By default, this metric
    1. Scales off-diagonals by sqrt(2)
    2. Finds the squared error of the entire input without any reduction.
    3. Sums the error of each diffusion tensor component ("sum over the channels dim")
    4. Calculates the mean squared error for each element in the batch, over all spatial
       dimensions; if a mask is given, the sum squared-error is divided by the number of
       elements in the mask.
    5. Takes the mean between each MSE in the batch.

    Parameters
    ----------
    input : torch.Tensor
        Tensor with dimensions `batch x 6 x sp_dim_1 [x sp_dim_2 ...]`

        The 6 component dims are assumed to be the lower triangular elements of a full
        3 x 3 symmetric diffusion tensor.
    target : torch.Tensor
        Tensor with dimensions `batch x 6 x sp_dim_1 [x sp_dim_2 ...]`

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

    reduction : str, optional
        Specifies reduction pattern of output, by default "mean"

        One of {None, 'none', 'mean', 'mean_dti'}

    Returns
    -------
    torch.Tensor
    """

    reduce = str(reduction).casefold()

    if scale_off_diags:
        off_diag_scaler = torch.sqrt(torch.as_tensor([1, 2, 1, 2, 2, 1]).to(input))
        shape = (1, 6) + ((1,) * len(input.shape[2:]))
        off_diag_scaler = off_diag_scaler.reshape(*shape)

        input = input * off_diag_scaler
        target = target * off_diag_scaler

    se = F.mse_loss(input, target, reduction="none")

    # Un-masked elements should not contribute to the error at all.
    if mask is not None:
        se = se * mask
    # Tensor component errors are *always* summed. A tensor's sum squared-error over all
    # tensor componenents at a particular voxel, in a particular volume is considered as
    # one "atomic unit" of error.
    tensor_se = se.sum(1, keepdim=True)
    if reduce == "none":
        result = tensor_se
    else:
        # Take sum of all tensor errors for each element in the batch.
        tensor_bsse = einops.reduce(tensor_se, "b ... -> b", "sum")
        # If a mask is present, average out the sum of tensor errors by the amount of
        # selected elements in the mask.
        if mask is not None:
            tensor_bmse = tensor_bsse / mask.reshape(mask.shape[0], -1).sum(-1)
        # Otherwise, average by the number of "spatial" elements (non-batch dims).
        else:
            tensor_bmse = tensor_bsse / tensor_se[0].numel()
        if reduce == "mean":
            result = tensor_bmse.mean()
        elif reduce == "mean_dti":
            result = tensor_bmse
        else:
            raise ValueError(f"ERROR: Invalid reduction {reduction}")

    return result


def dti_root_vec_fro_norm_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor = None,
    scale_off_diags=True,
    reduction="mean",
) -> torch.Tensor:
    """Calculate square root Frobenius norm of difference of vectorized DTIs.

    Analogous to root mean-squared error.

    By default, this metric
    1. Scales off-diagonals by sqrt(2)
    2. Finds the squared error of the entire input without any reduction.
    3. Sums the error of each diffusion tensor component ("sum over the channels dim")
    4. Calculates the mean squared error for each element in the batch, over all spatial
       dimensions; if a mask is given, the sum squared-error is divided by the number of
       elements in the mask.
       dimensions.
    5. Takes the square root of that per-batch MSE
    6. Takes the mean between each RMSE in the batch.

    Parameters
    ----------
    input : torch.Tensor
        Tensor with dimensions `batch x 6 x sp_dim_1 [x sp_dim_2 ...]`

        The 6 component dims are assumed to be the lower triangular elements of a full
        3 x 3 symmetric diffusion tensor.
    target : torch.Tensor
        Tensor with dimensions `batch x 6 x sp_dim_1 [x sp_dim_2 ...]`

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

    reduction : str, optional
        Specifies reduction pattern of output, by default "mean"

        One of {None, 'none', 'mean', 'mean_dti'}

    Returns
    -------
    torch.Tensor
    """

    reduce = str(reduction).casefold()
    fro_dist = dti_vec_fro_norm_loss(
        input, target, mask=mask, scale_off_diags=scale_off_diags, reduction="none"
    )

    if reduce == "none":
        result = torch.sqrt(fro_dist)
    else:

        # Take sum of all tensor errors for each element in the batch.
        fro_bsse = einops.reduce(fro_dist, "b ... -> b", "sum")
        # If a mask is present, average out the sum of tensor errors by the amount of
        # selected elements in the mask.
        if mask is not None:
            fro_bmse = fro_bsse / mask.reshape(mask.shape[0], -1).sum(-1)
        # Otherwise, average by the number of "spatial" elements (non-batch dims).
        else:
            fro_bmse = fro_bsse / fro_dist[0].numel()
        # Find the per-volume RMSE, with the mean properly calculated according to the
        # mask.
        fro_bmse = torch.sqrt(fro_bmse)

        # Take the mean across batches, kind of an "MRMSE"
        if reduce == "mean":
            result = fro_bmse.mean()
        elif reduce == "mean_dti":
            result = fro_bmse
        else:
            raise ValueError(f"ERROR: Invalid reduction {reduction}")

    return result

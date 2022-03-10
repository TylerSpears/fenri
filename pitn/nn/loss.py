# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import einops


def dti_vec_fro_norm_loss(
    input: torch.Tensor,
    target: torch.Tensor,
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
       dimensions.
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

    # Tensor component errors are *always* summed. A tensor's total error is
    # considered as one atomic "unit" of error.
    tensor_se = se.sum(1, keepdim=True)
    if reduce == "none":
        result = tensor_se
    else:
        if reduce == "mean":
            result = tensor_se.mean()
        elif reduce == "mean_dti":
            result = einops.reduce(tensor_se, "b ... -> b", "mean")
        else:
            raise ValueError(f"ERROR: Invalid reduction {reduction}")

    return result


def dti_root_vec_fro_norm_loss(
    input: torch.Tensor,
    target: torch.Tensor,
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
        input, target, scale_off_diags=scale_off_diags, reduction="none"
    )

    if reduce == "none":
        result = torch.sqrt(fro_dist)
    else:
        # Find the "MSE" in "RMSE"
        rm_fro_dist = torch.sqrt(einops.reduce(fro_dist, "b ... -> b", "mean"))

        # Take the mean across batches, kind of an "MRMSE"
        if reduce == "mean":
            result = rm_fro_dist.mean()
        elif reduce == "mean_dti":
            result = rm_fro_dist
        else:
            raise ValueError(f"ERROR: Invalid reduction {reduction}")

    return result

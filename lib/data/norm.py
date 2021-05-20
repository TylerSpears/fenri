import torch


def normalize_batch(batch, mean, var, eps=1 ** (-10)):
    """Normalize a batch of patches to have mean 0 and variance 1 in the given dimensions.

    The normalization occurs according to the dimensionality of the mean and var passed.
        For example, to perform channel-wise normalization, mean and var must have shape
        (1, C, 1, 1, 1)
    """

    return (batch - mean) / torch.sqrt(var + eps)


def denormalize_batch(normal_batch, mean, var, eps=1 ** (-10)):
    """De-normalize a batch to have given mean and variance.

    Inverse of `normalize_batch`.

    The normalization occurs according to the dimensionality of the mean and var passed.
    """

    return (normal_batch * torch.sqrt(var + eps)) + mean


def normalize_dti(dti, mean, var, eps=1 ** (-10)):
    """Normalize a 6-channel DTI tensor to have mean 0 and variance 1.

    The normalization occurs according to the dimensionality of the mean and var passed.
        To perform channel-wise normalization, mean and var must have shape
        (C x 1 x 1 x 1)
    """

    return (dti - mean) / torch.sqrt(var + eps)


def denormalize_dti(normal_dti, mean, var, eps=1 ** (-10)):
    """Inverse of DTI normalization."""

    return (normal_dti * torch.sqrt(var + eps)) + mean

import torch


def normalize_batch(batch, mean, var):
    """Normalize a batch of patches to have mean 0 and variance 1 in the given dimensions.

    The normalization occurs according to the dimensionality of the mean and var passed.
        For example, to perform channel-wise normalization, mean and var must have shape
        (1, C, 1, 1, 1)
    """

    epsilon = 1e-5
    return (batch - mean) / torch.sqrt(var + epsilon)


def denormalize_batch(normal_batch, mean, var):
    """De-normalize a batch to have mean 0 and variance 1 in the given dimensions.

    Inverse of `normalize_batch`.

    The normalization occurs according to the dimensionality of the mean and var passed.
        For example, to perform channel-wise normalization, mean and var must have shape
        (1, C, 1, 1, 1)
    """

    epsilon = 1e-5
    return (normal_batch * torch.sqrt(var + epsilon)) + mean


def normalize_dti(dti, mean, var):
    """Normalize a 6-channel DTI tensor to have mean 0 and variance 1.

    The normalization occurs according to the dimensionality of the mean and var passed.
        To perform channel-wise normalization, mean and var must have shape
        (C x 1 x 1 x 1)
    """

    epsilon = 1e-5
    return (dti - mean) / torch.sqrt(var + epsilon)


def denormalize_dti(normal_dti, mean, var):
    """Inverse of DTI normalization."""

    epsilon = 1e-5
    return (normal_dti * torch.sqrt(var + epsilon)) + mean

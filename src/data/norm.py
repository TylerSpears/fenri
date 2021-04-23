import torch


def normalize_batch(patches):
    """Normalize a batch of patches to have mean 0 and variance 1

    Computed over the batches, so means & vars are calculated as:
        (B x C x H x W x D) -> (1 x C x H x W x D)
    """

    mean = torch.mean(patches, dim=0, keepdim=True)
    var = torch.var(patches, dim=0, keepdim=True)
    epsilon = 1e-7

    return (patches - mean) / torch.sqrt(var + epsilon)


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

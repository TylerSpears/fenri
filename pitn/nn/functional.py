# -*- coding: utf-8 -*-
import itertools

import einops
import numpy as np
import torch.nn.functional as F


# Shuffle operation as a function.
def espcn_shuffle(x, channels):
    """Implements final-layer shuffle operation from ESPCN.

    x: 4D or 5D Tensor. Expects a shape of $C \times H \times W \times D$, or batched
        with a shape of $B \times C \times H \times W \times D$.

    channels: Integer giving the number of channels for the shuffled output.
    """
    batched = True if x.ndim == 5 else False

    if batched:
        downsample_factor = int(np.power(x.shape[1] / channels, 1 / 3))
        y = einops.rearrange(
            x,
            "b (c r1 r2 r3) h w d -> b c (h r1) (w r2) (d r3)",
            c=channels,
            r1=downsample_factor,
            r2=downsample_factor,
            r3=downsample_factor,
        )
    else:
        downsample_factor = int(np.power(x.shape[0] / channels, 1 / 3))
        y = einops.rearrange(
            x,
            "(c r1 r2 r3) h w d -> c (h r1) (w r2) (d r3)",
            c=channels,
            r1=downsample_factor,
            r2=downsample_factor,
            r3=downsample_factor,
        )

    return y


def unfold_3d(x, kernel, stride=1, **pad_kwargs):
    if stride != 1:
        raise NotImplementedError("Stride must be 1")
    if isinstance(stride, int):
        stride = (stride,) * 3
    if isinstance(kernel, int):
        kernel = (kernel,) * 3
    # By default, pad by one on each side, but allow user to override this.
    pad_kwargs = {**{"pad": tuple(itertools.repeat(1, len(kernel) * 2))}, **pad_kwargs}

    # Pad on each side.
    y = F.pad(x, **pad_kwargs)

    y = (
        y.unfold(2, kernel[0], stride[0])
        .unfold(3, kernel[1], stride[1])
        .unfold(4, kernel[2], stride[2])
    )
    y = einops.rearrange(
        y, "b c xmk ymk zmk k_x k_y k_z -> b c (xmk k_x) (ymk k_y) (zmk k_z)"
    )

    return y

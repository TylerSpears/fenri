# -*- coding: utf-8 -*-
from typing import Tuple

import numpy as np
import torch


def batched_tensor_indexing(
    t: torch.Tensor, index: Tuple[torch.Tensor]
) -> torch.Tensor:
    """Index a tensor by matching dimensions from right-to-left instead of left-to-right.

    This allows for broadcasting of tensor indexing across non-spatial dimensions. Ex.,
    sub-selecting the same patch across multiple channels, selecting the same timepoint
    in a set of timeseries data, etc.

    By default, pytorch indexes starting from left-to-right, meaning that a typical
    "B x C x H x W" tensor could not be indexed batch-and-channel-wise with only one
    definition of the index tuple; one would have to copy the index B or B*C times.
    This is a strange choice considering that B dims are the left-most dimension by
    convention.

    Parameters
    ----------
    t : torch.Tensor
        Tensor with ndim >= len(index)
    index : Tuple[torch.Tensor]
        Tuple of index Tensors, with length N_dims_to_index.

    Returns
    -------
    torch.Tensor
        Indexed Tensor broadcasted to the remaining left-most dimensions.
    """
    if not torch.is_tensor(index):
        idx = torch.stack(index, dim=0)
    else:
        idx = index
    # Pytorch requires long ints for indexing.
    idx = idx.long()
    flip_idx = torch.flip(idx, dims=(0,))
    return t.T[tuple(flip_idx)].T

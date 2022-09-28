# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Sequence, Tuple, Union

import nibabel as nib
import numpy as np
import torch
from more_itertools import collapse


def rerun_indicator_from_mtime(
    input_files: Sequence[Union[str, Path]], output_files: Sequence[Union[str, Path]]
):
    in_files = [Path(str(f)) for f in collapse(input_files)]
    out_files = [Path(str(f)) for f in collapse(output_files)]
    in_mtimes = [p.stat().st_mtime_ns if p.exists() else -1 for p in in_files]
    out_mtimes = [p.stat().st_mtime_ns if p.exists() else -1 for p in out_files]

    result: bool
    if any(map(lambda t: t < 0, out_mtimes)):
        result = True
    elif max(in_mtimes) > min(out_mtimes):
        result = True
    else:
        result = False

    return result


def rerun_indicator_from_nibabel(
    input_im_data: np.ndarray, input_im_affine: np.ndarray, output_nifti: Path
) -> bool:
    rerun = True
    out = Path(str(output_nifti)).resolve()
    if not out.exists():
        rerun = True
    else:
        out_im = nib.load(str(out))
        # Compare affines first to possibly avoid loading the full file.
        out_im_affine = out_im.affine
        try:
            if not np.isclose(out_im_affine, input_im_affine).all():
                rerun = True
            else:
                out_im_data = out_im.get_fdata()
                data_is_similar = np.isclose(out_im_data, input_im_data).all()
                rerun = not data_is_similar
        except ValueError:
            rerun = True

    return rerun


def union_parent_dirs(*paths, resolve=True) -> Tuple[Path]:
    ps = [Path(str(p)) for p in collapse(paths)]
    if resolve:
        ps = [p.resolve() for p in ps]
    parent_ps = set()
    for p in ps:
        if p.is_dir():
            parent_ps.add(str(p))
        elif p.is_file():
            parent_ps.add(str(p.parent))

    return tuple(parent_ps)


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

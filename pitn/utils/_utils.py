# -*- coding: utf-8 -*-
import collections
from pathlib import Path
from typing import Sequence, Tuple, Union

import dipy
import dipy.io
import dipy.io.streamline
import nibabel as nib
import numpy as np
import torch
from more_itertools import collapse


def dipy_save_trk2tck(
    trk_f,
    target_tck_f,
    trk_reference="same",
    load_trk_kwargs=dict(),
    save_tck_kwargs=dict(),
):
    trk_f = str(Path(trk_f).resolve())
    trk = dipy.io.streamline.load_trk(trk_f, trk_reference, **load_trk_kwargs)
    trk.to_rasmm()
    tck_f = str(Path(target_tck_f).resolve())
    dipy.io.streamline.save_tck(trk, tck_f, **save_tck_kwargs)
    return trk


def flatten(
    iterable: collections.abc.Iterable,
    parent_key=False,
    seperator: str = ".",
    as_dict: bool = False,
) -> collections.abc.Iterable:

    result = None
    if isinstance(iterable, collections.abc.MutableMapping):
        result = _flatten_dict(iterable, parent_key=parent_key, separator=seperator)
    elif isinstance(iterable, (list, tuple, set, frozenset, bytearray)):
        result = _flatten_dict(
            {"_": iterable}, parent_key=parent_key, separator=seperator
        )
        if not as_dict:
            result = tuple(result.values())
    else:
        result = iterable

    return result


def _flatten_dict(
    dictionary: collections.abc.MutableMapping,
    parent_key=False,
    separator: str = ".",
) -> collections.abc.MutableMapping:
    """
    Turn a nested dictionary into a flattened dictionary
    :param dictionary: The dictionary to flatten
    :param parent_key: The string to prepend to dictionary's keys
    :param separator: The string used to separate flattened keys
    :return: A flattened dictionary

    Taken from
    <https://github.com/ScriptSmith/socialreaper/blob/master/socialreaper/tools.py#L8>
    """

    items = []
    for key, value in dictionary.items():
        new_key = str(parent_key) + separator + key if parent_key else key
        if isinstance(value, collections.abc.MutableMapping):
            items.extend(flatten(value, new_key, separator).items())
        elif isinstance(value, (list, tuple, set, frozenset, bytearray)):
            for k, v in enumerate(value):
                items.extend(flatten({str(k): v}, new_key).items())
        else:
            items.append((new_key, value))
    return dict(items)


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
    ps = [Path(str(p)) for p in flatten(paths, as_dict=True).values()]
    if resolve:
        ps = [p.resolve() for p in ps]
    parent_ps = set()
    for p in ps:
        if p.is_dir():
            parent_ps.add(str(p))
        elif p.is_file():
            parent_ps.add(str(p.parent))
        elif not p.exists():
            # Check one level up from the non-existent Path, but only one level.
            if p.parent.exists():
                parent_ps.add(str(p.parent))
            else:
                raise RuntimeError(f"ERROR: Path {p} and {p.parent} do not exist.")

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

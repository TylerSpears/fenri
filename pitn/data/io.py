# -*- coding: utf-8 -*-
from pathlib import Path
from typing import TypedDict

import einops
import nibabel as nib
import numpy as np
import pandas as pd
import scipy
import torch

import pitn


class VolDataDict(TypedDict):
    vol: torch.Tensor
    affine: torch.Tensor
    header: dict


class DWIVolDict(TypedDict):
    dwi: torch.Tensor
    affine: torch.Tensor
    grad_table: pd.DataFrame
    header: dict


def reorient_nib_im(
    im: nib.spatialimages.DataobjImage, target_orientation: str = "same"
):
    target_ornt_str = target_orientation.strip()
    if target_ornt_str.lower() != "same":
        src_code = nib.orientations.aff2axcodes(im.affine)
        target_code = tuple(target_ornt_str.upper())
        if src_code != target_code:
            src_ornt = nib.orientations.axcodes2ornt(src_code)
            target_ornt = nib.orientations.axcodes2ornt(target_code)
            src2target_ornt = nib.orientations.ornt_transform(src_ornt, target_ornt)
            ret = im.as_reoriented(src2target_ornt)
        else:
            ret = im
    else:
        ret = im

    return ret


def load_vol(
    vol_f: Path, reorient_im_to: str = "same", ensure_channel_dim=False
) -> VolDataDict:

    vol_im = nib.load(vol_f)

    reorient_im_to = reorient_im_to.strip()
    if reorient_im_to.lower() != "same":
        src_code = nib.orientations.aff2axcodes(vol_im.affine)
        target_code = tuple(reorient_im_to.upper())
        src_ornt = nib.orientations.axcodes2ornt(src_code)
        target_ornt = nib.orientations.axcodes2ornt(target_code)
        src2target_ornt = nib.orientations.ornt_transform(src_ornt, target_ornt)
        vol_im = vol_im.as_reoriented(src2target_ornt)

    vol = torch.from_numpy(
        vol_im.get_fdata(caching="unchanged", dtype=vol_im.get_data_dtype())
    )
    if vol.ndim == 4:
        vol = einops.rearrange(vol, "a b c channel -> channel a b c")
    # Create a channel dimension if requested.
    elif ensure_channel_dim:
        vol = vol.unsqueeze(0)
    affine = torch.from_numpy(vol_im.header.get_best_affine())
    header = dict(vol_im.header)

    return {"vol": vol, "affine": affine, "header": header}


def load_dwi(
    dwi_vol_f: Path,
    grad_mrtrix_f: Path,
    reorient_im_to: str = "same",
) -> DWIVolDict:

    vol_data = load_vol(dwi_vol_f, reorient_im_to=reorient_im_to)
    dwi = vol_data["vol"]
    affine = vol_data["affine"]
    header = vol_data["header"]

    grad_table = np.loadtxt(grad_mrtrix_f, comments="#")
    grad_table = pd.DataFrame(grad_table, columns=("x", "y", "z", "b"))

    return {"dwi": dwi, "affine": affine, "grad_table": grad_table, "header": header}

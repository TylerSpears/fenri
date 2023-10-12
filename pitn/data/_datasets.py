# -*- coding: utf-8 -*-
import collections
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, TypedDict, Union

import einops
import monai
import nibabel as nib
import numpy as np
import pandas as pd
import torch

import pitn
import pitn.transforms.functional as p_tf


class LoadedSuperResSubjSampleDict(TypedDict):
    subj_id: str
    affine_vox2real: torch.Tensor
    dwi: torch.Tensor
    grad_table: pd.DataFrame
    odf: torch.Tensor
    brain_mask: torch.Tensor
    wm_mask: torch.Tensor
    gm_mask: torch.Tensor
    csf_mask: torch.Tensor


def load_super_res_subj_sample(
    subj_id: str,
    dwi_f: Path,
    grad_mrtrix_f: Path,
    odf_f: Path,
    fivett_seg_f: Path,
    brain_mask_f: Path,
) -> LoadedSuperResSubjSampleDict:

    target_im_orient = "RAS"

    dwi_data = pitn.data.io.load_dwi(
        dwi_f, grad_mrtrix_f=grad_mrtrix_f, reorient_im_to=target_im_orient
    )
    dwi = dwi_data["dwi"].to(torch.float32)
    affine_vox2real = dwi_data["affine"].to(torch.float32)
    grad_table = dwi_data["grad_table"]

    odf_data = pitn.data.io.load_vol(odf_f, reorient_im_to=target_im_orient)
    odf = odf_data["vol"].to(dwi)

    brain_mask_data = pitn.data.io.load_vol(
        brain_mask_f, reorient_im_to=target_im_orient, ensure_channel_dim=True
    )
    brain_mask = brain_mask_data["vol"].bool()

    fivett_data = pitn.data.io.load_vol(fivett_seg_f, reorient_im_to=target_im_orient)
    fivett_seg = fivett_data["vol"].bool()

    # Ensure that all vox-to-real affines and image shapes are the same.
    for vol in (odf, brain_mask, fivett_seg):
        assert tuple(vol.shape[1:]) == tuple(dwi.shape[1:])
    for d in (odf_data, brain_mask_data, fivett_data):
        assert torch.isclose(affine_vox2real, d["affine"].to(affine_vox2real)).all()

    # Construct the 3 tissue masks from the fivett segmentation.
    wm_mask = fivett_seg[2].unsqueeze(0)
    gm_mask = (fivett_seg[0] | fivett_seg[1]).unsqueeze(0)
    csf_mask = fivett_seg[3].unsqueeze(0)

    return LoadedSuperResSubjSampleDict(
        subj_id=subj_id,
        dwi=dwi,
        grad_table=grad_table,
        affine_vox2real=affine_vox2real,
        odf=odf,
        brain_mask=brain_mask,
        wm_mask=wm_mask,
        gm_mask=gm_mask,
        csf_mask=csf_mask,
    )

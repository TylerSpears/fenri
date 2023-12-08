#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

import nibabel as nib
import numpy as np

WM_PV_THRESH = 0.001
GM_PV_THRESH = 0.5
CSF_PV_THRESH = 0.5

if __name__ == "__main__":

    warped_fivett_f = Path(sys.argv[1])
    comp_1_pv_f = Path(sys.argv[2])
    comp_2_pv_f = Path(sys.argv[3])
    comp_3_pv_f = Path(sys.argv[4])
    comp_4_pv_f = Path(sys.argv[5])

    brain_mask_f = Path(sys.argv[6])
    out_fivett_f = Path(sys.argv[7])

    warped_fivett_im = nib.load(warped_fivett_f)
    warped_fivett = warped_fivett_im.get_fdata().astype(bool)
    affine = warped_fivett_im.affine
    brain_mask_im = nib.load(brain_mask_f)
    brain_mask = brain_mask_im.get_fdata().astype(bool)
    assert np.isclose(brain_mask_im.affine, affine).all()
    comp_1_pv_im = nib.load(comp_1_pv_f)
    comp_2_pv_im = nib.load(comp_2_pv_f)
    comp_3_pv_im = nib.load(comp_3_pv_f)
    comp_4_pv_im = nib.load(comp_4_pv_f)
    assert np.isclose(comp_1_pv_im.affine, affine).all()
    assert np.isclose(comp_2_pv_im.affine, affine).all()
    assert np.isclose(comp_3_pv_im.affine, affine).all()
    assert np.isclose(comp_4_pv_im.affine, affine).all()
    comp_1_pv = comp_1_pv_im.get_fdata()
    comp_2_pv = comp_2_pv_im.get_fdata()
    wm_pv = comp_1_pv + comp_2_pv
    gm_pv = comp_3_pv_im.get_fdata()
    csf_pv = comp_4_pv_im.get_fdata()

    # Threshold the WM mask. The threshold must be very low, given the relatively
    # high spatial resolution and the (relatively) sparse fiber count in the template
    # tractographies.
    wm_mask = (wm_pv >= WM_PV_THRESH) & brain_mask

    # Threshold the GM mask, but also split into cortical and sub-cortical masks.
    gm_mask = (gm_pv >= GM_PV_THRESH) & ~wm_mask & brain_mask
    # We only care about the sub-cortical gm tissue from the fivett mask.
    sub_cort_gm_mask = warped_fivett[..., 1] & gm_mask
    cort_gm_mask = gm_mask & ~sub_cort_gm_mask

    # Threshold CSF mask.
    csf_mask = (
        (csf_pv >= CSF_PV_THRESH)
        & ~cort_gm_mask
        & ~sub_cort_gm_mask
        & ~wm_mask
        & brain_mask
    )

    # Just zero out the pathological tissue.
    pathology_mask = wm_mask * 0

    new_fivett = np.stack(
        [cort_gm_mask, sub_cort_gm_mask, wm_mask, csf_mask, pathology_mask], axis=-1
    ).astype(np.uint8)

    new_fivett_im = nib.Nifti1Image(new_fivett, affine=affine)

    nib.save(new_fivett_im, Path(out_fivett_f))

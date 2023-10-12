#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy
import skimage

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--param_vol", type=Path, required=True)
    parser.add_argument("-m", "--mask", type=Path, required=True)
    parser.add_argument(
        "-t", "--tissue_mask", nargs="+", action="append", type=Path, required=True
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("-u", "--upper_quantile", type=float, default=0.99)
    parser.add_argument("-l", "--lower_quantile", type=float, default=0.0)
    parser.add_argument("-r", "--radius_select_outer", type=int, default=4)

    args = parser.parse_args()

    p_f = args.param_vol.resolve()
    p_im = nib.load(p_f)
    p = p_im.get_fdata().astype(np.float32)
    mask_f = args.mask.resolve()
    mask_im = nib.load(mask_f)
    mask = mask_im.get_fdata().astype(bool)

    tissue_mask = np.zeros_like(mask)
    tmp_fs = [t_f for t_f in args.tissue_mask]
    tissue_mask_fs = list()
    for t_f in tmp_fs:
        if isinstance(t_f, Path):
            tissue_mask_fs.append(t_f)
        else:
            tissue_mask_fs.extend(t_f)

    for t_f in tissue_mask_fs:
        t_im = nib.load(t_f.resolve())
        tissue_mask = tissue_mask | t_im.get_fdata().astype(bool)

    if len(p.shape) == 4:
        mask = mask[..., None]
        tissue_mask = tissue_mask[..., None]

    fluid_mask = mask & ~tissue_mask

    outer_shell_radius = args.radius_select_outer
    outer_shell_mask = mask ^ (
        skimage.morphology.binary_erosion(
            mask, skimage.morphology.ball(outer_shell_radius)
        )
    )
    outer_shell_fluid_mask = outer_shell_mask & fluid_mask

    fluid = p[fluid_mask]
    q_low = args.lower_quantile
    p_min = np.quantile(fluid, q_low)
    q_high = args.upper_quantile
    p_max = np.quantile(fluid, q_high)

    p_outer_fluid = p * outer_shell_fluid_mask
    valid_p_outer_fluid_mask = (p_outer_fluid >= p_min) & (p_outer_fluid <= p_max)
    invalid_p_outer_fluid_mask = outer_shell_fluid_mask & ~valid_p_outer_fluid_mask

    scale_min = p_outer_fluid.min()
    scale_max = p_outer_fluid.max() - scale_min
    p_of_int = p_outer_fluid - scale_min
    p_of_int = (p_of_int / scale_max) * np.iinfo(np.uint16).max
    p_of_int = np.round(p_of_int).astype(np.uint16)
    p_of_int_med = skimage.filters.rank.median(
        p_of_int, footprint=skimage.morphology.ball(2), mask=outer_shell_fluid_mask
    )
    p_of_median = (
        (p_of_int_med.astype(p_outer_fluid.dtype) / np.iinfo(np.uint16).max) * scale_max
    ) + scale_min

    p_out = np.where(invalid_p_outer_fluid_mask, p_of_median, p)

    nib.save(nib.Nifti1Image(p_out, p_im.affine, p_im.header), args.output.resolve())

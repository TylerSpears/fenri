#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split freesurfer segmentations into a set of binary masks according to the LUT"
    )

    parser.add_argument(
        "-s",
        "--fs_seg",
        type=Path,
        required=True,
        help="Freesurfer segmentation map file, must be an integer datatype (required)",
    )
    parser.add_argument(
        "-l",
        "--lut",
        type=Path,
        required=True,
        help="Freesurfer segmentation .txt lookup table (required)",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        type=Path,
        help="Output directory for all binary masks (required)",
    )

    args = parser.parse_args()

    seg_im = nib.Nifti1Image.from_image(nib.load(args.fs_seg))
    try:
        seg_dtype = np.iinfo(seg_im.get_data_dtype()).dtype
    except ValueError as e:
        raise ValueError(
            f"ERROR: volume dtype {seg_im.get_data_dtype()} is not an integer type"
        )
    seg = seg_im.get_fdata().astype(seg_im.get_data_dtype())

    lut_f = args.lut
    out_dir = args.output_dir
    out_dir.mkdir(exist_ok=True, parents=True)

    # Read LUT file
    lut = pd.read_table(
        lut_f, sep="\s+", comment="#", skip_blank_lines=True, header=None
    )
    lut = lut.set_axis(["No.", "Label Name", "R", "G", "B", "A"], axis=1)
    max_label_num_len = len(str(int(seg.max())))

    # Iterate over all unique segmentation indices.
    for l in np.unique(seg):
        l = int(l)
        if l == 0:
            continue
        # Correct labels that were removed from duplication. See the freesurfer standard
        # LUT for details.
        elif l == 75:
            l = 4
        elif l == 76:
            l = 76
        new_im = nib.Nifti1Image(seg == l, seg_im.affine, header=seg_im.header)
        new_im.set_data_dtype(np.uint8)

        readable_l = lut.loc[lut["No."] == l]["Label Name"].array[0]
        readable_l = str(readable_l).strip().replace("*", "")
        # Append index number to the front of the filename.
        l_fname = "_".join([str(l).zfill(max_label_num_len), readable_l])
        l_fname = ".".join([l_fname, "nii", "gz"])
        nib.save(new_im, str(out_dir / l_fname))
        print(readable_l, end=" | ")
    print("", flush=True)

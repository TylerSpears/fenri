#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import textwrap
from pathlib import Path

import dipy
import dipy.io
import nibabel as nib
import numpy as np
import pandas as pd
import torch

import pitn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate seed points based on the midpoint of given streamlines"
    )
    parser.add_argument(
        "tck",
        type=Path,
        help=".tck file containing streamline coordinates",
    )
    parser.add_argument(
        "ref",
        type=Path,
        help="Reference NIFTI image matching the real coordinates in the input tracks",
    )
    parser.add_argument(
        "output",
        type=Path,
        help=".csv output file with spatial and seed angle coordinates",
    )
    args = parser.parse_args()

    ref_im = nib.load(args.ref)

    print(f"Midpoint seeds & directions of streamlines found in {args.tck}")
    print(f"Reference space in {args.ref}")

    tractogram = dipy.io.streamline.load_tck(
        args.tck,
        reference=ref_im.header,
        to_space=dipy.io.stateful_tractogram.Space.RASMM,
    )
    seed_table = dict(
        x=list(),
        y=list(),
        z=list(),
        theta=list(),
        phi=list(),
    )

    for s in tractogram.streamlines:
        step_lens = np.linalg.norm(np.diff(s, axis=-1), ord=2, axis=-1)
        cumul_sum_lens = np.cumsum(step_lens)
        mid_len = cumul_sum_lens.max() / 2
        midpoint_idx = np.argmin(np.abs(mid_len - cumul_sum_lens)).flatten().item() + 1
        xyz = s[midpoint_idx]
        x, y, z = tuple(xyz.tolist())

        # midpoint_coord + peak_forward is the coord at midpoint_index + 1
        peak_forward = s[midpoint_idx + 1] - s[midpoint_idx]
        # midpoint_coord + peak_back is the coord at midpoint_index - 1
        peak_back = s[midpoint_idx - 1] - s[midpoint_idx]

        peaks_xyz = torch.from_numpy(np.stack([peak_forward, peak_back], axis=0))
        peaks_theta, peaks_phi = pitn.tract.xyz2unit_sphere_theta_phi(peaks_xyz)

        for i in range(2):
            peak_theta = peaks_theta[i].numpy().item()
            peak_phi = peaks_phi[i].numpy().item()
            seed_table["x"].append(x)
            seed_table["y"].append(y)
            seed_table["z"].append(z)
            seed_table["theta"].append(peak_theta)
            seed_table["phi"].append(peak_phi)

    seed_table = pd.DataFrame.from_dict(seed_table)

    affine_vox2real = ref_im.affine
    # Save some information about the original seed mask/space.
    csv_preamble = f"""
    # Real-space seed coordinates from streamlines
    # Streamline file '{str(args.tck)}'
    # Spatial reference from file '{str(args.ref)}'
    # voxel FOV shape {str(tuple(ref_im.shape[:-1])).replace(',', '')}
    # affine vox to real space:
    # Row 1 {str(affine_vox2real[0].tolist()).replace(',', '')}
    # Row 2 {str(affine_vox2real[1].tolist()).replace(',', '')}
    # Row 3 {str(affine_vox2real[2].tolist()).replace(',', '')}
    # Row 4 {str(affine_vox2real[3].tolist()).replace(',', '')}"""

    csv_preamble = textwrap.dedent(csv_preamble).strip()
    with open(args.output, "wt") as f:
        f.write(csv_preamble)
        f.write("\n")

    seed_table.to_csv(args.output, index=False, sep=",", float_format="%g", mode="a")

    print(f"Saved coordinates to {args.output}")

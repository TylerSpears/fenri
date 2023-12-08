#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import collections
import functools
import textwrap
from pathlib import Path

import einops
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from box import Box

import pitn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform tractography using trilinear interpolation of ODFs"
    )
    parser.add_argument(
        "mask",
        type=Path,
        help="NIFTI file that contains the mask for seed point creation",
    )
    parser.add_argument(
        "seeds_per_vox",
        type=int,
        help="Number of seed points within each voxel per dimension",
    )
    parser.add_argument(
        "output_seeds",
        type=Path,
        help="Output seeds .csv file that contains seed x, y, z real coordinates",
    )
    args = parser.parse_args()

    mask_im = nib.load(args.mask)

    mask = torch.from_numpy(mask_im.get_fdata().astype(bool))
    if mask.ndim == 4:
        mask.squeeze_(-1)

    affine_vox2real = torch.from_numpy(mask_im.affine)

    vol_real_coords = pitn.affine.affine_coordinate_grid(affine_vox2real, mask)
    select_real_coords = vol_real_coords[mask.bool()].to(torch.float64)
    del vol_real_coords

    if args.seeds_per_vox == 1:
        output_real_coords = select_real_coords
    else:
        select_vox_coords = pitn.affine.transform_coords(
            select_real_coords, pitn.affine.inv_affine(affine_vox2real)
        )
        del select_real_coords

        # Create voxel coordinate offsets to broadcast over the selected mask coords.
        vox_offsets = torch.stack(
            torch.meshgrid(
                [torch.linspace(-0.5, 0.5, steps=args.seeds_per_vox + 2)] * 3,
                indexing="ij",
            ),
            dim=-1,
        )
        # Take off voxel borders
        vox_offsets = vox_offsets[1:-1, 1:-1, 1:-1]
        vox_offsets = vox_offsets.reshape(1, -1, 3)
        expanded_vox_coords = select_vox_coords.unsqueeze(1) + vox_offsets
        del select_vox_coords
        expand_vox_coords = einops.rearrange(
            expanded_vox_coords,
            "offsets n_select coords -> (offsets n_select) coords",
        )
        output_real_coords = pitn.affine.transform_coords(
            expand_vox_coords, affine_vox2real
        )
        del expand_vox_coords

    output_real_coords = output_real_coords.cpu().numpy()
    output_real_coords = pd.DataFrame(output_real_coords, columns=["x", "y", "z"])

    print(f"Saving {output_real_coords.shape[0]} coordinates")

    # Try to save some information about the original seed mask/space.
    csv_preamble = f"""
    # Real-space seed coordinates
    # From mask file '{args.mask.name}'
    # voxel FOV shape {str(tuple(mask.shape[-3:])).replace(',', '')}
    # affine vox to real space:
    # Row 1 {str(affine_vox2real[0].tolist()).replace(',', '')}
    # Row 2 {str(affine_vox2real[1].tolist()).replace(',', '')}
    # Row 3 {str(affine_vox2real[2].tolist()).replace(',', '')}
    # Row 4 {str(affine_vox2real[3].tolist()).replace(',', '')}"""

    csv_preamble = textwrap.dedent(csv_preamble).strip()
    with open(args.output_seeds, "wt") as f:
        f.write(csv_preamble)
        f.write("\n")

    output_real_coords.to_csv(
        args.output_seeds, index=False, sep=",", float_format="%g", mode="a"
    )

    print(f"Saved coordinates to {args.output_seeds}")

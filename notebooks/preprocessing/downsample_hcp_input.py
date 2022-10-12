# -*- coding: utf-8 -*-
import shutil
from pathlib import Path
from typing import Tuple

import einops
import monai
import nibabel as nib
import numpy as np

import pitn


def downsample_hcp_input(
    hcp_diffusion_dir: Path,
    target_vox_size: Tuple[float],
    out_diffusion_dir: Path,
    b0_max: int = 100,
) -> Path:

    # Follow the standard HCP folder layout.
    dwi_f = hcp_diffusion_dir / "data.nii.gz"
    bval_f = hcp_diffusion_dir / "bvals"
    bvec_f = hcp_diffusion_dir / "bvecs"
    mask_f = hcp_diffusion_dir / "nodif_brain_mask.nii.gz"

    assert dwi_f.exists()
    assert bval_f.exists()
    assert bvec_f.exists()
    assert mask_f.exists()

    dwi = nib.load(dwi_f)
    mask = nib.load(mask_f)
    bval = np.loadtxt(bval_f)
    bvec = np.loadtxt(bvec_f)

    # Remove ~half of the b0s.
    b0_idx = np.where(bval <= b0_max)
    b0_idx = (b0_idx[0][::2],)
    b0_mask = np.ones_like(bval).astype(bool)
    b0_mask[b0_idx] = False
    # Remove all b2000s
    b2000_mask = (bval <= (1900 - 100)) | ((bval >= 2100))
    bval_mask = b0_mask & b2000_mask

    bval = bval[bval_mask]
    bvec = bvec[:, bval_mask]
    dwi_im = dwi.get_fdata()[..., bval_mask]
    dwi_im = einops.rearrange(dwi_im, "x y z c -> c x y z")

    mask_down_im, mask_down_aff = pitn.data.preproc.dwi.downsample_vol_voxwise(
        mask.get_fdata().astype(np.uint8)[None, ...],
        mask.affine,
        target_vox_size=target_vox_size,
        interp_mode="nearest-exact",
    )

    dwi_down_im, dwi_down_aff = pitn.data.preproc.dwi.downsample_vol_voxwise(
        dwi_im, dwi.affine, target_vox_size=target_vox_size, interp_mode="area"
    )

    # Crop both the DWI and the mask with the downsampled mask.
    crop_tf = monai.transforms.CropForeground(
        select_fn=lambda x: mask_down_im > 0, margin=2, k_divisible=2
    )
    meta_dwi = monai.data.MetaTensor(dwi_down_im, affine=dwi_down_aff)
    crop_meta_dwi = crop_tf(meta_dwi)
    dwi_down_crop_im = crop_meta_dwi.array.copy()
    dwi_down_crop_aff = crop_meta_dwi.affine.cpu().numpy().copy()

    meta_mask = monai.data.MetaTensor(mask_down_im, affine=mask_down_aff)
    crop_meta_mask = crop_tf(meta_mask)
    mask_down_crop_im = crop_meta_mask.array.copy()
    mask_down_crop_aff = crop_meta_mask.affine.cpu().numpy().copy()

    dwi_down_crop_im = einops.rearrange(dwi_down_crop_im, "c x y z -> x y z c")
    dwi_down = dwi.__class__(dwi_down_crop_im.astype(np.float32), dwi_down_crop_aff)
    dwi_down.set_data_dtype(np.float32)
    mask_down_crop_im = mask_down_crop_im[0]
    mask_down = mask.__class__(mask_down_crop_im.astype(np.uint8), mask_down_crop_aff)
    mask_down.set_data_dtype(np.uint8)

    out_dwi_f = out_diffusion_dir / "data.nii.gz"
    out_bval_f = out_diffusion_dir / "bvals"
    out_bvec_f = out_diffusion_dir / "bvecs"
    out_mask_f = out_diffusion_dir / "nodif_brain_mask.nii.gz"

    nib.save(dwi_down, out_dwi_f)
    nib.save(mask_down, out_mask_f)
    np.savetxt(out_bval_f, bval, fmt="%g")
    np.savetxt(out_bvec_f, bvec, fmt="%g")

    return out_diffusion_dir


if __name__ == "__main__":

    target_vox_size = (2.0, 2.0, 2.0)
    b0_max = 100
    hcp_root_dir = Path("/data/srv/data/pitn/hcp")
    output_root_dir = Path("/data/srv/outputs/pitn/hcp/downsample/scale-2.00mm/vol")
    ids_file = Path("../data/HCP_unique_ids.txt").resolve()

    with open(ids_file, "r") as f:
        subj_ids = list(map(lambda x: str(x).strip(), f.readlines()))

    for sid in subj_ids:

        print(f"\n====Starting subject {sid}.====\n")

        src_dir = hcp_root_dir / sid / "T1w" / "Diffusion"
        target_dir = output_root_dir / sid / "T1w" / "Diffusion"
        target_dir.mkdir(exist_ok=True, parents=True)

        dwi_f = target_dir / "data.nii.gz"
        bval_f = target_dir / "bvals"
        bvec_f = target_dir / "bvecs"
        mask_f = target_dir / "nodif_brain_mask.nii.gz"
        if dwi_f.exists() and bval_f.exists() and bvec_f.exists() and mask_f.exists():
            print(
                f"\n====Downsampled files found in {str(target_dir)} for subject {sid}",
                "skipping downsampling.====\n",
            )
            continue

        downsample_hcp_input(
            src_dir,
            target_vox_size=target_vox_size,
            out_diffusion_dir=target_dir,
            b0_max=b0_max,
        )

        # Save this code to the output directory
        code_dir = output_root_dir / sid / "code"
        code_dir.mkdir(exist_ok=True, parents=True)
        shutil.copy2(__file__, code_dir / Path(__file__).name)
        shutil.copy2(
            pitn.data.preproc.dwi.__file__,
            code_dir / Path(pitn.data.preproc.dwi.__file__).name,
        )
        print(f"\n====Completed subject {sid}.====\n")

# -*- coding: utf-8 -*-
import shutil
import textwrap
from pathlib import Path
from typing import Tuple

import einops
import monai
import nibabel as nib
import numpy as np

import pitn


def mrtrix_fit_fodf(
    dwi_f: Path,
    bval_f: Path,
    bvec_f: Path,
    freesurfer_seg_f: Path,
    target_fodf_f: Path,
    n_threads: int = 4,
) -> Path:

    docker_img = "mrtrix3/mrtrix3:3.0.3"

    dwi_f = Path(dwi_f).resolve()
    bval_f = Path(bval_f).resolve()
    bvec_f = Path(bvec_f).resolve()
    freesurfer_seg_f = Path(freesurfer_seg_f).resolve()
    target_dir = Path(target_fodf_f).parent.resolve()

    dwi_mif_f = target_dir / "_tmp.dwi.mif"
    five_tt_parc_f = target_dir / "5tt_parcellation.nii.gz"

    # It's unclear what the default lmax is for dwi2response, and there's a lot of
    # discussion about l_max:
    # <https://mrtrix.readthedocs.io/en/latest/concepts/sh_basis_lmax.html>
    # <https://github.com/MRtrix3/mrtrix3/pull/786> for a discussion about using only
    # zonal harmonics (m=0 for each even l) to estimate the tissue response functions.
    # Looking at other data, it seems that for the actual fod estimation, the
    # white matter gets an l_max of 8, while grey matter and CSF get an l_max of 0.
    # This seems overly restrictive on grey matter, but I'm not the expert here...
    script = rf"""\
    set -eou pipefail
    mrconvert -info -fslgrad \
        {bvec_f} \
        {bval_f} \
        {dwi_f} \
        {dwi_mif_f}
    5ttgen -info freesurfer -nthreads {n_threads} \
        {freesurfer_seg_f} \
        {five_tt_parc_f} \
        -force
    dwi2response msmt_5tt -info \
        -wm_algo tournier \
        -mask {mask_f} \
        {dwi_mif_f} \
        {five_tt_parc_f} \
        {target_dir / "wm_response.txt"} \
        {target_dir / "gm_response.txt"} \
        {target_dir / "csf_response.txt"}
    dwi2fod -info -nthreads {n_threads} \
        -lmax 8,4,0 \
        -niter 100 \
        -mask {mask_f} \
        msmt_csd \
        {dwi_mif_f} \
        {target_dir / "wm_response.txt"} {target_fodf_f} \
        {target_dir / "gm_response.txt"} {target_dir / "gm_msmt_csd_fod.nii.gz"} \
        {target_dir / "csf_response.txt"} {target_dir / "csf_msmt_csd_fod.nii.gz"} \
        -force
    rm {dwi_mif_f}
    """

    script = textwrap.dedent(script)
    script = pitn.utils.proc_runner.multiline_script2docker_cmd(script)
    vols = pitn.utils.union_parent_dirs(
        dwi_f, bval_f, bvec_f, freesurfer_seg_f, target_dir
    )
    vols = {v: pitn.utils.proc_runner.get_docker_mount_obj(v) for v in vols}
    docker_config = dict(volumes=vols)

    fodf_run_status = pitn.utils.proc_runner.call_docker_run(
        docker_img, cmd=script, run_config=docker_config
    )

    return target_fodf_f


def postproc_ground_truth(
    wm_fodf_f: Path,
    dwi_mask_f: Path,
    fivett_mask_f: Path,
    freesurfer_aseg_f: Path,
    out_fodf_f: Path,
    out_nodif_mask_f: Path,
    out_5tt_f: Path,
    out_freesurfer_aseg_f: Path,
    sh_coeff_min: float = 1e-8,
) -> Tuple[Path]:

    wm = nib.load(wm_fodf_f)
    wm_data = einops.rearrange(wm.get_fdata().astype(np.float32), "x y z c -> c x y z")
    # Threshold fodfs to remove non-negative and very small spherical harmonic
    # coefficients.
    wm_data[wm_data < sh_coeff_min] = 0.0

    ftt_mask = nib.load(fivett_mask_f)
    ftt_mask.set_data_dtype(np.uint8)
    fs_labels = nib.load(freesurfer_aseg_f)
    fs_labels.set_data_dtype(np.int32)
    dwi_mask = nib.load(dwi_mask_f)
    dwi_mask.set_data_dtype(np.uint8)

    # Resample transform into DWI space.
    resample_tf = monai.transforms.ResampleToMatch(mode="nearest", align_corners=True)
    dst_tensor = monai.data.MetaTensor(
        wm_data[0][None],
        affine=wm.affine,
        meta={monai.utils.misc.ImageMetaKey.FILENAME_OR_OBJ: str(wm_fodf_f)},
    )

    # Apply resample transform to 5tt and freesurfer segmentation vols.
    ftt_mask_data = einops.rearrange(
        ftt_mask.get_fdata().astype(np.uint8), "x y z c -> c x y z"
    )
    ftt_tensor = monai.data.MetaTensor(
        ftt_mask_data,
        affine=ftt_mask.affine,
        meta={monai.utils.misc.ImageMetaKey.FILENAME_OR_OBJ: str(fivett_mask_f)},
    )
    ftt_tensor = resample_tf(
        ftt_tensor,
        img_dst=dst_tensor,
    )

    fs_label_data = fs_labels.get_fdata().astype(np.int32)[None]
    fs_tensor = monai.data.MetaTensor(
        fs_label_data,
        affine=fs_labels.affine,
        meta={monai.utils.misc.ImageMetaKey.FILENAME_OR_OBJ: str(freesurfer_aseg_f)},
    )
    fs_tensor = resample_tf(
        fs_tensor,
        img_dst=dst_tensor,
    )

    # Build crop by the DWI mask transform.
    dwi_mask_data = dwi_mask.get_fdata().astype(np.uint8)[None]
    crop_tf = monai.transforms.CropForeground(
        select_fn=lambda x: dwi_mask_data > 0, margin=2, k_divisible=2
    )
    # All vols need to be cropped.
    # Start with the fodf volume.
    wm_tensor = monai.data.MetaTensor(
        wm_data,
        affine=wm.affine,
        meta={monai.utils.misc.ImageMetaKey.FILENAME_OR_OBJ: str(wm_fodf_f)},
    )
    wm_tensor = crop_tf(wm_tensor)

    # 5tt map
    ftt_tensor = crop_tf(ftt_tensor)
    # Freesurfer labels
    fs_tensor = crop_tf(fs_tensor)

    # Finally, the DWI mask
    dwi_mask_tensor = monai.data.MetaTensor(
        dwi_mask_data,
        affine=dwi_mask.affine,
        meta={monai.utils.misc.ImageMetaKey.FILENAME_OR_OBJ: str(dwi_mask_f)},
    )
    dwi_mask_tensor = crop_tf(dwi_mask_tensor)

    # Extract and save out processed vols.
    proc_wm = wm.__class__(
        einops.rearrange(wm_tensor.array.astype(np.float32), "c x y z -> x y z c"),
        affine=wm_tensor.affine.cpu().numpy(),
        header=wm.header,
    )
    proc_5tt = ftt_mask.__class__(
        einops.rearrange(ftt_tensor.array.astype(np.uint8), "c x y z -> x y z c"),
        affine=ftt_tensor.affine.cpu().numpy(),
        header=ftt_mask.header,
    )
    proc_fs = fs_labels.__class__(
        fs_tensor.array.astype(np.int32)[0],
        affine=fs_tensor.affine.cpu().numpy(),
        header=fs_labels.header,
    )
    proc_dwi_mask = dwi_mask.__class__(
        dwi_mask_tensor.array.astype(np.uint8)[0],
        affine=dwi_mask_tensor.affine.cpu().numpy(),
        header=dwi_mask.header,
    )
    nib.save(proc_wm, out_fodf_f)
    nib.save(proc_dwi_mask, out_nodif_mask_f)
    nib.save(proc_5tt, out_5tt_f)
    nib.save(proc_fs, out_freesurfer_aseg_f)

    return out_fodf_f, out_nodif_mask_f, out_5tt_f, out_freesurfer_aseg_f


if __name__ == "__main__":

    # hcp_root_dir = Path("/data/srv/data/pitn/hcp")
    # output_root_dir = Path("/data/srv/outputs/pitn/hcp/full-res/fodf")
    ids_file = Path("/home/tas6hh/Projects/pitn/notebooks/data/HCP_unique_ids.txt")
    hcp_root_dir = Path("/data/srv/outputs/pitn/hcp/downsample/scale-2.00mm/vol")
    output_root_dir = Path("/data/srv/outputs/pitn/hcp/downsample/scale-2.00mm/fodf")

    with open(ids_file, "r") as f:
        subj_ids = list(map(lambda x: str(x).strip(), f.readlines()))

    for sid in subj_ids:

        print(f"====Starting subject {sid}.====")

        src_dir = hcp_root_dir / sid / "T1w"
        target_dir = output_root_dir / sid / "T1w"
        target_dir.mkdir(exist_ok=True, parents=True)

        dwi_f = src_dir / "Diffusion" / "data.nii.gz"
        bval_f = src_dir / "Diffusion" / "bvals"
        bvec_f = src_dir / "Diffusion" / "bvecs"
        mask_f = src_dir / "Diffusion" / "nodif_brain_mask.nii.gz"
        freesurfer_seg_f = src_dir / "aparc.a2009s+aseg.nii.gz"
        target_postproc_fodf_f = target_dir / "postproc_wm_msmt_csd_fod.nii.gz"

        # Determine if all processing can be skipped.
        if target_postproc_fodf_f.exists():
            print(
                "====Processed fODF coefficients found in",
                f"{target_dir} for subject {sid}",
                "skipping subject.====\n",
            )
            continue

        target_fodf_f = target_dir / "wm_msmt_csd_fod.nii.gz"
        # Determine if fodf estimation with mrtrix can be skipped.
        if target_fodf_f.exists():
            print(
                f"====fODF coefficients found in {target_dir} for subject {sid},",
                "skipping fodf estimation.====\n",
            )
        else:
            out_fodf = mrtrix_fit_fodf(
                dwi_f,
                bval_f=bval_f,
                bvec_f=bvec_f,
                freesurfer_seg_f=freesurfer_seg_f,
                target_fodf_f=target_fodf_f,
                n_threads=17,
            )

        # Post-process the WM fodf and associated label/mask files.
        fivett_f = target_dir / "5tt_parcellation.nii.gz"
        (
            postproc_fodf_f,
            postproc_dwi_mask_f,
            postproc_5tt_f,
            postproc_freesurfer_f,
        ) = postproc_ground_truth(
            wm_fodf_f=target_fodf_f,
            dwi_mask_f=mask_f,
            fivett_mask_f=fivett_f,
            freesurfer_aseg_f=freesurfer_seg_f,
            out_fodf_f=target_postproc_fodf_f,
            out_5tt_f=target_dir / "postproc_5tt_parcellation.nii.gz",
            out_nodif_mask_f=target_dir / "postproc_nodif_brain_mask.nii.gz",
            out_freesurfer_aseg_f=target_dir / ("postproc_" + freesurfer_seg_f.name),
        )

        # Save this code to the output directory
        code_dir = output_root_dir / sid / "code"
        code_dir.mkdir(exist_ok=True, parents=True)
        shutil.copy2(__file__, code_dir / Path(__file__).name)

        print(f"====Completed subject {sid}.====\n")

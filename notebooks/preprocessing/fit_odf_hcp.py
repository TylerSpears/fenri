# -*- coding: utf-8 -*-
import shlex
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


if __name__ == "__main__":

    hcp_root_dir = Path("/data/srv/data/pitn/hcp")
    output_root_dir = Path("/data/srv/outputs/pitn/hcp/full-res/fodf")
    ids_file = Path("../data/HCP_unique_ids.txt").resolve()

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
        target_fodf_f = target_dir / "wm_msmt_csd_fod.nii.gz"

        if target_fodf_f.exists():
            print(
                f"====fODF coefficients found in {target_dir} for subject {sid}",
                "skipping.====\n",
            )
            continue

        out_fodf = mrtrix_fit_fodf(
            dwi_f,
            bval_f=bval_f,
            bvec_f=bvec_f,
            freesurfer_seg_f=freesurfer_seg_f,
            target_fodf_f=target_fodf_f,
            n_threads=19,
        )

        # Save this code to the output directory
        code_dir = output_root_dir / sid / "code"
        code_dir.mkdir(exist_ok=True, parents=True)
        shutil.copy2(__file__, code_dir / Path(__file__).name)

        print(f"====Completed subject {sid}.====\n")

# -*- coding: utf-8 -*-
import gzip
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import nibabel as nib
import numpy as np
import scipy
from redun import File, task

import pitn

if __package__ is not None:
    redun_namespace = str(__package__)


@task(
    check_valid="shallow",
    config_args=["tmp_dir", "script_exec_config"],
)
def bvec_flip_correct(
    dwi_data: np.ndarray,
    dwi_affine: np.ndarray,
    bval: np.ndarray,
    bvec: np.ndarray,
    tmp_dir: str,
    mask: Optional[np.ndarray] = None,
    script_exec_config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:

    dwi = nib.Nifti1Image(dwi_data, affine=dwi_affine)
    d = Path(tmp_dir)
    dwi_f = str(d / "dwi_data.nii.gz")
    nib.save(dwi, dwi_f)
    bval_f = str(d / "bval")
    np.savetxt(bval_f, bval)
    bvec_f = str(d / "bvec")
    np.savetxt(bvec_f, bvec)

    src_output_f = str(d / "dwi_data.src.gz")
    src_dwi = pitn.redun.dsi_studio.gen_src(
        source=File(dwi_f),
        output=src_output_f,
        bval=File(bval_f),
        bvec=File(bvec_f),
        log_stdout=False,
        script_exec_config=script_exec_config,
    )
    preproc_src_dwi_f = str(d / "dwi_data_preproc.src.gz")

    # If a mask is present, save out to a nifti file and pass to the recon command.
    if mask is not None:
        mask_f = File(str(d / "dwi_mask.nii.gz"))
        nib.save(nib.Nifti1Image(mask, affine=dwi_affine), mask_f.path)
    else:
        mask_f = None

    recon_files = pitn.redun.dsi_studio.recon(
        source=src_dwi,
        mask=mask_f,
        method="DTI",
        check_btable=True,
        save_src=preproc_src_dwi_f,
        align_acpc=False,
        other_output="md",
        record_odf=False,
        log_stdout=False,
        script_exec_config=script_exec_config,
    )
    preproc_src_dwi = recon_files["preproc"]

    corrected_btable = _extract_btable(preproc_src_dwi)
    corrected_bvec = _extract_bvec(corrected_btable)
    return corrected_bvec


@task(cache=False)
def _extract_btable(btable_f: File) -> np.ndarray:
    with gzip.open(btable_f.path, "r") as f:
        btable = dict(scipy.io.loadmat(f))["b_table"]
    return btable


@task(cache=False)
def _extract_bvec(btable: np.ndarray) -> np.ndarray:
    return btable[1:]


@task()
def eddy_apply_params():
    pass

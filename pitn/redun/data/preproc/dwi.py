# -*- coding: utf-8 -*-
import gzip
import os
from pathlib import Path
from typing import Any, Dict, Optional

import nibabel as nib
import numpy as np
import scipy
from redun import File, task

import pitn
from pitn.redun.utils import NDArrayValue, save_nib, save_np_txt

if __package__ is not None:
    redun_namespace = str(__package__)


@task(
    # check_valid="shallow",
    config_args=["tmp_dir", "script_exec_config"],
)
def bvec_flip_correct_files(
    dwi_f: File,
    bval_f: File,
    bvec_f: File,
    tmp_dir: str,
    mask_f: Optional[File] = None,
    script_exec_config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:

    d = Path(tmp_dir)
    # dwi_f_basename = Path(dwi_f.path).name.replace(
    #     "".join(Path(dwi_f.path).suffixes), ""
    # )
    # dwi_f_basename = str(dwi_f_basename)
    # bval_f = save_np_txt(str(d / f"{dwi_f_basename}.bval"), bval)
    # bvec_f = save_np_txt(str(d / f"{dwi_f_basename}.bvec"), bvec)
    # bvec_f = str(d / "bvec")
    # np.savetxt(bvec_f, bvec)

    src_output_f = str(d / "dwi_data.src.gz")
    src_dwi = pitn.redun.dsi_studio.gen_src.options(cache=False)(
        source=dwi_f,
        output=src_output_f,
        bval=bval_f,
        bvec=bvec_f,
        log_stdout=False,
        script_exec_config=script_exec_config,
    )
    preproc_src_dwi_f = str(d / "dwi_data_preproc.src.gz")

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


@task(
    check_valid="shallow",
    config_args=["tmp_dir", "script_exec_config"],
)
def bvec_flip_correct(
    bval: np.ndarray,
    bvec: np.ndarray,
    tmp_dir: str,
    dwi_f: Optional[File] = None,
    dwi_data: Optional[np.ndarray] = None,
    dwi_affine: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    script_exec_config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:

    if dwi_f is None:
        assert dwi_data is not None and dwi_affine is not None
    elif dwi_data is None or dwi_affine is None:
        assert dwi_f is not None
    else:
        raise ValueError("ERROR: Must only have one of {dwi_f, (dwi_data, dwi_affine)}")

    d = Path(tmp_dir)
    if dwi_f is None:
        dwi = nib.Nifti1Image(dwi_data, affine=dwi_affine)
        dwi_f = File(d / "dwi_data.nii.gz")
        nib.save(dwi, dwi_f.path)
    dwi_f_basename = Path(dwi_f.path).name.replace(
        "".join(Path(dwi_f.path).suffixes), ""
    )
    dwi_f_basename = str(dwi_f_basename)
    bval_f = save_np_txt(str(d / f"{dwi_f_basename}.bval"), bval)
    # np.savetxt(bval_f, bval)
    bvec_f = save_np_txt(str(d / f"{dwi_f_basename}.bvec"), bvec)
    # bvec_f = str(d / "bvec")
    # np.savetxt(bvec_f, bvec)

    src_output_f = str(d / "dwi_data.src.gz")
    src_dwi = pitn.redun.dsi_studio.gen_src(
        source=dwi_f,
        output=src_output_f,
        bval=bval_f,
        bvec=bvec_f,
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


@task(cache=False, hash_includes=[pitn.data.utils.least_distort_b0_idx])
def top_k_b0s(
    dwi: NDArrayValue,
    bval: np.ndarray,
    bvec: np.ndarray,
    n_b0s: int = 3,
    b0_max: float = 100,
) -> dict:
    b0_mask = bval <= b0_max
    b0s = dwi[..., b0_mask]
    b0_bvals = bval[b0_mask]
    b0_bvecs = bvec[:, b0_mask]
    top_b0s_idx = pitn.data.utils.least_distort_b0_idx(b0s, num_selections=n_b0s)
    output = dict(
        dwi=NDArrayValue(b0s[..., top_b0s_idx]),
        bval=b0_bvals[top_b0s_idx],
        bvec=b0_bvecs[:, top_b0s_idx],
    )

    return output


@task(config_args=["script_exec_config"])
def bet_mask_median_dwis(
    dwi_f: File,
    out_file: str,
    script_exec_config: Optional[Dict[str, Any]] = None,
    **bet_kwargs,
) -> File:
    dwi = nib.load(dwi_f.path)
    median_dwi_data = np.median(dwi.get_fdata(), axis=-1)
    out_path = Path(out_file).parent
    # dwi_path = Path(dwi_f.path)
    median_out_fname = "_tmp_median.nii.gz"
    median_out_path = out_path / median_out_fname
    median_dwi = nib.Nifti1Image(median_dwi_data, dwi.affine, dwi.header)
    median_dwi_f = save_nib(median_dwi, str(median_out_path))

    mask_out_basename = str(out_path / Path(out_file).name)

    bet_outputs = pitn.redun.fsl.bet(
        median_dwi_f,
        out_file_basename=mask_out_basename,
        mask=True,
        skip_brain_output=True,
        verbose=False,
        log_stdout=False,
        script_exec_config=script_exec_config,
        **bet_kwargs,
    )
    target_mask_f = File(out_file)
    mask_f = bet_outputs["mask"].copy_to(target_mask_f)
    # Copy the bet output mask to the given output location.
    mask_f = mask_f.copy
    # Delete the temporary median dwi.
    median_dwi_f.remove()
    return mask_f


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

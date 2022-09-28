# -*- coding: utf-8 -*-
import gzip
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import ants
import einops
import nibabel as nib
import numpy as np
import scipy
import SimpleITK as sitk

import pitn


def crop_nib_by_mask(nib_vol, mask):
    m = mask.astype(bool)
    min_extent = np.stack(np.where(m)).min(1)
    max_extent = np.stack(np.where(m)).max(1)
    slicer = tuple(
        slice(min_ext, max_ext + 1)
        for (min_ext, max_ext) in zip(min_extent, max_extent)
    )
    new_nib = nib_vol.slicer[slicer]
    return new_nib


def bvec_flip_correct(
    dwi_data: Optional[np.ndarray],
    dwi_affine: Optional[np.ndarray],
    bval: np.ndarray,
    bvec: np.ndarray,
    docker_img: str,
    mask: Optional[np.ndarray] = None,
    tmp_dir: Path = Path("/tmp/pitn_bvec_flip_correct"),
    docker_config: dict = dict(),
    docker_env: dict = dict(),
) -> np.ndarray:

    with tempfile.TemporaryDirectory(dir=tmp_dir) as t_dir:

        d = Path(t_dir)
        if "volumes" not in docker_config.keys():
            docker_config["volumes"] = dict()
        docker_config["volumes"][str(d)] = pitn.utils.proc_runner.get_docker_mount_obj(
            d, mode="rw"
        )

        dwi = nib.Nifti1Image(dwi_data, affine=dwi_affine)
        dwi_f = d / "dwi_data.nii.gz"
        nib.save(dwi, dwi_f)
        dwi_f_basename = Path(dwi_f).name.replace("".join(Path(dwi_f).suffixes), "")
        dwi_f_basename = str(dwi_f_basename)
        bval_f = d / f"{dwi_f_basename}.bval"
        bvec_f = d / f"{dwi_f_basename}.bvec"
        np.savetxt(bval_f, bval)
        np.savetxt(bvec_f, bvec)

        src_output_f = d / "dwi_data.src.gz"

        src_script = pitn.dsi_studio.src_cmd(
            source=dwi_f,
            output=src_output_f,
            bval=bval_f,
            bvec=bvec_f,
        )
        run_src_result = pitn.utils.proc_runner.call_docker_run(
            img=docker_img, cmd=src_script, env=docker_env, run_config=docker_config
        )

        preproc_src_dwi_f = d / "dwi_data_preproc.src.gz"

        # If a mask is present, save out to a nifti file and pass to the recon command.
        if mask is not None:
            mask_f = str(d / "dwi_mask.nii.gz")
            nib.save(nib.Nifti1Image(mask, affine=dwi_affine), mask_f)
        else:
            mask_f = None

        recon_script = pitn.dsi_studio.recon_cmd(
            source=src_output_f,
            mask=mask_f,
            method="DTI",
            check_btable=True,
            save_src=preproc_src_dwi_f,
            align_acpc=True,
            other_output="md",
            record_odf=False,
            thread_count=4,
        )
        run_recon_result = pitn.utils.proc_runner.call_docker_run(
            img=docker_img, cmd=recon_script, env=docker_env, run_config=docker_config
        )

        corrected_btable = _extract_btable(preproc_src_dwi_f)
        corrected_bvec = _extract_bvec(corrected_btable)

    return corrected_bvec


def _extract_btable(btable_f: Path) -> np.ndarray:
    with gzip.open(Path(btable_f), "r") as f:
        btable = dict(scipy.io.loadmat(f))["b_table"]
    return btable


def _extract_bvec(btable: np.ndarray) -> np.ndarray:
    return btable[1:]


# For use mainly with topup when sub-selecting AP/PA b0s, want the b0s with the lowest
# amount of head motion, i.e. the ones closest to the median b0.
def _least_distort_b0_idx(b0s: np.ndarray, num_selections=1, seed=None) -> np.ndarray:
    # Compare each b0 to the median of all b0s.
    median = np.median(b0s, axis=3).astype(b0s.dtype)
    sitk_median = sitk.GetImageFromArray(median)
    l_b0 = np.split(b0s, b0s.shape[-1], axis=3)
    sitk_mi = sitk.ImageRegistrationMethod()
    sitk_mi.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    if seed is not None:
        sitk_mi.SetMetricSamplingPercentage(
            float(sitk_mi.GetMetricSamplingPercentagePerLevel()[0]), seed=seed
        )
    sitk_mi.SetInterpolator(sitk.sitkNearestNeighbor)
    mis = list()
    # print("Mutual Information between b0s and median b0: ")
    sanity_mi = sitk_mi.MetricEvaluate(sitk_median, sitk_median)
    sanity_mi = -sanity_mi
    # print("MI between median and itself: ", sanity_mi)
    for b0 in l_b0:
        b0 = np.squeeze(b0)
        sitk_b0 = sitk.GetImageFromArray(b0)
        mi = sitk_mi.MetricEvaluate(sitk_median, sitk_b0)
        # MI is negated to work with a minimization optimization
        # (not neg log-likelihood?)
        mi = -mi
        mis.append(mi)
    # print(mis)
    # Sort from max to min for easier indexing.
    mi_order = np.flip(np.argsort(np.asarray(mis)))
    return mi_order[:num_selections]


def _b0_quick_rigid_reg(b0s: np.ndarray) -> np.ndarray:
    vox_isocenter = np.asarray(b0s.shape[1:]) / 2
    b0_com = np.stack([scipy.ndimage.center_of_mass(b0) for b0 in b0s], axis=0)
    b0_dists = [
        float(
            scipy.spatial.distance.pdist(
                np.stack([vox_isocenter, b0_com], axis=0), metric="euclidean"
            )
        )
        for b0_com in b0_com
    ]
    b0_close_to_isocenter_idx = b0_dists.index(min(b0_dists))

    # Perform ants rigid registration to the selected image.
    fixed_b0 = b0s[b0_close_to_isocenter_idx]
    fixed_b0 = ants.from_numpy(fixed_b0)
    list_b0s = list(b0s)
    moving_b0s = [ants.from_numpy(b0) for b0 in list_b0s]

    # Loop through all b0s.
    reg_b0s = list()
    for i, moving_b0 in enumerate(moving_b0s):
        # Ignore the chosen template b0, the registration would be at identity.
        if i == b0_close_to_isocenter_idx:
            reg_b0s.append(moving_b0.numpy())
            continue
        reg_results = ants.registration(
            fixed_b0, moving_b0, "Rigid", aff_metric="mattes"
        )
        reg_b0s.append(reg_results["warpedmovout"].numpy())

    return np.stack(reg_b0s, axis=0)


def top_k_b0s(
    dwi: np.ndarray,
    bval: np.ndarray,
    bvec: np.ndarray,
    n_b0s: int = 3,
    b0_max: float = 50,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    b0_mask = bval <= b0_max
    b0s = dwi[..., b0_mask]
    b0_bvals = bval[b0_mask]
    b0_bvecs = bvec[:, b0_mask]

    # Rigidly register b0s together for a more fair comparison.
    reg_b0s = _b0_quick_rigid_reg(einops.rearrange(b0s, "x y z b -> b x y z"))
    top_b0s_idx = _least_distort_b0_idx(reg_b0s, num_selections=n_b0s, seed=seed)
    output = dict(
        dwi=b0s[..., top_b0s_idx],
        bval=b0_bvals[top_b0s_idx],
        bvec=b0_bvecs[:, top_b0s_idx],
    )

    return output


def bet_mask_median_dwis(
    dwi_data: np.ndarray,
    dwi_affine: np.ndarray,
    # out_file: Path,
    docker_img: str,
    tmp_dir: Path = Path("/tmp/bet_mask_median_dwis"),
    docker_config: dict = dict(),
    docker_env: dict = dict(),
    **bet_kwargs,
) -> np.ndarray:

    with tempfile.TemporaryDirectory(dir=tmp_dir) as t_dir:

        d = Path(t_dir)
        if "volumes" not in docker_config.keys():
            docker_config["volumes"] = dict()
        docker_config["volumes"][str(d)] = pitn.utils.proc_runner.get_docker_mount_obj(
            d, mode="rw"
        )

        median_dwi_data = np.median(dwi_data, axis=-1)
        # dwi_path = Path(dwi_f.path)
        median_out_fname = "_tmp_median.nii.gz"
        median_out_path = d / median_out_fname
        # median_dwi = nib.Nifti1Image(median_dwi_data, dwi.affine, dwi.header)
        median_dwi = nib.Nifti1Image(median_dwi_data, dwi_affine)
        nib.save(median_dwi, str(median_out_path))

        mask_out_basename = pitn.utils.cli_parse.file_basename(median_out_path)

        bet_script = pitn.fsl.bet(
            median_out_path,
            out_file_basename=mask_out_basename,
            mask=True,
            skip_brain_output=True,
            verbose=False,
            **bet_kwargs,
        )

        run_bet_result = pitn.utils.proc_runner.call_docker_run(
            img=docker_img, cmd=bet_script, env=docker_env, run_config=docker_config
        )
        mask_f = pitn.utils.system.get_glob_file_unique(d, "*_mask_*")
        mask = nib.load(str(mask_f)).get_fdata().astype(np.uint8).copy()

    return mask

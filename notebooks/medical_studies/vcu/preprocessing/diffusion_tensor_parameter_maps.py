# -*- coding: utf-8 -*-
import concurrent
import datetime
import itertools
import os
from pathlib import Path

import dipy.core.gradients
import dipy.core.sphere as dps
import dipy.reconst.dki
import dipy.reconst.dti
import dipy.reconst.msdki
import nibabel as nib
import numpy as np
from dipy.reconst.dti import lower_triangular, mean_diffusivity
from dipy.reconst.vec_val_sum import vec_val_vect

import pitn

N_CPUS = 18

MIN_DIFFUSE = 0.0

# Theoretical min kurtosis of water.
MIN_KURTOSIS = -0.3 / 7
MAX_KURTOSIS = 10.0


def parallel_kurtosis_maximum(
    dki_params, sphere="repulsion100", gtol=1e-2, mask=None, max_workers=None
):
    shape = dki_params.shape[:-1]

    # load gradient directions
    if not isinstance(sphere, dps.Sphere):
        sphere = dipy.data.get_sphere("repulsion100")

    # select voxels where to find fiber directions
    if mask is None:
        mask = np.ones(shape, dtype="bool")
    else:
        if mask.shape != shape:
            raise ValueError("Mask is not the same shape as dki_params.")

    evals, evecs, kt = dipy.reconst.dki.split_dki_param(dki_params)

    # select non-zero voxels
    pos_evals = dipy.reconst.dki._positive_evals(
        evals[..., 0], evals[..., 1], evals[..., 2]
    )
    mask = np.logical_and(mask, pos_evals)

    dt_flat = lower_triangular(vec_val_vect(evecs, evals))[mask]
    md_flat = mean_diffusivity(evals)[mask]
    kt_flat = kt[mask]
    kt_max_flat = list()

    n_vox = len(md_flat)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        kt_max_results = executor.map(
            dipy.reconst.dki._voxel_kurtosis_maximum,
            dt_flat,
            md_flat,
            kt_flat,
            itertools.repeat(sphere, n_vox),
            itertools.repeat(gtol, n_vox),
            chunksize=n_vox // 100,
        )
        curr_vox_i = 0
        for res in kt_max_results:
            kt_max_flat.append(res[0].item())
            curr_vox_i += 1
            if curr_vox_i % (n_vox // 10) == 0:
                print(
                    f"{round((curr_vox_i/n_vox) * 100, 1)}% {curr_vox_i}/{n_vox} completed",
                    end=" | ",
                    flush=True,
                )
    print("")

    kt_max_flat = np.array(kt_max_flat)
    kt_max = np.zeros(mask.shape)
    kt_max[mask] = kt_max_flat
    return kt_max


def axonal_water_fraction(k_max: np.ndarray):
    return k_max / (k_max + 3)


if __name__ == "__main__":

    data_dir = Path("/data/srv/outputs/pitn/vcu/preproc")
    selected_subjs = (
        # "HC_01",
        "HC_02",
        "HC_03",
        "HC_04",
        "HC_05",
        "P_23",
        "P_03",
        "P_19",
        "P_27",
        "P_22",
        "P_21",
        "P_24",
        "P_13",
        "P_08",
        "P_33",
        "P_06",
        "P_01",
        "P_07",
        "P_05",
        "P_04",
        "P_26",
        "P_37",
        "P_25",
        "P_11",
        "P_20",
        "P_32",
        "P_02",
        "P_34",
        "P_14",
        "P_10",
        "P_29",
    )

    for subj_id in selected_subjs:
        t0 = datetime.datetime.now()
        print(
            f"{datetime.datetime.now().replace(microsecond=0)} {subj_id} |",
            "Starting subject",
        )

        dwi_dir = data_dir / subj_id / "diffusion" / "preproc" / "09_final"
        dwi_f = pitn.utils.system.get_file_glob_unique(dwi_dir, "*dwi.nii.gz")
        mask_f = pitn.utils.system.get_file_glob_unique(dwi_dir, "*mask.nii.gz")
        bval_f = pitn.utils.system.get_file_glob_unique(dwi_dir, "*bval*")
        bvec_f = pitn.utils.system.get_file_glob_unique(dwi_dir, "*bvec*")

        dwi_nib = nib.load(dwi_f)
        mask_nib = nib.load(mask_f)
        dwi = dwi_nib.get_fdata().astype(np.float32)
        mask = mask_nib.get_fdata().astype(bool)
        bval = np.loadtxt(bval_f)
        bvec = np.loadtxt(bvec_f)

        # Import gradients
        gtab = dipy.core.gradients.gradient_table_from_bvals_bvecs(
            bval, bvec.T, b0_threshold=50
        )

        ##### DTI
        out_dir = (
            data_dir / subj_id / "diffusion" / "preproc" / "parameter_maps" / "dti"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        dti_model = dipy.reconst.dti.TensorModel(gtab, fit_method="WLS")
        print(
            f"{datetime.datetime.now().replace(microsecond=0)} {subj_id} |",
            "Estimating DTI",
        )
        dti_fit = dti_model.fit(dwi, mask=mask)

        adc_test_sphere = dipy.data.get_sphere("repulsion100")
        dti_adc = dti_fit.adc(adc_test_sphere)
        dti_fa = np.clip(dti_fit.fa, a_min=0.0, a_max=1.0)
        dti_ga = dti_fit.ga
        dti_md = np.clip(dti_fit.md, a_min=MIN_DIFFUSE, a_max=None)
        dti_rd = np.clip(dti_fit.rd, a_min=MIN_DIFFUSE, a_max=None)
        dti_ad = np.clip(dti_fit.ad, a_min=MIN_DIFFUSE, a_max=None)
        dti_planarity = np.nan_to_num(dti_fit.planarity, nan=0.0)
        dti_linearity = np.nan_to_num(dti_fit.linearity, nan=0.0)
        dti_sphericity = np.nan_to_num(dti_fit.sphericity, nan=0.0)
        dti_eigvals = dti_fit.evals
        dti_tensor = dti_fit.lower_triangular()

        for arr, param_name in zip(
            (
                dti_adc,
                dti_fa,
                dti_ga,
                dti_md,
                dti_rd,
                dti_ad,
                dti_planarity,
                dti_linearity,
                dti_sphericity,
                dti_eigvals,
                dti_tensor,
            ),
            (
                "adc",
                "fa",
                "ga",
                "md",
                "rd",
                "ad",
                "planarity",
                "linearity",
                "sphericity",
                "eigenvals",
                "diffusion_tensor",
            ),
        ):
            im_nib = nib.Nifti1Image(
                arr.astype(np.float32), dwi_nib.affine, header=dwi_nib.header
            )
            nib.save(im_nib, out_dir / f"{subj_id}_{param_name}.nii.gz")

        ##### MSDKI
        out_dir = (
            data_dir / subj_id / "diffusion" / "preproc" / "parameter_maps" / "msdki"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        msdki_model = dipy.reconst.msdki.MeanDiffusionKurtosisModel(
            gtab, return_S0_hat=True
        )
        print(
            f"{datetime.datetime.now().replace(microsecond=0)} {subj_id} |",
            "Estimating MSDKI",
        )
        msdki_fit = msdki_model.fit(dwi, mask)

        s0_signal = msdki_fit.S0_hat
        msd = msdki_fit.msd
        msk = msdki_fit.msk

        s0_nib = nib.Nifti1Image(
            s0_signal.astype(np.float32), dwi_nib.affine, header=dwi_nib.header
        )
        nib.save(s0_nib, out_dir / f"{subj_id}_s0_non-diffusion-signal.nii.gz")
        msd_nib = nib.Nifti1Image(
            msd.astype(np.float32), dwi_nib.affine, header=dwi_nib.header
        )
        nib.save(msd_nib, out_dir / f"{subj_id}_mean_signal_diffusivity.nii.gz")
        msk_nib = nib.Nifti1Image(
            msk.astype(np.float32), dwi_nib.affine, header=dwi_nib.header
        )
        nib.save(msk_nib, out_dir / f"{subj_id}_mean_signal_kurtosis.nii.gz")

        ##### Standard DKI
        out_dir = (
            data_dir / subj_id / "diffusion" / "preproc" / "parameter_maps" / "dki"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        dki_model = dipy.reconst.dki.DiffusionKurtosisModel(gtab, fit_method="WLS")
        print(
            f"{datetime.datetime.now().replace(microsecond=0)} {subj_id} |",
            "Estimating DKI",
        )
        dki_fit = dki_model.fit(dwi, mask=mask)

        fa = np.clip(dki_fit.fa, a_min=0.0, a_max=1.0)
        md = np.clip(dki_fit.md, a_min=0.0, a_max=None)
        ad = np.clip(dki_fit.ad, a_min=0.0, a_max=None)
        rd = np.clip(dki_fit.rd, a_min=0.0, a_max=None)
        mk = dki_fit.mk(min_kurtosis=MIN_KURTOSIS, max_kurtosis=MAX_KURTOSIS)
        # ak = dki_fit.ak(min_kurtosis=MIN_KURTOSIS, max_kurtosis=MAX_KURTOSIS)
        # rk = dki_fit.rk(min_kurtosis=MIN_KURTOSIS, max_kurtosis=MAX_KURTOSIS)
        kfa = np.clip(dki_fit.kfa, a_min=0.0, a_max=1.0)
        mkt = dki_fit.mkt(min_kurtosis=MIN_KURTOSIS, max_kurtosis=MAX_KURTOSIS)
        kurtosis_tensor = dki_fit.kt

        # Calculate axonal water fract.
        k_test_sphere = dipy.data.get_sphere("repulsion100")
        k_max = parallel_kurtosis_maximum(
            dki_fit.model_params, sphere=k_test_sphere, mask=mask, max_workers=N_CPUS
        )
        k_max = np.clip(k_max, a_min=MIN_KURTOSIS, a_max=MAX_KURTOSIS)
        awf = axonal_water_fraction(k_max)

        for arr, param_name in zip(
            (fa, md, ad, rd, mk, kfa, mkt, kurtosis_tensor, k_max, awf),
            (
                "fa",
                "md",
                "ad",
                "rd",
                "mean_kurtosis",
                # "axial_kurtosis",
                # "radial_kurtosis",
                "kurtosis_fa",
                "mean_kurtosis_tensor",
                "kurtosis_tensor",
                "k_max",
                "axonal_water_fract",
            ),
        ):
            im_nib = nib.Nifti1Image(
                arr.astype(np.float32), dwi_nib.affine, header=dwi_nib.header
            )
            nib.save(im_nib, out_dir / f"{subj_id}_{param_name}.nii.gz")

        t1 = datetime.datetime.now()
        d_t = t1 - t0
        print(
            f"{datetime.datetime.now().replace(microsecond=0)} {subj_id} |",
            "Finished subject",
        )
        print(f"Total time {str(d_t).split('.')[0]} hours")

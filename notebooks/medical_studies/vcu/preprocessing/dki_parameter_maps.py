# -*- coding: utf-8 -*-
import datetime
import os
from pathlib import Path

import dipy.core.gradients
import dipy.reconst.dki
import dipy.reconst.msdki
import nibabel as nib
import numpy as np

import pitn

N_CPUS = 20

if __name__ == "__main__":

    data_dir = Path("/data/srv/outputs/pitn/vcu/preproc")
    selected_subjs = (
        "P_32",
        "P_38",
        "HC_01",
        "HC_02",
        "HC_03",
        "HC_04",
        "HC_05",
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

        fa = dki_fit.fa
        md = dki_fit.md
        ad = dki_fit.ad
        rd = dki_fit.rd
        mk = dki_fit.mk()
        ak = dki_fit.ak()
        rk = dki_fit.rk()
        kfa = dki_fit.kfa
        mkt = dki_fit.mkt()
        kurtosis_tensor = dki_fit.kt

        for arr, param_name in zip(
            (fa, md, ad, rd, mk, ak, rk, kfa, mkt, kurtosis_tensor),
            (
                "fa",
                "md",
                "ad",
                "rd",
                "mean_kurtosis",
                "axial_kurtosis",
                "radial_kurtosis",
                "kurtosis_fa",
                "mean_kurtosis_tensor",
                "kurtosis_tensor",
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

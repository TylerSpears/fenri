# -*- coding: utf-8 -*-
import datetime
import os
from pathlib import Path

import dipy.core.gradients
import dmipy
import dmipy.core.acquisition_scheme
import dmipy.core.modeling_framework
import dmipy.distributions.distribute_models
import dmipy.signal_models.cylinder_models
import dmipy.signal_models.gaussian_models
import nibabel as nib
import numpy as np

import pitn

N_CPUS = os.cpu_count()

if __name__ == "__main__":

    data_dir = Path("/data/srv/outputs/pitn/vcu/preproc")
    selected_subjs = (
        "P_32",
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

        out_dir = (
            data_dir / subj_id / "diffusion" / "preproc" / "parameter_maps" / "noddi"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

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
        acq_scheme = dmipy.core.acquisition_scheme.gtab_dipy2dmipy(
            gtab, b0_threshold=50, min_b_shell_distance=1000
        )

        print(f"{datetime.datetime.now().replace(microsecond=0)} {subj_id} |")
        acq_scheme.print_acquisition_info

        # Set up NODDI model.
        # Taken straight from
        # <https://nbviewer.org/github/AthenaEPI/dmipy/blob/master/examples/example_noddi_watson.ipynb>
        # Information at
        # <https://github.com/AthenaEPI/dmipy/blob/master/examples/example_watson_bingham.ipynb>
        ball = dmipy.signal_models.gaussian_models.G1Ball()
        stick = dmipy.signal_models.cylinder_models.C1Stick()
        zeppelin = dmipy.signal_models.gaussian_models.G2Zeppelin()

        watson_dispersed_bundle = (
            dmipy.distributions.distribute_models.SD1WatsonDistributed(
                models=[stick, zeppelin]
            )
        )
        print(
            f"{datetime.datetime.now().replace(microsecond=0)} {subj_id} |",
            watson_dispersed_bundle.parameter_names,
        )

        watson_dispersed_bundle.set_tortuous_parameter(
            "G2Zeppelin_1_lambda_perp", "C1Stick_1_lambda_par", "partial_volume_0"
        )
        watson_dispersed_bundle.set_equal_parameter(
            "G2Zeppelin_1_lambda_par", "C1Stick_1_lambda_par"
        )
        watson_dispersed_bundle.set_fixed_parameter("G2Zeppelin_1_lambda_par", 1.7e-9)
        noddi_model = dmipy.core.modeling_framework.MultiCompartmentModel(
            models=[ball, watson_dispersed_bundle]
        )
        print(noddi_model.parameter_names)

        noddi_model.set_fixed_parameter("G1Ball_1_lambda_iso", 3e-9)

        # Perform model fitting.
        print(
            f"{datetime.datetime.now().replace(microsecond=0)} {subj_id} |",
            f"Starting NODDI model fit over {mask.sum()} voxels",
        )

        model_fit = noddi_model.fit(
            acq_scheme,
            dwi,
            mask=mask,
            solver="brute2fine",
            Ns=5,
            maxiter=300,
            N_sphere_samples=30,
            use_parallel_processing=True,
            number_of_processors=N_CPUS,
        )

        print(
            f"{datetime.datetime.now().replace(microsecond=0)} {subj_id} |",
            "Done with model fitting",
        )

        for k, v in model_fit.fitted_parameters.items():
            im = nib.Nifti1Image(
                v.astype(np.float32), dwi_nib.affine, header=dwi_nib.header
            )
            nib.save(im, out_dir / f"{k}.nii.gz")

        print(
            f"{datetime.datetime.now().replace(microsecond=0)} {subj_id} |",
            "Estimating error metrics",
        )

        mse = model_fit.mean_squared_error(dwi)

        im = nib.Nifti1Image(
            mse.astype(np.float32), dwi_nib.affine, header=dwi_nib.header
        )
        nib.save(im, out_dir / "model_fit_mse.nii.gz")

        r2 = model_fit.R2_coefficient_of_determination(dwi)
        im = nib.Nifti1Image(
            r2.astype(np.float32), dwi_nib.affine, header=dwi_nib.header
        )
        nib.save(im, out_dir / "model_fit_R2-coeff-of-determination.nii.gz")

        t1 = datetime.datetime.now()
        d_t = t1 - t0
        print(
            f"{datetime.datetime.now().replace(microsecond=0)} {subj_id} |",
            "Finished subject",
        )
        print(f"Total time {str(d_t).split('.')[0]} hours")

        del model_fit, noddi_model, ball, stick, zeppelin, watson_dispersed_bundle

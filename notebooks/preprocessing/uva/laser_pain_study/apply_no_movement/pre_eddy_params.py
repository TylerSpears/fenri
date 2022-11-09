#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import re
from audioop import mul
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def select_one_glob(base, glob: str):
    b = Path(base).resolve()
    l = list(b.glob(glob))
    if len(l) == 0:
        raise RuntimeError(f"ERROR: No glob {str(b)}/{glob} found!")
    elif len(l) > 1:
        raise RuntimeError(f"ERROR: More than one glob {str(b)}/{glob} found!")

    return l[0]


def main(
    param_dir: Path,
    motion_ec_f_suffix,
    mbs_f_suffix,
    s2v_f_suffix,
    eddy_out_log_f_suffix,
    input_dwi_f,
    peas_f_suffix,
    acq_params_f,
    slspec_f,
    bvals_f,
    out_dir: Path,
):

    assert param_dir.exists()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Load files & parameters
    dwi_nib = nib.load(input_dwi_f.resolve())
    acq_params = np.loadtxt(acq_params_f)

    # Open slice acquisition groupings.
    # It may be possible that some rows do not have an equal number of slices (though
    # I'm not totally sure), so avoid using np.loadtxt for now.
    try:
        multi_band_slice_group_params = np.loadtxt(slspec_f)
        n_excitation_groups = multi_band_slice_group_params.shape[0]
    except ValueError:
        with open(slspec_f, "r") as f:
            multi_band_slice_group_params = f.read()
        multi_band_slice_group_params = multi_band_slice_group_params.strip().split(
            "\n"
        )
        n_excitation_groups = len(multi_band_slice_group_params)

    bvals = np.loadtxt(bvals_f)
    s2v_params = np.loadtxt(select_one_glob(param_dir, f"*{s2v_f_suffix}"))
    mbs_params_nib = nib.load(select_one_glob(param_dir, f"*{mbs_f_suffix}"))
    motion_ec_params = np.loadtxt(select_one_glob(param_dir, f"*{motion_ec_f_suffix}"))
    peas_f = select_one_glob(param_dir, f"*{peas_f_suffix}")
    log_f = select_one_glob(param_dir, f"*{eddy_out_log_f_suffix}")

    n_slices = dwi_nib.shape[2]
    n_scans = dwi_nib.shape[3]
    n_b0 = np.sum(bvals <= 100)
    n_dw = n_scans - n_b0

    #### Move MBS params into main directory as-is.
    mbs_new_fname = "pre_computed." + mbs_f_suffix
    nib.save(mbs_params_nib, out_dir / mbs_new_fname)

    #### Extract final GP hyperparams from output log.
    with open(log_f, "r") as f:
        matches = re.findall(
            r"estimated\s+hyperparameters\:((?:\s*\d+\.\d+\s*)+)",
            f.read(),
            flags=re.IGNORECASE | re.MULTILINE,
        )
        target_params = matches[-1]
        gp_hyperparams = np.fromstring(target_params, dtype=float, sep=" ")
    # Save to a text file. Must be a 1x6 comma-separated file. I don't know why.
    np.savetxt(
        out_dir / "pre_computed.gp_hyperparams.txt",
        gp_hyperparams.reshape(1, -1),
        fmt="%.13f",
        delimiter=",",
    )

    #### Set up for constructing a new affine matrix for each DWI.
    # Easier to keep track of things with labeled dims and more direct parameters.
    # new_aff_params = pd.DataFrame(
    #     np.zeros((n_scans, 6), dtype=float),
    #     columns=[
    #         "x_tran",
    #         "y_tran",
    #         "z_tran",
    #         "x_rot",
    #         "y_rot",
    #         "z_rot",
    #     ],
    # )

    ##### Modify motion and EC parameters.
    # Split motion from EC params.
    # x-z translate (mm), x-z rotation (radians)
    motion_params = motion_ec_params[:, :6]
    # x-z linear coefficients (Hz/mm),
    # x-z higher-order coefficients,
    # ...,
    # field offset from center of FOV (Hz)
    ec_params = motion_ec_params[:, 6:]

    # Zero-out head movement rigid terms.
    out_motion_params = np.zeros_like(motion_params)
    # Zero-out field offset terms.
    out_ec_params = np.copy(ec_params)
    out_ec_params[:, -1] = 0
    out_motion_ec_params = np.concatenate([out_motion_params, out_ec_params], axis=1)
    np.savetxt(
        out_dir / "pre_computed.eddy_parameters.txt",
        out_motion_ec_params,
        fmt="%.13g",
        delimiter=" ",
    )

    # Extract PEAS parameters
    # with open(peas_f, "r") as f:
    #     matches = re.findall()

    # Correct S2V motion params
    s2v_chunks = list()
    for i_scan, j_exc in enumerate(
        range(0, n_scans * n_excitation_groups, n_excitation_groups)
    ):
        print(i_scan, j_exc, j_exc + n_excitation_groups)
        s2v_scan = s2v_params[j_exc : j_exc + n_excitation_groups]
        motion_params_scan_i = motion_params[i_scan]
        s2v_scan = s2v_scan - motion_params_scan_i
        s2v_chunks.append(s2v_scan)
    s2v_chunks = np.concatenate(s2v_chunks, axis=0)
    # Save out new s2v parameters.
    np.savetxt(
        out_dir / "pre_computed.s2v_movement_parameters.txt",
        s2v_chunks,
        fmt="%.13f",
        delimiter=" ",
    )

    # new_aff_params = new_aff_params.rename(
    #     columns=dict(
    #         zip(
    #             ["x_tran", "y_tran", "z_tran", "x_rot", "y_rot", "z_rot"],
    #             [
    #                 "Translation X (mm)",
    #                 "Translation Y (mm)",
    #                 "Translation Z (mm)",
    #                 "Rotation X (radians)",
    #                 "Rotation Y (radians)",
    #                 "Rotation Z (radians)",
    #             ],
    #         )
    #     )
    # )


# PARAM_DIR = "raw_params"
# MOTION_EC_F_SUFFIX = "eddy_parameters"
# MBS_F_SUFFIX = ".eddy_mbs_first_order_fields.nii"
# S2V_F_SUFFIX = ".eddy_movement_over_time"
# OUT_LOG_F_SUFFIX = "out.log"
# INPUT_DWI_F_SUFFIX = ".eddy_outlier_free_data.nii"
# PEAS_F_SUFFIX = ".eddy_post_eddy_shell_alignment_parameters"
# ACQ_PARAMS_F = "../acqparams.txt"
# BVALS_F = "../bvals"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse, edit, and transform certain eddy output values to remove all translations & rotations."
    )
    parser.add_argument("--in-param-dir", dest="in_param_dir", required=True, type=Path)
    parser.add_argument("--motion-ec-suffix", dest="motion_ec_suffix", required=True)
    parser.add_argument(
        "--mbs-suffix", dest="mbs_suffix", default="eddy_mbs_first_order_fields.nii"
    )
    parser.add_argument(
        "--s2v-suffix", dest="s2v_suffix", default="eddy_movement_over_time"
    )
    parser.add_argument("--eddy-log-suffix", dest="eddy_log_suffix", default="out.log")
    parser.add_argument("--input-dwi", dest="input_dwi", required=True, type=Path)
    parser.add_argument(
        "--peas-suffix",
        dest="peas_suffix",
        default="eddy_post_eddy_shell_alignment_parameters",
    )
    parser.add_argument("--acq-params", dest="acq_params_f", required=True, type=Path)
    parser.add_argument("--slspec", dest="slspec_f", required=True, type=Path)
    parser.add_argument("--bvals", dest="bvals_f", required=True, type=Path)
    parser.add_argument("--out-dir", dest="out_dir", required=True, type=Path)

    args = parser.parse_args()
    main(
        param_dir=args.in_param_dir.resolve(),
        motion_ec_f_suffix=args.motion_ec_suffix,
        mbs_f_suffix=args.mbs_suffix,
        s2v_f_suffix=args.s2v_suffix,
        eddy_out_log_f_suffix=args.eddy_log_suffix,
        input_dwi_f=args.input_dwi.resolve(),
        peas_f_suffix=args.peas_suffix,
        acq_params_f=args.acq_params_f.resolve(),
        bvals_f=args.bvals_f.resolve(),
        slspec_f=args.slspec_f.resolve(),
        out_dir=args.out_dir.resolve(),
    )

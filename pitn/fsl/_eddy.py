# -*- coding: utf-8 -*-
import inspect
import re
import shlex
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from . import convert_seq_for_params, is_sequence


def eddy_cmd(
    imain: Path,
    bvecs: Path,
    bvals: Path,
    mask: Path,
    index: Path,
    acqp: Path,
    out: str,
    use_cuda: bool = True,
    session: Optional[Path] = None,
    slspec: Optional[Path] = None,
    json: Optional[Path] = None,
    mporder: int = 0,
    s2v_lambda=1,
    topup: Optional[str] = None,
    field: Optional[Path] = None,
    field_mat: Optional[str] = None,
    flm: str = "quadratic",
    slm: str = "none",
    b0_flm: str = "movement",
    b0_slm: str = "none",
    fwhm: Union[int, List[int]] = 0,
    s2v_fwhm: Union[int, List[int]] = 0,
    niter: int = 5,
    s2v_niter: int = 5,
    dwi_only: bool = False,
    b0_only: bool = False,
    fields: bool = True,
    rms: bool = False,
    dfields: bool = False,
    cnr_maps: bool = False,
    range_cnr_maps: bool = False,
    residuals: bool = False,
    with_outliers: bool = False,
    history: bool = False,
    fep: bool = False,
    dont_mask_output: bool = False,
    interp: str = "spline",
    s2v_interp: str = "trilinear",
    extrap: str = "periodic",
    epvalid: bool = False,
    resamp: str = "jac",
    covfunc: str = "spheri",
    hpcf: str = "CV",
    nvoxhp: int = 1000,
    initrand: int = 0,
    ff: float = 10,
    hypar: Optional[Sequence[float]] = None,
    wss: bool = False,
    repol: bool = False,
    rep_noise: bool = False,
    ol_nstd: int = 4,
    ol_nvox: int = 250,
    ol_ec: int = 1,
    ol_type: str = "sw",
    ol_pos: bool = False,
    ol_sqr: bool = False,
    estimate_move_by_susceptibility: bool = False,
    mbs_niter: int = 10,
    mbs_lambda: int = 10,
    mbs_ksp: int = 10,
    dont_sep_offs_move: bool = False,
    offset_model: str = "linear",
    dont_peas: bool = False,
    b0_peas: bool = False,
    data_is_shelled: bool = False,
    init: Optional[Path] = None,
    init_s2v: Optional[Path] = None,
    init_mbs: Optional[Path] = None,
    debug: int = 0,
    dbgindx: Optional[Sequence[int]] = None,
    lsr_lambda: float = 0.01,
    ref_scan_no: int = 0,
    rbvde: bool = False,
    test_rot: bool = False,
    pmiv: bool = False,
    pmip: bool = False,
    write_predictions: bool = False,
    write_scatter_brain_predictions: bool = False,
    log_timings: bool = False,
    very_verbose: bool = True,
    verbose: bool = False,
) -> str:
    """File containing all the images to estimate distortions for
    imain
    Mask to indicate brain
    mask
    File containing indices for all volumes in --imain into --acqp and --topup
    index
    File containing session indices for all volumes in --imain
    session=
    Multi-band factor
    mb=1
    Multi-band offset (-1 if bottom slice removed, 1 if top slice removed)
    mb_offs=0
    Name of text file defining slice/group order
    slorder=
    Name of text file completely specifying slice/group acuistion. N.B. --slspec and --json are mutually exclusive.
    slspec
    Name of .json text file with information about slice timing. N.B. --json and --slspec are mutually exclusive.
    json=
    Order of slice-to-vol movement model (default 0, i.e. vol-to-vol
    mporder=8
    Regularisation weight for slice-to-vol movement. (default 1, reasonable range 1--10
    s2v_lambda=2
    File containing acquisition parameters
    acqp
    Base name for output files from topup
    topup
    Name of file with susceptibility field (in Hz)
    field=
    Name of rigid body transform for susceptibility field
    field_mat=
    File containing the b-vectors for all volumes in --imain
    bvecs
    File containing the b-values for all volumes in --imain
    bvals
    First level EC model (movement/linear/quadratic/cubic, default quadratic)
    flm=quadratic
    Second level EC model (none/linear/quadratic, default none)
    slm=linear
    First level EC model for b0 scans (movement/linear/quadratic, default movement)
    b0_flm=movement
    Second level EC model for b0 scans (none/linear/quadratic, default none)
    b0_slm=none
    FWHM for conditioning filter when estimating the parameters (default 0)
    fwhm=0
    FWHM for conditioning filter when estimating slice-to-vol parameters (default 0)
    s2v_fwhm=0
    Number of iterations (default 5)
    niter=10
    Number of iterations for slice-to-vol (default 5)
    s2v_niter=5
    Basename for output
    out=/srv/tmp/data/pitn/uva/liu_laser_pain_study/derivatives/sub-001/eddy/eddy_full_run
    Switch on detailed diagnostic messages (default false)
    very_verbose=True
    Only register the dwi images (default false)
    dwi_only=False
    Only register the b0 images (default false)
    b0_only=False
    Write EC fields as images (default false)
    fields=True
    Write movement induced RMS (deprecated, its use will crash future versions)
    rms=False
    Write total displacement fields (default false)
    dfields=True
    Write shell-wise cnr-maps (default false)
    cnr_maps=True
    Write shell-wise range-cnr-maps (default false)
    range_cnr_maps=False
    Write residuals (between GP and observations), (default false)
    residuals=False
    Write corrected data (additionally) with outliers retained (default false)
    with_outliers=False
    Write history of mss and parameter estimates (default false)
    history=True
    Fill empty planes in x- or y-directions (default false)
    fep=False
    Do not mask output to include only voxels present for all volumes (default false)
    dont_mask_output=False
    Interpolation model for estimation step (spline/trilinear, default spline)
    interp=spline
    Slice-to-vol interpolation model for estimation step (spline/trilinear, default trilinear)
    s2v_interp=trilinear
    Extrapolation model for estimation step (periodic/mirror, default periodic)
    extrap=periodic
    Indicates that extrapolation is valid in EP direction (default false)
    epvalid=False
    Final resampling method (jac/lsr, default jac)
    resamp=jac
    Covariance function for GP (spheri/expo/old, default spheri)
    covfunc=spheri
    Cost-function for GP hyperparameters (MML/CV/GPP/CC, default CV)
    hpcf=CV
    # of voxels used to estimate the hyperparameters (default 1000)
    nvoxhp=1000
    Seeds rand for when selecting voxels (default 0=no seeding)
    initrand=0
    Fudge factor for hyperparameter error variance (default 10.0)
    ff=10
    User specified values for GP hyperparameters
    hypar=
    Write slice-wise stats for each iteration (default false)
    wss=False
    Detect and replace outlier slices (default false))
    repol=True
    Add noise to replaced outliers (default false)
    rep_noise=True
    Number of std off to qualify as outlier (default 4)
    ol_nstd=4
    Min # of voxels in a slice for inclusion in outlier detection (default 250)
    ol_nvox=250
    Error type (1 or 2) to keep constant for outlier detection (default 1)
    ol_ec=1
    Type of outliers, slicewise (sw), groupwise (gw) or both (both). (default sw)
    ol_type=both
    Consider both positive and negative outliers if set (default false)
    ol_pos=False
    Consider outliers among sums-of-squared differences if set (default false)
    ol_sqr=False
    Estimate how susceptibility field changes with subject movement (default false)
    estimate_move_by_susceptibility=True
    Number of iterations for MBS estimation (default 10)
    mbs_niter=10
    Weighting of regularisation for MBS estimation (default 10)
    mbs_lambda=10
    Knot-spacing for MBS field estimation (default 10mm)
    mbs_ksp=10
    Do NOT attempt to separate field offset from subject movement (default false)
    dont_sep_offs_move=False
    Second level model for field offset
    offset_model=linear
    Do NOT perform a post-eddy alignment of shells (default false)
    dont_peas=False
    Use interspersed b0s to perform post-eddy alignment of shells (default false)
    b0_peas=False
    Assume, don't check, that data is shelled (default false)
    data_is_shelled=False
    Text file with parameters for initialisation
    init=
    Text file with parameters for initialisation of slice-to-vol movement
    init_s2v=
    4D image file for initialisation of movement-by-susceptibility
    init_mbs=
    Level of debug print-outs (default 0)
    debug=0
    Indicies (zero offset) of volumes for debug print-outs
    dbgindx=
    Regularisation weight for LSR-resampling.
    lsr_lambda=0.01
    Zero-offset # of ref (for location) volume (default 0)
    ref_scan_no=0
    Rotate b-vecs during estimation (default false)
    rbvde=False
    Do a large rotation to test b-vecs
    test_rot=
    Write text file of MI values between shells (default false)
    pmiv=False
    Write text file of (2D) MI values between shells (default false)
    pmip=False
    Write predicted data (in addition to corrected, default false)
    write_predictions=False
    Write predictions obtained with a scattered data approach (in addition to corrected, default false)
    write_scatter_brain_predictions=False
    Write timing information (defaut false)
    log_timings=False
    switch on diagnostic messages
    -v,--verbose=False
    """
    # Most arg types need to be converted appropriately, so it's easier to loop through
    # and check arg values.
    args = locals()
    func_params_list = list(inspect.signature(eddy_cmd).parameters.keys())
    call_args = {k: args[k] for k in func_params_list}

    cmd = list()
    if use_cuda:
        cmd.append("eddy_cuda")
    else:
        cmd.append("eddy")
    call_args.pop("use_cuda")

    for k, v in call_args.items():
        # Clean up some string arguments.
        if k in {
            "flm",
            "slm",
            "b0_flm",
            "b0_slm",
            "interp",
            "s2v_interpinear",
            "extrap",
            "resamp",
            "covfunc",
            "hpcf",
        }:
            v = str(v).lower()
            if k in {"hpcf"}:
                v = v.upper()

        # None options won't be set at all, unless they fall under one of the enumerated
        # params (which have an option of 'none', but don't default to that value in
        # the eddy command itself).
        if v is None:
            continue
        # If non-string Sequence, convert to a comma-separated string.
        if is_sequence(v):
            cval = convert_seq_for_params(v)
        # Everything else should just be a string.
        else:
            cval = str(v)

        cmd.append(f"--{k}")
        cmd.append(cval)

    return shlex.join(cmd)


def parse_params_f(
    params_f: Path, as_dataframe: bool = True
) -> Union[np.ndarray, pd.DataFrame]:

    p = np.loadtxt(params_f)
    if as_dataframe:
        motion_cols = (
            "Translate X (mm)",
            "Translate Y (mm)",
            "Translate Z (mm)",
            "Rotate X (rads)",
            "Rotate Y (rads)",
            "Rotate Z (rads)",
        )
        ec_model_cols = (
            "X (Hz/mm)",
            "Y (Hz/mm)",
            "Z (Hz/mm)",
            "X^2 (Hz/mm)",
            "Y^2 (Hz/mm)",
            "Z^2 (Hz/mm)",
            "XY (Hz/mm)",
            "XZ (Hz/mm)",
            "YZ (Hz/mm)",
            "Isocenter Offset (Hz)",
        )
        p = pd.DataFrame(p, columns=motion_cols + ec_model_cols)
        p.index = p.index.rename("DWI idx")

    return p


def parse_gp_hyperparams_from_log(eddy_output_log_f: Path) -> np.ndarray:

    with open(Path(eddy_output_log_f), "r") as f:
        matches = re.findall(
            r"estimated\s+hyperparameters\:((?:\s*\d+\.\d+\s*)+)",
            f.read(),
            flags=re.IGNORECASE | re.MULTILINE,
        )

    target_params = matches[-1]
    gp_hyperparams = np.fromstring(target_params, dtype=float, sep=" ")

    return gp_hyperparams


def parse_s2v_params_f(
    move_over_time_f: Path, as_dataframe: bool = True
) -> Union[pd.DataFrame, np.ndarray]:
    p_s2v = np.loadtxt(move_over_time_f)

    if as_dataframe:
        cols = (
            "Translate X (mm)",
            "Translate Y (mm)",
            "Translate Z (mm)",
            "Rotate X (rads)",
            "Rotate Y (rads)",
            "Rotate Z (rads)",
        )
        p_s2v = pd.DataFrame(p_s2v, columns=cols)
        p_s2v.index = p_s2v.index.rename("Multiband Slice Group idx")

    return p_s2v


def slice_timing2slspec(slice_timing: np.ndarray) -> np.ndarray:
    groups = list()
    for time_i in np.unique(slice_timing):
        groups.append(np.where(slice_timing == time_i)[0].astype(int).flatten())

    return np.stack(groups)


def parse_post_eddy_shell_align_f(peas_f: Path) -> pd.DataFrame:
    with open(Path(peas_f), "rt") as f:
        peas_d = f.read()
    # TODO This algorithm could be simplified substantially.
    # Find all blocks of text that correspond to a report of parameters.
    matches = re.findall(
        r"((?:Shell.*to.*b0.*\n\s*x\-tr.*z.rot.*\n(?:\-?\s*\d+\.\d+\s*)+)+)",
        peas_d,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    # Find the "parameters applied to data" heading.
    applied_params_heading_match = re.search(
        r"^.*parameter.*appl.*data.*$",
        peas_d,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    heading_str_idx = applied_params_heading_match.span()[1]
    # Find the params string closest to the "parameters applied to data" heading.
    matches_under_applied_heading = [peas_d.find(m, heading_str_idx) for m in matches]
    applied_params_str_loc = min(filter(lambda x: x > 0, matches_under_applied_heading))
    applied_params_str = matches[
        matches_under_applied_heading.index(applied_params_str_loc)
    ]

    # Parse the params string/block.
    applied_params_str = applied_params_str.strip()
    # Should equal number of non-b0 shells in the DWI.
    shell_param_matches = list(
        re.finditer(
            r"Shell\s*(?P<shell>.+)\s*to\s*b0.*(?:\n.*){2}",
            applied_params_str,
            flags=re.IGNORECASE | re.MULTILINE,
        )
    )

    shell_params = list()
    cols = [
        "Shell val",
        "x-tr (mm)",
        "y-tr (mm)",
        "z-tr (mm)",
        "x-rot (deg)",
        "y-rot (deg)",
        "z-rot (deg)",
    ]
    # Convert columns to rads & rename labels.
    radian_cols = ["x-rot (rads)", "y-rot (rads)", "z-rot (rads)"]
    for match in shell_param_matches:
        shell = match.groupdict()["shell"]
        shell = float(shell)
        shell = int(shell) if shell.is_integer() else shell
        # Split the table into lines, ignore the first shell value line.
        val_str = match.group(0).split("\n")[-1]
        shell_vals = np.fromstring(val_str, sep=" ", dtype=float)
        # Convert rotation params from degrees to radians.
        shell_vals[3:] = shell_vals[3:] * np.pi / 180
        shell_params.append([shell] + shell_vals.tolist())
    # Store all shell & param values into a dataframe.
    shell_params_table = pd.DataFrame(shell_params, columns=cols[:4] + radian_cols)

    return shell_params_table


if __name__ == "__main__":
    print(
        eddy_cmd(
            "/tmp/sub 001.nii.gz",
            "../mask.nii.gz",
            "",
            "",
            "",
            "",
            "",
            hypar=[0.02, 0.42, 1.2, -0.2536],
        )
    )
    print(
        parse_post_eddy_shell_align_f(
            Path(
                "/srv/tmp/data/pitn/uva/liu_laser_pain_study/derivatives/sub-001/eddy_apply_no_move/raw_params/eddy_full_run.eddy_post_eddy_shell_alignment_parameters"
            )
        )
    )

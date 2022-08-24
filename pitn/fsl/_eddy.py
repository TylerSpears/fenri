# -*- coding: utf-8 -*-
import inspect
import shlex
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


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
    fwhm=10,8,4,2,1,0,0,0,0,0
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
    initrand=2985
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
    # Some arg types need to be converted appropriately, so it's easier to loop through
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
        # params (which have an option of 'none', but don't default to that value).
        if v is None:
            continue
        # If non-string Sequence
        if (
            not isinstance(v, str)
            and hasattr(type(v), "__len__")
            and hasattr(type(v), "__getitem__")
        ):
            v_arr = [str(e) for e in np.asarray(v).flatten()]
            cval = ",".join(v_arr)
        else:
            cval = str(v)

        cmd.append(f"--{k}")
        cmd.append(cval)

    return shlex.join(cmd)


def parse_params_f():
    pass


def parse_gp_hyperparams_from_log():
    pass


def parse_s2v_params_f():
    pass


def slice_timing2slspec() -> np.ndarray:
    pass


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

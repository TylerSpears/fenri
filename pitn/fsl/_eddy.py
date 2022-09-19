# -*- coding: utf-8 -*-
import collections
import inspect
import re
import shlex
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from box import Box

from pitn.utils.cli_parse import (
    add_equals_cmd_args,
    append_cmd_stdout_stderr_to_file,
    convert_seq_for_params,
    file_basename,
    is_sequence,
)

from . import FSL_OUTPUT_TYPE_SUFFIX_MAP


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
    field: Optional[str] = None,
    field_mat: Optional[Path] = None,
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
    fields: bool = False,
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
    fsl_output_type: str = "NIFTI_GZ",
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
    out=
    Switch on detailed diagnostic messages (default false)
    very_verbose=True
    Only register the dwi images (default false)
    dwi_only=False
    Only register the b0 images (default false)
    b0_only=False
    Write EC fields as images (default false)
    fields=True
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
    # Always avoid ambiguity regarding duplicate basenames. This is the default, but
    # it's good to be explicit.
    cmd.append("FSLMULTIFILEQUIT=TRUE")
    cmd.append(f"FSLOUTPUTTYPE={fsl_output_type.upper().strip()}")
    call_args.pop("fsl_output_type")
    if use_cuda:
        cmd.append("eddy_cuda")
    else:
        cmd.append("eddy")
    call_args.pop("use_cuda")

    # Can only set none or one of `dwi_only` and `b0_only`.
    if (not dwi_only) and (not b0_only):
        call_args.pop("dwi_only")
        call_args.pop("b0_only")

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
        }:
            v = str(v).lower().strip()
        elif k in {"hpcf"}:
            v = str(v).upper().strip()

        # None options won't be set at all, unless they fall under one of the enumerated
        # params (which have an option of 'none', but don't default to that value in
        # the eddy command itself).
        if v is None:
            continue
        # If non-string Sequence, convert to a comma-separated string.
        if is_sequence(v):
            cval = convert_seq_for_params(v)
        # boolean flags are only set by calling them, not assigning a value.
        if isinstance(v, bool):
            if not v:
                continue
            else:
                cval = None
        # Everything else should just be a string.
        else:
            cval = str(v)

        cmd.append(f"--{k}")
        if cval is not None:
            cmd.append(cval)
    cmd = shlex.join(cmd)
    cmd = add_equals_cmd_args(cmd)

    return cmd


def eddy_cmd_explicit_in_out_files(
    imain: Union[str, Path],
    bvecs: Union[str, Path],
    bvals: Union[str, Path],
    mask: Union[str, Path],
    index: Union[str, Path],
    acqp: Union[str, Path],
    out: str,
    use_cuda: bool = True,
    session: Optional[Union[str, Path]] = None,
    slspec: Optional[Union[str, Path]] = None,
    json: Optional[Union[str, Path]] = None,
    mporder: int = 0,
    s2v_lambda=1,
    topup_fieldcoef: Optional[Union[str, Path]] = None,
    topup_movpar: Optional[Union[str, Path]] = None,
    field: Optional[Union[str, Path]] = None,
    field_mat: Optional[Union[str, Path]] = None,
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
    fields: bool = False,
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
    init: Optional[Union[str, Path]] = None,
    init_s2v: Optional[Union[str, Path]] = None,
    init_mbs: Optional[Union[str, Path]] = None,
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
    fsl_output_type: str = "NIFTI_GZ",
    log_stdout: bool = True,
) -> Tuple[str, List[str], Dict[str, Union[List[str], str]]]:

    args = locals()
    # Take the kwargs of this function and intersect them with the parameters of
    # the associated "cmd" function.
    self_params = list(
        inspect.signature(eddy_cmd_explicit_in_out_files).parameters.keys()
    )
    func_params_list = list(inspect.signature(eddy_cmd).parameters.keys())
    # Params that are not common between these two cmd functions will be inserted later.
    eddy_cmd_kwargs = {k: args[k] for k in set(self_params) & set(func_params_list)}

    in_files = list()
    out_files = dict()

    # Handle topup inputs/basenames
    topup_basename = None
    top_field_basename = None
    mov_basename = None
    if topup_fieldcoef is not None:
        top_field_basename = file_basename(topup_fieldcoef).replace("_fieldcoef", "")
        in_files.append(str(Path(topup_fieldcoef)))
    if topup_movpar is not None:
        mov_basename = file_basename(topup_movpar).replace("_movpar", "")
        in_files.append(str(Path(topup_movpar)))

    if (top_field_basename is not None) and (mov_basename is not None):
        if mov_basename != top_field_basename:
            raise ValueError(
                f"""
                ERROR: Basenames of topup files
                {topup_fieldcoef} (base "{top_field_basename}")
                and
                {topup_movpar} (base "{mov_basename}")
                are not equal.""".replace(
                    "\n", " "
                )
            )

    if top_field_basename is not None:
        topup_basename = top_field_basename
    elif mov_basename is not None:
        topup_basename = mov_basename

    # Non-topup field files.
    field_basename = None
    field_mat_basename = None
    if field is not None:
        field_basename = file_basename(field)
        in_files.append(str(Path(field)))
    if field_mat is not None:
        field_mat_basename = file_basename(field_mat)
        in_files.append(str(Path(field_mat)))

    out_base = str(Path(out))
    log_f = out_base + ".stdout.log"
    # Stdout log file.
    if log_stdout:
        out_files["stdout"] = str(Path(log_f))

    # Grab the remaining input files.
    input_f_kwargs = eddy_cmd_kwargs.copy()
    input_f_kwargs["topup"] = None
    input_f_kwargs["field"] = None
    input_f_kwargs["field_mat"] = None
    in_files.extend(_eddy_unambiguous_input_files(**input_f_kwargs))

    # Create and modify the eddy command string.
    cmd_kwargs = eddy_cmd_kwargs.copy()
    cmd_kwargs["topup"] = topup_basename
    cmd_kwargs["field"] = field_basename
    cmd_kwargs["field_mat"] = field_mat_basename
    cmd = eddy_cmd(**cmd_kwargs)
    # Add stdout output piping, if needed.
    if log_stdout:
        cmd = append_cmd_stdout_stderr_to_file(
            cmd, str(Path(log_f)), overwrite_log=True
        )
    cmd = "\n".join([cmd, "sync"])
    # Add commands to zip all displacement field files, if present.
    if dfields:
        zip_cmd = f"""
        if compgen -G "{out_base}.eddy_displacement_fields.[0-9]*" > /dev/null; then
            sorted_fs="$($(which ls) {out_base}.eddy_displacement_fields.[0-9]* | sort -V)"
            tar --no-recursion --verify -cf {out_base}.eddy_displacement_fields.tar {out_base}.eddy_displacement_fields.[0-9]*
            gzip --force --verbose --rsyncable -9 {out_base}.eddy_displacement_fields.tar
            gzip --test {out_base}.eddy_displacement_fields.tar.gz && rm {out_base}.eddy_displacement_fields.[0-9]*
            sync
        fi
        """

        # tar --verify --no-recursion --remove-files -zcf {out_base}.eddy_displacement_fields.tar.gz $sorted_fs
        # zip --move --test {out_base}.eddy_displacement_fields.zip $sorted_fs
        zip_cmd = textwrap.dedent(zip_cmd)
        cmd = "".join([cmd, zip_cmd])
        out_files["displacement_fields"] = out_base + ".eddy_displacement_fields.tar.gz"

    out_files.update(_eddy_unambiguous_output_files(**cmd_kwargs))

    return cmd, in_files, out_files


def _eddy_unambiguous_input_files(
    imain: Path,
    bvecs: Path,
    bvals: Path,
    mask: Path,
    index: Path,
    acqp: Path,
    session: Optional[Path],
    slspec: Optional[Path],
    json: Optional[Path],
    init: Optional[Path],
    init_s2v: Optional[Path],
    init_mbs: Optional[Path],
    **kwargs,
) -> List[str]:
    args = locals()
    func_params_list = list(
        inspect.signature(_eddy_unambiguous_input_files).parameters.keys()
    )
    call_args = {k: args[k] for k in func_params_list}
    call_args.pop("kwargs", None)

    in_files = list()
    for v in call_args.values():
        if v is None:
            continue
        f = str(Path(v))
        in_files.append(f)

    return in_files


def _eddy_unambiguous_output_files(
    out: str,
    mporder: int = 0,
    dwi_only: bool = False,
    b0_only: bool = False,
    fields: bool = True,
    cnr_maps: bool = False,
    range_cnr_maps: bool = False,
    residuals: bool = False,
    repol: bool = False,
    with_outliers: bool = False,
    history: bool = False,
    wss: bool = False,
    estimate_move_by_susceptibility: bool = False,
    dont_peas: bool = False,
    b0_peas: bool = False,
    write_predictions: bool = False,
    write_scatter_brain_predictions: bool = False,
    log_timings: bool = False,
    fsl_output_type: str = "NIFTI_GZ",
    **kwargs,
) -> Dict[str, Union[List[str], str]]:
    args = locals()
    func_params_list = list(
        inspect.signature(_eddy_unambiguous_output_files).parameters.keys()
    )
    call_args = {k: args[k] for k in func_params_list}
    call_args.pop("kwargs", None)

    out_base = str(Path(out))
    call_args.pop("out")
    im_suffix = FSL_OUTPUT_TYPE_SUFFIX_MAP[fsl_output_type.strip().upper()]
    call_args.pop("fsl_output_type")
    b0_and_dwi = (not b0_only) and (not dwi_only)
    call_args.pop("b0_only")
    call_args.pop("dwi_only")

    def fname(f, suff=im_suffix, base=out_base):
        if suff is None:
            suff = ""
        return "".join([base, str(f), suff])

    # Construct dict of output filenames.
    out_files = collections.defaultdict(list)
    # These files are always given as outputs.
    out_files["corrected"] = fname("")
    out_files["motion_ec_parameters"] = fname(".eddy_parameters", None)
    out_files["rotated_bvecs"] = fname(".eddy_rotated_bvecs", None)
    out_files["movement_rms"] = fname(".eddy_movement_rms", None)
    out_files["restricted_movement_rms"] = fname(".eddy_restricted_movement_rms", None)
    out_files["peas"] = [
        fname(".eddy_post_eddy_shell_alignment_parameters", None),
        fname(".eddy_post_eddy_shell_PE_translation_parameters", None),
    ]
    out_files["outliers"] = [
        fname(".eddy_outlier_report", None),
        fname(".eddy_outlier_map", None),
        fname(".eddy_outlier_n_stdev_map", None),
        fname(".eddy_outlier_n_sqr_stdev_map", None),
    ]

    for k, v in call_args.items():
        if v is None:
            continue
        # Reuse the key name by default, but allow it to be overridden if the arg name
        # is not clearly related to the actual content(s) of the file(s).
        l = k
        if k == "fields" and v:
            f = fname(".eddy_fields")
        elif k == "estimate_move_by_susceptibility" and v:
            f = fname(".eddy_mbs_first_order_fields")
            l = "mbs_first_order_fields"
        elif k == "history" and v:
            f = list()
            # Base parameter history.
            if b0_and_dwi or dwi_only:
                f.extend(
                    [
                        fname(".eddy_dwi_mss_history", None),
                        fname(".eddy_dwi_parameter_history", None),
                    ]
                )
            if b0_and_dwi or b0_only:
                f.extend(
                    [
                        fname(".eddy_b0_mss_history", None),
                        fname(".eddy_b0_parameter_history", None),
                    ]
                )
            # s2v parameter history, but still stored in the "history" entry.
            if mporder > 0:
                if b0_and_dwi or dwi_only:
                    f.extend(
                        [
                            fname(".eddy_slice_to_vol_b0_mss_history", None),
                            fname(".eddy_slice_to_vol_b0_parameter_history", None),
                        ]
                    )
                if b0_and_dwi or b0_only:
                    f.extend(
                        [
                            fname(".eddy_slice_to_vol_dwi_mss_history", None),
                            fname(".eddy_slice_to_vol_dwi_parameter_history", None),
                        ]
                    )
        elif k == "mporder" and v > 0:
            f = fname(".eddy_movement_over_time", None)
            l = "s2v"
        elif k == "repol" and v:
            out_files["outliers"].append(fname(".eddy_outlier_free_data"))
            l = None
        elif k == "cnr_maps" and v:
            f = fname(".eddy_cnr_maps")
        elif k == "residuals" and v:
            f = fname(".eddy_residuals")
        elif (
            k
            in {
                "range_cnr_maps",
                "with_outliers",
                "write_predictions",
                "write_scatter_brain_predictions",
                "log_timings",
                "wss",
            }
            and v
        ):
            raise NotImplementedError(f"ERROR: Param {k} not accounted for.")
        else:
            l = None
            f = None

        # Allow skipping assignment with l = None
        if l is not None:
            out_files[l] = f

    return dict(out_files)


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


def estimate_slspec(json_sidecar: dict, n_slices: int) -> Optional[np.ndarray]:
    """Estimate slspec timing information for FSL eddy.

    This process is highly error-prone, even from experts and with the help of
    DICOM to Nifti convertors. You can thank MR machine manufacturers. Useful links:

    <https://en.wikibooks.org/wiki/SPM/Slice_Timing>
    <https://web.archive.org/web/20180718215057/http://dbic.dartmouth.edu/wiki/index.php?title=Slice_Acquisition_Order&oldid=1246>
    <https://web.archive.org/web/20161123082626/https://nifti.nimh.nih.gov/nifti-1/documentation/faq/#Q20>
    <https://github.com/rordenlab/dcm2niix>
    <https://www.nitrc.org/plugins/mwiki/index.php/dcm2nii:MainPage#Slice_timing_correction>
    <https://crnl.readthedocs.io/stc/index.html>
    <https://practicalfmri.blogspot.com/2012/07/siemens-slice-ordering.html>
    <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy/UsersGuide#A--slspec>

    Parameters
    ----------
    json_sidecar : dict
    n_slices : int

    Returns
    -------
    Optional[np.ndarray]

    """
    if "SliceTiming" in json_sidecar.keys():
        slspec = slice_timing2slspec(np.asarray(json_sidecar["SliceTiming"]))
    elif (
        "ParallelReductionOutOfPlane" in json_sidecar.keys()
        or "MultibandAccelerationFactor" in json_sidecar.keys()
    ):
        mb_k = list(
            filter(
                lambda k: k
                in {"ParallelReductionOutOfPlane", "MultibandAccelerationFactor"},
                json_sidecar.keys(),
            )
        )[0]
        mb_factor = int(round(json_sidecar[mb_k]))
        mb_groups = n_slices // mb_factor

        slspec_rows = list()
        slice_idx = np.arange(n_slices)

        if mb_groups % 2 == 0:
            for odd_idx in slice_idx[1::2]:
                group = slice_idx[odd_idx::mb_groups]
                if len(group) < mb_factor:
                    break
                slspec_rows.append(group)
            for even_idx in slice_idx[::2]:
                group = slice_idx[even_idx::mb_groups]
                if len(group) < mb_factor:
                    break
                slspec_rows.append(group)

            slspec = np.stack(slspec_rows, axis=0)
        else:
            for even_idx in slice_idx[::2]:
                group = slice_idx[even_idx::mb_groups]
                if len(group) < mb_factor:
                    break
                slspec_rows.append(group)
            for odd_idx in slice_idx[1::2]:
                group = slice_idx[odd_idx::mb_groups]
                if len(group) < mb_factor:
                    break
                slspec_rows.append(group)

            slspec = np.stack(slspec_rows, axis=0)
    else:
        # Assume single-band, which still has a slice acqusition order.
        slspec = np.arange(n_slices).reshape(-1, 1)

    # An inverted SliceEncodingDirection indicates that the SliceTiming info is
    # reversed. If the encoding direction is set and is negative, then we must
    # flip the slspec to get the correct ordering (though the grouping is still
    # correct). See
    # <https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/01-magnetic-resonance-imaging-data.html#timing-parameters>
    # for details.
    if "SliceEncodingDirection" in json_sidecar.keys():
        direct = str(json_sidecar["SliceEncodingDirection"])
        if "-" in direct:
            slspec = np.flip(slspec, axis=0)

    return slspec


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

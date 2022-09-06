# -*- coding: utf-8 -*-
import inspect
from typing import Any, Dict, List, Optional, Sequence, Union

from redun import File, script, task

import pitn

if __package__ is not None:
    redun_namespace = str(__package__)


@task(
    hash_includes=[
        pitn.fsl.eddy_cmd_explicit_in_out_files,
        pitn.fsl.eddy_cmd,
        pitn.fsl._eddy._eddy_unambiguous_input_files,
        pitn.fsl._eddy._eddy_unambiguous_output_files,
    ],
    config_args=["script_exec_config"],
    version="1",
)
def eddy(
    imain: File,
    bvecs: File,
    bvals: File,
    mask: File,
    index: File,
    acqp: File,
    out: str,
    use_cuda: bool = True,
    session: Optional[File] = None,
    slspec: Optional[File] = None,
    json: Optional[File] = None,
    mporder: int = 0,
    s2v_lambda=1,
    topup_fieldcoef: Optional[File] = None,
    topup_movpar: Optional[File] = None,
    field: Optional[File] = None,
    field_mat: Optional[File] = None,
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
    init: Optional[File] = None,
    init_s2v: Optional[File] = None,
    init_mbs: Optional[File] = None,
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
    script_exec_config: Optional[Dict[str, Any]] = None,
):
    args = locals()
    # Take the kwargs of this function and intersect them with the parameters of
    # the associated "cmd" function.
    func_params_list = list(
        inspect.signature(pitn.fsl.eddy_cmd_explicit_in_out_files).parameters.keys()
    )
    call_kwargs = {k: args[k] for k in func_params_list}
    # Convert over to strings for functions outside of redun tasks.
    for k in call_kwargs.keys():
        if isinstance(call_kwargs[k], File):
            call_kwargs[k] = str(call_kwargs[k].path)

    cmd, in_files, out_files = pitn.fsl.eddy_cmd_explicit_in_out_files(**call_kwargs)

    for k, v in out_files.items():
        if isinstance(v, list):
            new_v = [File(f).stage(f) for f in v]
        else:
            new_v = File(v).stage(v)
        out_files[k] = new_v

    script_executor = dict()
    if script_exec_config is not None:
        script_executor.update(script_exec_config)

    cmd_task_outs = script(
        cmd,
        inputs=[File(f).stage(f) for f in in_files],
        outputs=out_files,
        **script_executor,
    )

    return cmd_task_outs


@task(
    hash_includes=[
        pitn.fsl.topup_cmd_explicit_in_out_files,
        pitn.fsl.topup_cmd,
    ],
    config_args=["script_exec_config"],
)
def topup(
    imain: File,
    datain: File,
    out: Optional[str] = None,
    iout: Optional[File] = None,
    fout: Optional[File] = None,
    warpres: Union[int, Sequence[int]] = 10,
    subsamp: Union[int, Sequence[int]] = 1,
    fwhm: Union[int, Sequence[int]] = 8,
    miter: Union[int, Sequence[int]] = 5,
    lambd: Optional[Union[float, Sequence[float]]] = None,
    ssqlambda: Union[bool, Sequence[bool]] = True,
    estmov: Union[bool, Sequence[bool]] = True,
    minmet: Union[str, Sequence[str]] = "LM",
    regmod: str = "bending_energy",
    splineorder: int = 3,
    numprec: str = "double",
    interp: str = "spline",
    scale: Union[bool, Sequence[bool]] = False,
    regrid: Union[bool, Sequence[bool]] = True,
    verbose: bool = True,
    builtin_logout: Optional[File] = None,
    log_stdout: bool = True,
    fsl_output_type: str = "NIFTI_GZ",
    script_exec_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Union[File, List[File]]]:
    args = locals()
    # Take the kwargs of this function and intersect them with the parameters of
    # the associated "cmd" function.
    func_params_list = list(
        inspect.signature(pitn.fsl.topup_cmd_explicit_in_out_files).parameters.keys()
    )
    call_kwargs = {k: args[k] for k in func_params_list}
    # Convert over to strings for functions outside of redun tasks.
    for k in call_kwargs.keys():
        if isinstance(call_kwargs[k], File):
            call_kwargs[k] = str(call_kwargs[k].path)

    cmd, in_files, out_files = pitn.fsl.topup_cmd_explicit_in_out_files(**call_kwargs)
    in_files = [File(str(f)).stage(f) for f in in_files]
    for k, v in out_files.items():
        if pitn.utils.cli_parse.is_sequence(v):
            out_files[k] = [File(str(f)).stage(f) for f in v]
        else:
            out_files[k] = File(str(v))

    script_executor = dict()
    if script_exec_config is not None:
        script_executor.update(script_exec_config)

    cmd_task_outs = script(
        cmd,
        inputs=in_files,
        outputs=out_files,
        **script_executor,
    )

    return cmd_task_outs


@task(
    config_args=["script_exec_config"],
    hash_includes=[pitn.fsl.bet_cmd, pitn.fsl._bet._bet_output_files],
)
def bet(
    in_file: File,
    out_file_basename: str,
    mask: bool = False,
    skip_brain_output: bool = False,
    robust_iters: bool = False,
    eye_cleanup: bool = False,
    verbose: bool = True,
    fsl_output_type: str = "NIFTI_GZ",
    log_stdout: bool = True,
    script_exec_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Union[File, List[File]]]:
    args = locals()
    # Take the kwargs of this function and intersect them with the parameters of
    # the associated "cmd" function.
    func_params_list = list(inspect.signature(pitn.fsl.bet_cmd).parameters.keys())
    call_kwargs = {k: args[k] for k in func_params_list}
    call_kwargs["in_file"] = str(in_file.path)

    # Build the command and determine output files based on inputs/parameters.
    cmd = pitn.fsl.bet_cmd(**call_kwargs)
    out_files = pitn.fsl._bet_output_files(**call_kwargs)
    if log_stdout:
        out_files["stdout"] = str(out_file_basename) + "_stdout.log"
        cmd = pitn.utils.cli_parse.append_cmd_stdout_stderr_to_file(
            cmd, out_files["stdout"], overwrite_log=True
        )

    # Stage & convert all files in the dict before flattening.
    for k, v in out_files.items():
        if isinstance(v, list):
            new_v = [File(f).stage(f) for f in v]
        else:
            new_v = File(v).stage(v)
        out_files[k] = new_v

    script_executor = dict()
    if script_exec_config is not None:
        script_executor.update(script_exec_config)
    # Run the command.
    cmd_task_outs = script(
        cmd,
        inputs=[in_file.stage(in_file)],
        outputs=out_files,
        **script_executor,
    )

    return cmd_task_outs

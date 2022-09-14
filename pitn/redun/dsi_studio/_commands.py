# -*- coding: utf-8 -*-
import inspect
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from redun import File, script, task

import pitn

if __package__ is not None:
    redun_namespace = str(__package__)


@task(
    hash_includes=[pitn.dsi_studio.src_cmd],
    config_args=["script_exec_config"],
)
def gen_src(
    source: File,
    output: str,
    bval: Optional[File] = None,
    bvec: Optional[File] = None,
    b_table: Optional[File] = None,
    other_sources: Optional[Union[File, Sequence[File]]] = None,
    log_stdout: bool = True,
    script_exec_config: Optional[Dict[str, Any]] = None,
) -> Union[File, Tuple[File, File]]:

    args = locals()
    # Output must *not* be a File, otherwise there will be a circular dependency on
    # input vs. output, and the task will never hit cache.
    assert not isinstance(output, File)
    # Take the kwargs of this function and intersect them with the parameters of
    # the associated "cmd" function.
    func_params_list = list(
        inspect.signature(pitn.dsi_studio.src_cmd).parameters.keys()
    )
    call_kwargs = {k: args[k] for k in set(func_params_list) & set(args.keys())}
    # Convert over to strings for functions outside of redun tasks.
    for k in call_kwargs.keys():
        if isinstance(call_kwargs[k], File):
            call_kwargs[k] = str(call_kwargs[k].path)

    cmd = pitn.dsi_studio.src_cmd(**call_kwargs)

    # Stage input files.
    in_files = [source.stage(source.path)]
    if bval is not None:
        in_files.append(bval.stage(bval.path))
    if bvec is not None:
        in_files.append(bvec.stage(bvec.path))
    if b_table is not None:
        in_files.append(b_table.stage(b_table.path))
    if other_sources is not None:
        if pitn.utils.cli_parse.is_sequence(other_sources):
            in_files.extend([f.stage(f.path) for f in other_sources])
        else:
            in_files.append(other_sources.stage(other_sources.path))

    # Convert and stage output files.
    src_out_f = File(str(output)).stage(str(output))
    if log_stdout:
        # Add command to save stdout to file.
        log_f = "/".join([os.path.dirname(output), "stdout.log"])
        cmd = pitn.utils.cli_parse.append_cmd_stdout_stderr_to_file(
            cmd, str(log_f), overwrite_log=True
        )
        cmd = "\n".join([cmd, "sync"])

        # Convert the log file to a `File` object.
        log_f = File(log_f).stage(log_f)

        out_files = (src_out_f, log_f)
        nout = len(out_files)
    else:
        out_files = src_out_f
        nout = None

    script_executor = dict()
    if script_exec_config is not None:
        script_executor.update(script_exec_config)
    if nout is not None:
        cmd_task_outs = script(
            cmd,
            inputs=in_files,
            outputs=out_files,
            nout=nout,
            **script_executor,
        )
    else:
        cmd_task_outs = script(
            cmd,
            inputs=in_files,
            outputs=out_files,
            **script_executor,
        )

    return cmd_task_outs


@task(
    hash_includes=[pitn.dsi_studio.recon_cmd],
    config_args=["thread_count", "script_exec_config"],
)
def recon(
    source: File,
    method: str = "GQI",
    param0: float = 1.25,
    align_acpc: bool = True,
    check_btable: bool = True,
    other_output: Union[str, Sequence[str]] = (
        "fa",
        "ad",
        "rd",
        "md",
        "nqa",
        "iso",
        "rdi",
        "nrdi",
    ),
    record_odf: bool = False,
    qsdr_reso: Optional[float] = None,
    mask: Optional[File] = None,
    rev_pe: Optional[File] = None,
    rotate_to: Optional[File] = None,
    align_to: Optional[File] = None,
    other_image: Optional[File] = None,
    save_src: Optional[str] = None,
    cmd: Optional[str] = None,
    cmd_files: Optional[Sequence[File]] = None,
    thread_count: Optional[int] = None,
    dti_no_high_b: Optional[bool] = None,
    r2_weighted: bool = False,
    log_stdout: bool = True,
    script_exec_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, File]:

    args = locals()
    # Take the kwargs of this function and intersect them with the parameters of
    # the associated "cmd" function.
    func_params_list = list(
        inspect.signature(pitn.dsi_studio.recon_cmd).parameters.keys()
    )
    call_kwargs = {k: args[k] for k in set(func_params_list) & set(args.keys())}
    # Convert over to strings for functions outside of redun tasks.
    for k in call_kwargs.keys():
        if isinstance(call_kwargs[k], File):
            call_kwargs[k] = str(call_kwargs[k].path)

    cmd = pitn.dsi_studio.recon_cmd(**call_kwargs)

    in_files = [source.stage(source.path)]
    for file_arg in (mask, rev_pe, rotate_to, align_to, other_image):
        if file_arg is not None:
            in_files.append(file_arg.stage(file_arg.path))
    if cmd_files is not None:
        in_files.extend(f.stage(f.path) for f in cmd_files)

    out_files = dict()

    main_im_out_extensions = list()
    method_name = method.upper().strip()
    if method_name == "DTI":
        main_im_out_extensions.append(method.lower().strip())
    elif method_name == "GQI":
        if record_odf:
            main_im_out_extensions.append("odf")
        main_im_out_extensions.append(method.lower().strip())
        main_im_out_extensions.append(str(param0))
    elif method_name == "QSDR":
        raise NotImplementedError(method_name)
    else:
        main_im_out_extensions.append(method.lower().strip())
    main_im_out_extensions.extend(["fib", "gz"])

    main_im = File(".".join([source.path] + main_im_out_extensions))
    out_files["fib"] = main_im.stage(main_im.path)

    if save_src is not None:
        out_files["preproc"] = File(save_src).stage(save_src)

    if log_stdout:
        # Add command to save stdout to file.
        log_f = ".".join([main_im.path, "stdout.log"])
        cmd = pitn.utils.cli_parse.append_cmd_stdout_stderr_to_file(
            cmd, str(log_f), overwrite_log=True
        )
        cmd = "\n".join([cmd, "sync"])

        # Convert the log file to a `File` object.
        log_f = File(log_f).stage(log_f)
        out_files["stdout"] = log_f

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

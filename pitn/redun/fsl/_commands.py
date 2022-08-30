# -*- coding: utf-8 -*-
import functools
import inspect
from typing import Any, Dict, List, Optional, Union

import redun
from redun import File, script, task

import pitn

if __package__ is not None:
    redun_namespace = str(__package__)


@task()
def eddy():
    pass


@task()
def topup():
    pass


@task(
    config_args=["docker_exec_config"],
    hash_includes=[pitn.fsl.bet_cmd, pitn.fsl._bet_output_files],
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
    stdout_log_f: Optional[str] = None,
    docker_image: Optional[str] = None,
    docker_exec_config: Optional[Dict[str, Any]] = None,
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
    if stdout_log_f is not None:
        out_files["stdout"] = str(stdout_log_f)
        cmd = pitn.redun.utils.append_cmd_stdout_stderr_to_file(
            cmd, out_files["stdout"], overwrite=True
        )

    # Stage & convert all files in the dict before flattening.
    for k, v in out_files.items():
        if isinstance(v, list):
            new_v = [File(f).stage(f) for f in v]
        else:
            new_v = File(v).stage(v)
        out_files[k] = new_v

    # Flatten for easier management of the output expression from the script task.
    flat_file_list, idx_map = pitn.redun.utils.flatten_dict_depth_1(out_files)
    nouts = len(flat_file_list)

    if docker_exec_config is not None:
        script_exec_conf = dict(docker_exec_config)
    else:
        script_exec_conf = dict()
    if docker_image is not None:
        script_exec_conf["image"] = docker_image
    # Run the command.
    cmd_task_outs = script(
        cmd,
        inputs=[in_file.stage(in_file.path)],
        outputs=flat_file_list,
        nouts=nouts,
        **script_exec_conf,
    )

    # Unflatten the script's output files as the original dict, for easier usability
    # of the output.
    out_expr = pitn.redun.utils.unflatten_dict_depth_1(cmd_task_outs, idx_map)

    return out_expr

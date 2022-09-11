# -*- coding: utf-8 -*-
import inspect
import os
from typing import Any, Dict, Optional, Tuple, Union

from redun import File, script, task

import pitn

if __package__ is not None:
    redun_namespace = str(__package__)


@task(
    hash_includes=[pitn.mrtrix.dwi_bias_correct_cmd],
    config_args=["script_exec_config", "nthreads"],
)
def dwi_bias_correct(
    algorithm: str,
    input: File,
    output_name: str,
    grad: Optional[File] = None,
    fslgrad: Optional[Tuple[File, File]] = None,
    mask: Optional[File] = None,
    bias: Optional[File] = None,
    algorithm_options: Optional[Dict[str, Any]] = None,
    nthreads: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    log_stdout: bool = True,
    script_exec_config: Optional[Dict[str, Any]] = None,
) -> Union[File, Tuple[File, File]]:

    args = locals()
    # Output must *not* be a File, otherwise there will be a circular dependency on
    # input vs. output, and the task will never hit cache.
    assert not isinstance(output_name, File)
    # Take the kwargs of this function and intersect them with the parameters of
    # the associated "cmd" function.
    func_params_list = list(
        inspect.signature(pitn.mrtrix.dwi_bias_correct_cmd).parameters.keys()
    )

    args["output"] = output_name
    call_kwargs = {k: args[k] for k in set(func_params_list) & set(args.keys())}

    # Convert over to strings for functions outside of redun tasks.
    for k in call_kwargs.keys():
        if k == "fslgrad" and call_kwargs[k] is not None:
            call_kwargs[k] = (str(call_kwargs[k][0].path), str(call_kwargs[k][1].path))
        elif isinstance(call_kwargs[k], File):
            call_kwargs[k] = str(call_kwargs[k].path)

    cmd = pitn.mrtrix.dwi_bias_correct_cmd(**call_kwargs)

    in_files = list()
    in_files.append(input.stage(input.path))
    for file_arg in (grad, fslgrad, mask, bias):
        if file_arg is not None:
            if file_arg is fslgrad:
                in_files.append(file_arg[0].stage(file_arg[0].path))
                in_files.append(file_arg[1].stage(file_arg[1].path))
            else:
                in_files.append(file_arg.stage(file_arg.path))

    main_im_out = File(str(output_name)).stage(str(output_name))

    if log_stdout:
        out_files = dict()
        out_files["output"] = main_im_out
        # Add command to save stdout to file.
        log_f = "/".join([os.path.dirname(output_name), "stdout.log"])
        cmd = pitn.utils.cli_parse.append_cmd_stdout_stderr_to_file(
            cmd, str(log_f), overwrite_log=True
        )
        cmd = "\n".join([cmd, "sync"])

        # Convert the log file to a `File` object.
        log_f = File(log_f).stage(log_f)
        out_files["stdout"] = log_f
    else:
        out_files = main_im_out

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

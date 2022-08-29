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


def _bet_output_files(
    in_file: Any,
    out_file_basename: str,
    mask: bool = False,
    skip_brain_output: bool = False,
    robust_iters: bool = False,
    eye_cleanup: bool = False,
    verbose: bool = True,
    fsl_output_type: str = "NIFTI_GZ",
) -> Dict[str, Union[str, List[str]]]:

    base = str(out_file_basename)
    out_files = dict()
    suffix = pitn.fsl.FSL_OUTPUT_TYPE_SUFFIX_MAP[fsl_output_type.upper()]

    if not skip_brain_output:
        out_files["brain"] = "".join([base, suffix])

    if mask:
        out_files["mask"] = "".join([base, "_mask", suffix])

    return out_files


@task(hash_includes=[pitn.fsl.bet_cmd, _bet_output_files])
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
) -> Dict[str, Union[File, List[File]]]:
    args = locals()
    # Take the kwargs of this function and intersect them with the parameters of
    # the associated "cmd" function.
    func_params_list = list(inspect.signature(pitn.fsl.bet_cmd).parameters.keys())
    call_kwargs = {k: args[k] for k in func_params_list}
    call_kwargs["in_file"] = str(in_file.path)
    cmd = pitn.fsl.bet_cmd(**call_kwargs)
    out_files = _bet_output_files(**call_kwargs)
    if stdout_log_f is not None:
        out_files["stdout"] = str(stdout_log_f)
        cmd = pitn.redun.utils.append_cmd_stdout_stderr_to_file(
            cmd, out_files["stdout"], overwrite=True
        )

    # Get a dict from the output file function, but we need to flatten into a list.
    flat_file_list = list()
    idx_map = dict()
    idx = 0
    for k, v in out_files.items():
        if isinstance(v, list):
            new_v = [File(f).stage(f) for f in v]
            flat_file_list.extend(new_v)
            idx_map[k] = slice(idx, idx + len(new_v))
            idx = idx + len(new_v)
        else:
            new_v = File(v).stage(v)
            flat_file_list.append(new_v)
            idx_map[k] = idx
            idx = idx + 1

    nouts = len(flat_file_list)

    cmd_task_outs = script(
        cmd,
        inputs=[in_file.stage(in_file.path)],
        outputs=flat_file_list,
        nouts=nouts,
    )

    out_expr = dict()
    for k, i in idx_map.items():
        if isinstance(i, slice):
            out_expr[k] = [cmd_task_outs[j] for j in range(i.start, i.stop)]
        else:
            out_expr[k] = cmd_task_outs[i]

    return out_expr

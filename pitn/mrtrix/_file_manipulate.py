# -*- coding: utf-8 -*-
import inspect
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


def mr_grid_cmd(
    input: Path,
    operation: str,
    output: Path,
    template: Optional[Path] = None,
    interp: str = "cubic",
    as_reference: Optional[Path] = None,
    uniform: Optional[int] = None,
    mask: Optional[Path] = None,
    crop_unbound: Optional[bool] = None,
    all_axes: Optional[bool] = None,
    fill: float = 0.0,
    nthreads: Optional[int] = None,
    force: bool = False,
) -> str:
    args = locals()
    func_params_list = list(inspect.signature(mr_grid_cmd).parameters.keys())
    call_args = {k: args[k] for k in func_params_list}

    cmd = list()
    cmd.append("mrgrid")
    cmd.append(str(Path(input)))
    call_args.pop("input")
    cmd.append(str(operation))
    call_args.pop("operation")
    cmd.append(str(Path(output)))
    call_args.pop("output")

    for k, v in call_args.items():
        if v is None:
            continue
        # Boolean indicator flags.
        if k in {"crop_unbound", "all_axes", "force"}:
            if v:
                cmd.append(f"-{k}")
        else:
            cmd.append(f"-{k}")
            cmd.append(str(v))

    cmd = shlex.join(cmd)
    return cmd


def dwi_cat_cmd(
    inputs: Sequence[Path],
    output: Path,
    mask: Optional[Path] = None,
    nthreads: Optional[int] = None,
    force: bool = False,
) -> str:
    args = locals()
    func_params_list = list(inspect.signature(dwi_cat_cmd).parameters.keys())
    call_args = {k: args[k] for k in func_params_list}

    cmd = list()
    cmd.append("dwicat")
    for i in inputs:
        cmd.append(str(Path(i)))
    call_args.pop("inputs")
    cmd.append(str(Path(output)))
    call_args.pop("output")

    for k, v in call_args.items():
        if v is None:
            continue
        elif k in {"force"}:
            if v:
                cmd.append(f"-{k}")
        else:
            cmd.append(f"-{k}")
            cmd.append(str(v))

    cmd = shlex.join(cmd)
    return cmd


def mr_convert_cmd(
    input: Path,
    output: Path,
    fslgrad: Optional[Tuple[Path, Path]] = None,
    json_import: Optional[Path] = None,
    export_grad_fsl: Optional[Tuple[Path, Path]] = None,
    nthreads: Optional[int] = None,
    force: bool = False,
) -> str:
    args = locals()
    func_params_list = list(inspect.signature(mr_convert_cmd).parameters.keys())
    call_args = {k: args[k] for k in func_params_list}

    cmd = list()
    cmd.append("mrconvert")
    cmd.append(str(Path(input)))
    call_args.pop("input")
    cmd.append(str(Path(output)))
    call_args.pop("output")

    for k, v in call_args.items():
        if v is None:
            continue
        if k in {"fslgrad", "export_grad_fsl"}:
            bvec, bval = v[0], v[1]
            bvec = str(Path(bvec))
            bval = str(Path(bval))

            cmd.append(f"-{k}")
            cmd.append(bvec)
            cmd.append(bval)
        elif k in {"force"}:
            if v:
                cmd.append(f"-{k}")
        else:
            cmd.append(f"-{k}")
            cmd.append(str(v))

    cmd = shlex.join(cmd)
    return cmd


def dwi_extract_cmd(
    input: Path,
    output: Path,
    bzero: bool = False,
    no_bzero: bool = False,
    singleshell: bool = False,
    fslgrad: Optional[Tuple[Path, Path]] = None,
    export_grad_fsl: Optional[Tuple[Path, Path]] = None,
    nthreads: Optional[int] = None,
    force: bool = False,
) -> str:
    args = locals()
    func_params_list = list(inspect.signature(mr_convert_cmd).parameters.keys())
    call_args = {k: args[k] for k in func_params_list}

    cmd = list()
    cmd.append("dwiextract")
    cmd.append(str(Path(input)))
    call_args.pop("input")
    cmd.append(str(Path(output)))
    call_args.pop("output")

    for k, v in call_args.items():
        if v is None:
            continue
        if k in {"fslgrad", "export_grad_fsl"}:
            bvec, bval = v[0], v[1]
            bvec = str(Path(bvec))
            bval = str(Path(bval))

            cmd.append(f"-{k}")
            cmd.append(bvec)
            cmd.append(bval)
        elif k in {"bzero", "no_bzero", "singleshell", "force"}:
            if v:
                cmd.append(f"-{k}")
        else:
            cmd.append(f"-{k}")
            cmd.append(str(v))

    cmd = shlex.join(cmd)
    return cmd

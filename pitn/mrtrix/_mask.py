# -*- coding: utf-8 -*-
import inspect
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


def dwi2mask_cmd(
    input: Path,
    output: Path,
    clean_scale: int = 2,
    fslgrad: Optional[Tuple[Path, Path]] = None,
    nthreads: Optional[int] = None,
    force: bool = False,
) -> str:
    args = locals()
    func_params_list = list(inspect.signature(dwi2mask_cmd).parameters.keys())
    call_args = {k: args[k] for k in func_params_list}

    cmd = list()
    cmd.append("dwi2mask")
    cmd.append(str(Path(input)))
    call_args.pop("input")
    cmd.append(str(Path(output)))
    call_args.pop("output")

    for k, v in call_args.items():
        if v is None:
            continue
        if k in {"fslgrad"}:
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


def mask_filter_cmd(
    input: Path,
    filter: str,
    output: Path,
    scale: Optional[int] = None,
    largest: Optional[bool] = None,
    connectivity: Optional[int] = None,
    npass: Optional[int] = None,
    nthreads: Optional[int] = None,
    force: bool = False,
) -> str:
    args = locals()
    func_params_list = list(inspect.signature(mask_filter_cmd).parameters.keys())
    call_args = {k: args[k] for k in func_params_list}

    cmd = list()
    cmd.append("maskfilter")
    cmd.append(str(Path(input)))
    call_args.pop("input")
    cmd.append(str(filter))
    call_args.pop("filter")
    cmd.append(str(Path(output)))
    call_args.pop("output")

    for k, v in call_args.items():
        if v is None:
            continue
        if k in {"largest", "force"}:
            if v:
                cmd.append(f"-{k}")
        else:
            cmd.append(f"-{k}")
            cmd.append(str(v))

    cmd = shlex.join(cmd)
    return cmd

# -*- coding: utf-8 -*-
import inspect
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# from pitn.utils.cli_parse import convert_seq_for_params, is_sequence, add_equals_cmd_args


def dwi_bias_correct_cmd(
    algorithm: str,
    input: Path,
    output: Path,
    grad: Optional[Path] = None,
    fslgrad: Optional[Tuple[Path, Path]] = None,
    mask: Optional[Path] = None,
    bias: Optional[Path] = None,
    algorithm_options: Optional[Dict[str, Any]] = None,
    nthreads: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:

    args = locals()
    func_params_list = list(inspect.signature(dwi_bias_correct_cmd).parameters.keys())
    call_args = {k: args[k] for k in func_params_list}
    alg_params = call_args.pop("algorithm_options")
    config_params = call_args.pop("config")

    cmd = list()
    cmd.append("dwibiascorrect")

    cmd.append(str(algorithm))
    call_args.pop("algorithm")
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
        else:
            cmd.append(f"-{k}")
            cmd.append(str(v))

    if alg_params is not None:
        for k, v in alg_params.items():
            cmd.append(f"-{k}")
            cmd.append(str(v))

    if config_params is not None:
        for k, v in config_params.items():
            cmd.append("-config")
            cmd.append(str(k))
            cmd.append(str(v))

    cmd = shlex.join(cmd)
    return cmd

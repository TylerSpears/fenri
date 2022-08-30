# -*- coding: utf-8 -*-
import shlex
from typing import Union

import redun
from redun import File, script, task
from redun.file import Dir

import pitn

if __package__ is not None:
    redun_namespace = str(__package__)


@task(cache=False)
def stringify_shell_log(cmd_out: Union[str, bytes]) -> str:
    if isinstance(cmd_out, str):
        ret = cmd_out
    else:
        ret = cmd_out.decode()
    return ret


def flatten_dict_depth_1(d: dict):
    flat_list = list()
    idx_map = dict()
    idx = 0
    for k, v in d.items():
        if pitn.utils.cli_parse.is_sequence(v):
            l_v = list(v)
            flat_list.extend(l_v)
            idx_map[k] = slice(idx, idx + len(l_v))
            idx = idx + len(l_v)
        else:
            flat_list.append(v)
            idx_map[k] = idx
            idx = idx + 1

    return flat_list, idx_map


def unflatten_dict_depth_1(flat_sequence, idx_map: dict):
    out_dict = dict()
    for k, i in idx_map.items():
        if isinstance(i, slice):
            out_dict[k] = [flat_sequence[j] for j in range(i.start, i.stop)]
        else:
            out_dict[k] = flat_sequence[i]

    return out_dict


def append_cmd_stdout_stderr_to_file(cmd: str, log_file: str, overwrite=True) -> str:
    log_tokens = list()
    log_tokens.append("2>&1")
    log_tokens.append("|")
    log_tokens.append("tee")
    if not overwrite:
        log_tokens.append("--append")
    log_f = str(log_file)
    log_tokens.append(shlex.quote(log_f))
    joined_cmd = " ".join([cmd] + log_tokens)

    return joined_cmd

# -*- coding: utf-8 -*-
from typing import Union

from redun import task

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

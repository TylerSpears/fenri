# -*- coding: utf-8 -*-
import collections.abc
import re
import shlex
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np


def add_equals_cmd_args(cmd: str) -> str:
    """Replace the whitespace ` ` when setting a CLI argument to an equals `=`.

    This is necessary for CLI programs that don't accept arguments of the form
    `--argument1 value1 --argument2 value2 ...`, but instead require the form
    `--argument1=value1 --argument2=value2 ...`. If the `=` is added before shlex-ing,
    then shlex won't properly quote/escape tokens.

    Parameters
    ----------
    cmd : str

    Returns
    -------
    str
        Command with argument assignment done by an equals sign.
    """
    # Simple pattern to match any `--` flags, plus the whitespace after them. Excludes
    # flags that do not have an assignment value, like `--verbose`.
    pattern = r"(\-\-[\w\-\.]+)(\s+)(?=[^\-\s])"
    new_cmd = re.sub(pattern, lambda m: m.group(1) + "=", cmd, flags=re.MULTILINE)

    return new_cmd


def convert_seq_for_params(
    seq, element_format_callable: Optional[Callable[[Any], str]] = None
) -> str:

    if element_format_callable is None:
        element_format_callable = lambda a: a

    if is_sequence(seq):
        v_arr = [str(element_format_callable(e)) for e in np.asarray(seq).flatten()]
        cval = ",".join(v_arr)
    elif isinstance(seq, str):
        cval = seq.strip().replace(" ", "")
    else:
        raise ValueError(
            f"ERROR: Argument {seq} with type {type(seq)} "
            "not interpreted as a sequence type."
        )

    return cval


def is_sequence(seq: Any) -> bool:

    if isinstance(seq, (str, bytes)):
        ret = False
    # Technically, this should define every Sequence, but not even np.ndarray inherits
    # from this abstract class.
    elif isinstance(seq, collections.abc.Sequence):
        ret = True
    # Case for ndarray.
    elif isinstance(seq, np.ndarray):
        ret = True
    # Never consider strings as the type of Sequence we're interested in.
    # The most broad definition, may include some false positives.
    elif hasattr(type(seq), "__len__") and hasattr(type(seq), "__getitem__"):
        ret = True
    else:
        ret = False

    return ret


def append_cmd_stdout_stderr_to_file(
    cmd: str, log_file: str, overwrite_log=True
) -> str:
    log_tokens = list()
    log_tokens.append("2>&1")
    log_tokens.append("|")
    log_tokens.append("tee")
    if not overwrite_log:
        log_tokens.append("--append")
    log_f = str(log_file)
    log_tokens.append(shlex.quote(log_f))
    joined_cmd = " ".join([cmd] + log_tokens)

    return joined_cmd


def file_basename(fname: Union[str, Path]) -> str:
    f = Path(fname)
    return str(f).replace("".join(f.suffixes), "")

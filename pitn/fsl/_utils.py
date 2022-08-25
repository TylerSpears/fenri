# -*- coding: utf-8 -*-
import collections.abc
from typing import Any, Callable, Optional

import numpy as np


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

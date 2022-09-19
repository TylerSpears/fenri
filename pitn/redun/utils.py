# -*- coding: utf-8 -*-
import io
from collections import namedtuple
from typing import Any, Optional, Union

import nibabel as nib
import numpy
import numpy as np
import redun
import redun.value
from redun import File, task

import pitn

if __package__ is not None:
    redun_namespace = str(__package__)

NibImageTuple = namedtuple(
    "NibImageTuple",
    ("dataobj", "affine", "header", "extra", "file_map"),
    defaults=(None, None, None),
)


class NDArrayValue(redun.value.ProxyValue):
    pass
    # MIME_TYPE_NDARRAY = "application/x-python-numpy-ndarray"
    # type: numpy.ndarray
    # type_name = "numpy.ndarray"
    # _NPZ_KEY = "array"

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self._hash = None

    # def __repr__(self) -> str:
    #     return (
    #         "<NDArrayValue "
    #         f"size '{self.instance.shape}', "
    #         f"dtype '{self.instance.dtype}', "
    #         f"at {hex(id(self))}>"
    #     )

    # def __getstate__(self) -> dict:
    #     return {"hash": self.get_hash(), "serialized": self.serialize()}

    # def __setstate__(self, state: dict) -> None:
    #     """
    #     Populates the value from state during deserialization.
    #     """
    #     arr_bytes = state["serialized"]
    #     self.instance = self.deserialize(None, arr_bytes)
    #     self._hash = state["hash"]

    # ### Hash only state
    # # def __getstate__(self) -> dict:
    # #     return {"hash": self.get_hash()}  # , "serialized": self.serialize()}

    # # def __setstate__(self, state: dict) -> None:
    # #     """
    # #     Populates the value from state during deserialization.
    # #     """
    # #     arr_bytes = io.BytesIO(state["serialized"])
    # #     self.instance = self.deserialize(None, arr_bytes)
    # #     self._hash = state["hash"]
    # ###

    # def _serialize(self) -> bytes:
    #     arr = self.instance
    #     mem_bytes_file = io.BytesIO()
    #     np.save(mem_bytes_file, arr, allow_pickle=True)
    #     return mem_bytes_file.getvalue()

    # @classmethod
    # def _deserialize(cls, data: bytes) -> Any:
    #     # User defined deserialization.
    #     arr_bytes = io.BytesIO(data)
    #     arr = np.load(arr_bytes, allow_pickle=True)
    #     return arr

    # def get_hash(self, data: Optional[bytes] = None) -> str:
    #     """
    #     Returns a hash for the value.
    #     """
    #     # The hash is calculated from the serialize() output, while the state dict data
    #     # is made up of the .npz compressed data.
    #     if data is None:
    #         if self._hash is None:
    #             data = self.serialize()
    #             # Cache the hash.
    #             self._hash = super().get_hash(data)
    #         ret = self._hash
    #     else:
    #         ret = super().get_hash(data)

    #     return ret

    # def serialize(self) -> bytes:
    #     return self._serialize()

    # @classmethod
    # def deserialize(cls, raw_type: type, data: bytes) -> Any:
    #     """
    #     Returns a deserialization of bytes `data` into a new Value.
    #     """
    #     return cls._deserialize(data)

    # @classmethod
    # def get_serialization_format(cls) -> str:
    #     """
    #     Returns mimetype of serialization.
    #     """
    #     return cls.MIME_TYPE_NDARRAY

    # @classmethod
    # def parse_arg(cls, raw_type: type, arg: str) -> Any:
    #     """
    #     Parse a command line argument in a new Value.
    #     """
    #     return np.fromstring(arg)


@task(cache=False)
def save_np_txt(f: str, arr: np.ndarray, **kwargs) -> File:
    out_f = File(str(f))
    np.savetxt(out_f.path, arr, **kwargs)
    return out_f


@task(cache=False)
def load_np_txt(f: File, **kwargs) -> np.ndarray:
    return np.loadtxt(f.path, **kwargs)


@task(cache=False)
def save_nib(im: NibImageTuple, f: str, **kwargs) -> File:
    nib.save(nib.Nifti1Image(**im._asdict()), str(f), **kwargs)
    out_f = File(str(f))
    return out_f


@task(cache=False)
def save_np_to_nib(arr: np.ndarray, affine: np.ndarray, f: str, **kwargs) -> File:
    im = NibImageTuple(arr, affine=affine, **kwargs)
    return save_nib(im, f)


@task(cache=False)
def load_nib(f: File, **kwargs) -> NibImageTuple:
    im = nib.load(f.path, **kwargs)
    return NibImageTuple(
        np.ndarray(im.get_fdata()), im.affine, dict(im.header), im.extra
    )


@task(cache=False)
def join_save_dwis(
    dwis: list,
    affine: np.ndarray,
    dwis_out_f: str,
    bvals: list = None,
    bvals_out_f: str = None,
    bvecs: list = None,
    bvecs_out_f: str = None,
) -> dict:

    print("Joining and saving DWIs and parameters")
    join_out = dict()
    join_dwi = np.concatenate(list(dwis), axis=-1)
    join_out["dwi"] = save_np_to_nib(join_dwi, affine=affine, f=dwis_out_f)
    if bvals is not None and bvals_out_f is not None:
        join_bvals = np.concatenate(list(bvals), axis=0)
        join_out["bval"] = save_np_txt(str(bvals_out_f), join_bvals)
    if bvecs is not None and bvecs_out_f is not None:
        join_bvecs = np.concatenate(list(bvecs), axis=1)
        join_out["bvec"] = save_np_txt(str(bvecs_out_f), join_bvecs)

    return join_out


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
# -*- coding: utf-8 -*-
import ast
import inspect
import io
import os
import pickle
from collections import namedtuple
from typing import Any, Optional, Tuple, Union

import nibabel as nib
import numpy
import numpy as np
import redun
import redun.hashing
import redun.value
from redun import File, task

import pitn

if __package__ is not None:
    redun_namespace = str(__package__)

banner = "=!" * 10


class BigNDArrayValue(redun.value.FileCache):

    type: numpy.ndarray
    type_name = "numpy.ndarray"

    def __init__(self, instance: Any):
        super().__init__(instance)
        self._base_path = None
        self._hash = None

    @property
    def base_path(self) -> str:
        if self._base_path is None:
            from redun.scheduler import get_current_scheduler

            config = get_current_scheduler(required=True).config.get_config_dict()
            print(config)
            if "value_store_path" in config["backend"]:
                self._base_path = "/".join(
                    [str(config["backend"]["value_store_path"]), "BigNDArray"]
                )
            else:
                self._base_path = "/".join(
                    [str(config["backend"]["config_dir"]), "BigNDArray"]
                )
        os.makedirs(self._base_path, exist_ok=True)
        return self._base_path

    def _serialize(self) -> bytes:
        arr = self.instance
        mem_bytes_file = io.BytesIO()
        np.save(mem_bytes_file, arr, allow_pickle=False)
        return mem_bytes_file.getvalue()

    def serialize(self) -> bytes:
        from redun.file import File

        if self._hash is None:
            # Serialize data.
            self._hash = self.get_hash()
        # Write (possibly large) data to file.
        filename = os.path.join(self.base_path, self._hash)
        # If the file already exists, then by the definition of the hash function this
        # data already exists.
        if not os.path.exists(filename):
            File(filename).write(bytes, mode="wb")

        # Only cache filename.
        return filename.encode("utf8")

    def get_hash(self, data: Optional[bytes] = None) -> str:
        if self._hash is None:
            from redun.hashing import hash_bytes

            bytes = self._serialize()
            self._hash = hash_bytes(bytes)
        return self._hash

    @classmethod
    def _deserialize(cls, data: bytes) -> Any:
        arr_bytes = io.BytesIO(data)
        arr = np.load(arr_bytes, allow_pickle=False)
        return arr

    def __getstate__(self) -> dict:
        return {"base_path": self.base_path, "hash": self._hash}

    def __setstate__(self, data: dict):
        self._base_path = data["base_path"]
        self._hash = data["hash"]


# class BeyondPickleValue(redun.value.ProxyValue):

#     type: numpy.ndarray
#     type_name = "numpy.ndarray"
#     PICKLE_PROTOCOL = 4

#     def __init__(self, instance: Any):
#         super().__init__(instance)
#         self._proxy_hash = None
#         self._instance_hash = None
#         self._backend_db = None

#     @property
#     def val(self):
#         return self.instance

#     def __getstate__(self) -> dict:
#         print(
#             banner, f"__getstate__ {type(self.instance)}, {self.instance.shape}", banner
#         )
#         return {"instance_hash": self._get_instance_hash()}

#     def __setstate__(self, state: dict) -> None:
#         self._backend_db = None
#         self._proxy_hash = None
#         self._instance_hash = state["instance_hash"]
#         self.instance, self._backend_db = self._get_instance(
#             self._instance_hash, self._backend_db
#         )
#         self._proxy_hash = self._get_proxy_hash()
#         print(
#             banner,
#             f"__setstate__ state={state}, {type(self.instance)}, {self.instance.shape}",
#             banner,
#         )

#     @classmethod
#     def _stable_instance_hash(cls, instance: Any) -> str:
#         # This *should* re-create the hash algorithm as done by redun internally
#         # but without calling the bottleneck serialization (pickle dumps), but with
#         # our own serialization.
#         from redun.value import Value

#         instance_hash = Value(instance).get_hash(
#                 data=cls._beyond_pickle_serialize(instance)
#          )

#         return instance_hash

#     def _get_instance_hash(self) -> str:
#         if self._instance_hash is None:
#             self._instance_hash = self._stable_instance_hash(self.instance)

#         return self._instance_hash

#     @classmethod
#     def _get_instance(cls, instance_hash: str, backend_db=None) -> Tuple[Any, Any]:
#         if backend_db is None:
#             from redun.value import InvalidValueError
#             from redun.scheduler import get_current_scheduler
#             from redun.backends.db import RedunBackendDb

#             backend_db = get_current_scheduler(required=True).backend
#             # The base RedunBackend class does not have the functions we need to
#             # set/get cache items, so it must be a database backend.
#             assert isinstance(backend_db, RedunBackendDb)
#         instance, valid = backend_db.get_value(instance_hash)
#         if instance is None and not valid:
#             raise InvalidValueError

#         return instance, backend_db

#     @classmethod
#     def _beyond_pickle_serialize(cls, obj: Any) -> bytes:
#         # User defined serialization.
#         return pickle.dumps(obj, protocol=cls.PICKLE_PROTOCOL)

#     def _record_instance(self) -> str:
#         from redun.value import Value
#         if self._backend_db is None:
#             from redun.scheduler import get_current_scheduler
#             from redun.backends.db import RedunBackendDb

#             self._backend_db = get_current_scheduler(required=True).backend
#             # The base RedunBackend class does not have the functions we need to
#             # set/get cache items, so it must be a database backend.
#             assert isinstance(self._backend_db, RedunBackendDb)

#         recorded_instance_hash = self._backend_db.record_value(
#             value=self.instance,
#             data=self._beyond_pickle_serialize(self.instance),
#         )
#         if self._instance_hash is None:
#             self._instance_hash = recorded_instance_hash

#         if self._instance_hash != recorded_instance_hash:
#             raise RuntimeError(
#                 "\n".join(
#                     [
#                         f"ERROR: Set instance hash {self._instance_hash}",
#                         "!=",
#                         f"calculated {recorded_instance_hash}.",
#                         "Check _get_instance_hash()"
#                         + " vs. redun.backends.db.RedunBackendDb.record_value()",
#                     ]
#                 )
#             )

#         return self._instance_hash

#     def _get_proxy_hash(self) -> str:
#         if self._proxy_hash is None:
#             from redun.hashing import hash_struct

#             # Including the proxy class' source code allows us to make changes to how
#             # the class functions without worrying about hash/cache collisions. And the
#             # `unparse(parse(...))` processing normalizes spacing, removes comments,
#             # etc.
#             normalized_source = str(
#                 ast.unparse(ast.parse(inspect.getsource(self.__class__)))
#             )
#             self._proxy_hash = hash_struct(
#                 [
#                     f"Value.{type(self).__name__}",
#                     "instance_hash",
#                     self._get_instance_hash(),
#                     "proxy_unparsed_ast",
#                     normalized_source,
#                 ]
#             )

#         return self._proxy_hash

#     def get_hash(self, data: Optional[bytes] = None) -> str:
#         print(
#             banner,
#             "get_hash",
#             f"data={str(data)}, instance={type(self.instance)} | {self.instance.shape}",
#             banner,
#         )
#         if data is None:
#             hash = self._get_proxy_hash()
#         else:
#             hash = super().get_hash(data)

#         return hash

#     def get_serialization_format(self) -> str:
#         return f"application/x-python-pickle{self.PICKLE_PROTOCOL}"

#     def serialize(self) -> bytes:
#         instance_hash = self._record_instance()

#         print(
#             banner,
#             "serialize",
#             f"instance hash={instance_hash}, instance={type(self.instance)} | {self.instance.shape}",
#             banner,
#         )
#         return instance_hash.encode("utf8")

#     @staticmethod
#     def _beyond_pickle_deserialize(data: bytes) -> Any:
#         return pickle.loads(data)

#     @classmethod
#     def deserialize(cls, raw_type: type, data: bytes) -> Any:
#         # The deserialization has to do double duty for deserializing both the 'pointer'
#         # instance hash, and the instance data itself.
#         try:
#             # `data` is the instance hash string in a bytes object.
#             instance_hash = data.decode("utf8")
#             instance, _ = cls._get_instance(instance_hash, None)
#         except UnicodeDecodeError:
#             # `data` is the instance itself.
#             instance = cls._beyond_pickle_deserialize(data)
#         print(
#             banner,
#             "deserialize",
#             f"data length={len(data)}",
#             f"instance={instance}",
#             banner,
#         )
#         return instance

#     def preprocess(self, preprocess_args) -> redun.value.Value:
#         print(
#             banner,
#             f"preprocess, preproc args {preprocess_args}, instance {self.instance.shape}",
#             banner,
#         )
#         return self

#     def postprocess(self, postprocess_args) -> redun.value.Value:
#         print(
#             banner,
#             f"postprocess, postproc args {postprocess_args}, instance {self.instance.shape}",
#             banner,
#         )
#         return self


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

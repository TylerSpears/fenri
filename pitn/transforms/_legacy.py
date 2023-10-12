# -*- coding: utf-8 -*-

from typing import Dict, Hashable, Mapping, Optional

import dipy
import dipy.core
import dipy.reconst
import dipy.reconst.dti
import monai
import numpy as np
import skimage
import skimage.morphology
import skimage.transform
import torch
import torch.nn.functional as F
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.transforms.transform import MapTransform, Transform
from monai.utils import TransformBackends, convert_data_type, convert_to_tensor
from monai.utils.type_conversion import convert_to_dst_type

# Use lazy-loader of slow, unoptimized, or rarely-used module imports.
from pitn._lazy_loader import LazyLoader

scipy = LazyLoader("scipy", globals(), "scipy")
nib = LazyLoader("nib", globals(), "nibabel")


class BinaryDilate(Transform):

    backend = [TransformBackends.NUMPY]

    def __init__(self, footprint: Optional[NdarrayOrTensor] = None) -> None:
        self.footprint = footprint

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            img: shape must be (C, spatial_dim1[, spatial_dim2, ...]).
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        img_np, *_ = convert_data_type(img, np.ndarray)
        if self.footprint is not None:
            footprint, *_ = convert_data_type(self.footprint, np.ndarray)
        else:
            footprint = self.footprint
        out_np_list = list()
        # First dim is always the channel.
        for i in range(img_np.shape[0]):
            out = skimage.morphology.binary_dilation(img_np[i], footprint=footprint)
            out_np_list.append(out)
        out_np = np.stack(out_np_list, axis=0)
        out, *_ = convert_to_dst_type(out_np, img)
        return out


class BinaryDilated(MapTransform):
    """
    Dictionary-based wrapper of Dilate.
    """

    backend = BinaryDilate.backend

    def __init__(
        self,
        keys: KeysCollection,
        footprint: Optional[NdarrayOrTensor] = None,
        write_to_keys: Optional[KeysCollection] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        self.converter = BinaryDilate(footprint=footprint)
        self.write_to_keys = write_to_keys

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for i, key in enumerate(self.key_iterator(d)):
            new_v = self.converter(d[key])
            if self.write_to_keys is not None:
                d[self.write_to_keys[i]] = new_v
            else:
                d[key] = new_v
        return d

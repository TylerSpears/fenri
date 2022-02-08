# -*- coding: utf-8 -*-
import numpy as np

def conv3d_out_shape(
    out_channels,
    input_shape,
    kernel_size,
    stride=(1, 1, 1),
    padding=(0, 0, 0),
    dilation=(1, 1, 1),
):
    in_shape = np.asarray(input_shape)
    kernel = np.asarray(kernel_size)
    stride = np.asarray(stride)
    pad = np.asarray(padding)
    dilate = np.asarray(dilation)

    spatial = np.floor(((in_shape + 2 * pad - dilate * (kernel - 1) - 1) / stride) + 1)

    return (out_channels,) + tuple(spatial)

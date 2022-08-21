# -*- coding: utf-8 -*-
import functools

import einops
import torch


class ESPCNShuffle3d(einops.layers.torch.Rearrange):

    _pattern = "b (c r1 r2 r3) d w h -> b c (d r1) (w r2) (h r3)"

    def __init__(self, num_channels, upscale_factor):
        self._num_channels = num_channels
        self._upscale_factor = upscale_factor
        self._rearrange_kwargs = {
            "c": self._num_channels,
            "r1": self._upscale_factor,
            "r2": self._upscale_factor,
            "r3": self._upscale_factor,
        }
        super().__init__(self._pattern, **self._rearrange_kwargs)


# Based on implementations from fastai at <https://github.com/fastai/fastai> and
# torchlayers <https://github.com/szymonmaszke/torchlayers>.
def _icnr_init(w, scale, init_fn):
    """Initialize weight tensor according to ICNR nearest-neighbor scheme.

    Assumes w has shape 'chan_out x chan_in x spatial_dim_1 (x optional_spatial_dim_2 x ...)'
        and that 'chan_out' is divisible by `scale^N_spatial_dims`.

    Parameters
    ----------
    w : weight Tensor, usually from a convolutional layer.
    scale : int
    init_fn : Callable

    Returns
    -------
    Tensor
        Initialized Tensor with the same shape as `w`
    """
    n_spatial_dims = len(w.shape[2:])
    new_shape = (w.shape[0] // (scale**n_spatial_dims),) + tuple(w.shape[1:])
    set_sub_w = torch.zeros(*new_shape)
    set_sub_w = init_fn(set_sub_w)

    w_nn = einops.repeat(
        set_sub_w,
        "c_lr_out c_lr_in ... -> (c_lr_out repeat) c_lr_in ...",
        repeat=scale**n_spatial_dims,
    )

    return w_nn


@torch.no_grad()
def _conv_icnr_subkernel_init(
    m: torch.nn.Module,
    scale: int,
    init_fn=torch.nn.init.kaiming_normal_,
    zero_bias=True,
):

    if isinstance(
        m,
        (
            torch.nn.Conv3d,
            torch.nn.ConvTranspose3d,
            torch.nn.LazyConv3d,
            torch.nn.LazyConvTranspose3d,
        ),
    ):

        m.weight.data.copy_(_icnr_init(m.weight.data, scale, init_fn=init_fn))

        # The bias term is typically initialized to 0, so allow that as an option.
        if zero_bias:
            m.bias.data.zero_()
        else:
            # Add a dimension to the bias to give it 2 dimensions.
            m.bias.data.copy_(
                init_fn(
                    m.bias.data[
                        None,
                    ]
                )[0]
            )


class ICNRUpsample3d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upscale_factor: int,
        activate_fn,
        blur: bool,
        zero_bias: bool,
        conv_init_fn=torch.nn.init.kaiming_normal_,
    ):
        """3D upsampling layer with ICNR initialization and ESPCN shuffling.

        Based work found in:

            A. Aitken, C. Ledig, L. Theis, J. Caballero, Z. Wang, and W. Shi,
            "Checkerboard artifact free sub-pixel convolution: A note on sub-pixel
            convolution, resize convolution and convolution resize," arXiv:1707.02937
            [cs], Jul. 2017, Accessed: Jan. 06, 2022. [Online].
            Available: http://arxiv.org/abs/1707.02937

            Y. Sugawara, S. Shiota, and H. Kiya, "Super-Resolution Using Convolutional
            Neural Networks Without Any Checkerboard Artifacts,” in 2018 25th
            IEEE International Conference on Image Processing (ICIP), Oct. 2018,
            pp. 66-70. doi: 10.1109/ICIP.2018.8451141.

        and on the `PixelShuffle_ICNR` implementation from fastai at
        https://github.com/fastai/fastai/blob/351f4b9314e2ea23684fb2e19235ee5c5ef8cbfd/fastai/layers.py


        Parameters
        ----------
        in_channels : int
        out_channels : int
        upscale_factor : int
        activate_fn : Callable
            Activation function used after the convolutional layer, but before shuffling.
        blur : bool
            Use a 2x2x2 avg pooling after the ESPCN shuffling.
        zero_bias : bool
            Set the bias term of the convolutional layer to a 0-vector.
        conv_init_fn : Callable, optional
            Function to initialize conv params, by default torch.nn.init.kaiming_normal_
        """

        super().__init__()
        pre_shuffle_c = out_channels * (upscale_factor**3)

        self.pre_conv = torch.nn.Conv3d(
            in_channels=in_channels,
            out_channels=pre_shuffle_c,
            kernel_size=1,
        )

        # Apply ICNR weight initialization.
        self.pre_conv.apply(
            functools.partial(
                _conv_icnr_subkernel_init,
                scale=upscale_factor,
                init_fn=conv_init_fn,
                zero_bias=zero_bias,
            )
        )

        if activate_fn is None:
            self.activate_fn = torch.nn.Identity()
        else:
            self.activate_fn = activate_fn()

        self.shuffle = ESPCNShuffle3d(
            num_channels=out_channels, upscale_factor=upscale_factor
        )

        self.blur = blur
        if self.blur:
            p = torch.nn.ReplicationPad3d((1, 0, 1, 0, 1, 0))
            avg = torch.nn.AvgPool3d(kernel_size=2, stride=1)
            self.blur_layers = torch.nn.Sequential(p, avg)
        else:
            self.blur_layers = torch.nn.Identity()

    def forward(self, x):
        y = self.pre_conv(x)
        y = self.activate_fn(y)
        y = self.shuffle(y)
        if self.blur:
            y = self.blur_layers(y)

        return y

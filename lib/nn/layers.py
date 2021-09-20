# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F


def gaussian_kernel_3d(kernel_radius: int, sigma: float) -> torch.Tensor:

    x = y = z = torch.arange(-kernel_radius, kernel_radius + 1, 1)
    xx, yy, zz = torch.meshgrid(x, y, z)

    k = (1 / 2 * np.pi * sigma ** 2) * torch.exp(
        -(xx ** 2 + yy ** 2 + zz ** 2) / (2 * sigma ** 2)
    )

    k = k / k.sum()

    return k


def discrete_gaussian_kernel_3d() -> torch.Tensor:
    k_1d = torch.as_tensor([1, 2, 4, 2, 1])
    kx, ky, kz = torch.meshgrid(k_1d, k_1d, k_1d)

    k = kx * ky * kz
    k = 1 / k.sum() * k

    return k


class LaplacePyramid3d(torch.nn.Module):
    def __init__(self, num_high_freq=3):
        """A 3D Laplacian pyramid implemented as a pytorch Module.

        Parameters
        ----------
        num_high_freq : int, optional
            Number of high frequency levels, by default 3
                The total number of levels in the pyramid is `num_high_freq + 1`, to
                include the low-frequency residual.

        Pyramids are listed in *decreasing* order of frequency.
        """
        super().__init__()

        self._num_h_freq = num_high_freq
        self._L = self._num_h_freq + 1
        self._kernel_radius = 2

        # Set the gaussian kernel as a buffer so pytorch cannot change its weights, but
        # still syncs up with the module's device (cuda/cpu).
        self.register_buffer("kernel", gaussian_kernel_3d(self._kernel_radius, 1.0))

    @property
    def L(self):
        """Number of levels in the pyramid, including both high-freq and low-freq."""
        return self._L

    @staticmethod
    def _ensure_5d(vol: torch.Tensor) -> torch.Tensor:
        """Reshapes tensor to be `B x C x D x H x W`.

        Parameters
        ----------
        vol : torch.Tensor

        Returns
        -------
        torch.Tensor
            Reshaped tensor, with same data as the input.
        """

        if vol.ndim < 3 or vol.ndim > 5:
            raise RuntimeError(
                f"ERROR: volume shape {tuple(vol.shape)} invalid, "
                + "expected 3 or 5 dimensions."
            )

        if vol.ndim == 4:
            raise RuntimeError(
                "ERROR: volume with 4 dimensions is ambiguous, "
                + "reshape to 5 dimensions."
            )

        if vol.ndim == 3:
            vol = vol[None, None, ...]

        return vol

    @staticmethod
    def _spatial_downsample(x):
        return x[..., ::2, ::2, ::2]

    @staticmethod
    def _gaussian_convolution(x, kernel):
        """Convolve 5D volume with a 3D Gaussian kernel.

        Parameters
        ----------
        x : torch.Tensor
            Input volume, of shape `B x C x D x H x W`
        kernel : torch.Tensor
            Kernel to perform convolution, assumed to be a Gaussian kernel of shape ]
            `D x H x W`.

        Note that with a symmetric Gaussian kernel, convolution is equivalent to cross-
        correlation. The latter is actually the operation performed by
        torch convolutional networks, confusingly enough. But the symmetry of the
        Gaussian makes them equivalent.

        Returns
        -------
        torch.Tensor
        """

        # Pad to maintain the same shape after covolution. Use replication to maintain
        # the average intensity of the gaussian weighting; reflection would be better,
        # but it is not implemented in pytorch for 3 spatial dimension tensors.
        y = F.pad(x, (kernel.shape[-1] // 2,) * 6, mode="replicate")
        num_channels = num_groups = x.shape[1]
        k_expand = kernel.repeat(num_channels, num_channels // num_groups, 1, 1, 1)
        # Perform convolution such that each channel is independently processed.
        y = F.conv3d(y, k_expand, groups=num_groups)
        return y

    def _downsample(self, x, kernel):
        y = self._gaussian_convolution(x, kernel)
        y = self._spatial_downsample(y)

        return y

    def _upsample(self, x, kernel):
        # See <http://www.cs.cornell.edu/courses/cs4670/2019sp/lec07-resampling3.pdf>
        # for an explanation of upsampling with 0-padding and re-scaled gaussian
        # convolution.

        y = torch.zeros(*x.shape[:2], *tuple(torch.as_tensor(x.shape[2:]) * 2)).to(x)
        y[..., ::2, ::2, ::2] = x

        # Rescale the kernel values to account for the 0-value voxels in the window of
        # each convolution step.
        y = self._gaussian_convolution(y, 4 * kernel)
        return y

    def decompose(self, vol):

        vol = self._ensure_5d(vol)

        # Pyramids are listed in *decreasing* order of frequency.
        gauss_pyramid = list()
        # The first level is just the original volume.
        gauss_pyramid.append(vol)
        # Iteratively blur and downsample for the number of desired high frequency
        # volumes.
        for k in range(1, self._L):
            gauss_pyramid.append(self._downsample(gauss_pyramid[k - 1], self.kernel))

        # Set the size of the laplace pyramid so indexing by k can be used.
        laplace_pyramid = [
            None,
        ] * self._L
        # Add the low-frequency residual to the pyramid as the *last* element.
        laplace_pyramid[-1] = gauss_pyramid[-1]

        # Subtract gauss pyramid level from upsampled previous level, indexing from the
        # end to the start.
        for k in reversed(range(0, self._num_h_freq)):
            g_k = gauss_pyramid[k]
            l_kp1 = laplace_pyramid[k + 1]
            l_kp1_upsampled = self._upsample(l_kp1, self.kernel)

            laplace_pyramid[k] = g_k - l_kp1_upsampled

        l_freq_residual = laplace_pyramid[-1]
        # Separate the high frequency diff vols from the low frequency residual, for
        # clarity.
        return laplace_pyramid[:-1], l_freq_residual

    def reconstruct(
        self, high_freq: list, low_freq_residual: torch.Tensor
    ) -> torch.Tensor:

        low_freq_residual = self._ensure_5d(low_freq_residual)
        high_freq = list(map(self._ensure_5d, high_freq))

        gauss_kp1 = low_freq_residual

        for k in reversed(range(len(high_freq))):
            laplace_k = high_freq[k]
            gauss_k = self._upsample(gauss_kp1, self.kernel)
            curr_reconstruction = laplace_k + gauss_k
            gauss_kp1 = curr_reconstruction

        return curr_reconstruction


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_features):

        super().__init__()
        self.conv1 = torch.nn.Conv3d(in_features, in_features, 3, padding=1)
        self.conv2 = torch.nn.Conv3d(in_features, in_features, 3, padding=1)

    def forward(self, x):
        y = self.conv1(x)
        y = F.leaky_relu(y)
        y = self.conv2(y)

        return y


class LowFreqTranslateNet(torch.nn.Module):
    def __init__(self, num_channels=6, num_residual_blocks=5):
        super().__init__()
        self.conv_pre_res_1 = torch.nn.Conv3d(num_channels, 16, 3, padding=1)
        self.conv_pre_res_2 = torch.nn.Conv3d(16, 64, 3, padding=1)

        self.res_blocks = torch.nn.ModuleList()
        for _ in range(num_residual_blocks):
            self.res_blocks.append(ResidualBlock(64))

        self.conv_post_res_1 = torch.nn.Conv3d(64, 16, 3, padding=1)
        self.conv_post_res_2 = torch.nn.Conv3d(16, num_channels, 3, padding=1)

    def forward(self, low_freq: torch.Tensor) -> torch.Tensor:

        # Three levels of the term "residual":
        # 1. The low-frequency image I_L is a residual of the high-frequencies of the
        #    original image
        # 2. This module computes a residual of I_L, I_L^R, adding it at the end of the
        #    forward pass.
        # 3. The residual blocks within this module compute residuals over I_L^R, namely
        #    (I_L^R)^R!
        # How confusing.

        # Pre-residual blocks.
        # This is also a sort of "residual" to the original low-frequency residual.
        y_res = self.conv_pre_res_1(low_freq)
        y_res = F.instance_norm(y_res)
        y_res = F.leaky_relu(y_res)
        y_res = self.conv_pre_res_2(y_res)
        y_res = F.leaky_relu(y_res)

        # Residual blocks.
        for i_res_block in range(len(self.res_blocks)):
            res_block = self.res_blocks[i_res_block]
            # This is the residual to the residual to the residual!
            y_res_block_i = res_block(y_res)
            y_res = y_res + y_res_block_i

        # Post residual blocks.
        y_res = self.conv_post_res_1(y_res)
        y_res = F.leaky_relu(y_res)
        y_res = self.conv_post_res_2(y_res)

        # Combine the res blocks and the original input.
        y = low_freq + y_res
        y = torch.tanh(y)

        return y


class HighFreqTranslateNet(torch.nn.Module):
    def __init__(self, num_input_channels, num_residual_blocks=3):
        super().__init__()
        self.conv_pre_res = torch.nn.Conv3d(num_input_channels, 16, 3, padding=1)

        self.res_blocks = torch.nn.ModuleList()
        for _ in range(num_residual_blocks):
            self.res_blocks.append(ResidualBlock(16))

        self.conv_post_res = torch.nn.Conv3d(16, 1, 3, padding=1)

    def forward(
        self, high_freq_level_L: torch.Tensor, processed_level_Lm1: torch.Tensor
    ) -> torch.Tensor:

        # Compute the new mask given information from the level below the current level.
        # Pre-residual blocks.
        m = self.conv_pre_res(processed_level_Lm1)
        m = F.leaky_relu(m)

        # Residual blocks, if there are any.
        for i_res_block in range(len(self.res_blocks)):
            res_block = self.res_blocks[i_res_block]
            m_res_i = res_block(m)
            m = m + m_res_i

        m = self.conv_post_res(m)
        # No leaky-relu afterwards.

        # The full prediction is level L of a new Laplacian pyramid that will be
        # reconstructed into the final prediction.
        y = torch.multiply(high_freq_level_L, m)
        # Return both the Laplacian level L prediction and the mask generated.
        return y, m


class DiscriminatorBlock(torch.nn.Module):
    def __init__(
        self, num_input_channels: int, num_output_channels: int, normalize: bool = True
    ):
        super().__init__()
        self.conv = torch.nn.Conv3d(
            num_input_channels, num_output_channels, kernel_size=3, stride=2, padding=1
        )

        if normalize:
            self.normalizer = torch.nn.InstanceNorm3d(num_output_channels, affine=True)
        else:
            self.normalizer = None

    def forward(self, x):
        y = self.conv(x)
        if self.normalizer:
            # Normalization requires spatial inputs greater than size 1, which can
            # happen when several discriminator blocks are performing subsequent
            # downsampling (stride > 1).
            if torch.prod(torch.as_tensor(y.shape[-3:])) > 1:
                y = self.normalizer(y)
        y = F.leaky_relu(y, negative_slope=0.2)

        return y

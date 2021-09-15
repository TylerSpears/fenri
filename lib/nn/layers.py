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
        self._kernel = gaussian_kernel_3d(self._kernel_radius, 1.0)

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

        Returns
        -------
        torch.Tensor
        """

        # Pad with a reflection to maintain the same shape after covolution.
        y = F.pad(x, padding=(kernel.shape[-1] // 2,) * 6, mode="reflect")
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
        y = self._gaussian_convolution(y, 8 * kernel)
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
            gauss_pyramid.append(self._downsample(gauss_pyramid[k - 1], self._kernel))

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
            l_kp1_upsampled = self._upsample(l_kp1, self._kernel)

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
            gauss_k = self._upsample(gauss_kp1, self._kernel)
            curr_reconstruction = laplace_k + gauss_k
            gauss_kp1 = curr_reconstruction

        return curr_reconstruction

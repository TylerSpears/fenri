# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

from .. import layers


class LPTN(torch.nn.Module):
    def __init__(self, num_input_channels, num_high_freq_levels=3):
        """Laplacian Pyramid Translation Network implementation.

        Parameters
        ----------
        num_input_channels : int

        num_pyramid_levels : int, optional
            Number of high-frequency levels in the pyramid, by default 3.

            Must be >= 1. Note that this is only for the number of *high frequency*
            levels; the total number of levels will always be num_high_freq_levels + 1,
            for the low-frequency residual.
        """

        super().__init__()

        self.num_input_channels = num_input_channels

        self.laplace_pyramid = layers.LaplacePyramid3d(num_high_freq_levels)
        self.low_freq_net = layers.LowFreqTranslateNet(
            self.num_input_channels, num_residual_blocks=5
        )

        # Set up the high-frequency processing networks.
        high_freq_nets = list()
        high_freq_nets.append(
            layers.HighFreqTranslateNet(
                self.num_input_channels * 3, num_residual_blocks=3
            )
        )
        for _ in range(num_high_freq_levels - 1):
            high_freq_nets.append(layers.HighFreqTranslateNet(1, num_residual_blocks=0))
        # Reverse to put in order of decreasing frequency, to match the laplacian
        # pyramid.
        self.high_freq_nets = torch.nn.ModuleList(reversed(high_freq_nets))

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

    def upsample(self, x):
        # Convenience function, for brevity and in case we want to change the upsampling
        # method.
        return F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self._ensure_5d(x)
        high_freqs, low_freq = self.laplace_pyramid.decompose(x)

        low_freq_reconstruct = self.low_freq_net(low_freq)

        # Refine high-frequency components.
        # Prepare first lower-level high-freq input, which is different than all the
        # other levels.
        upsample_low_freq = self.upsample(low_freq)
        upsample_low_freq_reconstruct = self.upsample(low_freq_reconstruct)
        high_freq_km1_input = torch.cat(
            [high_freqs[-1], upsample_low_freq, upsample_low_freq_reconstruct], dim=1
        )
        high_freqs_reconstruct = [
            None,
        ] * len(high_freqs)

        for k, high_freq_net_k in reversed(list(enumerate(self.high_freq_nets))):
            laplacian_level_k, mask_k = high_freq_net_k(
                high_freqs[k], high_freq_km1_input
            )
            high_freqs_reconstruct[k] = laplacian_level_k
            if k > 0:
                high_freq_km1_input = self.upsample(mask_k)

        y = self.laplace_pyramid.reconstruct(
            high_freqs_reconstruct, low_freq_reconstruct
        )

        return y

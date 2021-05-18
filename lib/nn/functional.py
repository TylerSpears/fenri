import numpy as np
import einops

# Shuffle operation as a function.
def espcn_shuffle(x, channels):
    """Implements final-layer shuffle operation from ESPCN.

    x: 4D or 5D Tensor. Expects a shape of $C \times H \times W \times D$, or batched
        with a shape of $B \times C \times H \times W \times D$.

    channels: Integer giving the number of channels for the shuffled output.
    """
    batched = True if x.ndim == 5 else False

    if batched:
        downsample_factor = int(np.power(x.shape[1] / channels, 1 / 3))
        y = einops.rearrange(
            x,
            "b (c r1 r2 r3) h w d -> b c (h r1) (w r2) (d r3)",
            c=channels,
            r1=downsample_factor,
            r2=downsample_factor,
            r3=downsample_factor,
        )
    else:
        downsample_factor = int(np.power(x.shape[0] / channels, 1 / 3))
        y = einops.rearrange(
            x,
            "(c r1 r2 r3) h w d -> c (h r1) (w r2) (d r3)",
            c=channels,
            r1=downsample_factor,
            r2=downsample_factor,
            r3=downsample_factor,
        )

    return y

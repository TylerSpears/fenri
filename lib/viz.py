import numpy as np
import torch
import torchio

import dipy
import dipy.core
import dipy.reconst
import dipy.reconst.dti
import dipy.segment.mask

# Create FA map from DTI's
def fa_map(dti, channels_first=True) -> np.ndarray:
    if torch.is_tensor(dti):
        t = dti.cpu().numpy()
    else:
        t = np.asarray(dti)
    # Reshape to work with dipy.
    if channels_first:
        t = t.transpose(1, 2, 3, 0)

    # Re-create the symmetric DTI's (3x3) from the lower-triangular portion (6).
    t = dipy.reconst.dti.from_lower_triangular(t)
    eigvals, eigvecs = dipy.reconst.dti.decompose_tensor(t)

    fa = dipy.reconst.dti.fractional_anisotropy(eigvals)

    return fa


# Generate FA-weighted diffusion direction map.
def direction_map(dti, channels_first=True) -> np.ndarray:

    if torch.is_tensor(dti):
        t = dti.cpu().numpy()
    else:
        t = np.asarray(dti)
    # Reshape to work with dipy.
    if channels_first:
        t = t.transpose(1, 2, 3, 0)

    # Re-create the symmetric DTI's (3x3) from the lower-triangular portion (6).
    t = dipy.reconst.dti.from_lower_triangular(t)
    eigvals, eigvecs = dipy.reconst.dti.decompose_tensor(t)

    fa = dipy.reconst.dti.fractional_anisotropy(eigvals)
    direction_map = dipy.reconst.dti.color_fa(fa, eigvecs)

    if channels_first:
        return direction_map.transpose(3, 0, 1, 2)

    return direction_map

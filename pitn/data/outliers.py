# -*- coding: utf-8 -*-
import numpy as np
import torch

import pitn

# This is computed as the median of the eigenvalue cutoffs found in the jupyter notebook
# "notebooks/data/dti_thresholding.ipynb".
# Basically, any eigenvalue of a DTI greater than this constant can be considered
# as an outlier and should be clipped.
EIGVAL_CUTOFF = 0.00332008
# This is also a sane minimum eigenvalue; anything less than this is either noise, an
# error, or background.
EIGVAL_MIN = 1e-5


def clip_dti_eigvals(
    dti: torch.Tensor,
    tensor_components_dim: int = 1,
    eigval_min: float = EIGVAL_MIN,
    eigval_max: float = EIGVAL_CUTOFF,
    **eigh_kwargs,
) -> torch.Tensor:
    def clip_fn(eigvals, eigvecs):
        eigvals = torch.clip(eigvals, min=eigval_min, max=eigval_max)
        return eigvals, eigvecs

    m_dti = pitn.eig.tril_vec2sym_mat(dti, tril_dim=tensor_components_dim)

    clipped_m_dti = pitn.eig.eigh_decompose_apply_recompose(
        m_dti, eigvals_eigvecs_fn=clip_fn, **eigh_kwargs
    )

    # Indexing the lower triangle requires putting the matrix dims as the first dims.
    # So, two transposes will allow for easier batched indexing.
    clipped_dti = clipped_m_dti.T[
        tuple(
            torch.tril_indices(
                clipped_m_dti.shape[-2],
                clipped_m_dti.shape[-1],
                device=clipped_m_dti.device,
            )
        )
    ].T

    # Move the tensor component dim back to its original location.
    clipped_dti = clipped_dti.movedim(-1, tensor_components_dim)

    return clipped_dti

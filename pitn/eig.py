# -*- coding: utf-8 -*-
import collections
from typing import Callable, Tuple

import einops
import numpy as np
import torch


def eigh_workaround(A: torch.Tensor, *args, chunk_size=100000, **kwargs):
    """Wrapper function of torch.linalg.eigh that avoids CUDA errors.

    When decomposing too many matrices at once (>2,500,000, for example), the CUDA
        implementation of torch.linalg.eigh throws a cryptic error. This can be solved
        by only operating on some fixed chunk of matrices at one time, while keeping
        the speedup of decomposing matrices on the GPU.

    All args and kwargs are the same as torch.linalg.eigh, except the chunk_size.
    Parameters
    ----------
    chunk_size : int, optional
        Number of matrices to decompose at each step, by default 100000

    """
    try:
        result = torch.linalg.eigh(A, *args, **kwargs)
    except RuntimeError:
        n = A.shape[-1]
        # Flatten to be a batch of matrices.
        M = A.view(-1, n, n)
        n_mats = M.shape[0]

        # Initialize eigval and eigvec tensors, and write into them with each chunk.
        eigvals = torch.empty(n_mats, n).to(M)
        eigvecs = torch.empty_like(M)
        # Operate on chunk_size matrices at each step.
        start_idx = range(0, n_mats, chunk_size)
        end_idx = range(chunk_size, n_mats + chunk_size, chunk_size)
        for start, stop in zip(start_idx, end_idx):
            evals, evecs = torch.linalg.eigh(M[start:stop], *args, **kwargs)
            eigvals[start:stop] = evals
            eigvecs[start:stop] = evecs
        eigvals = eigvals.view(*A.shape[:-2], n)
        eigvecs = eigvecs.view(*A.shape)
        # Match pytorch named tuple output.
        EighResult = collections.namedtuple("EighResult", ("eiganvalues, eiganvectors"))
        result = EighResult(eiganvalues=eigvals, eiganvectors=eigvecs)

    return result


def eigvalsh_workaround(
    A: torch.Tensor, *args, chunk_size=100000, **kwargs
) -> torch.Tensor:
    """Wrapper function of torch.linalg.eigvalsh that avoids CUDA errors.

    When decomposing too many matrices at once (>2,500,000, for example), the CUDA
        implementation of torch.linalg.eigvalsh throws a cryptic error. This can be solved
        by only operating on some fixed chunk of matrices at one time, while keeping
        the speedup of decomposing matrices on the GPU.

    All args and kwargs are the same as torch.linalg.eigvalsh, except the chunk_size.

    Parameters
    ----------
    chunk_size : int, optional
        Number of matrices to decompose at each step, by default 100000

    """

    try:
        result = torch.linalg.eigvalsh(A, *args, **kwargs)
    except RuntimeError:
        n = A.shape[-1]
        # Flatten to be a batch of matrices.
        M = A.view(-1, n, n)
        n_mats = M.shape[0]

        # Initialize eigval tensor and write into it one chunk at a time.
        eigvals = torch.empty(n_mats, n).to(M)
        # Operate on chunk_size matrices at each step.
        start_idx = range(0, n_mats, chunk_size)
        end_idx = range(chunk_size, n_mats + chunk_size, chunk_size)
        for start, stop in zip(start_idx, end_idx):
            evals = torch.linalg.eigvalsh(M[start:stop], *args, **kwargs)
            eigvals[start:stop] = evals
        eigvals = eigvals.view(*A.shape[:-2], n)
        # Match pytorch named tuple output.
        result = eigvals

    return result


def tril_vec2sym_mat(x: torch.Tensor, tril_dim=1) -> torch.Tensor:
    tril_size = x.shape[tril_dim]
    # Solve quadratic equation of sum of consecutive numbers to find size of matrix.
    mat_size = (-1 + np.sqrt(1 + (4 * 2 * tril_size))) / 2
    # If mat_size is non-integer, then the shape of the full matrix cannot be
    # determined/does not exist.
    if not np.isclose(int(round(mat_size)), mat_size):
        raise ValueError(
            f"ERROR: Invalid upper triangle number of elements {tril_size}"
        )
    mat_size = int(round(mat_size))
    # Move tril elements to the first dimension.
    v = torch.movedim(x, tril_dim, 0)
    batch_shape = tuple(v.shape[1:])
    full_mat_shape = (mat_size, mat_size) + batch_shape

    m = torch.zeros(full_mat_shape).to(x)
    m = m.reshape(mat_size, mat_size, -1)

    # Handle lower triangle with the tril elements of the input.
    m[
        tuple(torch.tril_indices(mat_size, mat_size, device=x.device))
    ] = v.contiguous().view(tril_size, -1)
    # Handle the upper triangle (sans the diagonal) one element at a time.
    for r, c in zip(
        *torch.triu_indices(mat_size, mat_size, offset=1, device=x.device).tolist()
    ):
        m[r, c] = m[c, r]
    m = m.view(*full_mat_shape)
    # Many algorithms require the matrix dims to be the last two dims, so rearrange
    # to that shape.
    m = einops.rearrange(m, "r c ... -> ... r c", r=mat_size, c=mat_size)

    return m


def sym_mat2tril_vec(x: torch.Tensor, dim1=-2, dim2=-1, tril_dim=1) -> torch.Tensor:

    dims = [[f"d{i}", x.shape[i]] for i in range(len(x.shape))]
    dims[dim1][0] = "md1"
    dims[dim2][0] = "md2"
    in_shape_str = " ".join(d[0] for d in dims)
    batch_shape = list(filter(lambda sd: "md" not in sd[0], dims))
    if tril_dim < 0:
        tril_dim = len(batch_shape) + tril_dim
    bat_shape_str = " ".join(d[0] for d in batch_shape)

    bat_mat = einops.rearrange(
        x,
        f"{in_shape_str} -> ({bat_shape_str}) md1 md2",
        **dict(d for d in dims),
    )
    mat_shape = (x.shape[-1], x.shape[-2])
    tril_idx = torch.tril_indices(*mat_shape, device=x.device)
    tril_idx = (slice(None),) + tuple(tril_idx)
    bat_vec = bat_mat[tril_idx]
    vec_shape = bat_vec.shape[-1]
    out_shape = batch_shape.copy()
    out_shape.insert(tril_dim, ("v", vec_shape))
    out_shape_str = " ".join(sd[0] for sd in out_shape)
    y = einops.rearrange(
        bat_vec, f"({bat_shape_str}) v -> {out_shape_str}", **dict(d for d in out_shape)
    )

    return y


def eigh_decompose_apply_recompose(
    A: torch.Tensor,
    eigvals_eigvecs_fn: Callable[
        [torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
    ],
    **eigh_kwargs,
) -> torch.Tensor:

    # Decompose the matrices into their eigenvalues and eigenvectors.
    # Input needs to be a float, there's no getting around it.
    eigvals, eigvecs = eigh_workaround(A.float(), **eigh_kwargs)

    # Apply the function to the eigenvalues and eigenvectors.
    eigvals, eigvecs = eigvals_eigvecs_fn(eigvals, eigvecs)

    # Reconstruct the tensor from the eigenvalues and eigenvectors.
    # The eigh() function expects the matrix dims to be the final two dimensions of
    # A, so we can assume that the transpose can be performed on the last two dims
    # of the eigenvectors.
    # conj_physical() computes the element-wise conjugate only if the eigenvectors
    # are complex. Otherwise, it is an identity function.
    A_transform = (
        eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-2, -1).conj_physical()
    )

    return A_transform

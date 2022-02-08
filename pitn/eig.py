# -*- coding: utf-8 -*-
import collections
import numpy as np
import torch
import einops


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


def eigvalsh_workaround(A: torch.Tensor, *args, chunk_size=100000, **kwargs):
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

        # Initialize eigval and eigvec tensors, and write into them with each chunk.
        eigvals = torch.empty(n_mats, n).to(M)
        # Operate on chunk_size matrices at each step.
        start_idx = range(0, n_mats, chunk_size)
        end_idx = range(chunk_size, n_mats + chunk_size, chunk_size)
        for start, stop in zip(start_idx, end_idx):
            evals, evecs = torch.linalg.eigh(M[start:stop], *args, **kwargs)
            eigvals[start:stop] = evals
        eigvals = eigvals.view(*A.shape[:-2], n)
        # Match pytorch named tuple output.
        result = eigvals

    return result


def tril_vec2sym_mat(x: torch.Tensor, tril_dim=1):
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
    m = m.view(mat_size, mat_size, -1)

    # Handle upper triangle with the tril elements of the input.
    m[tuple(torch.tril_indices(mat_size, mat_size))] = v.view(tril_size, -1)
    # Handle the lower triangle (sans the diagonal) one element at a time.
    for r, c in zip(*torch.triu_indices(mat_size, mat_size, offset=1).tolist()):
        m[r, c] = m[c, r]
    m = m.view(*full_mat_shape)
    # Many algorithms require the matrix dims to be the last two dims, so rearrange
    # to that shape.
    m = einops.rearrange(m, "r c ... -> ... r c", r=mat_size, c=mat_size)

    return m

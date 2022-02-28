# -*- coding: utf-8 -*-
# Module for working with log and exp maps of SPD matrices & log-euclidean metrics.
import torch

import pitn.eig

EPSILON_LOG = 1e-5


def log_map(x: torch.Tensor, **eigh_kwargs) -> torch.Tensor:
    def eigval_eigvec_log_map(eigvals: torch.Tensor, eigvecs: torch.Tensor):
        log_eigvals = torch.log(torch.clamp_min(eigvals, min=EPSILON_LOG))
        return log_eigvals, eigvecs

    log_mapped = pitn.eig.eigh_decompose_apply_recompose(
        x, eigval_eigvec_log_map, **eigh_kwargs
    )

    return log_mapped


def exp_map(x: torch.Tensor, **eigh_kwargs) -> torch.Tensor:
    def eigval_eigvec_exp_map(eigvals: torch.Tensor, eigvecs: torch.Tensor):
        exp_eigvals = torch.exp(eigvals)
        exp_eigvals[
            torch.isclose(exp_eigvals, torch.as_tensor(EPSILON_LOG).to(exp_eigvals))
        ] = 0.0
        return exp_eigvals, eigvecs

    exp_mapped = pitn.eig.eigh_decompose_apply_recompose(
        x, eigval_eigvec_exp_map, **eigh_kwargs
    )

    return exp_mapped

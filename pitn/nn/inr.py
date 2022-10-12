# -*- coding: utf-8 -*-
from typing import Callable, List, Optional

import einops
import monai
import numpy as np
import torch
import torch.nn.functional as F

import pitn


class ResMLP(torch.nn.Module):
    def __init__(self, in_size, out_size, internal_size, n_layers, activate_fn):
        self.in_size = in_size
        self.out_size = out_size
        self.internal_size = internal_size
        self.n_layers = n_layers
        assert self.n_layers >= 3

        if isinstance(activate_fn, str):
            activate_fn = pitn.utils.torch_lookups.activation[activate_fn]
        self.activate_fn = activate_fn

        n_pre = (self.n_layers - 1) // 2
        pre_res_mlp = list()
        pre_res_mlp.append(torch.nn.Linear(self.in_size, self.out_size))
        pre_res_mlp.append(self.activate_fn())

        for _ in range(1, n_pre):
            pre_res_mlp.append(torch.nn.Linear(self.internal_size, self.internal_size))
            pre_res_mlp.append(self.activate_fn())

        self.pre_res = torch.nn.Sequential(*pre_res_mlp)
        self.res = torch.nn.linear(self.internal_size, self.internal_size)
        self.activate_callable = self.activate_fn()

        n_post = self.n_layers - n_pre - 1
        post_res_mlp = list()
        for _ in range(1, n_post):
            post_res_mlp.append(torch.nn.Linear(self.internal_size, self.internal_size))
            post_res_mlp.append(self.activate_fn())
        post_res_mlp.append(torch.nn.Linear(self.internal_size, self.out_size))
        self.post_res = torch.nn.Sequential(*post_res_mlp)

    def forward(self, x):

        y = self.pre_res(x)
        y_res = self.res(y)
        y_res = self.activate_callable(y_res)
        y = y + y_res
        y = self.post_res(y)

        return y

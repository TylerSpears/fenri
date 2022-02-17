# -*- coding: utf-8 -*-
from functools import partial

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr
import monai
from box import Box


class _CaseInsensitiveBox(Box):
    """Almost exactly like ConfigBox!"""

    _protected_keys = dir(Box)

    def __getitem__(self, item, *args, **kwargs):
        if isinstance(item, str):
            item = item.casefold()
        return super().__getitem__(item, *args, **kwargs)

    def __setitem__(self, key, value):
        if isinstance(key, str):
            key = key.casefold()
        return super().__setitem__(key, value)

    def __delitem__(self, key):
        if isinstance(key, str):
            key = key.casefold()
        return super().__delitem__(key)


loss_fn = _CaseInsensitiveBox(
    {
        "mse": partial(nn.MSELoss, reduction="mean"),
        "sse": partial(nn.MSELoss, reduction="sum"),
        "l1": partial(nn.L1Loss, reduction="mean"),
        "rmse": partial(monai.metrics.RMSEMetric, reduction="mean"),
        "psnr": partial(monai.metrics.PSNRMetric, reduction="mean"),
        "mae": partial(monai.metrics.MAEMetric, reduction="mean"),
    },
    frozen_box=True,
)

activate_fn = _CaseInsensitiveBox(
    {
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
    },
    frozen_box=True,
)

optim = _CaseInsensitiveBox(
    {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
        "sgd": torch.optim.SGD,
        "adadelta": torch.optim.Adadelta,
    },
    frozen_box=True,
)

lr_scheduler = _CaseInsensitiveBox(
    {
        "constant": lr.ConstantLR,
        "linear": lr.LinearLR,
        "exponential": lr.ExponentialLR,
        "cyclic": lr.CyclicLR,
        "onecycle": lr.OneCycleLR,
        'step': lr.StepLR,
        'multistep': lr.MultiStepLR,
        'reduce_on_plateau': lr.ReduceLROnPlateau,
    },
    frozen_box=True,
)

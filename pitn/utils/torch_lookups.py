# -*- coding: utf-8 -*-
from functools import partial

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr
import monai
import pyrsistent


class _CaseInsensitiveDict(pyrsistent.CheckedPMap):

    __key_type__ = str

    def __getitem__(self, key, *args, **kwargs):
        try:
            v = super().__getitem__(key, *args, **kwargs)
        except KeyError:
            v = super().__getitem__(key.casefold(), *args, **kwargs)
        return v


loss_fn = _CaseInsensitiveDict.create(
    {
        "mse": partial(nn.MSELoss, reduction="mean"),
        "sse": partial(nn.MSELoss, reduction="sum"),
        "l1": partial(nn.L1Loss, reduction="mean"),
        "rmse": partial(monai.metrics.RMSEMetric, reduction="mean"),
        "psnr": partial(monai.metrics.PSNRMetric, reduction="mean"),
        "mae": partial(monai.metrics.MAEMetric, reduction="mean"),
        "frobenius_norm": lambda: (
            lambda x, y: torch.linalg.matrix_norm((x - y).float(), ord="fro").mean()
        ),
    },
)

activation = _CaseInsensitiveDict.create(
    {
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
    },
)

optim = _CaseInsensitiveDict.create(
    {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
        "sgd": torch.optim.SGD,
        "adadelta": torch.optim.Adadelta,
    },
)

lr_scheduler = _CaseInsensitiveDict.create(
    {
        "constant": lr.ConstantLR,
        "linear": lr.LinearLR,
        "exponential": lr.ExponentialLR,
        "cyclic": lr.CyclicLR,
        "onecycle": lr.OneCycleLR,
        "step": lr.StepLR,
        "multistep": lr.MultiStepLR,
        "reduce_on_plateau": lr.ReduceLROnPlateau,
        "cosine_annealing_warm_restarts": lr.CosineAnnealingWarmRestarts,
        "chained_scheduler": lr.ChainedScheduler,
        "sequential": lr.SequentialLR,
    },
)

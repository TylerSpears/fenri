# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
import einops

import pitn
import pitn.nn.layers as layers
import pitn.riemann.log_euclid as log_euclid

DEBUG_RAND_PROB = 0.01


class CascadeUpsample(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        interior_channels: int,
        upscale_factor: int,
        n_res_units: int,
        n_dense_units: int,
        activate_fn,
        upsample_activate_fn,
        center_crop_output_side_amt=None,
        input_scaler=None,
        output_descaler=None,
    ):
        super().__init__()

        self.channels = channels
        self.interior_channels = interior_channels
        self.upscale_factor = upscale_factor

        if isinstance(activate_fn, str):
            activate_fn = pitn.utils.torch_lookups.activation[activate_fn]
        if isinstance(upsample_activate_fn, str):
            upsample_activate_fn = pitn.utils.torch_lookups.activation[
                upsample_activate_fn
            ]

        self.input_scaler = (
            input_scaler if input_scaler is not None else torch.nn.Identity()
        )
        self.output_descaler = (
            output_descaler if output_descaler is not None else torch.nn.Identity()
        )

        # Pad to maintain the same input shape.
        self.pre_conv = torch.nn.Conv3d(
            self.channels,
            self.interior_channels,
            kernel_size=3,
            padding="same",
            padding_mode="reflect",
            groups=self.channels,
        )

        # Construct the densely-connected cascading layers.
        # Create n_dense_units number of dense units.
        top_level_units = list()
        for _ in range(n_dense_units):
            # Create n_res_units number of residual units for every dense unit.
            res_layers = list()
            for _ in range(n_res_units):
                res_layers.append(
                    layers.ResBlock3dNoBN(
                        self.interior_channels,
                        kernel_size=3,
                        activate_fn=activate_fn,
                        padding="same",
                        padding_mode="reflect",
                    )
                )
            top_level_units.append(
                layers.DenseCascadeBlock3d(self.interior_channels, *res_layers)
            )

        self.activate_fn = activate_fn()

        # Wrap everything into a densely-connected cascade.
        self.cascade = layers.DenseCascadeBlock3d(
            self.interior_channels, *top_level_units
        )

        self.upsample = layers.upsample.ICNRUpsample3d(
            self.interior_channels,
            self.channels,
            self.upscale_factor,
            activate_fn=upsample_activate_fn,
            blur=True,
            zero_bias=True,
        )

        self.post_conv = torch.nn.Conv3d(
            self.channels,
            self.channels,
            kernel_size=3,
            padding="same",
            padding_mode="reflect",
            groups=self.channels,
        )

        # "Padding" by a negative amount will perform cropping!
        # <https://github.com/pytorch/pytorch/issues/1331>
        if center_crop_output_side_amt is not None and center_crop_output_side_amt > 0:
            crop = -center_crop_output_side_amt
            self.output_cropper = torch.nn.ConstantPad3d(crop, 0)
            self._crop = True
        else:
            self._crop = False
            self.output_cropper = torch.nn.Identity()

    def crop_full_output(self, x):
        if self._crop:
            x = self.output_cropper(x)
        return x

    def transform_input(self, x):
        return self.input_scaler(x.float()).float()

    def transform_output(self, y_pred, crop=True):
        y_pred = self.output_descaler(y_pred.float())
        if crop:
            y_pred = self.crop_full_output(y_pred)
        return y_pred.float()

    def transform_ground_truth_for_training(self, y, crop=True):
        y = self.input_scaler(y.float())
        if crop:
            y = self.crop_full_output(y)
        return y.float()

    def forward(
        self,
        x: torch.Tensor,
        transform_x: bool = True,
        transform_y: bool = True,
        debug=False,
    ):
        # debug = (torch.rand(1).item() <= DEBUG_RAND_PROB) or debug
        # if debug:
        #     breakpoint()
        if transform_x:
            x = self.transform_input(x)
        y = self.pre_conv(x)
        y = self.activate_fn(y)
        y = self.cascade(y)
        y = self.upsample(y)

        y = self.activate_fn(y)
        y = self.post_conv(y)

        if transform_y:
            y = self.transform_output(y, crop=self._crop)

        return y


class CascadeUpsampleLogEuclid(CascadeUpsample):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Coefficients that scale the off-diagonal tensor componenents such that
        # the matrix log space vectors can be computed with regular Euclidean distance.
        mat_norm_coeffs = torch.ones(6, dtype=torch.float)
        mat_norm_coeffs[torch.as_tensor([1, 3, 4])] = np.sqrt(2)
        mat_norm_coeffs = mat_norm_coeffs.reshape(1, -1, 1, 1, 1)
        self.register_buffer("mat_norm_coeffs", mat_norm_coeffs, persistent=True)

    def transform_input(self, x):
        x = x.float()
        if x.ndim == 5:
            x = pitn.eig.tril_vec2sym_mat(x, tril_dim=1)
        x = log_euclid.log_map(x.float())
        x = pitn.eig.sym_mat2tril_vec(x, dim1=-2, dim2=-1, tril_dim=1)
        # Scale off-diagonals to use regular L2 norm.
        x = x * self.mat_norm_coeffs
        x = self.input_scaler(x.float()).float()
        return x

    def transform_output(self, y_pred, crop=True):
        # Need to convert to a the lower triangular vector to perform scaling as a
        # log-euclidean metric.
        y_pred = y_pred.float()
        if y_pred.ndim > 5:
            y_pred = pitn.eig.sym_mat2tril_vec(y_pred, dim1=-2, dim2=-1, tril_dim=1)
        # Descale off-diagonals used to make log-euclidean domain L2 distance.
        y_pred = self.output_descaler(y_pred).float()
        y_pred = (y_pred / self.mat_norm_coeffs).float()
        y_pred = pitn.eig.tril_vec2sym_mat(y_pred, tril_dim=1)
        y_pred = log_euclid.exp_map(y_pred.float()).float()
        y_pred = pitn.eig.sym_mat2tril_vec(y_pred, dim1=-2, dim2=-1, tril_dim=1)
        if crop:
            y_pred = self.crop_full_output(y_pred)
        return y_pred.float()

    def transform_ground_truth_for_training(self, y, crop=True):
        # Assume that the ground truth is already the matrix log, scaled for the
        # euclidean norm.
        y = y.float()
        if y.ndim > 5:
            y = pitn.eig.sym_mat2tril_vec(y, tril_dim=1).float()
        y = self.input_scaler(y).float()
        if crop:
            y = self.crop_full_output(y)
        return y.float()


class CascadeUpsampleModeRefine(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        interior_channels: int,
        upscale_factor: int,
        n_res_units: int,
        n_dense_units: int,
        activate_fn,
        upsample_activate_fn,
        center_crop_output_side_amt=None,
        input_scaler=None,
        multi_modal_input_scaler=None,
        output_descaler=None,
    ):
        super().__init__()

        self.channels = channels
        self.interior_channels = interior_channels
        self.upscale_factor = upscale_factor

        if isinstance(activate_fn, str):
            activate_fn = pitn.utils.torch_lookups.activation[activate_fn]
        if isinstance(upsample_activate_fn, str):
            upsample_activate_fn = pitn.utils.torch_lookups.activation[
                upsample_activate_fn
            ]

        self.input_scaler = (
            input_scaler if input_scaler is not None else torch.nn.Identity()
        )
        self.multi_modal_input_scaler = (
            multi_modal_input_scaler
            if multi_modal_input_scaler is not None
            else torch.nn.Identity()
        )
        self.output_descaler = (
            output_descaler if output_descaler is not None else torch.nn.Identity()
        )

        # Pad to maintain the same input shape.
        self.pre_conv = torch.nn.Conv3d(
            self.channels,
            self.interior_channels,
            kernel_size=3,
            padding="same",
            padding_mode="reflect",
            groups=self.channels,
        )

        # Construct the densely-connected cascading layers.
        # Create n_dense_units number of dense units.
        top_level_units = list()
        for _ in range(n_dense_units):
            # Create n_res_units number of residual units for every dense unit.
            res_layers = list()
            for _ in range(n_res_units):
                res_layers.append(
                    layers.ResBlock3dNoBN(
                        self.interior_channels,
                        kernel_size=3,
                        activate_fn=activate_fn,
                        padding="same",
                        padding_mode="reflect",
                    )
                )
            top_level_units.append(
                layers.DenseCascadeBlock3d(self.interior_channels, *res_layers)
            )

        self.activate_fn = activate_fn()

        # Wrap everything into a densely-connected cascade.
        self.cascade = layers.DenseCascadeBlock3d(
            self.interior_channels, *top_level_units
        )

        self.upsample = layers.upsample.ICNRUpsample3d(
            self.interior_channels,
            self.channels,
            self.upscale_factor,
            activate_fn=upsample_activate_fn,
            blur=True,
            zero_bias=True,
        )

        self.residual_mode_refine = torch.nn.Sequential(
            torch.nn.LazyConv3d(out_channels=self.channels, kernel_size=1),
            self.activate_fn,
            torch.nn.Conv3d(
                self.channels,
                self.channels,
                kernel_size=3,
                padding="same",
                padding_mode="reflect",
            ),
            self.activate_fn,
            torch.nn.Conv3d(
                self.channels,
                self.channels,
                kernel_size=3,
                padding="same",
                padding_mode="reflect",
            ),
        )

        self.post_conv = torch.nn.Conv3d(
            self.channels,
            self.channels,
            kernel_size=3,
            padding="same",
            padding_mode="reflect",
            groups=self.channels,
        )

        # "Padding" by a negative amount will perform cropping!
        # <https://github.com/pytorch/pytorch/issues/1331>
        if center_crop_output_side_amt is not None and center_crop_output_side_amt > 0:
            crop = -center_crop_output_side_amt
            self.output_cropper = torch.nn.ConstantPad3d(crop, 0)
            self._crop = True
        else:
            self._crop = False
            self.output_cropper = torch.nn.Identity()

    def crop_full_output(self, x):
        if self._crop:
            x = self.output_cropper(x)
        return x

    def transform_input(self, x):
        return self.input_scaler(x)

    def transform_input_mode_refine(self, x_mode_refine):
        return self.multi_modal_input_scaler(x_mode_refine)

    def transform_output(self, y_pred):
        return self.output_descaler(y_pred)

    def transform_ground_truth_for_training(self, y):
        return self.input_scaler(y)

    def forward(
        self,
        x: torch.Tensor,
        x_mode_refine: torch.Tensor,
        transform_x: bool = True,
        transform_x_mode_refine: bool = True,
        transform_y: bool = True,
        debug=False,
    ):
        # debug = (torch.rand(1).item() <= DEBUG_RAND_PROB) or debug
        # if debug:
        #     breakpoint()
        if transform_x:
            x = self.transform_input(x)
        y = self.pre_conv(x)
        y = self.activate_fn(y)
        y = self.cascade(y)
        y = self.upsample(y)

        # Integrate the extra HR multi-modal refinement input (i.e. a T2 or T1 patch).
        # Size of the modal refinement input may be one voxel different in shape due to
        # an uneven division `hr_size / downscale_factor`.
        if x_mode_refine.shape[2:] != y.shape[2:]:
            raise RuntimeError(
                f"ERROR: Mode refine input shape {x_mode_refine.shape[2:]} "
                + f"incompatible with upsample output shape {y.shape[2:]}. "
                + "This may be due to roundoff error in downsampling of FR data. "
                + f"Does {x.shape[2:]} x {self.upscale_factor} "
                + f"=|{x_mode_refine.shape[2:]}| ?"
            )
        if transform_x_mode_refine:
            x_mode_refine = self.transform_input_mode_refine(x_mode_refine)

        # Concatinate the multi-mode refinement input and the upsampled x to create
        # an additive residual to refine the upsample output.
        input_residual_mode_refine = torch.cat([x_mode_refine, y], dim=1)
        multi_mode_residual = self.residual_mode_refine(input_residual_mode_refine)
        y = y + multi_mode_residual

        y = self.activate_fn(y)
        y = self.post_conv(y)
        if self._crop:
            y = self.crop_full_output(y)
        if transform_y:
            y = self.transform_output(y)
        return y


class CascadeLogEuclid(CascadeUpsampleModeRefine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mat_norm_coeffs = torch.ones(6)
        mat_norm_coeffs[torch.as_tensor([1, 3, 4])] = np.sqrt(2)
        mat_norm_coeffs = mat_norm_coeffs.reshape(1, -1, 1, 1, 1)
        self.register_buffer("mat_norm_coeffs", mat_norm_coeffs, persistent=True)

    def transform_input(self, x):
        x = x.float()
        x = pitn.eig.tril_vec2sym_mat(x, tril_dim=1)
        x = log_euclid.log_map(x.float())
        x = pitn.eig.sym_mat2tril_vec(x, dim1=-2, dim2=-1, tril_dim=1)
        # Scale off-diagonals to use regular L2 norm.
        x = x * self.mat_norm_coeffs
        return self.input_scaler(x.float()).float()

    def transform_input_mode_refine(self, x_mode_refine):
        return self.multi_modal_input_scaler(x_mode_refine.float()).float()

    def transform_output(self, y_pred):
        # Need to convert to a the lower triangular vector to perform scaling as a
        # log-euclidean metric.
        y_pred = y_pred.float()
        if y_pred.ndim > 5:
            y_pred = pitn.eig.sym_mat2tril_vec(y_pred, dim1=-2, dim2=-1, tril_dim=1)
        # Descale off-diagonals used to make log-euclidean domain L2 distance.
        y_pred = (y_pred / self.mat_norm_coeffs).float()
        y_pred = self.output_descaler(y_pred)
        y_pred = pitn.eig.tril_vec2sym_mat(y_pred, tril_dim=1)
        y_pred = log_euclid.exp_map(y_pred.float())
        y_pred = pitn.eig.sym_mat2tril_vec(y_pred, dim1=-2, dim2=-1, tril_dim=1)
        return y_pred.float()

    def transform_ground_truth_for_training(self, y):
        y = y.float()
        if y.ndim == 5:
            y = pitn.eig.tril_vec2sym_mat(y, tril_dim=1)
        y = log_euclid.log_map(y.float())
        y = pitn.eig.sym_mat2tril_vec(y, dim1=-2, dim2=-1, tril_dim=1)
        # Scale off-diagonals to use regular L2 norm.
        y = (y * self.mat_norm_coeffs).float()
        y = self.input_scaler(y).float()
        return y


class CascadeLogEuclidMultiOutput(CascadeLogEuclid):
    def forward(
        self,
        x: torch.Tensor,
        x_mode_refine: torch.Tensor,
        return_y_upsample=False,
        transform_x: bool = True,
        transform_x_mode_refine: bool = True,
        transform_y: bool = True,
        debug=False,
    ):
        # debug = (torch.rand(1).item() <= DEBUG_RAND_PROB) or debug
        # if debug:
        #     breakpoint()
        if transform_x:
            x = self.transform_input(x)
        y = self.pre_conv(x)
        y = self.activate_fn(y)
        y = self.cascade(y)
        y_upsampled = self.upsample(y)

        # Integrate the extra HR multi-modal refinement input (i.e. a T2 or T1 patch).
        # Size of the modal refinement input may be one voxel different in shape due to
        # an uneven division `hr_size / downscale_factor`.
        if x_mode_refine.shape[2:] != y_upsampled.shape[2:]:
            raise RuntimeError(
                f"ERROR: Mode refine input shape {x_mode_refine.shape[2:]} "
                + f"incompatible with upsample output shape {y_upsampled.shape[2:]}. "
                + "This may be due to roundoff error in downsampling of FR data. "
                + f"Does {x.shape[2:]} x {self.upscale_factor} "
                + f"=|{x_mode_refine.shape[2:]}| ?"
            )
        if transform_x_mode_refine:
            x_mode_refine = self.transform_input_mode_refine(x_mode_refine)

        # Concatinate the multi-mode refinement input and the upsampled x to create
        # an additive residual to refine the upsample output.
        input_residual_mode_refine = torch.cat([x_mode_refine, y_upsampled], dim=1)
        multi_mode_residual = self.residual_mode_refine(input_residual_mode_refine)
        y = y_upsampled + multi_mode_residual

        y = self.activate_fn(y)
        y = self.post_conv(y)
        if self._crop:
            y = self.crop_full_output(y)
            if return_y_upsample:
                y_upsampled = self.crop_full_output(y_upsampled)
        if transform_y:
            y = self.transform_output(y)
            if return_y_upsample:
                y_upsampled = self.transform_output(y_upsampled)

        if return_y_upsample:
            return y, y_upsampled
        else:
            return y

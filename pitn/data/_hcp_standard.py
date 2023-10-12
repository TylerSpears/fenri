# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch

# From <https://wiki.humanconnectome.org/display/PublicData/Gradient+Vector+Direction+table+for+HCP+3T+dMRI>
# <https://wiki.humanconnectome.org/download/attachments/88901015/HCP_Diffusion_Vectors.txt?version=1&modificationDate=1484832206965&api=v2>
# Gradient direction table for Human Connectome Project (HCP) dMRI (WU-Minn
# consortium). Each table acquired once with right-to-left and left-to-right
# phase encoding polarities, using the CMRR multiband diffusion sequence,
# with bmax = 3000 s/mm^2.  Shells at 1000, 2000, and 3000 s/mm^2 are
# obtained through appropriate weighting of the vector norm.
#
# The diffusion directions were obtained using a toolbox available from
# INRIA that returns uniformly distributed directions in multiple q-space
# shells.  The directions are optimized so that every subset of the first M
# directions is also isotropic. References and the INRIA toolbox can be
# found at:
# http://www-sop.inria.fr/members/Emmanuel.Caruyer/q-space-sampling.php
#
# NOTE: These gradient directions were tested for use on the HCP customized
# Skyra scanner with 100 mT/m gradients and may not be optimal for use with
# other scanner systems.
#
# Further questions can be directed to the HCP Data Users mailing list
# (hcp-user@humanconnectome.org) by signing up at
# http://www.humanconnectome.org/contact

HCP_STANDARD_3T_BVEC = torch.tensor(
    [
        # segment 1/3 with interleaved shells. Interleaved bval ratio 1:2:3
        # [directions=95]
        # CoordinateSystem = xyz
        # Normalisation = none
        (0.000, 0.000, 0.000),
        (-0.533, 0.147, -0.167),
        (-0.303, -0.642, 0.404),
        (0.106, -0.563, -0.819),
        (0.110, 0.550, 0.137),
        (0.430, -0.713, 0.554),
        (0.590, -0.467, -0.317),
        (-0.193, -0.092, 0.536),
        (0.483, 0.180, 0.633),
        (0.898, 0.410, -0.161),
        (-0.106, 0.462, -0.330),
        (-0.212, 0.048, 0.787),
        (-0.852, 0.483, 0.203),
        (-0.234, -0.234, -0.473),
        (0.795, 0.184, 0.027),
        (0.281, 0.960, -0.011),
        (0.000, 0.000, 0.000),
        (-0.519, -0.143, 0.208),
        (-0.356, 0.686, -0.264),
        (0.102, 0.251, -0.963),
        (-0.753, -0.183, -0.632),
        (-0.318, 0.393, 0.280),
        (-0.146, -0.714, -0.369),
        (0.453, 0.339, 0.117),
        (0.252, -0.343, 0.697),
        (-0.697, -0.007, 0.717),
        (-0.255, 0.154, -0.494),
        (0.647, 0.145, -0.476),
        (0.318, -0.929, -0.189),
        (-0.276, -0.435, 0.260),
        (0.711, -0.308, 0.256),
        (0.508, 0.690, 0.516),
        (0.000, 0.000, 0.000),
        (0.293, 0.713, -0.638),
        (-0.347, 0.460, -0.036),
        (-0.500, -0.640, -0.085),
        (-0.130, 0.242, 0.507),
        (-0.198, 0.516, 0.601),
        (0.847, -0.308, 0.433),
        (-0.521, 0.150, 0.199),
        (0.282, 0.105, 0.954),
        (-0.033, 0.815, -0.041),
        (0.579, -0.515, -0.632),
        (0.478, 0.107, 0.305),
        (0.198, 0.364, -0.704),
        (0.003, -0.567, 0.107),
        (0.645, 0.346, 0.362),
        (-1.000, -0.001, -0.011),
        (0.000, 0.000, 0.000),
        (0.022, 0.275, -0.507),
        (0.778, -0.181, -0.172),
        (-0.082, 0.935, -0.345),
        (0.638, 0.726, -0.258),
        (-0.029, -0.442, -0.371),
        (0.131, 0.265, 0.761),
        (0.387, -0.062, -0.424),
        (0.449, -0.357, 0.819),
        (0.368, -0.711, -0.163),
        (0.376, -0.323, 0.296),
        (0.553, -0.214, 0.561),
        (-0.044, -0.867, -0.496),
        (0.333, 0.471, -0.028),
        (-0.585, -0.379, 0.717),
        (0.611, 0.465, -0.277),
        (0.000, 0.000, 0.000),
        (0.073, 0.047, 0.571),
        (0.496, -0.216, -0.611),
        (0.682, -0.719, 0.132),
        (0.572, 0.071, 0.035),
        (-0.060, 0.647, -0.495),
        (-0.832, -0.474, -0.288),
        (-0.154, 0.533, 0.160),
        (-0.331, -0.514, -0.541),
        (0.917, -0.076, -0.392),
        (0.355, 0.254, -0.378),
        (0.623, -0.525, 0.052),
        (0.276, -0.186, -0.943),
        (0.324, 0.364, 0.309),
        (-0.101, 0.091, -0.805),
        (0.277, 0.762, -0.095),
        (0.000, 0.000, 0.000),
        (0.493, -0.300, -0.023),
        (-0.330, -0.483, -0.811),
        (-0.551, 0.755, 0.355),
        (-0.381, 0.021, -0.433),
        (0.757, 0.015, 0.305),
        (-0.079, 0.614, -0.785),
        (0.098, 0.438, -0.363),
        (0.442, 0.191, -0.660),
        (-0.553, -0.816, -0.167),
        (-0.308, 0.249, 0.420),
        (0.154, -0.713, -0.367),
        (0.696, -0.137, 0.705),
        (0.357, -0.920, 0.161),
        (0.476, 0.308, -0.109),
        # segment 2/3 with interleaved shells. Interleaved bval ratio 1:2:3
        # [directions=96]
        # CoordinateSystem = xyz
        # Normalisation = none
        (0.000, 0.000, 0.000),
        (-0.813, -0.341, 0.472),
        (0.208, -0.514, 0.159),
        (-0.772, -0.123, 0.235),
        (0.427, -0.503, 0.481),
        (-0.055, -0.276, -0.504),
        (-0.678, -0.452, -0.055),
        (-0.286, -0.890, 0.354),
        (-0.197, 0.319, -0.439),
        (0.135, -0.311, -0.743),
        (-0.935, 0.333, -0.119),
        (-0.263, -0.482, -0.178),
        (-0.030, -0.762, 0.292),
        (-0.013, -0.247, -0.969),
        (0.568, -0.099, -0.034),
        (-0.370, 0.033, -0.727),
        (0.934, 0.143, 0.327),
        (0.000, 0.000, 0.000),
        (-0.110, 0.048, 0.565),
        (-0.265, 0.759, 0.595),
        (0.667, -0.211, -0.421),
        (-0.166, -0.540, 0.121),
        (-0.408, -0.631, -0.319),
        (-0.416, -0.141, 0.898),
        (-0.430, -0.088, 0.375),
        (0.431, 0.450, -0.528),
        (0.026, 0.985, 0.169),
        (0.455, -0.147, 0.324),
        (0.396, -0.561, -0.442),
        (-0.776, 0.316, 0.546),
        (0.141, -0.460, -0.319),
        (0.798, -0.151, 0.087),
        (-0.205, 0.251, -0.946),
        (0.000, 0.000, 0.000),
        (-0.398, 0.397, 0.131),
        (-0.688, 0.577, -0.439),
        (0.029, -0.454, 0.678),
        (0.195, 0.281, -0.465),
        (0.522, 0.187, 0.162),
        (-0.089, -0.534, -0.611),
        (0.613, 0.441, 0.656),
        (0.290, -0.762, 0.048),
        (-0.767, -0.641, -0.009),
        (-0.125, 0.564, 0.008),
        (-0.639, -0.043, -0.506),
        (0.205, 0.718, 0.665),
        (0.245, 0.061, 0.519),
        (0.165, 0.789, 0.130),
        (-0.245, -0.512, 0.823),
        (0.000, 0.000, 0.000),
        (0.059, -0.139, 0.557),
        (-0.569, 0.513, -0.282),
        (0.507, -0.288, -0.813),
        (-0.433, 0.234, 0.302),
        (-0.430, 0.027, 0.694),
        (-0.969, -0.142, 0.203),
        (-0.183, -0.423, -0.348),
        (0.475, 0.636, -0.192),
        (-0.534, -0.082, -0.841),
        (0.444, -0.338, 0.149),
        (-0.115, -0.159, 0.793),
        (0.120, -0.810, 0.575),
        (0.707, -0.394, -0.109),
        (-0.404, -0.336, 0.239),
        (-0.671, 0.731, 0.120),
        (0.000, 0.000, 0.000),
        (0.389, 0.214, 0.370),
        (0.583, 0.630, -0.514),
        (0.503, 0.398, 0.505),
        (-0.312, -0.895, -0.320),
        (-0.552, -0.010, 0.170),
        (-0.726, -0.325, 0.185),
        (0.028, 0.517, -0.255),
        (0.243, -0.568, 0.533),
        (0.615, -0.447, 0.650),
        (-0.094, 0.992, -0.082),
        (-0.255, 0.091, 0.510),
        (0.360, -0.378, -0.628),
        (0.270, -0.428, 0.278),
        (0.099, -0.791, -0.177),
        (0.958, -0.247, -0.146),
        (0.000, 0.000, 0.000),
        (0.002, -0.531, -0.226),
        (-0.031, -0.011, 0.999),
        (0.176, 0.565, -0.563),
        (0.549, 0.003, 0.180),
        (0.292, 0.183, 0.740),
        (-0.754, -0.216, -0.229),
        (0.529, 0.232, -0.006),
        (-0.347, 0.537, 0.769),
        (0.942, 0.316, 0.113),
        (-0.107, 0.362, 0.437),
        (-0.530, 0.606, 0.138),
        (-0.882, 0.067, -0.467),
        (-0.479, -0.866, 0.141),
        (0.205, 0.539, 0.021),
        # segment 3/3 with interleaved shells. Interleaved bval ratio 1:2:3
        # [directions=97]
        # CoordinateSystem = xyz
        # Normalisation = none
        (0.000, 0.000, 0.000),
        (-0.711, 0.022, 0.401),
        (-0.039, 0.378, -0.435),
        (0.510, -0.800, 0.317),
        (0.607, -0.329, 0.435),
        (-0.318, -0.102, 0.471),
        (0.003, -0.109, -0.809),
        (-0.826, -0.112, 0.553),
        (-0.290, 0.489, 0.102),
        (-0.578, -0.522, -0.246),
        (-0.205, 0.898, 0.390),
        (0.345, -0.214, 0.410),
        (-0.163, 0.752, -0.273),
        (0.274, -0.524, 0.806),
        (-0.123, -0.162, -0.540),
        (0.681, 0.617, 0.395),
        (-0.581, -0.344, 0.460),
        (0.000, 0.000, 0.000),
        (-0.131, -0.437, -0.890),
        (0.484, -0.059, -0.309),
        (-0.029, 0.644, 0.502),
        (-0.371, -0.427, -0.113),
        (-0.402, 0.225, -0.674),
        (-0.739, 0.532, 0.413),
        (-0.238, -0.373, 0.371),
        (0.815, -0.009, -0.055),
        (-0.127, -0.833, 0.539),
        (-0.528, 0.225, -0.066),
        (-0.336, -0.735, -0.116),
        (-0.400, -0.349, 0.848),
        (0.833, -0.524, 0.178),
        (-0.173, 0.066, -0.547),
        (-0.535, 0.381, 0.485),
        (0.000, 0.000, 0.000),
        (0.016, 0.576, 0.036),
        (-0.489, -0.294, -0.821),
        (-0.053, 0.283, -0.764),
        (0.433, 0.284, 0.256),
        (0.211, 0.744, -0.264),
        (0.779, 0.529, -0.337),
        (-0.501, 0.072, 0.862),
        (-0.468, 0.287, 0.178),
        (0.231, 0.411, 0.667),
        (-0.490, 0.871, 0.046),
        (-0.087, -0.156, 0.549),
        (0.475, -0.656, 0.102),
        (-0.238, 0.359, 0.385),
        (-0.705, 0.136, -0.389),
        (-0.424, 0.132, -0.896),
        (0.000, 0.000, 0.000),
        (-0.968, 0.099, -0.229),
        (-0.385, -0.412, 0.124),
        (-0.582, 0.027, 0.572),
        (0.477, -0.019, 0.325),
        (-0.291, -0.197, 0.737),
        (-0.134, -0.975, 0.180),
        (-0.340, 0.437, -0.164),
        (-0.611, -0.534, 0.091),
        (-0.029, 0.724, 0.689),
        (0.151, 0.342, 0.439),
        (-0.323, 0.672, 0.332),
        (0.812, 0.342, 0.473),
        (0.451, 0.207, -0.295),
        (-0.004, 0.433, -0.901),
        (0.617, 0.217, 0.489),
        (0.000, 0.000, 0.000),
        (0.044, -0.157, -0.554),
        (-0.738, 0.344, -0.061),
        (0.324, 0.804, 0.499),
        (-0.092, 0.531, -0.207),
        (-0.292, 0.217, 0.731),
        (-0.301, 0.854, -0.425),
        (0.560, 0.119, -0.078),
        (0.461, 0.559, -0.376),
        (-0.709, 0.200, 0.676),
        (0.360, 0.107, 0.439),
        (0.252, 0.630, 0.454),
        (0.456, 0.784, -0.421),
        (-0.122, 0.251, -0.505),
        (0.453, -0.713, -0.535),
        (-0.384, 0.402, -0.598),
        (0.000, 0.000, 0.000),
        (0.141, 0.498, 0.257),
        (0.112, 0.806, -0.066),
        (0.889, -0.305, -0.342),
        (0.168, 0.493, -0.249),
        (-0.251, 0.472, 0.219),
        (0.713, 0.676, 0.185),
        (-0.235, -0.045, 0.971),
        (0.704, -0.309, -0.275),
        (0.212, 0.029, 0.788),
        (-0.466, 0.244, -0.237),
        (0.740, -0.299, 0.602),
        (0.252, -0.665, 0.402),
        (0.386, -0.167, -0.395),
        (0.804, 0.040, 0.138),
        (0.462, 0.553, -0.693),
    ]
    # Transpose to be 3 x 288
).T

HCP_STANDARD_3T_BVAL = (
    torch.Tensor(
        [
            5,
            1000,
            1995,
            3005,
            995,
            2990,
            2005,
            990,
            1990,
            3000,
            1000,
            1985,
            2995,
            1005,
            1995,
            2995,
            5,
            995,
            2000,
            3010,
            3005,
            995,
            2005,
            995,
            1990,
            2985,
            1005,
            2000,
            3000,
            995,
            1995,
            2990,
            5,
            3005,
            1000,
            2000,
            990,
            1990,
            2990,
            995,
            2985,
            1995,
            3005,
            995,
            2005,
            1000,
            1990,
            2995,
            5,
            1005,
            2005,
            2995,
            3000,
            1005,
            1985,
            1005,
            2985,
            2005,
            995,
            1990,
            3005,
            1000,
            2990,
            2000,
            5,
            990,
            2005,
            2995,
            1000,
            2005,
            3000,
            995,
            2005,
            3005,
            1005,
            2000,
            3010,
            990,
            2005,
            1995,
            5,
            1000,
            3010,
            2990,
            1005,
            1990,
            3005,
            1000,
            2010,
            3000,
            990,
            2005,
            2990,
            2995,
            1000,
            5,
            2990,
            995,
            1995,
            1995,
            1005,
            2000,
            2995,
            1005,
            2010,
            2995,
            1005,
            1995,
            3010,
            1000,
            2005,
            2990,
            5,
            990,
            2985,
            2005,
            1000,
            2005,
            2985,
            995,
            2005,
            2990,
            995,
            2005,
            2990,
            1005,
            2000,
            3010,
            5,
            995,
            2995,
            1990,
            1005,
            995,
            2005,
            2990,
            2000,
            2995,
            1000,
            2005,
            2985,
            990,
            1995,
            2985,
            5,
            990,
            2000,
            3010,
            995,
            1990,
            2995,
            1005,
            2000,
            3005,
            1000,
            1990,
            2995,
            2000,
            995,
            2990,
            5,
            995,
            3005,
            1990,
            3005,
            1000,
            2000,
            1000,
            1990,
            2990,
            2995,
            990,
            2010,
            995,
            2005,
            3000,
            5,
            1005,
            2980,
            2005,
            1000,
            1985,
            2005,
            1000,
            2985,
            2995,
            990,
            1995,
            3005,
            2995,
            995,
            5,
            1995,
            1005,
            3000,
            1990,
            995,
            2010,
            2990,
            995,
            2005,
            2990,
            995,
            2000,
            2985,
            1005,
            2990,
            1995,
            5,
            3010,
            1005,
            1990,
            1000,
            2005,
            2990,
            995,
            2000,
            2995,
            1000,
            2000,
            2990,
            2995,
            1005,
            1990,
            5,
            995,
            3010,
            2005,
            995,
            2000,
            3000,
            2985,
            995,
            1990,
            2995,
            990,
            2000,
            995,
            2005,
            3010,
            5,
            3000,
            1000,
            1990,
            995,
            1990,
            3000,
            1000,
            2000,
            2985,
            990,
            1990,
            2990,
            1000,
            3005,
            1990,
            5,
            1005,
            2000,
            2990,
            1000,
            1990,
            3000,
            1000,
            2000,
            2985,
            995,
            1990,
            3000,
            1005,
            3005,
            2005,
            5,
            995,
            1995,
            3005,
            1000,
            995,
            2990,
            2985,
            2005,
            1990,
            1000,
            2990,
            1995,
            1005,
            1995,
            3005,
        ]
    )
    .int()
    .reshape(-1)
)


HCP_STANDARD_3T_GRAD_MRTRIX = torch.Tensor(
    [
        [0.0, 0.0, 0.0, 5.0],
        [0.9228320096, 0.2545146443, -0.2891424871, 1000.0],
        [0.3709524039, -0.7859783607, 0.4946032052, 1995.0],
        [-0.1060548445, -0.5632912969, -0.8194237516, 3005.0],
        [-0.1905153965, 0.9525769827, 0.2372782666, 995.0],
        [-0.4299602305, -0.7129340566, 0.5539487621, 2990.0],
        [-0.7225933321, -0.5719509934, -0.3882408242, 2005.0],
        [0.3344485547, -0.159426254, 0.9288312192, 990.0],
        [-0.5916798854, 0.2205018207, 0.7754314027, 1990.0],
        [-0.8978092358, 0.4099129028, -0.1609657984, 3000.0],
        [0.1835294922, 0.7999115606, -0.5713654004, 1000.0],
        [0.2596555851, 0.05878994379, 0.9639101201, 1985.0],
        [0.8518287996, 0.4829029463, 0.2029592093, 2995.0],
        [0.4053560347, -0.4053560347, -0.8193735231, 1005.0],
        [-0.9737135565, 0.2253626345, 0.03306951701, 1995.0],
        [-0.280904228, 0.9596728073, -0.01099625092, 2995.0],
        [0.0, 0.0, 0.0, 5.0],
        [0.8992842588, -0.2477796705, 0.3604067935, 995.0],
        [0.4358910637, 0.8399473869, -0.3232450585, 2000.0],
        [-0.1019605489, 0.2509029194, -0.9626275352, 3010.0],
        [0.7530293687, -0.1830071374, -0.6320246494, 3005.0],
        [0.5502644311, 0.6800437781, 0.484509562, 995.0],
        [0.1787314915, -0.874070445, -0.4517254821, 2005.0],
        [-0.7840480447, 0.5867379408, 0.2025024751, 995.0],
        [-0.3085673675, -0.4199944724, 0.8534581553, 1990.0],
        [0.6970184712, -0.007000185507, 0.7170190013, 2985.0],
        [0.442041994, 0.2669586944, -0.8563480198, 1005.0],
        [-0.7926814717, 0.1776488615, -0.5831783316, 2000.0],
        [-0.3180181275, -0.9290529575, -0.1890107739, 3000.0],
        [0.4782845072, -0.7538179733, 0.4505578691, 995.0],
        [-0.8712810038, -0.3774325586, 0.3137101786, 1995.0],
        [-0.5078933536, 0.6898551456, 0.5158916741, 2990.0],
        [0.0, 0.0, 0.0, 5.0],
        [-0.2928152918, 0.7125505224, -0.6375978026, 3005.0],
        [0.6010471753, 0.7967772352, -0.06235647928, 1000.0],
        [0.6122997294, -0.7837436537, -0.104090954, 2000.0],
        [0.2254441849, 0.4196730212, 0.8792323213, 990.0],
        [0.2425005152, 0.6319710395, 0.736074796, 1990.0],
        [-0.847100811, -0.3080366585, 0.4330515362, 2990.0],
        [0.9021161526, 0.2597263395, 0.3445702771, 995.0],
        [-0.2819062817, 0.1049651049, 0.9536829531, 2985.0],
        [0.04040663185, 0.9979213623, -0.05020217896, 1995.0],
        [-0.5790318476, -0.5150283273, -0.6320347629, 3005.0],
        [-0.8283867991, 0.1854338651, 0.5285731668, 995.0],
        [-0.2423814735, 0.4455901836, -0.8618007946, 2005.0],
        [-0.005199166213, -0.9826424143, 0.1854369283, 1000.0],
        [-0.7898903424, 0.4237241217, 0.4433183007, 1990.0],
        [0.9999390056, -0.0009999390056, -0.01099932906, 2995.0],
        [0.0, 0.0, 0.0, 5.0],
        [-0.03811514337, 0.4764392921, -0.878380804, 1005.0],
        [-0.952164533, -0.221518998, -0.2105042412, 2005.0],
        [0.08200106602, 0.9350121552, -0.3450044851, 2995.0],
        [-0.6377819159, 0.7257518353, -0.2579118092, 3000.0],
        [0.05019089145, -0.7649784146, -0.6420972665, 1005.0],
        [-0.1604607945, 0.3245962636, 0.9321424779, 1985.0],
        [-0.6702476992, -0.1073781844, -0.7343282286, 1005.0],
        [-0.4490424365, -0.3570337413, 0.8190774065, 2985.0],
        [-0.4504199617, -0.8702407412, -0.1995066678, 2005.0],
        [-0.6512631521, -0.559462761, 0.512696524, 995.0],
        [-0.6774570263, -0.2621623935, 0.6872574896, 1990.0],
        [0.04400790013, -0.8671556684, -0.496089056, 3005.0],
        [-0.576616677, 0.8155749395, -0.04848428515, 1000.0],
        [0.5850131629, -0.3790085278, 0.717016133, 2990.0],
        [-0.7485390321, 0.5696737314, -0.3393540293, 2000.0],
        [0.0, 0.0, 0.0, 5.0],
        [-0.1263931417, 0.08137640627, 0.9886367656, 990.0],
        [-0.607780615, -0.2646786549, -0.7486974914, 2005.0],
        [-0.6821674927, -0.7191765795, 0.1320324179, 2995.0],
        [-0.9905597291, 0.1229540923, 0.06061117224, 1000.0],
        [0.07345344449, 0.7920729764, -0.605990917, 2005.0],
        [0.8320649036, -0.4740369763, -0.2880224666, 3000.0],
        [0.2667071549, 0.9230838543, 0.2770983427, 995.0],
        [0.4054600948, -0.6296268542, -0.6627006383, 2005.0],
        [-0.9168491907, -0.07598750108, -0.3919355319, 3005.0],
        [-0.6147935084, 0.4398804257, -0.6546252005, 1005.0],
        [-0.7631354949, -0.6430917092, 0.06369670263, 2000.0],
        [-0.275997102, -0.185998047, -0.9429900987, 3010.0],
        [-0.5615048916, 0.6308264832, 0.5355092948, 990.0],
        [0.1237140477, 0.111465132, -0.9860377066, 2005.0],
        [-0.3393227098, 0.9334436999, -0.1163742145, 1995.0],
        [0.0, 0.0, 0.0, 5.0],
        [-0.8535878389, -0.5194246484, -0.03982255638, 1000.0],
        [0.330014851, -0.4830217365, -0.8110364975, 3010.0],
        [0.5510961747, 0.755131782, 0.3550619637, 2990.0],
        [0.6601513663, 0.03638629578, -0.7502507654, 1005.0],
        [-0.9273876301, 0.01837624102, 0.373650234, 1990.0],
        [0.07902125958, 0.6141652327, -0.7852112502, 3005.0],
        [-0.1697706067, 0.7587706708, -0.628844186, 1000.0],
        [-0.541021504, 0.2337898354, -0.8078601643, 2010.0],
        [0.5531233603, -0.8161820289, -0.1670372535, 3000.0],
        [0.533526338, 0.4313248641, 0.7275359154, 990.0],
        [-0.1885955289, -0.8731728057, -0.4494451889, 2005.0],
        [-0.6959269315, -0.1369856173, 0.7049259867, 2990.0],
        [-0.3570410621, -0.9201058183, 0.1610185182, 2995.0],
        [-0.8244714373, 0.5334815182, -0.1887970308, 1000.0],
        [0.0, 0.0, 0.0, 5.0],
        [0.8129861794, -0.3409942031, 0.4719919762, 2990.0],
        [-0.3605870921, -0.8910661794, 0.2756410944, 995.0],
        [0.9457361934, -0.1506807666, 0.2878860174, 1995.0],
        [-0.5229533786, -0.6160317317, 0.5890879979, 1995.0],
        [0.09527942213, -0.4781294638, -0.8731059773, 1005.0],
        [0.8301614164, -0.5534409442, -0.06734347773, 2000.0],
        [0.2861127506, -0.8903508674, 0.3541395585, 2995.0],
        [0.3412356782, 0.5525592962, -0.7604185927, 1005.0],
        [-0.1652998576, -0.3808018941, -0.9097614383, 2010.0],
        [0.9353391219, 0.3331207782, -0.119043161, 2995.0],
        [0.4556362216, -0.8350443301, -0.3083773667, 1005.0],
        [0.03673845207, -0.9331566826, 0.3575876002, 1995.0],
        [0.01299909659, -0.2469828353, -0.9689326615, 3010.0],
        [-0.9834395787, -0.1714093632, -0.05886786211, 1000.0],
        [0.4532061377, 0.04042108796, -0.8904888165, 2005.0],
        [-0.9341242468, 0.1430190228, 0.3270434997, 2990.0],
        [0.0, 0.0, 0.0, 5.0],
        [0.1904411469, 0.08310159138, 0.9781749818, 990.0],
        [0.2649561534, 0.7588744167, 0.5949015519, 2985.0],
        [-0.816914428, -0.2584242044, -0.5156236495, 2005.0],
        [0.287320672, -0.9346576076, 0.209432538, 1000.0],
        [0.4998011267, -0.7729767426, -0.390775881, 2005.0],
        [0.4161371398, -0.1410464825, 0.8982960373, 2985.0],
        [0.7448537291, -0.1524351818, 0.6495817405, 995.0],
        [-0.5277153216, 0.550978874, -0.6464818789, 2005.0],
        [-0.02600699682, 0.985265072, 0.1690454794, 2990.0],
        [-0.7877562664, -0.2545058707, 0.560951715, 995.0],
        [-0.484913753, -0.68696115, -0.5412421182, 2005.0],
        [0.7759425824, 0.3159766186, 0.5459596005, 2990.0],
        [-0.2442526288, -0.7968525478, -0.552599919, 1005.0],
        [-0.9769747392, -0.1848661474, 0.1065122836, 2000.0],
        [0.2050059453, 0.2510072793, -0.9460274352, 3010.0],
        [0.0, 0.0, 0.0, 5.0],
        [0.6895210366, 0.6877885717, 0.226952904, 995.0],
        [0.6883463253, 0.5772904502, -0.4392209837, 2995.0],
        [-0.035518285, -0.5560448756, 0.830393008, 1990.0],
        [-0.3378119016, 0.4867956121, -0.8055514578, 1005.0],
        [-0.9036377215, 0.3237169616, 0.2804392929, 995.0],
        [0.1090242642, -0.6541455849, -0.7484699483, 2005.0],
        [-0.6128204699, 0.4408708438, 0.6558078764, 2990.0],
        [-0.3550744759, -0.9329887954, 0.05877094774, 2000.0],
        [0.767287403, -0.6412401894, -0.009003372395, 2995.0],
        [0.2163598168, 0.9762154934, 0.01384702827, 1000.0],
        [0.7828825055, -0.05268223433, -0.6199351295, 2005.0],
        [-0.2050231689, 0.7180811478, 0.6650751577, 2985.0],
        [-0.4244965889, 0.1056909874, 0.8992397129, 990.0],
        [-0.2020860361, 0.9663386819, 0.1592193012, 1995.0],
        [0.2450615182, -0.5121285604, 0.8232066508, 2985.0],
        [0.0, 0.0, 0.0, 5.0],
        [-0.102234303, -0.2408570868, 0.9651611322, 990.0],
        [0.6969910108, 0.6283943559, -0.3454331547, 2000.0],
        [-0.5067563088, -0.2878615719, -0.8126092289, 3010.0],
        [0.7498479124, 0.4052295877, 0.5229886132, 995.0],
        [0.5264041238, 0.03305328219, 0.8495917719, 1990.0],
        [0.9688382175, -0.1419762919, 0.2029661075, 2995.0],
        [0.3168756347, -0.7324502375, -0.6025831741, 1005.0],
        [-0.5816149699, 0.7787518334, -0.2350948931, 2000.0],
        [0.5342241541, -0.08203442066, -0.8413530216, 3005.0],
        [-0.7687450228, -0.5852158057, 0.2579797486, 1000.0],
        [0.1407730049, -0.1946339807, 0.9707216772, 1990.0],
        [-0.1199325569, -0.8095447591, 0.5746768351, 2995.0],
        [-0.8657002962, -0.4824411834, -0.133467231, 2000.0],
        [0.6998538622, -0.5820566775, 0.4140224581, 995.0],
        [0.6713350798, 0.7313650423, 0.1200599249, 2990.0],
        [0.0, 0.0, 0.0, 5.0],
        [-0.6730778763, 0.3702793459, 0.6402026073, 995.0],
        [-0.5827130844, 0.629689954, -0.5137470419, 3005.0],
        [-0.6161523495, 0.4875320777, 0.6186022594, 1990.0],
        [0.3118801051, -0.8946560709, -0.3198770309, 3005.0],
        [0.9555609015, -0.0173108859, 0.2942850603, 1000.0],
        [0.8889918845, -0.3979646866, 0.2265337447, 2000.0],
        [-0.04851455127, 0.895786536, -0.4418289491, 1000.0],
        [-0.2978151399, -0.6961275699, 0.6532323851, 1990.0],
        [-0.6148358607, -0.4468806988, 0.6498265195, 2990.0],
        [0.09401767699, 0.9921865486, -0.08201542035, 2995.0],
        [0.4416248214, 0.1575994461, 0.8832496429, 990.0],
        [-0.4408415916, -0.4628836712, -0.7690236654, 2010.0],
        [-0.4676294019, -0.7412792001, 0.4814850879, 995.0],
        [-0.1212366191, -0.9686683406, -0.2167563796, 2005.0],
        [-0.9579573719, -0.2469890092, -0.1459935034, 3000.0],
        [0.0, 0.0, 0.0, 5.0],
        [-0.003465621624, -0.920122541, -0.3916152435, 1005.0],
        [0.03101422328, -0.01100504697, 0.9994583568, 2980.0],
        [-0.2154737714, 0.6917197774, -0.6892712118, 2005.0],
        [-0.9502167311, 0.005192441154, 0.3115464692, 1000.0],
        [-0.3577096635, 0.2241810563, 0.9065244898, 1985.0],
        [0.9228029059, -0.2643573311, -0.2802677261, 2005.0],
        [-0.9157499803, 0.4016143581, -0.01038657823, 1000.0],
        [0.346975886, 0.5369626824, 0.7689465601, 2985.0],
        [-0.942005181, 0.316001738, 0.1130006215, 2995.0],
        [0.1852936782, 0.6268814159, 0.7567601623, 990.0],
        [0.6488650169, 0.7419098117, 0.1689497591, 1995.0],
        [0.8817787013, 0.06698318933, -0.4668828271, 3005.0],
        [0.4791730127, -0.8663127954, 0.1410509286, 2995.0],
        [-0.3552550185, 0.934060756, 0.0363919775, 995.0],
        [0.0, 0.0, 0.0, 5.0],
        [0.8707026199, 0.02694157192, 0.4910713791, 1995.0],
        [0.0675199418, 0.6544240513, -0.7531070432, 1005.0],
        [-0.5098498713, -0.799764504, 0.3169066847, 3000.0],
        [-0.7438395948, -0.403168413, 0.533064619, 1990.0],
        [0.5507626918, -0.1766597313, 0.8157522888, 995.0],
        [-0.003675049647, -0.1335268038, -0.9910383881, 2010.0],
        [0.8257403455, -0.1119647926, 0.5528261635, 2990.0],
        [0.5020751244, 0.8466025373, 0.1765919403, 995.0],
        [0.7076810661, -0.6391168106, -0.3011929797, 2005.0],
        [0.204945799, 0.8977625732, 0.3898968859, 2990.0],
        [-0.5979274612, -0.3708883382, 0.7105804611, 995.0],
        [0.199643097, 0.9210528154, -0.3343715673, 2000.0],
        [-0.2740975961, -0.5241866437, 0.8062870893, 2985.0],
        [0.2131574921, -0.280744014, -0.93581338, 1005.0],
        [-0.6808383201, 0.6168535147, 0.3949062209, 2990.0],
        [0.7111340494, -0.4210501084, 0.5630321217, 1995.0],
        [0.0, 0.0, 0.0, 5.0],
        [0.1309849376, -0.4369497537, -0.8898976676, 3010.0],
        [-0.8384576566, -0.1022086813, -0.5352963138, 1005.0],
        [0.0354932701, 0.7881953774, 0.6144007445, 1990.0],
        [0.6431644866, -0.7402459185, -0.1958964609, 1000.0],
        [0.4924071463, 0.2756010147, -0.8255781507, 2005.0],
        [0.7391056997, 0.5320760923, 0.4130590717, 2990.0],
        [0.4121782217, -0.6459767928, 0.6425131102, 995.0],
        [-0.9976701048, -0.01101721588, -0.06732743039, 2000.0],
        [0.1269657873, -0.8327755972, 0.5388547982, 2995.0],
        [0.9139312474, 0.3894593384, -0.1142414059, 1000.0],
        [0.4115419539, -0.9002480242, -0.1420799603, 2000.0],
        [0.3998191228, -0.3488421846, 0.8476165403, 2990.0],
        [-0.8329379484, -0.5239609664, 0.1779867405, 2995.0],
        [0.2995726014, 0.1142878133, -0.9472035432, 1005.0],
        [0.6552658641, 0.4666472789, 0.5940260637, 1990.0],
        [0.0, 0.0, 0.0, 5.0],
        [-0.02771303463, 0.9976692465, 0.06235432791, 995.0],
        [0.4890983186, -0.2940591118, -0.8211650708, 3010.0],
        [0.06491501615, 0.34662169, -0.9357560818, 2005.0],
        [-0.7495871918, 0.4916461027, 0.4431739518, 995.0],
        [-0.2582108762, 0.9104686819, -0.3230695323, 2000.0],
        [-0.7789022539, 0.528933623, -0.3369577145, 3000.0],
        [0.5011932473, 0.07202777206, 0.8623324933, 2985.0],
        [0.8109116334, 0.4972898265, 0.3084236555, 995.0],
        [-0.282809113, 0.5031798503, 0.8165960102, 1990.0],
        [0.4897901699, 0.8706270162, 0.04598030166, 2995.0],
        [0.1506945989, -0.2702110049, 0.9509348825, 990.0],
        [-0.5818854804, -0.803614474, 0.1249522505, 2000.0],
        [0.4119706909, 0.621417975, 0.6664231765, 995.0],
        [0.8633316137, 0.1665434035, -0.4763631173, 2005.0],
        [0.423996608, 0.131998944, -0.8959928321, 3010.0],
        [0.0, 0.0, 0.0, 5.0],
        [0.9683554517, 0.09903635301, -0.2290840893, 3000.0],
        [0.6668278915, -0.713592445, 0.2147705417, 1000.0],
        [0.7128173755, 0.03306884732, 0.7005696543, 1990.0],
        [-0.8259631909, -0.03290000131, 0.5627631804, 995.0],
        [0.3564028069, -0.241276127, 0.9026421604, 1990.0],
        [0.1339343213, -0.9745221141, 0.1799117749, 3000.0],
        [0.5887810018, 0.7567567582, -0.2840002479, 1000.0],
        [0.7482678618, -0.653968966, 0.1114441496, 2000.0],
        [0.02900379975, 0.7240948626, 0.6890902767, 2985.0],
        [-0.261872724, 0.5931157059, 0.7613385816, 990.0],
        [0.3957498363, 0.8233556966, 0.4067769215, 1990.0],
        [-0.8119849784, 0.3419936732, 0.4729912497, 2990.0],
        [-0.7812232742, 0.3585658931, -0.5109997027, 1000.0],
        [0.004001388723, 0.4331503293, -0.9013128098, 3005.0],
        [-0.755535945, 0.2657233388, 0.598795911, 1990.0],
        [0.0, 0.0, 0.0, 5.0],
        [-0.07619107589, -0.2718636117, -0.95931491, 1005.0],
        [0.9038384415, 0.4213013874, -0.07470751346, 2000.0],
        [-0.3239363528, 0.8038420606, 0.4989019754, 2990.0],
        [0.1593628582, 0.9198008447, -0.358566431, 1000.0],
        [0.3576128074, 0.2657602028, 0.8952567199, 1990.0],
        [0.3009184621, 0.85376866, -0.4248848718, 3000.0],
        [-0.9692048727, 0.2059560355, -0.134996393, 1000.0],
        [-0.5647296629, 0.6847806542, -0.4606038032, 2000.0],
        [0.7091216248, 0.2000343088, 0.6761159638, 2985.0],
        [-0.623130274, 0.1852081648, 0.7598727508, 995.0],
        [-0.3086696631, 0.7716741578, 0.5560953455, 1990.0],
        [-0.4560380808, 0.7840654722, -0.4210351579, 3000.0],
        [0.2114445084, 0.4350210788, -0.8752416128, 1005.0],
        [-0.4530446271, -0.7130702409, -0.5350527053, 3005.0],
        [0.4703029712, 0.492348423, -0.7323988979, 2005.0],
        [0.0, 0.0, 0.0, 5.0],
        [-0.2439994193, 0.8617851832, 0.4447365303, 995.0],
        [-0.1371848704, 0.9872411207, -0.08084108433, 1995.0],
        [-0.888862237, -0.304952736, -0.3419470023, 3005.0],
        [-0.2910104368, 0.8539770555, -0.4313190402, 1000.0],
        [0.4344758953, 0.8170224007, 0.3790845461, 995.0],
        [-0.7131533445, 0.6761453869, 0.1850397878, 2990.0],
        [0.2349893082, -0.04499795264, 0.9709558225, 2985.0],
        [-0.8621846095, -0.3784304607, -0.3367908631, 2005.0],
        [-0.2596337751, 0.03551594094, 0.9650538435, 1990.0],
        [0.8077081404, 0.4229201422, -0.4107871873, 1000.0],
        [-0.7402202483, -0.2990889922, 0.602179175, 2990.0],
        [-0.3084815824, -0.8140486203, 0.492101572, 1995.0],
        [-0.6689965595, -0.2894363353, -0.6845949249, 1005.0],
        [-0.9844045201, 0.04897534926, 0.1689649549, 1995.0],
        [-0.462115081, 0.5531377485, -0.6931726215, 3005.0],
    ]
)

HCP_STANDARD_3T_GRAD_MRTRIX_TABLE = pd.DataFrame(
    HCP_STANDARD_3T_GRAD_MRTRIX.detach().contiguous().cpu().numpy(),
    columns=["x", "y", "z", "b"],
)

# Necessary to know the reference space of the mrtrix gradient table.
HCP_STANDARD_3T_AFFINE_VOX2WORLD = torch.Tensor(
    [
        [-1.25, 0.0, 0.0, 90.0],
        [0.0, 1.25, 0.0, -126.0],
        [0.0, 0.0, 1.25, -72.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

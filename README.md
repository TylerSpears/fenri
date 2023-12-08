# FENRI - Fiber Orientations from Explicit Neural Representations Project Code

This is the official project repository for FENRI - Fiber Orientations from Explicit Neural Representations. Model implementations are in Pytorch, with some jax functions used to fill in implementation gaps in the pytorch library.

## Environment Installation

### Quickstart

To quickly recreate the development environment, install the anaconda packages found in
`pitn.txt` and the pypi packages found in `requirements.txt`. For example:

```bash
# Make sure to install mamba in your root anaconda env for package installation.
# Explicit anaconda packages with versions and platform specs. Only works on the same
# platform as development.
mamba create --name pitn --file pitn.txt
# Move to the new environment.
conda activate pitn
# Install pip packages, try to constrain by the anaconda package versions, even if pip
# does not detect some of them.
pip install --requirement requirements.txt --constraint pitn_pip_constraints.txt
# Install the pitn as a local editable package.
pip install -e .
```

### Detailed Installation Notes

If the previous commands fail to install the environment (which they likely will), then
the following notes should be sufficient to recreate the environment.

* All package versions are recorded and kept up-to-date in the `environment/` directory. If you encounter issues, check these files for the exact versions used in this code. Further instructions are found in the directory's `README.md`.
* All packages were installed and used on a Linux x86-64 system with Nvida GPUs. Using this code on Windows or Mac OSX is not supported.
* This environment is managed by [`mamba`](https://github.com/mamba-org/mamba), which wraps `anaconda`. `mamba` requires that no packages come from the `defaults` anaconda channel (see <https://mamba.readthedocs.io/en/latest/user_guide/troubleshooting.html#using-the-defaults-channels> for details). All anaconda packages come from the following anaconda channels:
  * `conda-forge`
  * `pytorch`
  * `nvidia`
  * `simpleitk`
  * `mrtrix3`
  * `nodefaults` (simply excludes the `defaults` channel)
* Various packages conflict between `anaconda` and pypi, and there's no great way to resolve this problem. Generally, you should install `anaconda` packages first, then `pip install` packages from pypi, handling conflicts on a case-by-case basis. Just make sure that pip does not override `pytorch` packages that make use of the GPU.
* The `jax` and `jaxlib` packages are installed with `pip`, but are hosted on Google's repositories. So, installing from the `requirements.txt` will usually fail. See the `jax` installation docs at <https://jax.readthedocs.io/en/latest/installation.html#pip-installation-gpu-cuda-installed-via-pip-easier> for details on installing `jax` and `jaxlib`.
* The `antspyx` package is not versioned because old versions of this package get deleted from pypi. See <https://github.com/ANTsX/ANTsPy#note-old-pip-wheels-will-be-deleted>

### Package Installation

To install as a python package, install directly from this repository:

```bash
pip install git+ssh://git@github.com/TylerSpears/fenri.git
```

To install an editable version for development:

```bash
pip install -e .
```

## Directory Layout

This repository has the following top-level directory layout:

```bash
./ ## Project root
├── README.md
├── notebooks/ ## Notebooks and scripts for training, testing, and results analysis
├── environment/ ## Detailed specs for package versions
├── pitn/ ## Python package containing data loading/processing, metrics, etc.
├── results/ ## Experiment results directory; contents not tracked by git
├── sources/ ## Projects and sub-modules referenced in this project repository
├── docker/ ## Directory for any auxilary custom docker containers
├── tests/ ## Unit test scripts run by `pytest`
├── pitn.txt ## Anaconda environment package specs
├── requirements.txt ## Pypi-installed package specs
└── pip_constraints.txt ## Constraints on pypi packages to help (slightly) differences between conda and pip
```

## Notebooks

While the `pitn` local package contains helper functions and classes, the actual training and testing of model code is in `notebooks/`. This directory is laid out as follows:

```bash
notebooks/
├── continuous_sr/ ## Contains ODF prediction models
│   ├── fenri.py ## FENRI training script
│   ├── inr_networks.py ## FENRI and Fixed-Net network class definitions
│   ├── test_fenri_native-res.py ## Test FENRI on native image resolution
│   ├── test_fenri_super-res.py ## Predict ODF at arbitrary resolution with FENRI
│   └── baselines/ ## Comparison and baseline model scripts
│       ├── train_fixed_net.py ## Fixed-Net training script
│       ├── test_fixed_net.py ## Fixed-Net native-resolution testing script
│       └── batch_test_trilinear-dwi.py ## Trilinear-DWI testing script
├── tractography/
│   └── trax.py ## Perform tractography with FENRI or trilinear interp on GPU or CPU
├── preprocessing/ ## Data preprocessing scripts
│   ├── fit_odf_hcp2.sh ## Script for creating ODF SH images from HCP data
│   └── fit_odf_ismrm-2015-sims.sh ## Script for creating ODF SH images from ISMRM-sim data
├── data_analysis/ ## Scripts and notebooks for analysing prediction results
│   ├── hcp/ ## Directory for results on HCP data
│   │   ├── quant_analysis.ipynb ## Quantitative voxel-wise metrics on HCP ODF predictions
│   │   └── qualitative_viz_529549/ ## Qualitative results on a particular HCP subject
│   ├── ismrm_sim/ ## Directory for results on ISMRM-sim data
│   │   ├── scilpy_score_bundle_as_tracto.py ## Helper script that calls scilpy bundle rating script
│   │   ├── config_score_tractogram.json ## Config file for scilpy scoring
│   │   └── bundle_rating_analysis.ipynb ## Notebook to compile scilpy bundle rating scores
│   ├── figs/ ## Result figs location
│   └── figs.ipynb ## Notebook to gather result files and create final figures
├── data/ ## Directory for scripts and notebooks pertaining to data generation
│   └── ISMRM-sim/ ## Directory for creating ISMRM-sim dataset; see directory README.md for more info
└── sandbox/ ## Testing directory, not tracked by git
```

## Misc. Notes

### Jax & Pytorch on the GPU

This code makes use of Pytorch for network training and inference and Jax for some of the more steps in tractography. Sometimes, using the Nvidia CUDA distributions of pytorch and jax together will cause an error due to incompatibilities between the CUDA versions of each library. Importing jax *before* pytorch seems to resolve this issue, and lets both libraries run functions on the GPU. This is used, for example, in the `notebooks/tractography/trax.py` script:

```python
try:
    torch
except NameError:
    import jax
    jax.devices()
else:
    raise RuntimeError(
        "ERROR: Must import jax and instantiate devices before importing pytorch"
    )
import jax.numpy as jnp
import torch
```

Additionally, the default jax behavior is to pre-allocate almost all of the GPU memory, but that leaves pytorch very little to work with. You can disable the default behavior with an environment variable set *before* importing jax. For example:

```python
import os
# This env var should be set as early as possible in the import steps
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
```

### Installing Packages

When installing a new python package, always use [`mamba`](https://github.com/mamba-org/mamba)
for installation; this will save you so much time and effort. For example:

```bash
# conda install numpy
# replaced by
mamba install numpy
```

If a package is not available on the anaconda channels, or a package must be built from
a git repository, then use `pip`:

```bash
pip install ipyvolume
```

# FENRI - Fiber Orientations from Explicit Neural Representations Project Code

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

If the previous commands fail to install the environment (which it likely will), then
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
* Various packages conflict between `anaconda` and pypi, and there's no great way to resolve this problem. Generally, you should install `anaconda` packages first, then `pip install` packages from pypi, handling conflicts on a case-by-case basis.
* The `jax` and `jaxlib` packages are installed with `pip`, but are hosted on Google's repositories. So, installing from the `requirements.txt` will usually fail. See the `jax` installation docs at <https://jax.readthedocs.io/en/latest/installation.html#pip-installation-gpu-cuda-installed-via-pip-easier> for details on installing `jax` and `jaxlib`.


### Package Installation

To install as a python package, install directly from this repository:

```bash
pip install git+ssh://git@github.com/TylerSpears/fenri.git
```

To install an editable version for development:

```bash
pip install -e .
```

## Directory Structure



## Misc. Notes

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


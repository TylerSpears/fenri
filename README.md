# Pain in the Net (PITN) Project Code

# Setup

## Environment

### Required Packages

This project requires the python packages:

- `pytorch`
- `conda`
- `tensorboard`
- `jupyter`
- `numpy`
- ...and many others

It is *strongly* recommended that you create a new `conda` environment with the
`environment.yml` file.

Please see the `environment/` directory for other environment description files.

### Environment Variables

Several configuration options in the form of environment variables are used in this code.
In particular, env vars that define directories are needed to properly locate data
and write results. For example, the `DATA_DIR`, `WRITE_DATA_DIR`, and `RESULTS_DIR`
all designate directories for reading and writing.

It is recommended that these variables be stored in a `.env` file. For convenience, you
may want to set up [`direnv`](<https://direnv.net/>) for automatic variable loading. Your
`.env` file should be specific to your system and may contain sensitive data or keys, so
it is explicitly *not* version-controlled.

See the `.env.template` file for all env vars and example values, and for a starting
point for your own `.env` file.

## Developers

### Installing Packages

Installing new python packages to the conda environment requires the following anaconda
channels:

- `pytorch`
- `conda-forge`

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

### pre-commit Hooks

This repository relies on [`pre-commit`](<https://pre-commit.com/>) to run basic cleanup
and linting utilities before a commit can be made. Hooks are defined in the
`.pre-commit-config.yaml` file. To set up pre-commit hooks:

```bash
# If pre-commit is not already in your conda environment
mamba install -c conda-forge pre-commit
pre-commit install
# (Optional) run against all files
pre-commit run --all-files
```

### git Filters

The [`nbstripout`](<https://github.com/kynan/nbstripout>) application is set up as
a git repository filter that strips jupyter/ipython notebooks (*.ipynb files) of output
and metadata. Install `nbstripout` with:

```bash
# If not installed in your conda env already
mamba install -c conda-forge nbstripout
nbstripout --install --attributes .gitattributes
```

You may also install `nbstripout` as a `pre-commit` hook (see <https://github.com/kynan/nbstripout#using-nbstripout-as-a-pre-commit-hook>),
but this causes your local working version to be stripped of output.

You may selectively keep cell outputs in jupyter itself by tagging a cell with the
`keep_output` tag. See <https://github.com/kynan/nbstripout#keeping-some-output> for
details.

## Docker Containers

This project contains many custom-made `Dockerfile`s, found in the `docker/` directory.
Most of these container definitions assume that you have an nvidia GPU installed and
configured, with an up-to-date version of
[`nvidia-container-runtime`](<https://github.com/NVIDIA/nvidia-container-runtime>) installed.

Using the `run.sh` scripts for any GUI-based containers also requires
[`x11docker`](<https://github.com/mviereck/x11docker>).

Do **not** run these containers/scripts if security is a concern to you, these were not
developed with security practices in mind.

# Make sure to "source" this file, not "dot-slash" execute it!

# Required linux terminal applications:
# * jq
# * awk
# * sed
# * conda/anaconda
# * mamba
# * conda-lock

ACTIVE_PREFIX=$(conda info --json | jq -e -r '.active_prefix')
ACTIVE_CONDA_ENV=$(conda info --json | jq -e -r '.active_prefix_name')
# The explicit, channel-specific environment will be the default environment file.
# Use sed to remove all pip-installed packages, as --no-pip does not seem to work
# with --export.
conda list --explicit --export --no-pip --prefix "$ACTIVE_PREFIX" \
    | sed -E '/.*=pypi.*$/d' \
    > "$ACTIVE_CONDA_ENV.txt"
# Also export cross-platform env file.
# Use sed to remove the "prefix:" field from the environment.yml file.
conda env export --from-history --prefix "$ACTIVE_PREFIX" \
    | sed '/prefix:/d' \
    > "cross-platform_from-history-conda-deps_${ACTIVE_CONDA_ENV}_environment.yml"
# conda-lock lock \
#     --mamba \
#     --strip-auth \
#     --file environment.yml \
#     --extras environment.yml \
#     --platform linux-64
# Export the pypi-only packages into a requirements.txt-like file.
# Note that file contains *only* the pip-installed packges, not the conda/mamba packages.
# Taken from <https://stackoverflow.com/a/62617146/13225248>
# Some edits must be made to avoid future errors:
#   * comment out the pitn package, as it is a local installed editable package that
#     should be included in a separate step
#   * comment out the antspyx version number and just install the latest available on
#     pypi; antspyx keeps hitting their storage limit on pypi, so only the latest
#     few versions actually stay online
conda list \
    | awk '$4 ~ /pypi/ { print $1 "==" $2 }' \
    | sed '/^[^#]/ s/\(^.*pitn==.*$\)/#\ \1/' \
    | sed '/^[^#]/ s/\(^.*antspyx==.*$\)/#\ \1/' \
    | sed '/antspyx==.*$/a antspyx' \
    > "${ACTIVE_CONDA_ENV}_pip_only_requirements.txt"

# Create a constraints file from the anaconda packages to be used in the pip install.
# This can help (but not totally solve) the problem of pip packages pulling extra
# package versions from pypi, even when those packages are already installed with conda.
# The --no-pip flag doesn't seem to work with --export, so the pypi packages must be
# filtered out manually. Then, the package list must be cleaned up for pip.
conda list --no-pip --export --prefix "$ACTIVE_PREFIX" \
    | sed -E '/.*=pypi.*$/d' \
    | sed -E '/^_.*/d' \
    | sed -E 's/([[:alnum:]\!_\-\+\.]+)\=([[:alnum:]\!_\-\+\.]+)\=([[:alnum:]\!_\-\+\.]+)$/\1==\2/' \
    > "${ACTIVE_CONDA_ENV}_pip_constraints.txt"

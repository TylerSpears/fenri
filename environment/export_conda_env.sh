# Make sure to "source" this file, not "dot-slash" execute it!
# Use sed to remove the "prefix:" field from the environment.yml file.
ACTIVE_PREFIX=$(conda info --json | jq -e -r '.active_prefix')
conda env export -p "$ACTIVE_PREFIX" \
    | sed '/prefix:/d' \
    | sed 's/name: null/name: pitn/' \
    > environment.yml
conda env export --from-history -p "$ACTIVE_PREFIX" \
    | sed '/prefix:/d' \
    | sed 's/name: null/name: pitn/' \
    > from_history_environment.yml
# conda-lock lock \
#     --mamba \
#     --strip-auth \
#     --file environment.yml \
#     --extras environment.yml \
#     --platform linux-64
# Export the pypi-only packages into a requirements.txt-like file.
# Note that file contains *only* the pip-installed packges, not the conda/mamba packages.
# Taken from <https://stackoverflow.com/a/62617146/13225248>
conda list | awk '$4 ~ /pypi/ { print $1 "==" $2 }' > pip_only_deps_requirements.txt

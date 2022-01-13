# Make sure to "source" this file, not "dot-slash" execute it!
# Use sed to remove the "prefix:" field from the environment.yml file.
ACTIVE_PREFIX=$(conda info --json | jq -e -r '.active_prefix')
conda env export -p "$ACTIVE_PREFIX" \
    | sed '/prefix:/d' \
    | sed 's/name: null/name: pitn/' \
    > environment.yml
conda-lock lock \
    --mamba \
    --strip-auth \
    --file environment.yml \
    --extras environment.yml \
    --platform linux-64

# Make sure to "source" this file, not "dot-slash" execute it!
set -e
# Use sed to remove the "prefix:" field from the environment.yml file.
conda env export | sed '/prefix:/d' > environment.yml
conda-lock lock \
    --mamba \
    --strip-auth \
    --file environment.yml \
    --extras environment.yml \
    --platform linux-64
set +e

#!/usr/bin/bash

set -eou pipefail

# Create config file based on tracula config template, replacing env variables in the
# template.
# Using envsubst:
# envsubst < "$TRACULA_CONF_TEMPLATE" > "$TRACULA_CONF"

# Using perl:
# perl -p -i -e 's/\$\{([^}]+)\}/defined $ENV{$1} ? $ENV{$1} : $&/eg' < template.txt

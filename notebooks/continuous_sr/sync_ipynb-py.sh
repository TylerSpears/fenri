#!/bin/bash

set -eou pipefail

jupytext --sync "$1"

# To do the initial export:
# jupytext --to ipynb --from py:percent \
#     --opt comment_magics=true \
#     inr.py
# jupytext --set-formats ipynb,py:percent inr.py

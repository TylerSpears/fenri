#!/bin/bash

set -eou pipefail

jupytext --sync --show-changes "$1" | bat --language diff --pager never

# To do the initial export:
# jupytext --to ipynb --from py:percent \
#     --opt comment_magics=true \
#     inr.py
# jupytext --set-formats ipynb,py:percent inr.py

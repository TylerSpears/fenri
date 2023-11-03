#!/bin/bash

set -eou pipefail

sync_f="$1"

case "$sync_f" in
*.ipynb)
    in_f_ext="ipynb"
    pair_f_ext="py"
    ;;
*.py)
    in_f_ext="py"
    pair_f_ext="ipynb"
    ;;
*)
    exit 1
    ;;
esac

sync_basename="$(basename -s ".${in_f_ext}" "$sync_f")"
sync_dirname="$(dirname "$sync_f")"
paired_f="${sync_dirname}/${sync_basename}.${pair_f_ext}"

if [ "$in_f_ext" == "ipynb" ] && [ ! -s "$paired_f" ]; then
    jupytext --to py:percent --from ipynb --opt comment_magics=true "$sync_f"
    jupytext --set-formats ipynb,py:percent "$sync_f"
elif [ "$in_f_ext" == "py" ] && [ ! -s "$paired_f" ]; then
    jupytext --to ipynb --from py:percent --opt comment_magics=true "$sync_f"
    jupytext --set-formats ipynb,py:percent "$sync_f"
fi

pyscript_f="${sync_dirname}/${sync_basename}.py"
run_pyscript_f="${sync_dirname}/run_${sync_basename}.py"

jupytext --sync --show-changes "$sync_f" | bat --language diff --pager never

# ./add_main.py "$pyscript_f" "$run_pyscript_f"
# black "$run_pyscript_f"

# To do the initial export:
# jupytext --to ipynb --from py:percent \
#     --opt comment_magics=true \
#     inr.py
# jupytext --set-formats ipynb,py:percent inr.py

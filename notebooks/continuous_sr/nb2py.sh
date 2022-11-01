#!/bin/bash

set -eou pipefail

jupytext --to py:percent --opt cell_metadata_filter=-all "$1"

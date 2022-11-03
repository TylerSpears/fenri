#!/bin/bash

set -eou pipefail

jupytext --sync "$1"

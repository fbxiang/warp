#!/bin/bash
set -e

SCRIPT_DIR=$(dirname ${BASH_SOURCE})
"$SCRIPT_DIR/repo.sh" build --fetch-only --no-docker $@ 

echo "Installing test dependencies"
./_build/target-deps/python/python -m pip install matplotlib
./_build/target-deps/python/python -m pip install usd-core

echo "Installing Warp to Python"
./_build/target-deps/python/python -m pip install -e .

echo "Running tests"
./_build/target-deps/python/python warp/tests/test_all.py

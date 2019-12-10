#!/usr/bin/env bash
python setup.py build
PYTHONPATH=""
export PYTHONPATH
PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONPATH
python setup.py develop --install-dir $(pwd)
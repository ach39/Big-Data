#!/usr/bin/env bash

set -o errexit
set -o pipefail

CWD=$(pwd)
parentdir="$(dirname "$CWD")"
export PYTHONPATH="${PYTHONPATH}:${parentdir}/src"

cd ../src
python features/etl_sepsis_data.py
python rnn/train_variable_rnn.py


#!/usr/bin/env bash
set -e

rm -rf /workspace/TensorOps


git clone https://github.com/andreaslam/TensorOps.git /workspace/TensorOps
cd /workspace/TensorOps
git fetch origin
git reset --hard origin/main

pip install -r requirements.txt

cd tensorops
maturin develop --release

cd /workspace/TensorOps
pip install .

exec "$@"
#!/bin/bash
PY=/opt/conda/bin/python; S=/root/claude64b_cpuinfer/serve64b.py
export OMP_NUM_THREADS=12 TORCH_THREADS64=4 PORT64=8770 CUDA_VISIBLE_DEVICES=
exec nice -n 5 $PY -u $S >> /workspace/claude64b_infer/serve.log 2>&1

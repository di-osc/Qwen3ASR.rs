#!/usr/bin/env bash

# Source this file before CUDA builds/runs:
#   source setup.sh

_vasr_cuda_root="${CUDA_TOOLKIT:-/home/featurize/.local/conda-cuda-12.8}"

export CUDA_TOOLKIT="${_vasr_cuda_root}"
export CUDA_HOME="${_vasr_cuda_root}"
export CUDA_PATH="${_vasr_cuda_root}"
export CUDA_ROOT="${_vasr_cuda_root}"
export CUDA_TOOLKIT_ROOT_DIR="${_vasr_cuda_root}"
export PATH="${_vasr_cuda_root}/bin:${PATH}"

_vasr_workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_vasr_py_cuda_libs="/home/featurize/work/.local/lib/python3.10/site-packages/nvidia/curand/lib:/home/featurize/work/.local/lib/python3.10/site-packages/nvidia/cublas/lib:/home/featurize/work/.local/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib"
_vasr_link_libs="${_vasr_workspace}/target/cuda-lib-links:/home/featurize/.local/cuda-12.8-merged/lib:${_vasr_py_cuda_libs}"
_vasr_cuda_libs="${_vasr_link_libs}:${_vasr_cuda_root}/lib64:${_vasr_cuda_root}/lib"
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  export LD_LIBRARY_PATH="${_vasr_cuda_libs}:${LD_LIBRARY_PATH}"
else
  export LD_LIBRARY_PATH="${_vasr_cuda_libs}"
fi
if [ -n "${LIBRARY_PATH:-}" ]; then
  export LIBRARY_PATH="${_vasr_cuda_libs}:${LIBRARY_PATH}"
else
  export LIBRARY_PATH="${_vasr_cuda_libs}"
fi

unset _vasr_workspace
unset _vasr_py_cuda_libs
unset _vasr_link_libs
unset _vasr_cuda_root
unset _vasr_cuda_libs

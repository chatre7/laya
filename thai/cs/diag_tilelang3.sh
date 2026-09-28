#!/usr/bin/env bash
# Which CUDA pieces are installed in laya-train-0320 and at which versions; which nvcc/ptxas versions pip offers.
cd ~/laya-0320
docker run --rm --gpus '"device=1"' -w /work -v "$PWD":/work laya-train-0320 bash -c '
echo "-- pip nvidia packages:"; pip list 2>/dev/null | grep -i nvidia
echo "-- binaries:"; nvcc --version | tail -1; ptxas --version | tail -1; which nvcc ptxas
SP=$(python -c "import site; print(site.getsitepackages()[0])")
echo "-- cu13/bin owners:"; for f in nvcc ptxas cudafe++ nvlink; do pip show -f nvidia-cuda-nvcc 2>/dev/null | grep -q "bin/$f" && echo "$f: nvidia-cuda-nvcc" || echo "$f: other"; done
ls -la $SP/nvidia/cu13/bin | head -20
echo "-- versions available:"; pip index versions nvidia-cuda-nvcc 2>/dev/null | head -2; pip index versions nvidia-cuda-cccl 2>/dev/null | head -2; pip index versions nvidia-cuda-nvcc-cu13 2>/dev/null | head -2
'

#!/usr/bin/env bash
# Why does tilelang not find a CUDA target inside laya-train-0320? Run on the GPU box.
cd ~/laya-0320
docker run --rm --gpus '"device=1"' -w /work -v "$PWD":/work laya-train-0320 bash -c '
which nvcc || echo "no nvcc"; ls -d /usr/local/cuda* 2>/dev/null || echo "no /usr/local/cuda"; echo "CUDA_HOME=$CUDA_HOME"
python - <<EOF
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "cap", torch.cuda.get_device_capability(0))
try:
    from tilelang import tvm
    print("tvm cuda(0).exist =", tvm.cuda(0).exist)
except Exception as e: print("tvm probe:", type(e).__name__, str(e)[:200])
try:
    from tilelang.utils.target import determine_target
    print("determine_target(auto) =", determine_target("auto"))
except Exception as e: print("determine_target:", type(e).__name__, str(e)[:300])
try:
    from tilelang import tvm
    print("nvcc via tvm:", tvm.contrib.nvcc.find_cuda_path())
except Exception as e: print("find_cuda_path:", type(e).__name__, str(e)[:200])
EOF
pip list 2>/dev/null | grep -i -E "nvidia-cuda|tilelang|tvm|apache" '

#!/usr/bin/env bash
# Can tilelang compile when nvcc comes from the pip wheel (no system CUDA toolkit)? Throwaway container, nothing baked yet.
cd ~/laya-0320
docker run --rm --gpus '"device=1"' -w /work -v "$PWD":/work -v ~/laya/thai/out:/work/thai/out -v ~/laya/thai/data:/work/thai/data:ro laya-train-0320 bash -c '
set -e
pip install -q --no-cache-dir "nvidia-cuda-nvcc" 2>&1 | tail -1
SP=$(python -c "import site; print(site.getsitepackages()[0])")
ls $SP/nvidia/ | tr "\n" " "; echo
echo "-- cu13 layout:"; ls $SP/nvidia/cu13 | tr "\n" " "; echo; ls $SP/nvidia/cu13/bin 2>/dev/null | tr "\n" " "; echo
export CUDA_HOME=$SP/nvidia/cu13 PATH=$SP/nvidia/cu13/bin:$PATH
mkdir -p $CUDA_HOME/lib64 && ln -sf $CUDA_HOME/lib/* $CUDA_HOME/lib64/ 2>/dev/null || true
nvcc --version | tail -1
ls $CUDA_HOME/include | head -5
python - <<EOF
import time, torch
from tilelang import tvm
print("find_cuda_path:", tvm.contrib.nvcc.find_cuda_path())
import laya
t=time.perf_counter()
agent = laya.Agent("/work/thai/out/laya-th-run8", device="cuda")
try:
    agent.accelerate(strict=True); print("accelerate OK in %.1fs" % (time.perf_counter()-t))
    q = {"urgency": {"type": "score", "instructions": "เรื่องนี้เร่งด่วนแค่ไหน", "criteria": ["ไม่รีบ", "ควรตอบวันนี้", "ด่วน"]}}
    print(agent.predict("เน็ตล่มทั้งบ้าน ทำงานไม่ได้เลย", q)["answers"]["urgency"])
except Exception as e:
    import subprocess; print("host compilers:", subprocess.run("which gcc g++ c++ clang; gcc --version 2>&1 | head -1", shell=True, capture_output=True, text=True).stdout)
    print("accelerate FAILED:", type(e).__name__, "... tail of message:\n", str(e)[-1800:])
EOF
' 2>&1 | grep -v Warning | tail -25

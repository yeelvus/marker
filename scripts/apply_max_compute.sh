#!/bin/zsh
# 在当前 shell 中启用「尽量吃满本机算力」的环境变量（M1 Pro 16GB 保守配置）
# 用法:
#   source /Users/wyattsun/Projects/marker/scripts/apply_max_compute.sh
#   conda activate marker
#   python process_docs.py -p file.pdf -o ./output --preset speed

export SURYA_INFERENCE_BACKEND=llamacpp
export TORCH_DEVICE=mps
export PYTORCH_ENABLE_MPS_FALLBACK=1
export LLAMA_CPP_NGL=99
export SURYA_INFERENCE_KEEP_ALIVE=1
export SURYA_INFERENCE_PARALLEL=2
export SURYA_INFERENCE_CTX_PER_SLOT=8192
export LLAMA_CPP_EXTRA_ARGS="--threads 8 --threads-batch 8 --flash-attn on --prio 2"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PATH="/opt/homebrew/bin:$PATH"

echo "已设置本机算力相关环境变量："
echo "  SURYA_INFERENCE_PARALLEL=$SURYA_INFERENCE_PARALLEL"
echo "  LLAMA_CPP_NGL=$LLAMA_CPP_NGL"
echo "  SURYA_INFERENCE_KEEP_ALIVE=$SURYA_INFERENCE_KEEP_ALIVE"
echo "  LLAMA_CPP_EXTRA_ARGS=$LLAMA_CPP_EXTRA_ARGS"
echo "提示：16GB 内存请先关掉占内存的 App；内存压力发红时不要把 PARALLEL 调到 4+"

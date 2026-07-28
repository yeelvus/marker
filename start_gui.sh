#!/bin/zsh
# 启动 Marker 图形界面
# 用法: 双击本文件，或在终端执行 ./start_gui.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 初始化 conda
if [ -f /opt/anaconda3/etc/profile.d/conda.sh ]; then
  source /opt/anaconda3/etc/profile.d/conda.sh
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "未找到 conda，请确认 Anaconda/Miniconda 已安装。"
  exit 1
fi

conda activate marker

# Apple Silicon：surya VLM 走 llama.cpp
export SURYA_INFERENCE_BACKEND="${SURYA_INFERENCE_BACKEND:-llamacpp}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export PYTHONUNBUFFERED=1

# 确保 brew 的 llama-server 可用
export PATH="/opt/homebrew/bin:$PATH"

mkdir -p "$SCRIPT_DIR/logs" "$SCRIPT_DIR/output"

echo "启动 Marker GUI..."
echo "Python: $(which python)"
echo "marker_single: $(which marker_single 2>/dev/null || echo '未安装')"
echo "llama-server: $(which llama-server 2>/dev/null || echo '未安装')"
echo "后端: $SURYA_INFERENCE_BACKEND"
python "$SCRIPT_DIR/marker_gui.py"

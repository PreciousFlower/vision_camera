#!/bin/bash
#SBATCH --job-name=drvscan
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2g
#SBATCH --time=0-00:05:00
#SBATCH --array=0-18
# 刻意不加 --exclude：就是要探出哪些节点不能跑
#SBATCH --output=/share/home/u00009/apps/civility-kit/logs/drvscan-%A_%a.out

# 每个数组任务探一个节点。数组索引 → 节点名由本脚本自己按 sinfo 顺序解析。
# （Slurm 的数组不会自动绑定节点，所以用 NODELIST 环境变量取实际分到的节点。）

WORKDIR=/share/home/u00009/apps/civility-kit
VLM_PY=/share/home/u00009/apps/conda/envs/vlm/bin/python
cd "$WORKDIR" || exit 1

NODE="${SLURM_NODELIST:-unknown}"
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
# 关键判据：torch 是 cu130 编译，需要驱动 ≥580。驱动 550 的节点上这里会是 0。
CUDA_OK=$("$VLM_PY" -c 'import torch;print(1 if torch.cuda.is_available() else 0)' 2>/dev/null || echo 0)

echo "RESULT|${NODE}|${DRV:-NONE}|${CUDA_OK}|${GPU:-NONE}"

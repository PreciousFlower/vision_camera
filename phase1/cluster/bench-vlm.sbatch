#!/bin/bash
#SBATCH --job-name=civility-bench
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
# 平台按「ntasks × cpus-per-task」算核数，每卡上限 7 核；默认 ntasks=2，
# 所以必须显式 -n1，否则 7×2=14 会被拒。
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64g
# 微基准只要几分钟 —— 时限压到 30 分钟，避免万一卡住白烧卡时
# （对比常驻服务的 12 小时，这里"跑完就退"本身就是成本控制）
#SBATCH --time=0-00:30:00
# 排除驱动过旧（550.54.14）的节点
#SBATCH --exclude=gpu8002,gpu8003,gpu8005,gpu8007,gpu8011,gpu8012,gpu8015
#SBATCH --output=/share/home/u00009/apps/civility-kit/logs/bench-%j.out
#SBATCH --error=/share/home/u00009/apps/civility-kit/logs/bench-%j.err

# 推理栈微基准：查清"34 token/s 到底慢在哪"，并对两种加载配置做 A/B。
# 跑完即退，不常驻 —— 一次约 0.1~0.2 元。

set -uo pipefail
WORKDIR=/share/home/u00009/apps/civility-kit
VLM_PY=/share/home/u00009/apps/conda/envs/vlm/bin/python
export MODEL_ROOT=${MODEL_ROOT:-/ssdfs/datahome/u00009/models}
MODEL=${MODEL:-Qwen/Qwen3-VL-4B-Instruct}
# ⚠️ SLURM 的 --export 用**逗号**分隔变量，所以配置列表只能用 `+` 传进来，
#    在这里换成逗号。直接传 "B,C,D" 的话只有第一个生效（实测踩到）。
CONFIGS=${CONFIGS:-B}
CONFIGS="${CONFIGS//+/,}"

cd "$WORKDIR" || { echo "❌ 工作目录不存在: $WORKDIR"; exit 1; }
mkdir -p logs
export PYTHONPATH="$WORKDIR/pkg:${PYTHONPATH:-}"
export HF_HOME="$MODEL_ROOT/hf"
export HF_HUB_CACHE="$MODEL_ROOT/hf/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export BENCH_OUT_DIR="$WORKDIR/logs"

echo "════════════════════════════════════════════════════════"
echo "推理栈微基准  作业 ${SLURM_JOB_ID:-?}  节点 $(hostname)"
echo "模型 $MODEL   配置 $CONFIGS"
echo "开始 $(date '+%F %T')"
echo "════════════════════════════════════════════════════════"

DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
echo "GPU 驱动 = ${DRV:-未知}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 坏节点立刻退出，别占着卡
"$VLM_PY" -c 'import sys, torch; ok = torch.cuda.is_available(); print(f"torch={torch.__version__} cuda={ok}"); sys.exit(0 if ok else 3)'
RC=$?
if [ $RC -ne 0 ]; then
  echo "❌ CUDA 不可用（rc=$RC）—— 换节点"
  exit 3
fi

BENCH="$WORKDIR/pkg/cluster/vlm_bench.py"
[ -f "$BENCH" ] || BENCH="$WORKDIR/cluster/vlm_bench.py"

"$VLM_PY" "$BENCH" --pkg "$WORKDIR/pkg" --model "$MODEL" --configs "$CONFIGS"
echo "结束 $(date '+%F %T')"

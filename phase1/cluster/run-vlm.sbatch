#!/bin/bash
#SBATCH --job-name=civility-vlm
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=0-03:00:00

# ⚠️ 必须排除驱动过旧的节点：这些节点是 550.54.14，而 torch 是 cu130 编译，
#    在上面 torch.cuda.is_available() 返回 False。
#    列表来自 2026-10-05 全节点实测（tools/scan_all_nodes.sh），
#    比旧文档多出 gpu8007/gpu8011/gpu8015。
#SBATCH --exclude=gpu8002,gpu8003,gpu8005,gpu8007,gpu8011,gpu8012,gpu8015

# 被 TimeLimit 杀掉或驱动不兼容时自动重排；脚本内做断点续跑，重排后接着跑
#SBATCH --requeue

# ⚠️ 输出路径必须用绝对路径：相对路径是相对 WorkDir 的，
#    若目录不存在作业会 0 秒 FAILED 且没有任何日志。
#SBATCH --output=/share/home/u00009/apps/civility-kit/logs/vlm-%j.out
#SBATCH --error=/share/home/u00009/apps/civility-kit/logs/vlm-%j.err

set -uo pipefail

# ── 路径与解释器（注意：不能用 conda activate，profile 脚本不存在）──
WORKDIR=/share/home/u00009/apps/civility-kit
VLM_PY=/share/home/u00009/apps/conda/envs/vlm/bin/python
export MODEL_ROOT=${MODEL_ROOT:-/ssdfs/datahome/u00009/models}

cd "$WORKDIR" || { echo "❌ 工作目录不存在: $WORKDIR"; exit 1; }
mkdir -p logs

echo "════════════════════════════════════════════════════════"
echo "作业      ${SLURM_JOB_ID:-local}  数组 ${SLURM_ARRAY_TASK_ID:-none}"
echo "节点      $(hostname)"
echo "开始      $(date '+%F %T')"
echo "工作目录  $WORKDIR"
echo "模型      ${MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
echo "════════════════════════════════════════════════════════"

# ── 驱动预检：坏节点 30 秒内退出，不白跑几小时 ──
# ⚠️ 用 python -c 单行，不要用 heredoc：
#    sbatch 从 spool 文件读脚本，stdin 不是脚本自身，heredoc 会静默拿到空输入。
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
echo "GPU 驱动 = ${DRV:-未知}"
"$VLM_PY" -c 'import sys, torch; ok = torch.cuda.is_available(); print(f"torch={torch.__version__} cuda_available={ok}"); sys.exit(0 if ok else 3)'
RC=$?
if [ $RC -ne 0 ]; then
  echo "❌ CUDA 不可用（rc=$RC），退出让 Slurm 换节点"
  exit 3
fi

# ── 让集群脚本能 import 到 civility 包（本地 prompts.py 的同一份副本）──
export PYTHONPATH="$WORKDIR/pkg:${PYTHONPATH:-}"
export HF_HOME="$MODEL_ROOT/hf"
export HF_HUB_CACHE="$MODEL_ROOT/hf/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# 避免 tokenizer 并行告警刷屏
export TOKENIZERS_PARALLELISM=false

# ── 跑推理 ──
MODEL="${MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
LIMIT="${LIMIT:-0}"        # 冒烟测试时用 LIMIT=5 sbatch ...
MAXTOK="${MAXTOK:-512}"    # 自回归上限：写满会显著拖慢，实时场景要盯这个数

echo "── 启动推理 ──"
"$VLM_PY" pkg/cluster/run_cluster_vlm.py \
  --pkg pkg \
  --model "$MODEL" \
  --out results.jsonl \
  --limit "$LIMIT" \
  --max-new-tokens "$MAXTOK"
RC=$?

echo "── 结果统计 ──"
if [ -f results.jsonl ]; then
  echo "结果行数: $(wc -l < results.jsonl)"
  "$VLM_PY" -c '
import json, collections
c = collections.Counter()
for line in open("results.jsonl", encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
    except Exception:
        continue
    c["失败" if r.get("error") else "成功"] += 1
print("计数:", dict(c))
'
else
  echo "⚠️ 没有生成 results.jsonl"
fi

echo "结束      $(date '+%F %T')"
echo "退出码    $RC"
echo "════════════════════════════════════════════════════════"
exit $RC

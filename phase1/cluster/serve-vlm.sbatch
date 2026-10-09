#!/bin/bash
#SBATCH --job-name=civility-serve
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
# 平台按「ntasks × cpus-per-task」算 CPU 核数，每卡上限 7 核。
# 默认 ntasks=2，所以必须显式 -n1，否则 7×2=14 核会被拒。
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64g
#SBATCH --time=0-12:00:00
# 排除驱动过旧（550.54.14）的节点 —— torch 是 cu130 编译，需驱动 ≥580。
# 列表来自 2026-10-05 全节点实测（tools/scan_all_nodes.sh），
# 比旧文档多出 gpu8007/gpu8011/gpu8015。脚本内另有运行时预检兜底。
#SBATCH --exclude=gpu8002,gpu8003,gpu8005,gpu8007,gpu8011,gpu8012,gpu8015
#SBATCH --output=/share/home/u00009/apps/civility-kit/logs/serve-%j.out
#SBATCH --error=/share/home/u00009/apps/civility-kit/logs/serve-%j.err

# 计算节点常驻 VLM 推理服务 —— 网页实时交互的后端。
#
# 与 batch 推理的区别：
#   批处理（run-vlm.sbatch）：跑完就退，适合大批量评测
#   本脚本  ：常驻不退，模型一直在显存里，单窗口延迟只有推理时间本身
#              —— 这是"网页能实时看到结果"的前提
#
# 由 webui/cluster_client.py 通过 sbatch --export 传入 MODEL 与 PORT。

set -uo pipefail

WORKDIR=/share/home/u00009/apps/civility-kit
VLM_PY=/share/home/u00009/apps/conda/envs/vlm/bin/python
export MODEL_ROOT=${MODEL_ROOT:-/ssdfs/datahome/u00009/models}
MODEL=${MODEL:-Qwen/Qwen3-VL-4B-Instruct}
PORT=${PORT:-18080}
# ── 同卡多 worker：一张卡跑多份模型，吞吐接近翻倍而**计费不变** ──
# 依据：网关监听 PORT，把请求轮询分给 PORT+1 … PORT+WORKERS。
#      4B 模型 bf16 约占 8~9GB，A800 有 80GB，放 3 份很宽裕。
#      实测单窗口推理 10~12 秒，是"结论来得慢"的主因，多 worker 能直接摊薄它。
# 想回退成单进程：起服务时传 VLM_WORKERS=1，或改这里的默认值。
WORKERS=${VLM_WORKERS:-3}

cd "$WORKDIR" || { echo "❌ 工作目录不存在: $WORKDIR"; exit 1; }
mkdir -p logs

echo "════════════════════════════════════════════════════════"
echo "常驻推理服务  作业 ${SLURM_JOB_ID:-?}  节点 $(hostname)"
echo "模型 $MODEL   网关端口 $PORT   worker 数 $WORKERS"
echo "开始 $(date '+%F %T')"
echo "════════════════════════════════════════════════════════"

# ── 驱动预检：坏节点立刻退出换节点，不浪费一次长时间占用 ──
# 用 python -c 单行而不是 heredoc：sbatch 从 spool 读脚本，stdin 不是脚本自身。
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
echo "GPU 驱动 = ${DRV:-未知}（torch 需 ≥580）"
"$VLM_PY" -c 'import sys, torch; ok = torch.cuda.is_available(); print(f"torch={torch.__version__} cuda_available={ok}"); sys.exit(0 if ok else 3)'
RC=$?
if [ $RC -ne 0 ]; then
  echo "❌ CUDA 不可用（rc=$RC）—— 这是驱动过旧的节点，退出让 Slurm 换节点"
  exit 3
fi

# ── 环境 ──
export PYTHONPATH="$WORKDIR/pkg:${PYTHONPATH:-}"
export HF_HOME="$MODEL_ROOT/hf"
export HF_HUB_CACHE="$MODEL_ROOT/hf/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CIVILITY_TOKEN="${CIVILITY_TOKEN:-}"

if [ ! -f "$WORKDIR/pkg/behaviors.json" ]; then
  echo "❌ 缺少 $WORKDIR/pkg/behaviors.json —— 先在本机跑 civility.cli cluster-prepare 并上传"
  exit 1
fi

# 把节点名写到固定文件，方便本机不查 squeue 也能知道服务在哪
echo "$(hostname) $PORT $(date +%s)" > "$WORKDIR/logs/serving-node.txt"

echo "── 启动服务 ──"
# 作业被 scancel 时要连带收掉所有 worker（否则会留孤儿进程占着显存）
cleanup() {
  echo "[serve] 收到退出信号，清理 worker…"
  kill 0 2>/dev/null
}
trap cleanup EXIT INT TERM

BACKENDS=""
PIDS=""
if [ "$WORKERS" -le 1 ]; then
  # 单进程：直接对外提供服务（不经过网关，省一层转发）
  echo "[serve] 单 worker 模式，端口 $PORT"
  exec "$VLM_PY" pkg/cluster/vlm_service.py \
    --pkg "$WORKDIR/pkg" --model "$MODEL" --port "$PORT" \
    --bind 0.0.0.0 --behaviors "$WORKDIR/pkg/behaviors.json"
fi

for i in $(seq 1 "$WORKERS"); do
  WP=$((PORT + i))
  echo "[serve] 启动 worker #$i → 端口 $WP"
  "$VLM_PY" pkg/cluster/vlm_service.py \
    --pkg "$WORKDIR/pkg" --model "$MODEL" --port "$WP" \
    --bind 0.0.0.0 --behaviors "$WORKDIR/pkg/behaviors.json" \
    > "logs/worker-$WP.log" 2>&1 &
  PIDS="$PIDS $!"
  BACKENDS="${BACKENDS:+$BACKENDS,}$WP"
done

# 等 worker 的**模型**真正就绪 —— 只等"端口能连"是不够的（实测踩到）。
# ⚠️ 但**不能阻塞在等待上再启动网关**：本机侧在作业进入 RUNNING 后就会
#    连 PORT 做隧道自测，端口没人听就会把整个启动判成失败
#    （实测报 `隧道自测失败：ChannelException Connect failed`）。
#    所以正确顺序是：**网关立刻起来**（它自己会按 /health 过滤掉未就绪的
#    worker，并对"全都没就绪"如实返回 503），就绪等待放到后台只做日志。
WAIT="pkg/cluster/wait_ready.py"
[ -f "$WAIT" ] || WAIT="cluster/wait_ready.py"
(
  if "$VLM_PY" "$WAIT" --ports "$BACKENDS" --timeout 900; then
    echo "[serve] ✅ 全部 worker 模型就绪，可以开始推理"
  else
    echo "[serve] ⚠️ 有 worker 长时间未就绪，网关会继续跳过它们"
  fi
) &
echo "[serve] worker PID：$PIDS"

echo "[serve] 启动网关 $PORT → $BACKENDS"
# ⚠️ 路径优先取 pkg/cluster/ 下的那份：作业包（pkg.tar.gz）解包后**一定**会更新
#    pkg/cluster/*，而 $WORKDIR/cluster/ 只在手工/脚本同步过时才是新的。
#    踩过的坑：从 pkg 里跑脚本却引用 cluster/ 下的文件，改完不生效还以为没改。
GW="pkg/cluster/vlm_gateway.py"
[ -f "$GW" ] || GW="cluster/vlm_gateway.py"
"$VLM_PY" "$GW" --port "$PORT" --backends "$BACKENDS" --host 127.0.0.1

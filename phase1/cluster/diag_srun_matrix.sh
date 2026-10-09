#!/bin/bash
# 找出"能给指定节点分配资源"的 srun 写法。
# 背景：srun -w <node> --gres=gpu:a800:1 一律报
#   "必须为每个 GPU 设置不超过 7 个 CPU 核心"
# 但不钉节点却成功 —— 说明是配额/参数解析问题，不是"节点没资源"。
cd /share/home/u00009/apps/civility-kit || exit 1

TRY() {
  desc="$1"; shift
  out=$("$@" 2>&1)
  rc=$?
  echo "--- $desc"
  echo "    rc=$rc  $(echo "$out" | tr '\n' ' ' | cut -c1-160)"
}

TRY "A: -w + gres + cpus=7"        srun -p A800 -w gpu8001 --gres=gpu:a800:1 --cpus-per-task=7 --mem=8g --time=0-00:03:00 nvidia-smi -L
TRY "B: -w + gres + -n1 -c7"       srun -p A800 -w gpu8001 --gres=gpu:a800:1 -n1 -c7 --mem=8g --time=0-00:03:00 nvidia-smi -L
TRY "C: -w + gres + cpus=7 + excl" srun -p A800 -w gpu8001 --exclusive --gres=gpu:a800:1 --cpus-per-task=7 --mem=8g --time=0-00:03:00 nvidia-smi -L
TRY "D: -w 不要 gres（纯 CPU）"    srun -p A800 -w gpu8001 --cpus-per-task=1 --mem=1g --time=0-00:02:00 nvidia-smi -L
TRY "E: --nodelist 长选项"          srun -p A800 --nodelist=gpu8001 --gres=gpu:a800:1 --cpus-per-task=7 --mem=8g --time=0-00:03:00 nvidia-smi -L
TRY "F: -w 不带任何资源"            srun -p A800 -w gpu8001 nvidia-smi -L
TRY "G: salloc 钉节点（分配后跑）"  bash -c 'salloc -p A800 -w gpu8001 --gres=gpu:a800:1 --cpus-per-task=7 --mem=8g --time=0-00:02:00 nvidia-smi -L'

echo "### 环境里的 SLURM 相关变量"
env | grep -i -E 'slurm|sbatch|srun' || echo "(无)"
echo "### 我的账户与限额"
sacctmgr show assoc where user=$USER format=Account,Partition,GrpTRES,MaxTRES,MaxJobs,MaxSubmit 2>/dev/null | head -10
echo DIAGDONE

#!/bin/bash
# srun 钉节点失败原因诊断 —— 不用 2>/dev/null，把真实错误落盘。
cd /share/home/u00009/apps/civility-kit || exit 1
echo "### 0. 本脚本确实是最新版（带这行就是）DIAG-V2"
echo "### 1. 单节点 srun 原始输出（7 核）"
srun -p A800 -w gpu8001 --gres=gpu:a800:1 --cpus-per-task=7 --mem=8g \
     --time=0-00:03:00 nvidia-smi --query-gpu=driver_version --format=csv,noheader
echo "rc=$?"
echo "### 2. 单节点 srun 原始输出（默认核数，不指定 cpus）"
srun -p A800 -w gpu8001 --gres=gpu:a800:1 --mem=8g --time=0-00:03:00 \
     nvidia-smi --query-gpu=driver_version --format=csv,noheader
echo "rc=$?"
echo "### 3. 不钉节点的 srun（对照）"
srun -p A800 --gres=gpu:a800:1 --cpus-per-task=7 --mem=8g --time=0-00:03:00 \
     nvidia-smi --query-gpu=driver_version --format=csv,noheader
echo "rc=$?"
echo "### 4. gpu8001 当前状态"
sinfo -p A800 -h -n gpu8001 -o "%t %C %G"
scontrol show node gpu8001 | grep -E 'State|CfgTRES|AllocTRES|Reason'
echo DIAGDONE

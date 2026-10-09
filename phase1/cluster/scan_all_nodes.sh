#!/bin/bash
# 逐节点钉住扫描驱动与 CUDA 可用性。
# 为什么不用数组作业：Slurm 会按资源空闲度把多个数组任务塞到同一个节点，
# 拿不到"每个节点一条"的结果（第一版就踩了这个坑）。
# 这里用 srun -w 逐个钉住，慢但准。
cd /share/home/u00009/apps/civility-kit || exit 1
VLM_PY=/share/home/u00009/apps/conda/envs/vlm/bin/python

for n in $(sinfo -p A800 -N -h -o "%N" | sort -u); do
  st=$(sinfo -p A800 -h -n "$n" -o "%t" 2>/dev/null)
  case "$st" in
    drng|drain|down)
      echo "RESULT|$n|$st|SKIPSKIP"
      continue
      ;;
  esac
  # 在目标节点上取驱动版本 + torch CUDA 可用性
  # ⚠️⚠️ 必须写成 `-n1 -c7`，不能写 `--cpus-per-task=7`。
  #   平台按「ntasks × cpus-per-task」算总核数，而 srun 默认 ntasks=2，
  #   于是 --cpus-per-task=7 实际要 14 核/卡 → 报
  #   "必须为每个 GPU 设置不超过 7 个 CPU 核心"。
  #   实测：-w + --cpus-per-task=7 ❌ ；-w + -n1 -c7 ✅（2026-10-05）
  out=$(timeout 240 srun -p A800 -w "$n" -n1 -c7 --gres=gpu:a800:1 \
          --mem=8g --time=0-00:03:00 \
          bash -c "d=\$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1); \
                   ok=\$(/share/home/u00009/apps/conda/envs/vlm/bin/python -c \
                        'import torch;print(1 if torch.cuda.is_available() else 0)' 2>/dev/null || echo 0); \
                   echo \"\${d}|\${ok}\"" 2>/dev/null)
  echo "RESULT|$n|$st|${out:-TIMEOUT}"
done
echo ALLDONE

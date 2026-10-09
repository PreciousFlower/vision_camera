#!/bin/bash
# 扫描所有 A800 节点的驱动版本与 CUDA 可用性 —— 生成准确的 --exclude 列表。
#
# 为什么需要它：现有文档（tju/11-cluster-connection.md §3）给出了坏节点列表
# gpu8002,gpu8003,gpu8012，但 2026-10-05 实测发现 gpu8015 也是 550.54.14。
# 硬编码一个过期的列表会导致作业随机失败 —— 而且失败得很难查
# （作业能跑起来，只是 torch.cuda.is_available() 返回 False）。
#
# 做法：对每个空闲节点各起一个极轻的 srun（1 核、1G、2 分钟、不申请 GPU），
# 只读 nvidia-smi。不申请 GPU 也能看到驱动版本，成本几乎为零。
#
# 由 tools/scan_drivers.py 上传并调用。

PARTITION=${1:-A800}
echo "### 节点清单"
# ⚠️ 必须用 sinfo -N（每节点一行）。
#    `sinfo -h -o "%N"` 返回的是压缩区间如 "gpu8001-8015"，
#    tr ',' '\n' 拆不开它 —— 第一版就踩了这个坑，结果只拿到 2 个"节点名"。
sinfo -p "$PARTITION" -N -h -o "%N" | sort -u > /tmp/nodes.txt
echo "节点数: $(wc -l < /tmp/nodes.txt)"
head -3 /tmp/nodes.txt

echo "### 逐节点驱动（并行）"
while read -r n; do
  [ -z "$n" ] && continue
  (
    drv=$(timeout 20 srun -p "$PARTITION" -w "$n" --gres=none --cpus-per-task=1 \
            --mem=1g --time=0-00:02:00 -o /dev/null \
            nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    state=$(sinfo -p "$PARTITION" -h -n "$n" -o "%t" 2>/dev/null)
    echo "${n}|${state}|${drv:-QUERY_FAILED}"
  ) &
done < /tmp/nodes.txt
wait

echo "### DONE"

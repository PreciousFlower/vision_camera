#!/bin/bash
# 计算节点环境探测 —— 在 A800 计算节点上执行，判定"常驻服务 + 隧道"是否可行。
# 由 tools/tunnel_test.py 上传并通过 srun 调用。
echo "### 我是谁"; hostname; whoami; date
echo "### 计算节点 → 登录节点 可达性"
for hp in "logini01:22" "logini04:22" "192.168.212.236:10022"; do
  h=${hp%%:*}; p=${hp##*:}
  if timeout 5 bash -c "echo > /dev/tcp/$h/$p" 2>/dev/null; then
    echo "  $h:$p  ✅ 可达"
  else
    echo "  $h:$p  ❌ 不可达"
  fi
done
echo "### 计算节点 → 外网（预期不通）"
if timeout 8 curl -s -o /dev/null -w "  hf-mirror  HTTP %{http_code}  %{time_total}s\n" https://hf-mirror.com 2>/dev/null; then :; else echo "  hf-mirror  ❌ 不通（符合预期）"; fi
if timeout 8 curl -s -o /dev/null -w "  pypi       HTTP %{http_code}\n" https://pypi.tuna.tsinghua.edu.cn/simple/ 2>/dev/null; then :; else echo "  pypi       ❌ 不通"; fi
echo "### GPU"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -2
echo "### Python 与回环绑定"
VLM_PY=/share/home/u00009/apps/conda/envs/vlm/bin/python
$VLM_PY -c "import sys,socket;print('  python',sys.version.split()[0]);s=socket.socket();s.bind(('127.0.0.1',0));print('  回环绑定 ✅',s.getsockname()[1]);s.close()"
echo "### pip 能否装包（离线，预期失败）"
timeout 30 $VLM_PY -m pip install --dry-run --no-input fastapi 2>&1 | tail -2
echo "### 内存与共享内存"
free -g | head -2
df -h /dev/shm | tail -1

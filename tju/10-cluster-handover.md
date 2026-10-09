# 同济超算 A800 使用手册（AI 交接文档）

> **本文用途**：把同济大学超算 A800 的**全部使用信息**写成一份自包含文档。
> **读者**：另一个 AI 助手（或接手的人）。
> **写法约定**：区分「**已实测**」和「**未验证/需确认**」，不写没验证过的东西。
> **最后更新**：2026-10-04（全部事实在集群上实测采集，非凭记忆）
> **相关文档**：`07-defects-found.md`（评测缺陷，**必读**）、`docs/` 其余各篇（项目本身）

---

## 0. 给 AI 的三十秒速览

### 🔴 先记住这三条，否则一定会出错

| # | 陷阱 | 正确做法 |
|---|---|---|
| **1** | **原始结果 json 里的分数是假的** | 读**重判后**的文件（`*.regraded2.json` / `*.graded.json`）。原因：旧判分器缺裸字 `有`，把 1223~1332 道题的答案判成"无法解析"→算错。详见 §5.5、§7.1 |
| **2** | **A800 的分数是「精度天花板」** | 地瓜上 Qwen 是 **w4**、Gemma 是 **w8**（量化）。所以"Qwen 领先"里有一部分是量化不对称带来的，**上板后 Qwen 会被砍得更狠** |
| **3** | **A800 的速度绝对不能引用** | A800（80GB 企业卡）≠ 地瓜（32GB 端侧 SoC）。**只能用比值**，绝对值一个都不能用 |

### 基本信息

```
主机       logini.tongji.edu.cn  端口 10022
备用 IP    192.168.212.236 (logini01，实测可用) / .245 (logini04)
用户       u00009
前置条件   ⚠️ 必须先连学校 VPN，否则完全不通（无公网入口，实测两次断线即失联）
GPU        NVIDIA A800-SXM4-80GB × 8/节点，分区名 A800，19 个节点（gpu8001-gpu8020）
           另有 L40 分区（34 节点，也是 8 卡/节点）
环境       /share/home/u00009/apps/conda/envs/vlm/bin/python
           （python 3.12.14 / torch 2.13.0+cu130 / transformers 5.14.1 / pillow 12.3.0）
工作目录   /share/home/u00009/apps/vlm-kit/
存储       /share（HOME）1.6 TB 配额   /ssdfs/datahome/u00009（NVMe）513 GB 配额
计费       4 元/卡时（按 billing TRES 计，cpus-per-task=4 → billing=4）
提交方式   sbatch（登录节点不能跑 GPU 计算，会被监控杀）
```

### 最短上手路径

```powershell
# 1) 确认 VPN 已连（不连则完全不通）
ping 192.168.212.236

# 2) 本地看集群状态（用仓库自带工具，自动绕 DNS 问题）
cd E:\vibecoding\aipc
python tools\s8-13-status.py      # 作业状态
python tools\s8-14-compare.py     # 三方结果对比
```

```bash
# 3) 或在集群上直接操作
ssh -p 10022 u00009@192.168.212.236
cd /share/home/u00009/apps/vlm-kit
squeue -u u00009
sbatch --array=0-2 slurm/bench-multi.sbatch
```

> **从 Windows 连它最省事的方式**：用仓库里的 `tools/cluster.py`（paramiko 封装，自动绕 DNS 问题）。

---

## 1. 登录

### 1.1 前置条件（最容易踩的坑）

| 条件 | 说明 |
|---|---|
| **必须挂学校 VPN** | 我实测过两次：VPN 通 → 秒连；VPN 断 → ping 100% 丢包、TCP 10022 拒绝。**没有公网入口。** |
| **DNS 可能解析不了内部域名** | 见 1.3。用 IP 直连可绕开。 |

### 1.2 登录信息

```
主机名   logini.tongji.edu.cn      (登录节点 i 系列)
         logina.tongji.edu.cn      (登录节点 a 系列，未实测)
端口     10022                      ← 不是默认 22
用户     u00009
认证     密码 / SSH 密钥            ← ⚠️ 密码不写在本文件里，单独保管
```

**解析到的实际 IP**（2026-09-22 实测）：

| 域名 | IP | 可用性 |
|---|---|---|
| logini.tongji.edu.cn | 192.168.212.236 | ✅ logini01，实测常用 |
| | 192.168.212.240 | ❌ Connection refused |
| | 192.168.212.241 | ❌ Connection refused |
| | 192.168.212.245 | ✅ logini04 |
| logina.tongji.edu.cn | 192.168.215.10 / .14 / .15 / .19 | 未实测 |

### 1.3 ⚠️ DNS 问题（会浪费很多时间，务必先读）

**症状**：`ssh logini.tongji.edu.cn` 报 `Could not resolve hostname`，
Python `socket.getaddrinfo` 报 `Errno 11001`，但 `nslookup` 加校园 DNS 却能查到记录。

**根因**：本机 Windows 解析器配置与校园 DNS 服务器矛盾。
服务器本身（`202.120.190.208` / `202.120.190.108`）能返回记录，但系统解析器用不上。

**三种解法**：

| 方案 | 做法 | 评价 |
|---|---|---|
| **A（推荐）IP 直连** | 直接用 `192.168.212.236` | 最省事，实测 0.7 秒连上 |
| B 换 DNS | 把网卡 DNS 改成 `202.120.190.208` / `.108` | 需管理员权限（`Set-DnsClientServerAddress` 会报"拒绝访问"） |
| C 改 hosts | `C:\Windows\System32\drivers\etc\hosts` 加一行 | 需管理员权限 |

> ⚠️ **另一个坑**：Xshell 里的「静态 DNS」如果设成 `119.29.29.29`（DNSPod，公网 DNS），
> **解析不了同济内网域名**。校园 DNS 才行。

### 1.4 三种连接方式

**方式 1：项目自带的 Python 封装（推荐给 AI）**

```python
import sys; sys.path.insert(0, r"E:\vibecoding\aipc\tools")
import cluster

# 执行命令
rc, out, err = cluster.run("hostname; whoami; date", timeout=120)
print(out)

# 传文件
c, sftp = cluster.sftp()
sftp.put(r"E:\local\file", "/share/home/u00009/remote/file")
sftp.close(); c.close()
```

`tools/cluster.py` 会自动：环境变量覆盖（`VLM_SSH_HOST/PORT/USER`）→ 逐个试候选 IP → 返回可用连接。
实测自动切到 `192.168.212.236`。

**方式 2：Xshell / 图形客户端**

新建会话 → 协议 SSH → 主机 `192.168.212.236` → 端口 `10022` → 用户 `u00009`。
（用户习惯用 Xshell。）

**方式 3：命令行 ssh / scp**

```bash
ssh -p 10022 u00009@192.168.212.236
scp -P 10022 local_file u00009@192.168.212.236:/share/home/u00009/
```

> ⚠️ **scp 没有断点续传**。传大文件（几 GB）中途断了要重来。
> 大文件优先在集群侧直接下载（见 5.3），或用 `tools/rput.py`（分块续传）。

---

## 2. 硬件与调度系统

### 2.1 硬件

**⚠️ 登录节点 ≠ 计算节点，规格不一样，别搞混**（我一开始就记错了）：

| | **登录节点** `logini01` | **A800 计算节点** |
|---|---|---|
| CPU | **96 核** | **56 核**（2 socket × 28 核） |
| 内存 | **629 GB** | **1031 GB**（`RealMemory=1031000M`） |
| GPU | 无 | **8 × A800-80GB**（`Gres=gpu:a800:8`） |
| OS | Anolis OS 8.6 | 同 |
| glibc | **2.28** | 同 |

### 2.1.1 整个集群的分区（`sinfo` 实测）

| 分区 | 可用性 | 时限 | 节点数(A/I/O/T) | 卡型 |
|---|---|---|---|---|
| `intel`（默认 `*`） | up | infinite | 99/11/119/229 | 纯 CPU |
| `amd` | up | infinite | 57/35/164/256 | 纯 CPU |
| `fata` | up | infinite | 2/2/1/5 | 纯 CPU |
| **`A800`** | up | infinite | **17/1/1/19** | `gpu:a800:8` |
| `L40` | up | infinite | 16/16/2/34 | `gpu:l40:8` |
| `L40-fbb` | up | infinite | 3/0/0/3 | `gpu:l40:8` |

**节点命名规律**（`sinfo` NODELIST 实测）：`cpua001-124`、`cpui003-255`、`fata01-05`、
`gpu4001-4042`（=L40）、`gpu8001-8020`（=A800）。

> ⚠️ `gpu8xxx` 里**没有 `gpu8016`**，编号不连续，别硬编码节点名。

### 2.1.2 A800 节点清单（实测状态，会变）

| 节点 | 状态 | CPUS(A/I/O/T) |
|---|---|---|
| gpu8001 | mix | 6/50/0/56 |
| gpu8002 | mix | 1/55/0/56 |
| gpu8003 | mix | 1/55/0/56 |
| gpu8004 | mix | 7/49/0/56 |
| gpu8005 | mix | 48/8/0/56 |
| gpu8006 | mix | 34/22/0/56 |
| gpu8007 | alloc | 56/0/0/56 |
| gpu8008 | drng | 56/0/0/56 |
| gpu8009 | mix | 7/49/0/56 |
| gpu8010 | idle | 0/56/0/56 |
| gpu8011 | mix | 7/49/0/56 |
| gpu8012 | mix | 34/22/0/56 |
| gpu8013 | mix | 28/28/0/56 |
| gpu8014 | drng | 56/0/0/56 |
| gpu8015 | mix | 37/19/0/56 |
| gpu8017 | **drain** | 0/0/56/56 |
| gpu8018 | drng | 56/0/0/56 |
| gpu8019 | drng | 56/0/0/56 |
| gpu8020 | mix | 20/36/0/56 |

（`A`=已分配 `I`=空闲 `O`=其他 `T`=总计；`idle`/`mix` 可排，`alloc`/`drng`/`drain` 别指望）

> ⚠️ **节点编号不连续**（没有 `gpu8016`）。跑之前用 `sinfo -N -p A800` 确认，别硬编码。

### 2.2 ⚠️⚠️ 驱动异构 —— 本环境最大的坑

**torch 是 cu130 编译的**（需要驱动 ≥ 580）。
但**部分节点的驱动是 550.54.14（CUDA 12.4）**，在这些节点上：

```python
torch.cuda.is_available()  # → False
```

**实测结果**：

| 节点 | 驱动 | CUDA 可用 |
|---|---|---|
| gpu8002 | 550.54.14 | ❌ |
| gpu8003 | 550.54.14 | ❌ |
| gpu8012 | 550.54.14 | ❌ |
| gpu8001 | 580.65.06 | ✅ |
| gpu8006 | 580.65.06 | ✅ |
| gpu8009 | 580.65.06 | ✅ |
| **gpu8013** | **580.65.06** | ✅ |
| gpu8020 | 580.65.06 | ✅ |

**应对（两条都要做）**：

1. **提交时排除坏节点**：
   ```
   #SBATCH --exclude=gpu8002,gpu8003,gpu8012
   ```
2. **脚本里做驱动预检**，坏节点 30 秒内退出，不白跑几小时：
   ```bash
   DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
   echo "GPU 驱动 = $DRV"
   "$VLM_PY" -c 'import sys,torch;ok=torch.cuda.is_available();print(f"torch={torch.__version__} cuda_available={ok}");sys.exit(0 if ok else 3)'
   [ $? -ne 0 ] && { echo "CUDA 不可用，退出换节点"; exit 3; }
   ```
   ⚠️ 用 `python -c` 而不是 heredoc —— sbatch 从 spool 文件读脚本，**stdin 不是脚本自身**，heredoc 会静默拿到空输入。

> ⚠️ **注意 `gpu8003` 有时显示 idle**，但它驱动是旧的，**绝不能**分到它。

### 2.3 容易踩的调度坑

| 坑 | 现象 | 解法 |
|---|---|---|
| **`--ntasks=2` 不会跑两遍脚本** | Slurm **不会**为每个 task 执行一遍批处理脚本！`SLURM_PROCID` 恒为 0，第二个任务零输出且无报错 | **必须用 `--array=0-1`**，数组才真起独立进程 |
| **`-w` 钉节点触发配额** | `MaxGRESPerAccount`，StartTime 被推到次日 | **不要用 `-w`**，让 Slurm 自选；要避开坏节点用 `--exclude` |
| **长时限排不上队** | 请求 18 小时 → StartTime 被推到次日 21:45 | 请求时长**短**（3–4 小时）能立刻排上 |
| **资源请求越大越难排** | 7 CPU/160 GB 排到次日；3 CPU/64 GB 立刻上 | 按需申请，别贪 |
| **输出路径相对 WorkDir** | `--output=logs/x.out` 若 WorkDir 下没有 `logs/`，作业 FAILED 且 `Elapsed 00:00:00`、无日志 | **用绝对路径**：`--output=$HOME/apps/vlm-kit/logs/x-%A-%a.out` |
| **`%a` 在非数组作业展开成怪值** | 出现 `multi-2821623-a4294967294.out` 空文件 | 无害，忽略 |
| **超时被 kill** | 作业到 TimeLimit 被杀 | 加 `--requeue`（自动重排）+ **脚本内断点续跑**（逐题落盘） |
| **单卡最多 7 核** | `srun: 必须为每个 GPU 设置不超过 7 个 CPU 核心` | `--cpus-per-task` ≤ 7 |

### 2.4 常用 Slurm 命令

```bash
# 看队列
squeue -u $USER -o '%.10i %.12j %.9T %.11M %.6D %R'

# 看分区
# ⚠️ 格式串里不能有裸 '-'：sinfo 会把 "-8a" 那部分当成"节点格式说明"报
#    "Invalid node format specification: -"。要么去掉对齐符，要么用 %-8a（数字前不写 -）
sinfo -o "%20P %10a %12l %16F %30G"
sinfo -N -p A800 -o "%12N %8t %12C %26G"

# 看单节点详细规格
scontrol show node gpu8013

# 看预计开始时间（排队时）
scontrol show job <jobid> | grep -E 'JobState|Reason|StartTime|SchedNodeList'

# 历史作业（今天）
sacct -u $USER -S today -o JobID,JobName%14,State,Elapsed,ExitCode,NodeList%12

# 取消
scancel <jobid>            # 单个
scancel <jobid>_<idx>      # 数组的某个
scancel -u $USER           # 全部

# 在计算节点上临时开 shell（调试用）
srun -p A800 --gres=gpu:a800:1 --cpus-per-task=4 --mem=64g --time=0-01:00:00 --pty bash -l
```

---

## 3. 环境

### 3.1 Python 环境（已在集群上装好）

```
路径   /share/home/u00009/apps/conda/envs/vlm/bin/python
```

| 包 | 版本（实测） |
|---|---|
| Python | 3.12.14 |
| torch | **2.13.0+cu130** |
| torchvision | 0.28.0 |
| transformers | **5.14.1** |
| accelerate | 1.15.0 |
| Pillow | 12.3.0 |
| numpy | 2.5.3 |
| pandas | 3.0.6 |
| pyarrow | 25.0.1 |
| **qwen-vl-utils** | **0.0.14** |
| sentencepiece | 0.2.2 |
| sentence-transformers | 6.1.0 |
| paramiko | 已装（本地侧也用） |

> ⚠️ **没有 `conda activate`**：`$HOME/apps/conda/etc/profile.d/conda.sh` **不存在**
> （只保留了 `envs/vlm/`）。所以：
> - ❌ 错误：`source ~/apps/conda/etc/profile.d/conda.sh && conda activate vlm`
> - ✅ 正确：**直接用绝对路径** `/share/home/u00009/apps/conda/envs/vlm/bin/python`

### 3.2 ⚠️ 为什么不用 vLLM

**glibc 2.28 太旧**：`vllm>=0.12` 的 wheel 要求 `manylinux_2_31`，装不上。
→ **本环境走纯 transformers + PyTorch 推理**，不用 vLLM。

这意味着：
- 没有 continuous batching，单请求串行
- 速度比 vLLM 慢，但对**精度评测**没影响（评测本来就要确定性输出，`do_sample=False`）

### 3.3 目录结构

```
/share/home/u00009/                     # HOME（注意是 /share，不是 /ssdfs）
├── apps/
│   ├── conda/envs/vlm/                 # Python 环境
│   └── vlm-kit/                        # ★ 评测工作目录
│       ├── bin/                        # 脚本（见 3.4）
│       ├── slurm/                      # sbatch 提交脚本
│       ├── bench/                      # 题目 + 图片
│       │   ├── questions.jsonl         # 4762 题
│       │   └── images/                 # 5298 张
│       ├── logs/                       # 作业日志与结果（每个作业一个子目录）
│       └── configs/models.json
└── ...

/ssdfs/datahome/u00009/                 # 大容量 NVMe
├── models/hf/hub/                      # HF 权重缓存（HF_HOME 指向这里）
│   ├── models--Qwen--Qwen3-VL-2B-Instruct
│   ├── models--Qwen--Qwen3-VL-4B-Instruct
│   └── models--google--gemma-4-E2B-it
└── ...
```

| 盘 | 集群总量（实测） | 本账号配额 | 已用 | 可用 |
|---|---|---|---|---|
| `/share`（HOME，HPC_HD） | 23 PB | 1.6 TB | 20 GB | 1.5 TB |
| `/ssdfs`（NVMe，HPCSSD_NVME） | 1.8 PB | 513 GB | 36 GB | 478 GB |
| `/`（本地盘） | 862 GB | — | 526 GB | 336 GB |

> ⚠️ **`df -h /share` 和 `df -h /ssdfs` 显示的"总量"是整个文件系统的**，
> 不是你个人的。个人配额见上表（1.6 TB / 513 GB）。

### 3.4 工作目录里的关键脚本

| 路径 | 用途 |
|---|---|
| `bin/31-bench-multi.py` | ★ **主力评测脚本**（多数据集、断点续跑、判分器自测） |
| `bin/30-bench-transformers.py` | 早期单数据集版（保留作对照） |
| `bin/16-prepare-bench-data.py` | 数据集 → 评测题 的转换器 |
| `slurm/bench-multi.sbatch` | ★ **主力提交脚本**（模型数组、驱动预检、--exclude） |
| `slurm/bench-mmstar.sbatch` | 只跑 MMStar、可覆盖 max_new_tokens |
| `slurm/tiny.sbatch` | 最小自检（验证能不能跑起来） |
| `slurm/drvprobe.sbatch` | 探测节点驱动版本 |
| `bin/03-download-models.sh` | 下载 HF 权重（走 hf-mirror） |

---

## 4. ⚠️ 网络与下载

### 4.1 huggingface.co 在国内被墙

| 域名 | 状态 |
|---|---|
| `huggingface.co` | ❌ **被污染**：解析到 `2a03:2880:...:face:b00c`（假地址） |
| `hf-mirror.com` | ✅ **完全可用**：API / raw / 文件下载 / 搜索全部正常 |

**在集群上下载 HF 权重的正确姿势**：

```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/ssdfs/datahome/u00009/models/hf"
os.environ["HF_HUB_CACHE"] = "/ssdfs/datahome/u00009/models/hf/hub"
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-VL-4B-Instruct",
                  cache_dir="/ssdfs/datahome/u00009/models/hf/hub",
                  max_workers=8)
```

### 4.2 下载速度实测（差异巨大，选路很重要）

| 路径 | 速度 | 备注 |
|---|---|---|
| 本地（Windows）→ hf-mirror | **0.02 ~ 0.32 MB/s** | 国际线路，极慢。CloudFront 缓存命中时能到几 MB/s，波动大 |
| **集群 → hf-mirror** | 起步 0.25，**峰值 8.66 MB/s** | 快得多（8.28 GiB 约 20 分钟） |
| 本地 → 集群（SFTP） | **1.25 MB/s** | 8.28 GiB 要 2 小时 |

**结论**：**大文件在集群侧直接下，不要从本地上传。**

### 4.3 下载的坑

| 坑 | 现象 | 解法 |
|---|---|---|
| `snapshot_download` 静默失败 | 元数据取不到时**返回已存在的本地目录**，打印"Returning existing local_dir ... as remote repo cannot be accessed"，看起来成功其实没下 | 用 `tools/hf-resume.py`（HTTP Range 续传 + SHA256 校验） |
| `is_complete()` 误判 | `.cache/huggingface/download/*.incomplete` 被当成权重文件 | 判断时**排除 `.cache`** |
| **透明 gzip 导致字节数对不上** | hf-mirror 前置 CDN 会 gzip。urllib 默认发 `Accept-Encoding: gzip`，拿到**解压后**的流，长度比原始大 0.1~13%，"size != size" 误报失败 | 请求头显式加 `"Accept-Encoding": "identity"`；或允许"偏大"并加 2% 容差 |
| 文件名含空格/中文 | `urllib` 报 `URL can't contain control characters` | `urllib.parse.quote(rel, safe="/")` |
| 并发过高被限流 | `HTTP 429 Too Many Requests` | 降到 6–8 线程，失败重试 |
| **本地图片可能超大** | 某些数据集单张 3 MB / 21 MB | 先 HEAD 测真实大小；评测按 448/384 用图，**可安全下采样到最大边 640px**（体积降 20 倍，且更贴近端侧） |

---

## 5. 跑评测的完整流程

### 5.1 一句话流程

```
准备数据(本地) → 上传/同步到集群 → sbatch 提交 → 监控 → 拉结果 → 重判(如需)
```

### 5.2 提交作业

```bash
cd /share/home/u00009/apps/vlm-kit

# 主力评测：数组 0=Qwen3-VL-2B, 1=gemma-4-E2B, 2=Qwen3-VL-4B
sbatch --array=0-2 slurm/bench-multi.sbatch
sbatch --array=2   slurm/bench-multi.sbatch      # 只跑某一个
```

**⚠️ 改脚本后必做**：

```bash
bash -n slurm/bench-multi.sbatch        # 语法检查
md5sum slurm/bench-multi.sbatch         # 与本地比对，确认传对了
```

### 5.3 结果在哪

```
$HOME/apps/vlm-kit/logs/multi-<jobid>-a<idx>/
├── bench-<model>.json                  # 最终结果（跑完才有）
├── bench-<model>.json.partial.jsonl    # 逐题落盘（跑的过程中实时写）
├── bench-<model>.json.partial.jsonl.done  # 跑完后 partial 改名
└── log-<model>.txt                     # 进度日志
```

**监控进度只要看 partial 的行数**（它是逐题追加的，是唯一可靠的"心跳"）：

```bash
wc -l $HOME/apps/vlm-kit/logs/multi-<jobid>-a<idx>/*.partial.jsonl
```

> ⚠️ **主日志是每 50 题才打印一次**，所以"日志不动"不等于卡住。
> 判断卡住要看 partial 文件的 mtime：
> ```bash
> stat -c '%y' .../xxx.partial.jsonl; date    # 差超过 5 分钟才可疑
> ```

### 5.4 断点续跑

`31-bench-multi.py` 把每题结果**立刻追加写**到 `<out>.partial.jsonl`，
重跑时 `load_partial()` 跳过已完成的题。

配合 Slurm 的 `--requeue`：被 TimeLimit kill → 自动重排 → 接着跑。

**退出码约定**：

| 退出码 | 含义 |
|---|---|
| 0 | 全部题跑完 → Slurm 视为成功，不再重排 |
| 1 | 未跑完（部分已落盘）→ 非零退出，`--requeue` 重排后续跑 |
| 3 | 驱动不兼容（`torch.cuda.is_available()==False`）→ 重排换节点通常就好 |
| 124 | 收到 SIGTERM（超时） |

### 5.5 ⚠️ 修判分器后如何"只重判、不重跑"

**关键**：`raw`（模型原始输出）已落盘，所以改判分器后**不需要重跑推理**。

```python
# 做法：用新判分器对每条 raw 重新 grade()
import importlib.util
spec = importlib.util.spec_from_file_location("b31", ".../bin/31-bench-multi.py")
b31 = importlib.util.module_from_spec(spec); spec.loader.exec_module(b31)

for r in results:
    q = qmap[r["id"]]            # ⚠️ 必须从题表补 expect/expect_all！
    ok, det = b31.grade(q, r["raw"])
    r["correct"], r["pred"] = bool(ok), det
```

**两个必须注意的点**：
1. **`expect` 字段必须从题表按 `id` 关联补回** —— 结果行里原本不存它，
   而 `contain`（OCRBench）和 `keyword` 判分完全依赖它。
   不补的话 OCRBench 会全判错（实测变成 `0/300`）。
2. 判分器改完**必须跑自测**：`tools/s8-6-grader-test.py`（28 个用例）。

### 5.6 登录节点禁忌

| ❌ 不要做 | 原因 |
|---|---|
| 在登录节点跑 GPU / 重计算 | **会被监控杀进程** |
| 在登录节点跑长 Python 分析 | 同上。要跑就用 `python3 /ssdfs/datahome/u00009/xxx.py`（放数据盘）或 `srun` |
| 用 `cat` 看 1 MB 的 JSON | 刷屏。用 `tail` / `wc -l` |
| 用 `srun -j <jobid>` | `srun` 没有 `-j`；用 `srun --jobid=` 或 `srun -w <node>` |

---

## 6. 这个环境特有的"验证过的坑"清单

> 全部是本次会话实测踩出来的，按"最容易浪费时间"排序。

| # | 坑 | 症状 | 解法 |
|---|---|---|---|
| 1 | **驱动异构** | `torch.cuda.is_available()=False` | `--exclude=gpu8002,gpu8003,gpu8012` + 脚本内预检 |
| 2 | **`--ntasks=2` 只跑一遍** | 第二个任务零输出无报错 | 用 `--array` |
| 3 | **VPN 断了全不通** | ping 100% 丢包 | 挂 VPN |
| 4 | **DNS 解析不了内网域名** | `Errno 11001` / `No such host` | IP 直连 |
| 5 | **`conda activate` 不可用** | `conda.sh` 不存在 | 用绝对路径 python |
| 6 | **`-w` 钉节点触发配额** | StartTime 推到次日 | 用 `--exclude` 代替 |
| 7 | **`--output` 相对路径** | 作业 0 秒 FAILED 无日志 | 用绝对路径 |
| 8 | **长时限/大资源排不上** | 排队到次日 | 请求 3–4 小时、小资源 |
| 9 | **透明 gzip 字节数虚高** | 下载"失败"但文件其实好的 | `Accept-Encoding: identity` 或加容差 |
| 10 | **`snapshot_download` 静默返回旧目录** | 以为下好了其实没有 | 独立脚本 + SHA256 校验 |
| 11 | **sbatch 里 heredoc 拿到空输入** | 预检逻辑静默失效 | 用 `python -c` 单行 |
| 12 | **文件名含空格** | `URL can't contain control characters` | `urllib.parse.quote` |
| 13 | **主日志有 50 题的缓冲** | 以为卡住 | 看 partial 文件 mtime |
| 14 | **`%a` 在非数组作业是怪值** | `-a4294967294.out` | 无害，忽略 |

---

## 7. 当前状态（截至 2026-10-04）

### 7.1 作业 —— **三个模型全部跑完，队列已空**

| 作业 | 模型 | 配置（照地瓜实际） | 状态 | 耗时 |
|---|---|---|---|---|
| 2832908_0 | Qwen3-VL-2B-Instruct | 448 / ctx 1024 | ✅ 完成 | 37 分 41 秒 |
| 2833137_2 | Qwen3-VL-4B-Instruct | 448 / ctx 1024 | ✅ 完成 | 约 12 分钟 |
| 2832908_1 | gemma-4-E2B-it | 384 / ctx 4096 | ✅ 完成 | 约 2 小时 6 分 |

**⚠️⚠️ 最重要的一条：必须用「重判后」的文件，原始 json 里的分数是假分数**

旧判分器缺裸字 `有`（详见 5.5 和 `07-defects-found.md`），导致新题型的
"有/没有"全判成 unparsed → 算错。

| 模型 | 原始 json（**假**） | **重判后（真）** | 修正题数 |
|---|---|---|---|
| Qwen3-VL-2B | 60.2% | **71.4%** | 1027 |
| Qwen3-VL-4B | 76.9%（本来就对） | **76.9%** | 0 |
| gemma-4-E2B | 47.2% | **66.0%** | 963 |

**该读哪个文件**：

```
2B   → logs/multi-2832908-a0/bench-Qwen--Qwen3-VL-2B-Instruct.regraded2.json
4B   → logs/multi-2833137-a2/bench-Qwen--Qwen3-VL-4B-Instruct.graded.json
Gemma→ logs/multi-2832908-a1/bench-google--gemma-4-E2B-it.graded.json
```

### 7.2 评测集（4762 题 / 5298 图 / 399 MB）

| 分组 | 题数 | 来源 |
|---|---|---|
| L1-MMStar | 1500 | Lin-Chen/MMStar |
| L1-OCRBench | 300 | Yuliang-Liu/MultimodalOCR |
| L2-PPE | 472 | njvisionpower/SHWD |
| L2-Danger | 2490 | Chappieut 工业场景 1388 + 烟火 1102 |

**已剔除的缺陷题**（详见 `07-defects-found.md`）：
- WorkerSafety25 全部 **166 题**（数据不可用：玩具照/合成图/错标，两模型都只考到 52%≈随机）
- SHWD **64 道**答案恒为 "1" 的坏计数题（问句有歧义，不是能力测试）

### 7.3 ⭐ 最终成绩（判分器 v3，三方对比）

```
════════════════════════════════════════════════════════════════════════════
  能力维度                    Qwen3-VL-2B        Qwen3-VL-4B       gemma-4-E2B
════════════════════════════════════════════════════════════════════════════
  看违规·安全帽 (SHWD)     98.7% (466/472)   98.9% (467/472)   98.7% (466/472)
  危险设施·工程机械/沟槽     65.1% (903/1388)  74.5% (1034/1388) 61.2% ( 849/1388)
  危险设施·明火/烟雾        92.5% (1019/1102) 93.6% (1032/1102) 91.5% (1008/1102)
  读字/OCR (OCRBench)      72.0% (216/300)   71.0% (213/300)   67.0% ( 201/300)
  综合推理 (MMStar)         52.9% (794/1500)  61.0% (915/1500)  41.4% ( 621/1500)
  ──────────────────────────────────────────────────────────────────────────
  总体                     71.4% (3398/4762) 76.9% (3661/4762) 66.0% (3145/4762)
   95% 置信区间            [70.1%,72.6%]     [75.7%,78.1%]     [64.7%,67.4%]
════════════════════════════════════════════════════════════════════════════
```

**读法要点**：
- **安全帽上三家打平（98.7/98.9/98.7）** —— 已到天花板，区分不出模型
- **4B 的收益几乎全在「推理」和「复杂工业场景」**：工程机械 +9.4 pp、MMStar +8.1 pp
- **⚠️ OCR 上 4B 没有提升（71.0% vs 72.0%，甚至微降）** —— 说明读字瓶颈在**分辨率/视觉编码**，不是语言模型容量，加参数无效
- Gemma 只在**明火/烟雾**上接近（91.5%），其余全面落后

### 7.4 速度与"写满 512 token"比例（部署可行性指标）

| 模型 | 单题中位 | 生成中位 | 输出中位 | **写满 512 token** |
|---|---|---|---|---|
| Qwen3-VL-2B | 0.07 s | 28.2 tok/s | 2 | 89 题 (1.9%) |
| **Qwen3-VL-4B** | 0.09 s | 23.6 tok/s | 2 | **0 题 (0.0%)** |
| gemma-4-E2B | 0.11 s | **17.8 tok/s** | 2 | **236 题 (5.0%)** |

> ⚠️ **"写满率"是部署关键指标**：地瓜上若每题写满 512 token，
> Qwen 需 512/96.79 ≈ **5.3 s/题**、Gemma 需 512/57.74 ≈ **8.9 s/题**。
> **Gemma 的 5% 写满率说明它有"停不下来"的倾向，端侧实时性风险更高。**
> 4B 反而是 0% —— 它不靠超长输出凑答案。

### 7.5 模型权重（都在 `/ssdfs/datahome/u00009/models/hf/hub/`）

| 模型 | 体积（实测） | 校验 |
|---|---|---|
| `models--Qwen--Qwen3-VL-2B-Instruct` | 4.0 G | ✅ |
| `models--Qwen--Qwen3-VL-4B-Instruct` | **8.3 G** | ✅ safetensors 头部可解析，713 张量 |
| `models--google--gemma-4-E2B-it` | 9.6 G | ✅ |

### 7.6 地瓜端侧对照（官方数据，来自 `04-s600-适配结论.md`）

| 模型 | 地瓜量化 | KV cache | 显存 | 解码速度 |
|---|---|---|---|---|
| Qwen3-VL-2B | w4 | 1024 | 3.7 GB | **96.79 tok/s** |
| Qwen3-VL-4B | w4 | 1024 | 6.0 GB | 53.45 tok/s |
| gemma-4-E2B | **w8** | 4096 | **8.8 GB** | 57.74 tok/s |

> ⚠️ **两个不可忽略的差异**（写报告时必须声明）：
> 1. **精度不对称**：A800 上是 bf16 vs bf16（公平），但地瓜上 Qwen 是 **w4**、
>    Gemma 是 **w8** → 上板后 Qwen 会被砍得更狠
> 2. **速度不可比**：A800 ≠ 地瓜，绝对值一个都不能引用，只能用比值

---

## 8. 本地侧（Windows）配套工具

工作区：`E:\vibecoding\aipc\`

| 文件 | 用途 |
|---|---|
| **`tools/cluster.py`** | ★ paramiko 封装：`connect()` / `sftp()` / `run()`，自动绕 DNS |
| `tools/rput.py` | 大文件分块续传上传 |
| `tools/hf-resume.py` | HTTP Range 续传 + SHA256 校验（替代 snapshot_download） |
| `tools/psftp-batch.py` | 按清单批量上传 |
| **`tools/s8-13-status.py`** | 一眼看清所有作业状态 |
| **`tools/s8-9-regrade2.py`** | 用新判分器重判（从题表补 expect） |
| **`tools/s8-14-compare.py`** | 多模型对比 |
| `tools/s8-6-grader-test.py` | 判分器自测（28 用例） |
| `hpc-kit/bin/31-bench-multi.py` | 评测脚本的**本地主副本**（改完要上传到集群） |
| `tools/bench-multi.sbatch` | 提交脚本的**本地主副本** |

> ⚠️ **改脚本后必须上传**：本地 `hpc-kit/bin/31-bench-multi.py`
> → 集群 `/share/home/u00009/apps/vlm-kit/bin/31-bench-multi.py`
> 传完用 `md5sum` 两边比对。

---

## 9. 待办 / 未验证项

### 9.1 ✅ 已闭环（不用再做）

| 项目 | 结果 |
|---|---|
| 三个模型评测 | ✅ 全部跑完（2B / 4B / Gemma），队列已空 |
| Gemma 重判 | ✅ 47.2% → **66.0%** |
| 集群连接 | ✅ VPN 已恢复，实测可用 |
| 4B 权重 | ✅ 已下载 + 逐字节校验 |
| 判分器缺陷 | ✅ 已修 + 28 项自测 + 三模型统一重判 |

### 9.2 ⚠️ 仍未验证 / 有风险

| 项目 | 状态 | 说明 |
|---|---|---|
| **密码** | ⚠️ 未写入本文档 | 单独保管，不要提交到 git |
| **量化 vs 地瓜** | ⚠️ **最大未验证项** | 本环境是 **bf16**；地瓜 Qwen=**w4**、Gemma=**w8**。**所有分数是"精度天花板"** |
| **速度 vs 地瓜** | ⚠️ 不可比 | A800 ≠ 地瓜。只能引用比值，绝对值一个都不能用 |
| **上板实测** | ❌ 未做 | 需要买 RDK S600（¥14,000 开发套件）。`.hbm` **在 A800 上跑不了** |
| `logina.tongji.edu.cn` | 未实测 | 只用过 logini |
| `gpu8016` | **不存在** | 已确认：A800 节点是 8001-8020 但中间跳过 8016 |
| L40 分区 | 未使用 | 探测时被 `MaxGRESPerAccount` 挡。**但 `sinfo` 显示它 34 节点、8 卡/节点，是 A800 之外的备选算力** |
| 工业场景数据缺口 | ⚠️ 部分 | **"工人抽烟"没找到可用公开数据**（3 个源全是影楼照/海报/合成白雾）→ 已明确标注"未覆盖" |
| 评测集的合成负例 | ⚠️ 需说明 | 烟火的负例是**从原图裁掉标注区域**得到的干净裁块（见 `s7-24-convert-fire.py`），不是独立采样的真实无火场景 |

### 9.3 下一步建议（按价值排序）

1. **写最终对比报告** `docs/06-final-comparison.md` —— 数据已齐（三模型 × 4762 题 × v3 判分器）
2. **补 L2 的 PPE 维度** —— `keremberke/protective-equipment-detection` 已下载（11,978 图，
   含 `glove/no_glove`、`goggles/no_goggles`、`mask/no_mask`、`shoes/no_shoes`），
   可把 L2 从"只有安全帽"扩展到全 PPE
3. **给地瓜发邮件**问三件事（免费，但可能决定排期）：
   ① OELLM 2.0 SDK 何时公开 ② 自行量化编译的申请流程 ③ 能否提供针对中文工业场景的校准图
4. 若要做实时监控，**必须先确认"写满 512 token"的比例**（见 7.4）

---

## 10. 一页速查（复制粘贴用）

```bash
# --- 连上后先做这三件事 ---
hostname; whoami; date
df -h /share /ssdfs/datahome/u00009
/share/home/u00009/apps/conda/envs/vlm/bin/python -c "import torch;print(torch.__version__, torch.cuda.is_available())"

# --- 看队列 / 历史 ---
squeue -u u00009 -o '%.10i %.12j %.9T %.11M %R'
sacct -u u00009 -S today -o JobID,JobName%14,State,Elapsed,ExitCode,NodeList%12

# --- 看分区 / 节点（注意格式串别写裸 '-'）---
sinfo -o "%20P %10a %12l %16F %30G"
sinfo -N -p A800 -o "%12N %8t %12C %26G"

# --- 提交评测（0=2B, 1=Gemma, 2=4B）---
cd /share/home/u00009/apps/vlm-kit
sbatch --array=0-2 slurm/bench-multi.sbatch

# --- 看进度（唯一可靠的心跳；主日志有 50 题缓冲，别被骗）---
wc -l /share/home/u00009/apps/vlm-kit/logs/multi-<jobid>-a<idx>/*.partial.jsonl
stat -c '%y' /share/home/u00009/apps/vlm-kit/logs/multi-<jobid>-a<idx>/*.partial.jsonl; date

# --- 看结果（⚠️ 注意用重判后的文件，见 7.1）---
tail -40 /share/home/u00009/apps/vlm-kit/logs/multi-<jobid>-a<idx>/log-*.txt

# --- 在计算节点开 shell 调试 ---
srun -p A800 --gres=gpu:a800:1 --cpus-per-task=4 --mem=64g --time=0-01:00:00 --pty bash -l
```

**本地侧（Windows）一条命令看全部状态**：

```powershell
cd E:\vibecoding\aipc
python tools\s8-13-status.py        # 三个模型的进度/状态
python tools\s8-14-compare.py       # 三方对比
```

---

## 11. AI 接手时的建议阅读顺序

| 顺序 | 文件 | 用途 |
|---|---|---|
| 1 | **本文档** | 集群怎么用、坑在哪 |
| 2 | `07-defects-found.md` | **三个评测缺陷**（MMStar 截断 / SHWD 计数题 / WorkerSafety25 数据不可用）。不读会误信旧分数 |
| 3 | `05-evaluation-plan.md` | 评测方案设计（三层结构） |
| 4 | `04-s600-适配结论.md` | 地瓜端侧约束（上下文 1024/4096、显存、量化） |
| 5 | `01-evaluation-vlm.md` | Qwen3-VL vs Gemma 的选型分析 |
| 6 | `02-drdk-s600.md` | 地瓜硬件/价格/采购建议 |

**⚠️ 三条最容易出错的认知**：
1. **原始 json 里的分数是假的** —— 必须读重判后的文件（7.1）
2. **A800 的分数是"精度天花板"** —— 地瓜会量化（Qwen w4 / Gemma w8），实际更低
3. **A800 的速度绝对不能引用** —— 硬件完全不同

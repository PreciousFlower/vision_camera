# 同济超算 A800 — 连接与内容速查

> **用途**：只讲两件事 —— **① 怎么连上服务器；② 服务器里有什么。**
> **读者**：另一个 AI 助手（或接手的人）。
> **所有内容均为在集群上实测采集**（2026-10-04），非凭记忆。
> **🔑 密码在 `11b-credentials.md`**（单独文件，含明文，勿外发）。
> **不包含**：评测方案、项目背景、经验教训 —— 那些在 `10-cluster-handover.md`。

---

## 1. 怎么连

### 1.1 ⚠️ 前置条件：必须先挂学校 VPN

**这是唯一的硬性前提。** 实测两次：

| VPN 状态 | 结果 |
|---|---|
| 已连 | 秒连 ✅ |
| 断开 | ping 100% 丢包、TCP 10022 拒绝 ❌ |

**服务器没有公网入口**，不挂 VPN 完全不通。

### 1.2 连接信息

```
主机名   logini.tongji.edu.cn
端口     10022                    ← 不是默认的 22
用户     u00009
密码     👉 见 `11b-credentials.md`（含明文密码，单独文件保管，不要外发）
```

> ⚠️ **密码含特殊字符 `}` 和 `&`**，在 shell / PowerShell 里有特殊含义，
> 拼接进命令行容易出错。**推荐用 Python + paramiko 传参**（绕过 shell 转义）。
> 具体写法见 `11b-credentials.md`。

**解析到的 IP**（2026-09-22 实测）：

| IP | 对应机器 | 可用性 |
|---|---|---|
| **192.168.212.236** | logini01 | ✅ **实测常用，首选** |
| **192.168.212.245** | logini04 | ✅ 实测可用 |
| 192.168.212.240 | logini02 | ❌ Connection refused |
| 192.168.212.241 | logini03 | ❌ Connection refused |

### 1.3 ⚠️ 域名解析不了时怎么办

**症状**：`ssh logini.tongji.edu.cn` 报 `Could not resolve hostname`；
Python 报 `socket.getaddrinfo ... Errno 11001` / `No such host is known`。
但用校园 DNS 查询却能拿到记录。

**根因**：本机 Windows 解析器配置与校园 DNS 矛盾。

**三种解法**（按推荐顺序）：

| 方案 | 做法 |
|---|---|
| **A. IP 直连（最省事）** | 直接用 `192.168.212.236`，实测 0.7 秒连上 |
| B. 改网卡 DNS | 设为校园 DNS `202.120.190.208` / `202.120.190.108`（需管理员权限） |
| C. 改 hosts | `C:\Windows\System32\drivers\etc\hosts` 加 `192.168.212.236 logini.tongji.edu.cn`（需管理员权限） |

> ⚠️ **另一个坑**：Xshell 的「静态 DNS」若设成 `119.29.29.29`（DNSPod，公网 DNS），
> **解析不了同济内网域名**。必须用校园 DNS 或直接 IP 直连。

### 1.4 三种连接方式

**方式 1：Python（推荐给 AI，自动绕 DNS 问题）**

```python
import sys
sys.path.insert(0, r"E:\vibecoding\aipc\tools")
import cluster

# 执行命令
rc, out, err = cluster.run("hostname; whoami; date", timeout=120)
print(rc, out)

# 传文件
c, sftp = cluster.sftp()
sftp.put(r"E:\local\file.txt", "/share/home/u00009/file.txt")
sftp.close(); c.close()
```

`tools/cluster.py` 会自动逐个试候选 IP，返回可用连接。
可用环境变量覆盖：`VLM_SSH_HOST` / `VLM_SSH_PORT` / `VLM_SSH_USER`。

**方式 2：Xshell（图形界面）**

新建会话 → 协议 `SSH` → 主机 `192.168.212.236` → 端口 `10022` → 用户 `u00009`

**方式 3：命令行**

```bash
ssh -p 10022 u00009@192.168.212.236

# 传文件（注意 scp 的端口参数是大写 -P）
scp -P 10022 local_file.txt u00009@192.168.212.236:/share/home/u00009/
```

> ⚠️ **`scp` 没有断点续传**，传几 GB 的大文件中途断了要重来。
> 大文件优先在集群侧直接下载（集群到外网比本地快得多）。

---

## 2. 服务器里有什么

### 2.1 系统

| 项目 | 值 |
|---|---|
| 登录节点 | `logini01` |
| OS | **Anolis OS 8.6** |
| glibc | **2.28** |
| 调度器 | **Slurm** |
| 用户 | `u00009` (uid=10745) |
| 用户组 | `u00009`, `root`, **`p_p00009`**（项目组） |
| Shell | `/bin/bash` |

### 2.2 硬件（⚠️ 登录节点 ≠ 计算节点）

| | **登录节点** `logini01` | **计算节点**（如 `gpu8013`） |
|---|---|---|
| CPU | **96 核** | **56 核**（2 socket × 28 核） |
| 内存 | **629 GB** | **1031 GB** |
| GPU | 无 | **8 × NVIDIA A800-SXM4-80GB** |

### 2.3 分区一览（`sinfo` 实测）

| 分区 | 时限 | 节点数(A/I/O/T) | 卡型 |
|---|---|---|---|
| `intel`（默认 `*`） | infinite | 99/11/119/229 | 纯 CPU |
| `amd` | infinite | 57/35/164/256 | 纯 CPU |
| `fata` | infinite | 2/2/1/5 | 纯 CPU |
| **`A800`** | infinite | **17/1/1/19** | `gpu:a800:8` |
| `L40` | infinite | 16/16/2/34 | `gpu:l40:8` |
| `L40-fbb` | infinite | 3/0/0/3 | `gpu:l40:8` |

**A800 节点清单**：`gpu8001`–`gpu8015`、`gpu8017`–`gpu8020`（**共 19 个，跳过 gpu8016**）

**节点命名规律**：`cpua001-124`、`cpui003-255`、`fata01-05`、
`gpu4001-4042`（=L40）、`gpu8001-8020`（=A800）

### 2.4 存储

| 路径 | 文件系统 | 集群总量 | **本账号配额** | 已用 | 可用 |
|---|---|---|---|---|---|
| **`/share/home/u00009`** | HPC_HD | 23 PB | **1.6 TB** | 20 GB | 1.5 TB |
| **`/ssdfs/datahome/u00009`** | HPCSSD_NVMe | 1.8 PB | **513 GB** | 36 GB | 478 GB |
| `/tmp` | `/dev/sda4` | 862 GB | 共享 | 527 GB | 336 GB |

> ⚠️ `df -h /share` 显示的"总量"是**整个文件系统**的（23 PB），不是你个人的。
> **个人配额是 1.6 TB**（`/share`）和 **513 GB**（`/ssdfs`）。

### 2.5 `$HOME` 目录内容（`/share/home/u00009`）

```
/share/home/u00009/
├── apps/                    ★ 主要工作区
│   ├── conda/envs/vlm/      Python 环境（见 2.6）
│   └── vlm-kit/             评测工具包（见 2.7）
├── backup/                  备份
├── data/                    数据
├── logs/                    日志
├── packages/                安装包
├── probes/                  探测脚本输出
├── hpc-kit.tar.gz           工具包压缩档
├── .bashrc / .bash_profile  shell 配置
├── .conda/  .condarc        conda 配置
├── .ssh/                    SSH 密钥
├── .singularity/            Singularity 容器
└── .triton/                 NVIDIA Triton
```

### 2.6 Python 环境（已在集群装好）

```
路径   /share/home/u00009/apps/conda/envs/vlm/bin/python
版本   Python 3.12.14
```

| 包 | 版本 |
|---|---|
| torch | **2.13.0+cu130** |
| torchvision | 0.28.0 |
| transformers | **5.14.1** |
| accelerate | 1.15.0 |
| Pillow | 12.3.0 |
| numpy | 2.5.3 |
| pandas | 3.0.6 |
| pyarrow | 25.0.1 |
| qwen-vl-utils | 0.0.14 |
| sentencepiece | 0.2.2 |
| sentence-transformers | 6.1.0 |

> ⚠️ **不能用 `conda activate`** —— `$HOME/apps/conda/etc/profile.d/conda.sh` **不存在**
> （只保留了 `envs/vlm/`）。必须用**绝对路径**调用 python。
>
> ⚠️ **不能用 vLLM** —— glibc 2.28 太旧，`vllm>=0.12` 的 wheel 要求 `manylinux_2_31`。
> 本环境走纯 `transformers + PyTorch`。

### 2.7 工作目录 `/share/home/u00009/apps/vlm-kit/`

```
apps/vlm-kit/
├── bin/          脚本
├── slurm/        Slurm 提交脚本
├── bench/        评测数据（questions.jsonl + images/）
├── logs/         作业日志与结果
├── configs/      配置
├── tests/        早期测试集
├── bench.tar     数据打包
└── README.md
```

**`bin/` 内容**：

| 文件 | 用途 |
|---|---|
| `00-preflight.sh` | 环境预检 |
| `01-setup-env.sh` | 环境安装 |
| `02-smoke-gpu.sh` | GPU 冒烟测试 |
| `03-download-models.sh` | 下载 HF 权重（走 hf-mirror） |
| `10-serve-vllm.sh` | 起 vLLM 服务（**本环境不可用**，见 2.6） |
| `11-bench-vlm.py` | VLM 基础评测 |
| `12-rag-stack.py` | RAG 栈 |
| `13-make-testdata.py` | 生成测试数据 |
| `14-make-industry-testset.py` | 生成工业测试集 |
| `15-merge-testsets.py` | 合并测试集 |
| `16-prepare-bench-data.py` | 数据集 → 评测题转换 |
| `21-serve-container.sh` | 容器服务 |
| `30-bench-transformers.py` | 单数据集评测（早期版） |
| **`31-bench-multi.py`** | ★ **主力评测脚本** |

**`slurm/` 内容**：`bench-multi.sbatch`（★ 主力）、`bench-mmstar.sbatch`、
`bench-transformers.sbatch`、`tiny.sbatch`（最小自检）、`drvprobe.sbatch`（探驱动）、
`probe-nodes.sbatch`、`diag-ranks.sbatch`、`serve-qwen3vl-8b.sbatch`、`20-pull-image.sbatch`

### 2.8 模型权重（`/ssdfs/datahome/u00009/models/`）

**HF 缓存位置**：`/ssdfs/datahome/u00009/models/hf/hub/`

| 模型 | 体积 |
|---|---|
| `models--Qwen--Qwen3-VL-2B-Instruct` | 4.0 G |
| `models--Qwen--Qwen3-VL-4B-Instruct` | 8.3 G |
| `models--google--gemma-4-E2B-it` | 9.6 G |

（另有 `models/part1.tar` 14.5 GB，是早期的一次性打包）

**环境变量约定**（脚本里都这么设）：

```bash
export MODEL_ROOT=/ssdfs/datahome/u00009/models
export HF_HOME=$MODEL_ROOT/hf
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1            # 离线加载已下好的权重
export HF_ENDPOINT=https://hf-mirror.com   # 需要联网下载时用
```

### 2.9 可用软件模块（`module avail` 节选）

```
/share/modules 下提供（节选）:
  anaconda3/24.1.2      lammps/22Jul2025-a800    matlab-runtime/2024a
  abaqus/3.10.1         ambertools/25            abinit/10.4.7
  AOCC/5.1.0            AOCL/5.2.0               lftp/4.9.2
```

> `PATH` 里已包含 `/share/apps/miniconda3/condabin` 和 `/usr/share/Modules/bin`。

### 2.10 常用路径速查

```
$HOME                          = /share/home/u00009
工作区                          = /share/home/u00009/apps/vlm-kit
Python                         = /share/home/u00009/apps/conda/envs/vlm/bin/python
HF 权重                        = /ssdfs/datahome/u00009/models/hf/hub
大容量盘                        = /ssdfs/datahome/u00009
```

---

## 3. ⚠️ 连上之后必读的三个坑

### 坑 1：部分 A800 节点驱动太旧，GPU 用不了

torch 是 **cu130** 编译（需驱动 ≥580），但以下节点驱动是 **550.54.14**：

| 节点 | 驱动 | CUDA 可用 |
|---|---|---|
| `gpu8002` | 550.54.14 | ❌ |
| `gpu8003` | 550.54.14 | ❌ |
| `gpu8012` | 550.54.14 | ❌ |
| `gpu8001/8006/8009/8013/8020` | 580.65.06 | ✅ |

**症状**：`torch.cuda.is_available()` 返回 `False`
**解法**：提交时 `#SBATCH --exclude=gpu8002,gpu8003,gpu8012`

### 坑 2：`--ntasks=2` 不会让脚本跑两遍

Slurm **不会**为每个 task 执行一遍批处理脚本 —— `SLURM_PROCID` 恒为 0，
第二个任务零输出且无报错。
**必须用 `--array=0-1`**（数组才真的起独立进程）。

### 坑 3：登录节点不能跑 GPU 计算

会被监控杀进程。计算必须通过 `sbatch` 提交，或临时开交互 shell：

```bash
srun -p A800 --gres=gpu:a800:1 --cpus-per-task=4 --mem=64g --time=0-01:00:00 --pty bash -l
```

---

## 4. 一页速查

```bash
# --- 连接 ---
ssh -p 10022 u00009@192.168.212.236       # 必须先挂 VPN

# --- 确认身份与资源 ---
hostname; whoami; date
df -h $HOME /ssdfs/datahome/u00009

# --- 确认 GPU 可用（登录节点无 GPU，返回 False 是正常的）---
/share/home/u00009/apps/conda/envs/vlm/bin/python -c "import torch;print(torch.__version__, torch.cuda.is_available())"

# --- 看分区 / 节点 ---
# ⚠️ 格式串里不能有裸 '-'（会报 Invalid node format specification）
sinfo -o "%20P %10a %12l %16F %30G"
sinfo -N -p A800 -o "%12N %8t %12C %26G"
scontrol show node gpu8013                # 单节点详细规格

# --- 看队列 / 历史 ---
squeue -u u00009 -o '%.10i %.12j %.9T %.11M %R'
sacct -u u00009 -S today -o JobID,JobName%14,State,Elapsed,ExitCode,NodeList%12

# --- 开交互 shell 调试 ---
srun -p A800 --gres=gpu:a800:1 --cpus-per-task=4 --mem=64g --time=0-01:00:00 --pty bash -l

# --- 看 HC 工具包 ---
ls /share/home/u00009/apps/vlm-kit/
ls /ssdfs/datahome/u00009/models/hf/hub/
```

---

## 5. 未确认项

| 项目 | 说明 |
|---|---|
| **密码** | 🔑 在 **`11b-credentials.md`**（单独文件，含明文，勿外发/勿提交 git） |
| `logina.tongji.edu.cn` | 未实测（只用过 logini） |
| `gpu8016` | **不存在**（节点编号 8001-8020 但跳过 8016） |
| L40 分区 | `sinfo` 显示 34 节点可用，但**未实际提交过作业** |
| `192.168.212.240` / `.241` | 实测 Connection refused（可能临时不可用） |

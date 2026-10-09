# 第一阶段完整流程 —— 本地开发 + 打通闭环

> 目标：**不买摄像头、不买显卡**，用你自己的 PC + 同济 A800，跑通
> 「视频 → 抽帧 → 视觉大模型 → 不文明行为判定 → 告警 → 评测」全链路，
> 并拿到第一份可信的准确率数字。
>
> 所有步骤都在本机实测验证过（2026-10-05）。凡是我没能实测的，都标了 ⚠️。

---

## 0. 先纠正一个会误导你的说法

你手上那批文档里，`03-tongji-a800-deploy.md` 写着"glibc ≥ 2.28 → 可以装 vLLM"，
并给了 vLLM 0.29.0 的完整部署步骤。**这条路走不通**，请以
`11-cluster-connection.md` 的实测结论为准：

| 说法 | 来源 | 结论 |
|---|---|---|
| glibc 2.28 ≥ 2.28，可装 vLLM | `03-tongji-a800-deploy.md`（计划稿） | ❌ **错**。`vllm>=0.12` 的 wheel 要求 `manylinux_2_31`，glibc 2.28 装不上 |
| 集群只能走纯 transformers | `11-cluster-connection.md`（实测稿） | ✅ **对**。已实测，环境里 torch 2.13.0+cu130 / transformers 5.14.1 |

**这个差别决定了整个架构**：没有 vLLM 就没有 continuous batching，
也就无法在集群上起一个"在线服务"给多路摄像头实时调用。
所以第一阶段必须用**批处理**模式，而不是流式服务。

还有三条实测约束直接约束架构设计：

1. **登录节点禁止跑 GPU 计算**，会被监控杀进程 → 必须 `sbatch` 提交。
2. **计算节点没有公网出口** → 集群上**不能**调用云端大模型 API，
   只能用已经下好的本地权重（HF 也要走 `hf-mirror.com`）。
3. **部分 A800 节点驱动是 550.54.14**，而 torch 是 cu130 编译 →
   在这些节点上 `torch.cuda.is_available()` 返回 `False`，
   必须 `--exclude=gpu8002,gpu8003,gpu8012`。

---

## 1. 第一阶段到底要产出什么

不是"一个能演示的系统"，而是**三份能拿来做决策的东西**：

| 产出 | 回答什么问题 | 没有它会怎样 |
|---|---|---|
| **Prompt 版本 + 准确率数字** | 哪几种行为能靠 Prompt 搞定？误报多少？ | 不知道该不该上微调，只能凭感觉 |
| **延迟与成本实测** | 一次判断几秒？一天花多少钱？ | 无法判断云端 API 还是本地部署 |
| **可复用的工程骨架** | 抽帧、去抖、续跑、评测都跑通了 | 第二阶段要从头搭 |

**明确不在第一阶段做的事**：真摄像头接入、实时语音对话、LoRA 微调、多路并发。
这些都依赖"Prompt 方案到底行不行"这个前置结论。

---

## 2. 架构：为什么第一阶段用批处理而不是流式

```
【你的 PC】                          【同济 A800】
 视频文件 / RTSP 模拟流
        │
        ▼  ffmpeg 抽帧（长边 640px，1 fps）
   窗口切分（4 秒 / 4 帧一个窗口）
        │
        ▼  manifest.jsonl + images/
   ┌────┴─────────────────────────────┐
   │                                  │
   ▼ 路径 A（主力）                    ▼ 路径 B（免费但要排队）
 云端 API（Qwen3-VL / GPT-4o）     打包上传 → sbatch → 批量推理
 延迟 1-3 秒，按 token 计费         纯 transformers，1 卡 3 小时约 12 元
   │                                  │
   └────────────┬─────────────────────┘
                ▼  结果都是同一份 windows.jsonl
        时序平滑（M/N 滑动窗口）
                ▼
          告警 alerts.jsonl
                ▼
        对着标注算 P/R/F1 → report.md
```

**为什么两条路径的 Prompt 必须逐字相同**：只有那样，两边的准确率数字才能直接对比，
"用云端还是用本地"才有依据。代码里这一点是强制的 —— 云端和集群**共用同一个
`civility/prompts.py`**（集群侧通过 `PYTHONPATH` 导入同一份代码，见
`tests/smoke_cluster_import.py` 的验证）。

**双路径怎么选**：

| | 路径 A：云端 API | 路径 B：A800 本地 |
|---|---|---|
| 迭代速度 | 改完 Prompt 立刻能跑 | 每次都要上传 + 排队 + 跑作业 |
| 成本 | 按 token 付费 | 1 卡约 4 元/卡时，几百个窗口几块钱 |
| 数据隐私 | 帧数据出本机 | 全程不出集群 |
| 适合 | **第 1-2 周调 Prompt** | 攒够量做批量评测、或数据敏感时 |

**建议**：前两周全用路径 A 调 Prompt，Prompt 稳定后再用路径 B 跑一次大样本，
确认两边数字一致 —— 这本身就是"以后能不能迁到本地/国产卡"的关键验证。

---

## 3. 逐步骤操作（复制即用）

### 步骤 1：环境体检（1 分钟）

```powershell
cd E:\vibecoding\camera\phase1
python -m civility.cli check
```

期望输出末尾是 `结论：✅ 可以开始`。它检查：行为定义是否完整、
`ffmpeg/ffprobe` 是否可用、`yaml/requests/paramiko` 是否装了、
数据目录有没有视频和标注、后端配置是否齐。

本机已经过一次：三个依赖都在，ffmpeg 在
`C:\Users\xiaohua\AppData\Local\Microsoft\WinGet\Links\`，可以直接开工。

> 缺 `yaml/requests/paramiko` 就 `pip install pyyaml requests paramiko`。

### 步骤 2：准备测试视频（不需要摄像头）

第一阶段**不需要买摄像头**。两个办法：

**办法一（已验证）**：用现成的合成视频跑通工程链路。

```powershell
python -m civility.cli extract   # 已经生成过 data/videos/ 下 3 个 mp4
```

**办法二（推荐，识别效果才有参考价值）**：手机拍 30 秒。
拍的时候刻意覆盖这几种情况，每个拍 15-30 秒就够：

| 拍什么 | 用途 |
|---|---|
| 真扔垃圾：有人把纸团扔到地上就走 | 正例 |
| 假扔垃圾：有人把东西**放进**垃圾桶 | 反例（测误报，比正例更重要） |
| 地上本来就有垃圾、但没人 | 反例（最经典的误报源） |
| 真吸烟：能看清手夹烟、口鼻有烟 | 正例 |
| 手靠近嘴（喝水/吃东西） | 反例 |
| 一段完全干净的场景 | 负例视频（算精确率必须有） |

放进 `data/videos/`，文件名带语义（如 `litter_real_pos.mp4`）。

> ⚠️ **合成视频只能验证"管子通不通"，不能用来判断"识别准不准"** ——
> 画面里是几何方块，模型对它的判断没有任何参考价值。

### 步骤 3：写标注（30 分钟，跳过这步后面全是自欺欺人）

编辑 `data/annotations.jsonl`，每行写"某视频的某时间段确实存在某行为"：

```jsonl
{"video": "litter_real_pos.mp4", "behavior_id": "littering", "start": 5.0, "end": 12.0}
{"video": "smoking_real_pos.mp4", "behavior_id": "smoking", "start": 2.0, "end": 20.0}
```

没写在里面的 `(视频, 行为)` 组合一律视为**负例**。

**为什么这步不能省**：不写标注，`score` 只能告诉你"跑了多少窗口"，
永远算不出误报率 —— 而误报率恰恰是"这系统能不能上线"的决定性指标。

### 步骤 4：抽帧 + 切窗口

```powershell
python -m civility.cli extract
```

输出形如 `litter_real_pos.mp4: 30.0s → 30 帧 → 7 个窗口`。

参数在 `config/settings.yaml`：

- `extract.max_side: 640` —— 长边缩到 640px。别再降：448 会明显丢
  "手里有没有拿东西"这类关键细节。
- `extract.fps: 1.0` —— 每秒抽 1 帧。
- `behaviors.yaml → defaults.window_stride_sec: 4.0` —— 每 4 秒一个判断窗口。

**抽帧是一次性成本**：抽完存在 `data/frames/`，之后改 Prompt 只重跑推理，
不重新解码视频。这是迭代速度的关键。

### 步骤 5：先看 Prompt（不用钱，但最能发现问题）

```powershell
python -m civility.cli prompt
```

把完整 Prompt 打到屏幕并存到 `runs/prompt-preview.txt`。

**这一步值得你花十分钟逐字读一遍**。绝大部分识别效果问题，
在读 Prompt 的时候就能看出来 —— 比如定义写得像社会规范判断而不是视觉描述、
反例列得太少、没告诉模型"证据不足就报没有"。

Prompt 由 `config/behaviors.yaml` 编译而来，**加行为不用改代码**：

```yaml
  - id: spitting          # 唯一标识，全程用做告警类型
    name: 随地吐痰
    enabled: true          # 从 false 改 true 就启用
    definition: >          # 只写"能看见什么"，不写"这合不合规矩"
      画面中有人做出明显的口部用力吐出的动作，并有液体/飞沫落到地面。
    evidence_required: >   # 判定所需的最低证据
      需要看到口部前伸用力的动作，最好同时可见落地飞沫。
    counter_examples:      # ★ 反例是压制误报的主力，比正例更重要
      - 咳嗽、说话、喝水（无吐出物）
      - 弯腰动作
```

`behaviors.yaml` 里已经写好两条**停用**的扩展示例（随地吐痰、驾驶用手机），
照着改就行。

### 步骤 6：用 mock 后端跑通链路（0 成本，10 秒）

```powershell
python -m civility.cli all
```

`mock` 是本地假模型，故意注入 20% 噪声。它的价值在于：
**把"工程问题"和"模型问题"彻底分开**。链路先跑通，再换真模型；
否则一旦结果不对，你分不清是 Prompt 差还是代码有 bug。

本机实测结果（这就是预期输出形态）：

```
[manifest] 共 9 个窗口
[run] 完成：{'ran_now': 9, 'errors': 0, ...}
[聚合] 9 个窗口 → 3 条告警
窗口级：合计 P 62.5% / R 83.3%
告警级：合计 P 66.7% / R 100.0%     ← 时序平滑把噪声压掉了
```

注意告警级指标**高于**窗口级 —— 这正是时序平滑该有的效果。
如果反过来（告警级明显更差），说明 `aggregate` 参数需要调。

### 步骤 7：接云端 API，开始真正调 Prompt

**这是第一阶段的主战场。**

```powershell
# 阿里云百炼（DashScope）OpenAI 兼容端点
$env:DASHSCOPE_API_KEY = "sk-你的key"

# 先小样本试水（比如只跑第一个视频），确认调用通了再全量
python -m civility.cli run --backend cloud
python -m civility.cli aggregate
python -m civility.cli score
```

换别的供应商只改 `config/settings.yaml`（都是 OpenAI 兼容协议，
不用改代码）：

```yaml
cloud:
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  model: "qwen3-vl-plus"
  api_key_env: "DASHSCOPE_API_KEY"
  concurrency: 2          # 被 429 就降到 1
  stitch_frames: false    # 端点报"不支持多图"时改 true
  price_in_per_mtok: 0.0  # 填单价后 run 结束会打印估算花费
  price_out_per_mtok: 0.0
```

> 模型选型：**Qwen3-VL 系列单次请求最多支持 256 张图**，
> 所以"一个窗口 4 帧"完全没问题（见
> [模型调优文档](https://help.aliyun.com/zh/model-studio/model-training-overview)）。
> 如果换成只支持单图的端点，把 `stitch_frames` 改成 `true`，
> 代码会把 4 帧拼成一张 2×2 网格图并在格子上标注时间戳。

**成本控制**：先跑一个视频（约 7 个窗口）估算单窗口成本，再乘总量。
`run` 结束会打印 token 数和估算金额。几百个窗口通常就是几块到几十块钱。

### 步骤 8：读报告，决定下一步改什么

```powershell
# runs/default/report.md（人看）+ report.json（机读）
```

报告分两层，**别只看一层**：

| 层次 | 含义 | 低了说明什么 |
|---|---|---|
| **窗口级** | 模型 + Prompt 的原始能力 | 要改 `behaviors.yaml` 的定义/反例 |
| **告警级** | 加时序平滑后的系统表现 | 要调 `settings.yaml` 的 `window_k`/`min_hits` |

报告末尾还会自动列出**下一步该改哪里**，比如：

- 召回率低 → 先查 `evidence_required` 和 `observable_cues` 够不够具体
- 误报多 → 去看被误报窗口的 `observable` 字段，把模型认错的场景**原话**
  写进 `counter_examples`
- 精确率整体 < 80% → **别急着调阈值**。阈值只能在 P/R 之间做交换，
  压不出真正的提升。回 Prompt 加反例才是收益最高的动作。

### 步骤 9：Prompt 一致性体检（容易被忽略，但很重要）

```powershell
python -m civility.cli stability --windows 8 --repeat 3
```

同一批窗口重复推理 3 次，看结论是否一致。**判断标准：一致率 < 85%
说明 Prompt 有歧义** —— 此时调阈值、加规则都是治标，必须回去改 Prompt。

这一步成本低（几十次调用）但信息量很大，建议每改一版 Prompt 跑一次。

### 步骤 10：（可选）跑一次 A800 本地模型对照

免费，但要排队。收益是拿到"本地 vs 云端"的同口径对比数据。

```powershell
# 10.1 先确认 VPN 已连，然后探一次集群状态（只读，不花机时）
python tools\cluster_probe.py
```

它会检查 VPN、登录节点、Python 环境、**已下好的权重**、A800 空闲卡数，
以及 `pyyaml / qwen-vl-utils` 在不在（决定作业包要不要额外装依赖）。

```powershell
# 10.2 打包（含 manifest、帧、Prompt、集群脚本、civility 包）
python -m civility.cli cluster-prepare

# 10.3 本地先验证集群侧的 import 逻辑，避免浪费一次排队
python tests\smoke_cluster_import.py

# 10.4 上传（需 VPN）
python -m civility.cli cluster-upload
```

然后 SSH 上去提交：

```bash
ssh -p 10022 u00009@192.168.212.236
cd /share/home/u00009/apps/civility-kit

# 冒烟：先跑 5 个窗口，确认能跑起来再放全量
LIMIT=5 sbatch cluster/run-vlm.sbatch

# 看队列和日志
squeue -u u00009 -o '%.10i %.20j %.9T %.11M %R'
tail -40 logs/vlm-<jobid>.out
```

跑通后再全量：`sbatch cluster/run-vlm.sbatch`

```powershell
# 10.5 拉回结果并评测
python -m civility.cli cluster-fetch
python -m civility.cli run --backend cluster
python -m civility.cli aggregate
python -m civility.cli score
```

**集群侧已经踩过的坑，脚本里都处理了**：

| 坑 | 脚本里的处理 |
|---|---|
| 三个老驱动节点 CUDA 不可用 | `#SBATCH --exclude=gpu8002,gpu8003,gpu8012` + 脚本内预检（不通过就 `exit 3` 让 Slurm 换节点） |
| 作业超时被杀 | `#SBATCH --requeue` + 逐窗口落盘 + 重跑时跳过已完成 |
| `--output` 用相对路径导致 0 秒 FAILED | 全部用绝对路径 |
| sbatch 里 heredoc 拿到空输入 | 预检用 `python -c` 单行 |
| `conda activate` 不可用 | 直接用绝对路径 `.../conda/envs/vlm/bin/python` |
| 集群环境没有 pyyaml | `civility/config.py` 惰性导入 yaml，`tests/smoke_cluster_import.py` 专门验证这条 |

---

## 4. 三个你一定会问的问题

### Q1：模型部署在哪里？

第一阶段**哪都不用部署**：

- **云端 API**：你只管发请求，模型在阿里云/OpenAI 那边。
- **A800 本地模型**：用 `sbatch` 交作业，跑完就释放，不占卡。
  已下好的权重在 `/ssdfs/datahome/u00009/models/hf/hub/`：
  `Qwen3-VL-2B-Instruct`(4.0G)、`Qwen3-VL-4B-Instruct`(8.3G)、
  `gemma-4-E2B-it`(9.6G)。**作业包默认用 Qwen3-VL-4B** —— 它在既往
  4762 题的评测里拿到 76.9%，明显好于 2B(71.4%) 和 Gemma(66.0%)。

⚠️ **计算节点没有公网出口**，所以集群上跑不了云端 API，也不能临场下新模型。
想换更大的模型（比如 Qwen3-VL-8B/32B），要**在登录节点**（有外网）
先下好权重再跑作业。

### Q2：需要买摄像头吗？

**第一阶段不需要。** 手机拍 30 秒就够，而且比摄像头更灵活（能凑近拍细节）。

什么时候才需要买，怎么选：

| 场景 | 选什么 | 关键要求 |
|---|---|---|
| 室内固定机位 | USB 摄像头 | 支持 UVC、1080P@30fps |
| 室外/大范围 | RTSP 网络摄像头 | **必须支持 RTSP 或 ONVIF**、≥2K 分辨率 |

**没有硬件也能先跑实时链路**：用 RTSP 模拟器把本地 mp4 变成实时 RTSP 流
（例如 `konekuto/vcam`），代码侧完全等价，`ffmpeg` 抽帧那层不用改。
这也是第二阶段"接真摄像头"的低成本预演。

### Q3：云端还是本地？成本和延迟怎么权衡？

第一阶段只需要一个答案：**先用云端把 Prompt 调到能看，再谈部署**。
理由是本地部署（A800 上甚至没有 vLLM）迭代一次要几十分钟排队，
而云端改完就能跑，这个差异在调 Prompt 阶段是决定性的。

拿到数字之后再决策，判断依据是这两条：

- **窗口级精确率**：如果 < 80%，什么部署方案都救不了，先回去改 Prompt。
- **单窗口延迟**：如果告警要求"3 秒内"，那本地 4B 模型在 A800 上单窗口
  通常在几秒量级（⚠️ 未实测，见下），可能压不进去，得考虑更小的模型
  或者把抽帧频率降下来。

---

## 5. 实测记录与未验证项

**已实测（本机，2026-10-05）**：

- `check` / `prompt` / `extract` / `run(mock)` / `aggregate` / `score` /
  `stability` / `all` / `cluster-prepare` 全部跑通，退出码 0。
- 端到端 mock 结果：窗口级 P 62.5% / R 83.3%，告警级 P 66.7% / R 100.0%。
- `tests/smoke_cluster_import.py` 7 项全通过，含"集群无 pyyaml"的模拟场景。
- 帧拼接回退可用：4 帧 640px → 单张 1292×732 网格图。

**⚠️ 未实测（需要你来做）**：

| 项目 | 为什么没做 | 怎么验 |
|---|---|---|
| **VPN 连通性** | 我这边 VPN 是断的（`ping 192.168.212.236` 100% 丢包） | 连上 VPN 后跑 `python tools\cluster_probe.py` |
| 云端 API 真实调用 | 需要你的 API Key | 设置 `DASHSCOPE_API_KEY` 后跑 `run --backend cloud` |
| A800 上 Qwen3-VL-4B 的真实延迟 | 要占卡、要排队 | 先 `LIMIT=5 sbatch ...` 冒烟，看日志里的 `本窗 X.Xs` |
| 真视频上的识别准确率 | 手上只有合成视频 | 拍 30 秒真视频替换 `data/videos/` |

**VPN 一通，建议立刻做这三件事**（顺序很重要）：

1. `python tools\cluster_probe.py` —— 确认权重还在、A800 有没有空卡
2. `LIMIT=5 sbatch cluster/run-vlm.sbatch` —— 5 个窗口的冒烟，几毛钱
3. 冒烟通过再放全量

---

## 6. 目录结构

```
phase1/
├── config/
│   ├── behaviors.yaml      ★ 不文明行为定义（改这个就能加/改行为）
│   └── settings.yaml       ★ 后端、抽帧、聚合、集群参数
├── civility/
│   ├── config.py           配置加载（惰性导入 yaml，集群也能用）
│   ├── prompts.py          ★ 行为定义 → Prompt（云端/集群共用，保证可比）
│   ├── vlm.py              三个后端：mock / cloud / cluster
│   ├── extract.py          ffmpeg 抽帧 + 窗口切分 + manifest
│   ├── aggregate.py        时序平滑 → 告警 + 一致性体检
│   ├── runner.py           编排 + 断点续跑 + 行为指纹
│   ├── score.py            窗口级/告警级 P/R/F1 + 报告
│   └── cli.py              命令行入口
├── cluster/
│   ├── run_cluster_vlm.py  集群侧批推理（纯 transformers）
│   └── run-vlm.sbatch      Slurm 提交脚本（含驱动预检与断点续跑）
├── tools/cluster_probe.py  集群只读探针
├── tests/smoke_cluster_import.py  集群导入路径冒烟（本地可跑，无需 GPU）
├── data/
│   ├── videos/             放视频（已含 3 个合成测试视频）
│   ├── frames/             抽帧产物
│   └── annotations.jsonl   ★ 标注（不用多，但必须有负例）
└── runs/                   结果、告警、报告
```

---

## 7. 进入第二阶段的条件

第一阶段做完，拿这三个问题的答案来决定下一步：

1. **Prompt 能不能搞定？** 窗口级精确率 ≥ 80% 且召回率 ≥ 75%
   → 不用微调，直接进第二阶段接真摄像头。
2. **哪些行为搞不定？** 如果只有 1-2 类行为差，收集误判案例做 LoRA 微调
   （500-1000 条标注样本）；如果全都差，先回去改 Prompt，别急着标数据。
3. **延迟够不够？** 单窗口延迟决定了能不能做实时告警，
   也决定第二阶段要不要精简抽帧策略。

**第二阶段（接真摄像头 + 实时告警）需要额外做的**：
RTSP 拉流（用 RTSP 模拟器先验证）、把批处理换成常驻进程、
多路摄像头的推理调度、TTS 语音播报。
**语音对话（ASR + LLM + TTS）建议放到第三阶段** —— 它依赖
"识别准不准"这个前置结论，现在做等于在流沙上盖楼。

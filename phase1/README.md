# 摄像头不文明行为识别 —— 第一阶段

本地开发用的完整工程骨架：视频 → 抽帧 → 视觉大模型 → 行为判定 → 告警 → 评测。

本机网页依赖：`python -m pip install -r requirements-web.txt`。
工位管理首次使用还需 `python tools/prepare_person_detector.py` 下载CPU人员模型；网页选择行为预设并绘制工位区域。多预设、人数/时长、人工关联及统计问答见 [工位功能说明](../工位人员管理与多套行为预设-2026-10-13.md)，当前控制台 http://127.0.0.1:8852/。
默认同时检查配置行为与 AI 开放风险（如倒地、明火、冲突），详见
[2026-10-10 优化说明](../优化说明-2026-10-10.md)。新风险准确率与推理延迟待真模型验证。

最新识别修复及用户4个视频的真模型结果见 [识别修复说明](../识别修复说明-2026-10-10.md)。
文件分析现为3fps、6秒上下文、2秒步长、最多18帧，图片可完整放大并逐帧回看。
文件分析默认描述人物动作过程，证据不完整的阳性回答自动由模型独立核验一次。
重启时请在原终端先停止旧服务；默认端口已占用会明确拒绝启动，避免新旧实例混用。

晚间实时加速保留窗口、抽帧和输出预算，默认单进程合批并复用重复帧图像特征。
采集持续运行，下一批使用近期画面；页面完成即显示。真模型对照与限制见
[延迟优化说明](../延迟优化说明-2026-10-10.md)，仍需重启本机网页及远端模型服务后生效。

**两种用法**：

- 🖱️ **可视化网页**（推荐）：`python -m webui.server` → http://127.0.0.1:8848
  —— 架构与实测数据见 [`../WEBUI-可视化界面.md`](../WEBUI-可视化界面.md)
- ⌨️ **命令行**（批量评测）：见下方速查，完整流程见 [`../PHASE1-开发流程.md`](../PHASE1-开发流程.md)

## 可视化网页

```powershell
$env:VLM_SSH_PASSWORD = '<集群密码>'       # 或先配免密
python -m civility.cli cluster-prepare      # 打包（首次一次）
python -m civility.cli cluster-upload       # 上传到 A800（首次一次）
python -m webui.server                      # 启动网页 → http://127.0.0.1:8848
```

页面上：① 启动服务 → ② 选择行为预设、拖视频进去看实时结果 → ③ 改行为定义（新任务使用保存版本，已开始任务保持独立快照）。切换预设不需要重启模型。

> ⚠️ 用完点「停止并释放节点」，否则一直占着一张 A800。

## 60 秒命令行跑通（不需要摄像头、GPU、API Key）

```powershell
cd E:\vibecoding\camera\phase1
python -m civility.cli check                 # 体检
python -m civility.cli prompt                # 看将要发给模型的 Prompt（值得读一遍）
python -m civility.cli all                   # 抽帧→推理(mock)→聚合→评测
```

跑完看 `runs/default/report.md`。

**预期结果**（mock 默认干净，用来验证管道正确性）：

以下百分比是旧协议的历史自检记录。2026-10-10 命令行已对齐当前 AI 告警请求，
本次默认模拟自检的告警 P=100%、R=50%；以当前生成的报告为准。
这些数字验证模拟管道与配置效果，不是视觉模型准确率。

```
窗口级：P 75.0% / R 100.0%
告警级：P 100.0% / R 100.0%   ← 告警级满分说明聚合逻辑正确
```

窗口级那 75% 不是 bug：`littering_pos_vis.mp4` 的标注从第 5 秒开始，
而第 1 个窗口（0–4 秒）确实不含该行为，所以它算一个真负例被正确排除。

### 想让 mock 变"吵"来观察时序平滑的作用

`config/settings.yaml` 里把 `mock.noise` 改成 `0.30`，清掉 `runs/` 重跑：

| mock 设置 | 窗口级 | 告警级 | 说明 |
|---|---|---|---|
| `noise: 0.0`（默认） | 75% P / 100% R | **100% P / 100% R** | 自检：管道与聚合正确 |
| `noise: 0.30` | 15.8% P / 100% R | 16.7% P / 100% R | 压力测试：指标低是**预期** |

> ⚠️ 噪声档是**故意让假模型乱答**，指标难看属正常。
> 早期版本把噪声默认设成 0.2，导致自检精确率只有 18% —— 看起来像模型很差，
> 实际是 mock 在乱答。一个"故意答错"的假模型对自检没有价值。

## 换成真模型

```powershell
# 云端（主力：迭代快）
$env:DASHSCOPE_API_KEY = "sk-..."
python -m civility.cli run --backend cloud
python -m civility.cli aggregate
python -m civility.cli score

# 同济 A800 本地模型（按卡时计费，需要排队 + VPN）
python tools\cluster_probe.py                # 先探集群状态
python -m civility.cli cluster-prepare
python tests\smoke_cluster_import.py         # 本地先验，别浪费排队
python -m civility.cli cluster-upload
# → ssh 上去 sbatch cluster/run-vlm.sbatch
python -m civility.cli cluster-fetch
python -m civility.cli run --backend cluster
```

## 加一种新的不文明行为

**只改 `config/behaviors.yaml`，不用动代码**：

```yaml
  - id: spitting
    name: 随地吐痰
    enabled: true
    definition: >       # 只写"能看见什么"，不写"这合不合规矩"
      画面中有人做出明显的口部用力吐出的动作，并有液体/飞沫落到地面。
    evidence_required: >
      需要看到口部前伸用力的动作，最好同时可见落地飞沫。
    counter_examples:   # ★ 反例是压制误报的主力
      - 咳嗽、说话、喝水（无吐出物）
```

改完先 `python -m civility.cli prompt` 看效果，再 `all` 重跑。

## 告警什么时候报？（三种策略）

`config/settings.yaml` 的 `aggregate.mode`，网页上也能随时切换：

| 策略 | 行为 | 延迟 | 误报 |
|---|---|---|---|
| **`ai`**（默认） | **模型自己判断该不该报警，并给出理由** | 中 | 低 |
| `immediate` | 判定命中就立刻报 | 最低 | 高 |
| `smoothed` | 多窗命中才报（M/N） | 最高 | 最低 |

`ai` 模式下模型额外输出 `alert` / `alert_reason`，告警里会显示它的理由。
**模型没给 `alert` 字段时回落到 `violation`**，不会静默不报。

对比三种策略（同一批结果、零额外推理）：

```powershell
python tools/compare_alert_modes.py
```

## 常用命令

| 命令 | 作用 |
|---|---|
| `check` | 体检：配置 / 依赖 / ffmpeg / 后端 |
| `prompt` | 打印并保存完整 Prompt |
| `extract` | 视频 → 抽帧 + manifest |
| `run` | 跑推理（`--backend mock\|cloud\|cluster`） |
| `aggregate` | 窗口结果 → 时序平滑 → 告警 |
| `score` | 对标注算 P/R/F1，出 `report.md` |
| `stability` | Prompt 一致性体检（一致率 <85% 说明 Prompt 有歧义） |
| `all` | 一把梭：extract → run → aggregate → score |
| `cluster-prepare` / `cluster-upload` / `cluster-fetch` | A800 作业的三步 |

## 关键参数在哪

- `config/behaviors.yaml` —— 行为定义、窗口长度、告警门槛
- `config/settings.yaml` —— 后端、抽帧分辨率、聚合参数、集群路径

## 三条别踩的坑

1. **不写 `data/annotations.jsonl` 就算不出误报率**，而误报率是能不能上线的决定性指标。
2. **别急着调阈值**。精确率低的时候调阈值只是在 P/R 之间交换，压不出真实提升 ——
   回去给 `counter_examples` 加反例才是正解。
3. **合成视频只能验证链路通不通**，不能用来判断识别准不准。

## 依赖

```
pyyaml requests paramiko      # paramiko 只在访问集群时需要
ffmpeg / ffprobe              # 必须在 PATH 里
```

生成测试视频：`ffmpeg`（本机已在 `WinGet\Links\` 下）。

"""统一视觉推理后端 —— 一份 Prompt，三种跑法。

    MockBackend    本地假模型：零成本、零网络、确定性输出，用来跑通链路和自检
    CloudBackend   OpenAI 兼容云端 API：第一阶段主力，快速迭代 Prompt
    ClusterBackend 同济 A800 本地模型：结果由 sbatch 作业产出后读回（见 cluster/）

三者接口完全一致，切换只改 settings.yaml 的 backend 字段，业务代码零改动。

⚠️ 一个必须记住的约束：**A800 计算节点没有公网出口**（见 tju/11-cluster-connection.md
   §4，HF 必须走 hf-mirror，且登录节点才可外网）。所以集群侧不能调用云端 API，
   ClusterBackend 走的是"离线批推理 + 结果回传"，不是在线服务。
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .config import BehaviorSet


@dataclass
class WindowRequest:
    """一个判断窗口的推理请求。"""

    window_id: str
    video: str
    t_start: float
    t_end: float
    frames: list[str]  # 图片路径或 data URL
    labels: list[str] = field(default_factory=list)
    # 是否要求模型自己判断"该不该告警"（对应 aggregate.mode: ai）。
    # 放在请求里而不是后端构造参数里，是因为同一批窗口可能用不同策略跑对照。
    with_alert: bool = False
    # 是否要求模型描述画面里每个人在做什么（对应网页的「摄像头监测」页）
    with_actions: bool = False
    # 开放风险识别：由 AI 主动发现配置之外的现场风险。
    with_risks: bool = False


@dataclass
class WindowResult:
    window_id: str
    video: str
    t_start: float
    t_end: float
    parsed: dict | None
    raw: str = ""
    error: str | None = None
    retries: int = 0
    latency_sec: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    backend: str = ""


class Backend(Protocol):
    name: str

    def infer(self, req: WindowRequest, bs: BehaviorSet) -> WindowResult: ...


def stitch_grid(paths: list[str], out_path: str | Path, cols: int = 2) -> str:
    """把多帧拼成一张 2×2 网格图，并**在格子上标注时间戳**。

    什么时候用它：
      - 云端端点不支持一条消息里放多张图（部分第三方 OpenAI 兼容服务如此）
      - 想压低 token：4 张独立图 vs 1 张拼图，后者常常便宜一半以上
      - 想降低请求体积（base64 后 4×80KB → 1×~200KB）

    代价：单格分辨率下降。4 帧拼 2×2 时每格约 320×180，
    "手里拿的是什么"这类细节会变模糊 —— 所以默认不启用，
    只在端点不兼容或成本吃紧时打开。

    时间戳直接画在图上，而不是只写在文字里：模型看图时能就近读到，
    描述"第几帧手在做什么"会明显更准确。
    """
    from PIL import Image, ImageDraw

    imgs = [Image.open(p).convert("RGB") for p in paths]
    if not imgs:
        raise ValueError("没有可拼接的帧")

    n = len(imgs)
    cols = max(1, min(cols, n))
    rows = (n + cols - 1) // cols
    w = max(i.width for i in imgs)
    h = max(i.height for i in imgs)

    pad = 4
    canvas = Image.new("RGB", (cols * w + pad * (cols + 1), rows * h + pad * (rows + 1)), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    for k, im in enumerate(imgs):
        r, c = divmod(k, cols)
        x = pad + c * (w + pad)
        y = pad + r * (h + pad)
        canvas.paste(im, (x, y))
        # 左上角画序号，与 frame_labels 的 t= 时间对应
        draw.rectangle([x, y, x + 34, y + 16], fill=(0, 0, 0))
        draw.text((x + 3, y + 3), f"#{k + 1}", fill=(255, 235, 59))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "JPEG", quality=88)
    return str(out_path)


def to_data_url(path_or_url: str) -> str:
    """本地路径 → data URL。已 HTTP(S)/data 开头的原样返回。

    为什么用 data URL 而不是 base64 字段：OpenAI 兼容接口普遍支持 data URL，
    且对我们的"图片已落盘、按窗口组装"的流程最省事。
    """
    if path_or_url.startswith(("http://", "https://", "data:")):
        return path_or_url
    p = Path(path_or_url)
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    with open(p, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


# ───────────────────────────── Mock ─────────────────────────────

# 演示模式的"剧本"。
#
# 为什么需要它（用户实测反馈）：
#     勾了模拟验证跑 RTSP 演示流时，页面上的结论是
#     「[mock] live @ 12.0s」+ 随机人数 —— 与画面完全对不上，
#     用户的第一反应是"AI 根本没识别出来丢垃圾"。
#     问题不在模型，在于演示模式的输出**本身不自洽**：
#     它只是随机占位文字，既没说画面里有什么，也不会产出对应的告警，
#     于是"演示"这件事根本不成立（看不到告警弹出的完整链路）。
#
# 设计：选一个场景 → 场景描述 / 人物动作 / 判定依据 / 是否告警
# 全部按同一份剧本产出，画面之外的文字与结论互相自洽。
# 所有文字都带「演示」前缀，避免被误当成真实识别结果。
MOCK_SCENES: dict[str, dict] = {
    "normal": {
        "label": "正常（无违规）",
        "scene": "小区门口，一人沿人行道正常走过，垃圾桶旁无人停留",
        "violation": None,
        "people": [
            {"where": "画面中央", "appearance": "浅色上衣、背包",
             "action": "沿人行道由左向右正常走过，未在垃圾桶旁停留",
             "changed": "位置持续右移"}
        ],
        "observable": "人由左向右匀速走过，手中物品一直提着，没有抛掷或丢弃动作",
    },
    "littering": {
        "label": "乱扔垃圾（扔在桶外）",
        "scene": "路边有一个绿色垃圾桶，一人拿着纸杯走近桶边",
        "violation": "littering",
        "people": [
            {"where": "画面中央、垃圾桶左侧", "appearance": "深色外套男性",
             "action": "把手中纸杯朝桶口方向抛出，纸杯落在垃圾桶外侧地面，随后转身离开",
             "changed": "由站立转为抛掷动作，抛掷后转身向画面右侧离开"}
        ],
        "observable": ("连续帧可见同一纸杯由手中脱离、向下落至桶外地面，"
                       "后续帧纸杯仍留地面，人物离开"),
    },
    "smoking": {
        "label": "吸烟",
        "scene": "建筑工地围挡旁，一人低头站立",
        "violation": "smoking",
        "people": [
            {"where": "画面左侧近处", "appearance": "蓝色工装、戴安全帽",
             "action": "手持可辨认的烟支送至嘴边吸入，随后有明显烟雾呼出",
             "changed": "右手由身侧抬至嘴边，口鼻前出现扩散的烟雾"}
        ],
        "observable": "可辨认的烟支送至嘴唇并吸入，随后口鼻前有烟雾扩散",
    },
    "queue_jumping": {
        "label": "插队",
        "scene": "窗口前排着一列队伍，一人从队伍侧面靠近",
        "violation": "queue_jumping",
        "people": [
            {"where": "画面右侧、队伍中段", "appearance": "黑色短袖男性",
             "action": "从队伍右侧走到队列中段并插入，后排人员向后退让",
             "changed": "位置由队伍外侧移动到队列内部，队列整体后移"}
        ],
        "observable": "人物从队伍外侧直接进入队列中段，未在队尾排队，后方人员被迫后退",
    },
}


class MockBackend:
    """确定性假模型：不联网、不花钱，用来验证"链路"而不是"模型"。

    真值约定（只为让端到端自检可复现，不依赖真视频内容）：
        文件名里含 `_pos_` → 视为确实存在该违规
        文件名里含 `_neg_` → 视为不存在
        命中哪个行为，由文件名里出现的行为 id 决定：
        `smoking_pos.mp4` → 只有 smoking 命中，`littering` 仍为负。

    故意注入噪声（noise 参数）：这样时序平滑逻辑才有东西可压。
    如果 mock 永远返回完美答案，aggregate 模块就永远测不出问题 ——
    一个测不出问题的自检，等于没有自检。
    """

    name = "mock"

    def __init__(self, noise: float = 0.0, seed: int = 7, latency: float = 0.02,
                 scene: str | None = None):
        """
        noise 默认 0.0 —— 这一点是踩过之后才定的。

        ⚠️ 第一版默认 0.2，想的是"有噪声才能测出时序平滑的价值"。
        结果是：16 个窗口 × 3 个行为 = 48 次判断，20% 均匀噪声必然产生
        十几个误报，端到端指标掉到精确率 18%。**看起来像模型很差，
        其实是 mock 在乱答。** 一个"故意答错"的假模型对自检毫无价值 ——
        自检要验证的是管道通不通、聚合逻辑对不对，不是抗噪能力。

        所以默认干净（指标应接近 100%，说明管道正确）；
        要专门观察时序平滑的抗噪效果时，再用 --noise 打开，
        并且应该知道那是**压力测试**，指标低是预期行为。
        """
        self.noise = noise
        self.seed = seed
        self.latency = latency
        # 演示场景（见 MOCK_SCENES）：给了就按剧本产出**自洽**的
        # 场景描述 / 人物动作 / 判定依据 / 告警，让"演示"这件事成立。
        self.scene = scene if scene in MOCK_SCENES else None

    def _truth(self, req: WindowRequest, bs: BehaviorSet) -> set[str]:
        """从文件名推断"真值"：**只有文件名点名的那个行为**为真。

        命名约定：`{行为id}_{pos|neg}_任意后缀.mp4`
            littering_pos_vis.mp4  → 只有 littering 为真
            motion_neg_vis.mp4     → 全部为假

        ⚠️ 第一版把所有含 `_pos_` 的视频都判成"**所有**行为都真"，
        而标注只标了其中一种 —— 结果剩下两种行为全被算成误报，
        精确率掉到 8%，自检彻底失去意义（看起来像模型很差，其实是 mock 的错）。
        必须精确到"哪个行为"，这个约定才有价值。
        """
        stem = Path(req.video).stem.lower()
        truth: set[str] = set()
        for b in bs.enabled:
            bid = b.id.lower()
            if f"{bid}_pos" in stem:
                truth.add(b.id)
            elif f"{bid}_neg" in stem:
                pass                       # 明确为负，不加
        if truth:
            return truth
        # 文件名没点名任何行为：退回旧的粗约定，并按行为 id 是否出现在
        # 文件名里来判断，避免"一个 _pos_ 让所有行为都变真"
        if "_neg_" in stem:
            return set()
        if "_pos_" in stem:
            hits = {b.id for b in bs.enabled if b.id.lower() in stem}
            return hits or ({bs.enabled[0].id} if bs.enabled else set())
        return set()

    def infer(self, req: WindowRequest, bs: BehaviorSet) -> WindowResult:
        rng = random.Random(f"{self.seed}:{req.window_id}")
        t0 = time.time()
        time.sleep(self.latency)

        script = MOCK_SCENES.get(self.scene) if self.scene else None
        if script:
            # 剧本模式：真相就是剧本点名的那一个行为（可能为 None = 无违规）
            truth = {script["violation"]} if script.get("violation") else set()
        else:
            truth = self._truth(req, bs)
            if req.with_risks and "fall_pos" in Path(req.video).stem.lower():
                truth = set()  # 开放风险演示不应同时虚构一个配置行为违规

        # 无图请求只用于离线 mock 契约测试，显式按四帧虚构剧本演示。
        # 一旦带了真实帧列表，就必须尊重其长度，不能制造越界证据。
        evidence_frame_count = len(req.frames) if req.frames else 4
        violations = []
        for b in bs.enabled:
            hit = b.id in truth
            if rng.random() < self.noise:
                hit = not hit  # 注入误报 / 漏报
            if b.evidence_sequence and evidence_frame_count < 2:
                hit = False
            evidence = {}
            if hit:
                evidence = {key: 1 for key in b.evidence_checks}
                for index, key in enumerate(b.evidence_sequence):
                    # 相邻步骤允许共用同帧，但首尾必须跨帧且整体不能倒序。
                    evidence[key] = 1 + index * (evidence_frame_count - 1) // (
                        len(b.evidence_sequence) - 1)
            conf = round(rng.uniform(0.55, 0.95) if hit else rng.uniform(0.0, 0.3), 2)
            # 剧本模式下 observable 用剧本里那段"看得见的事实"，
            # 这样页面上的"判定依据"和场景描述是对得上的
            obs = (f"[演示] {script['observable']}" if (script and hit)
                   else (f"[演示] 画面中未见「{b.name}」对应的动作" if script
                         else f"[mock] {req.window_id} {b.id}={'有' if hit else '无'}"))
            violations.append(
                {
                    "behavior_id": b.id,
                    "violation": hit,
                    "confidence": conf,
                    "observable": obs,
                    "evidence": evidence,
                    "person_desc": (script["people"][0]["appearance"]
                                    if (script and hit and script.get("people"))
                                    else ("mock 人物" if hit else "")),
                    "bbox": [0.1, 0.3, 0.4, 0.9] if hit else None,
                    # 模拟一个"配好了告警判断"的模型：
                    #   violation=true 才可能 alert=true；false 时明确说不告警。
                    # ⚠️ 这个字段不是可有可无 —— aggregate.mode 默认为 ai，
                    #    如果 mock 不产出它，自检时告警会全部退化成"回落 violation"，
                    #    等于没测到 ai 这条路径（实测告警级一度掉到 0%）。
                    "alert": bool(hit),
                    "alert_reason": (
                        (f"[演示] 动作链条完整（{b.name}），建议立即提醒" if hit
                         else "[演示] 无需提醒") if script else
                        (f"[mock] 动作链条完整，可以提醒" if hit else "[mock] 无需提醒")
                    ),
                }
            )

        obj = {"scene": (f"[演示] {script['scene']} @ t={req.t_start:.0f}s" if script
                         else f"[mock] {Path(req.video).name} @ {req.t_start:.1f}s"),
               "violations": violations}
        if req.with_risks:
            # mock 默认不凭随机动作制造风险，显式场景才产生演示风险。
            obj["risks"] = []
            stem = Path(req.video).stem.lower()
            if "fall_pos" in stem:
                obj["risks"] = [{"risk_id": "fall", "name": "人员摔倒", "severity": "high",
                                 "confidence": 0.93, "observable": "[演示] 中央人物由站立倒地，后续帧未起身",
                                 "alert": True, "alert_reason": "[演示] 持续倒地，需要现场确认",
                                 "person_desc": "画面中央"}]

        # 模拟"动作描述"（对应网页的摄像头监测页）。
        # 只有请求了 with_actions 才给 —— 与真实模型的行为一致，
        # 这样本地就能把监测页的渲染链路完整跑通。
        if req.with_actions:
            if script and script.get("people"):
                obj["people"] = [
                    {"where": p["where"], "appearance": p["appearance"],
                     "action": p["action"], "changed": p["changed"]}
                    for p in script["people"]
                ]
            elif script:
                obj["people"] = []
            else:
                n = rng.randint(0, 2)
                places = ["画面左侧近处", "画面中央", "画面右侧远处"]
                acts = ["由站立变为蹲下", "抬右手至嘴边", "向右走出画面",
                        "静止站立", "转身朝门口"]
                obj["people"] = [
                    {
                        "where": places[i % len(places)],
                        "appearance": "深色外套" if i % 2 == 0 else "",
                        "action": rng.choice(acts),
                        "changed": rng.choice(["无", "由站立变为蹲下", "由静止开始行走"]),
                    }
                    for i in range(n)
                ]

        raw = json.dumps(obj, ensure_ascii=False)
        return WindowResult(
            window_id=req.window_id,
            video=req.video,
            t_start=req.t_start,
            t_end=req.t_end,
            parsed=obj,
            raw=raw,
            latency_sec=time.time() - t0,
            backend=self.name,
        )


# ───────────────────────────── Cloud ─────────────────────────────


class CloudBackend:
    """OpenAI 兼容云端 API。

    为何选 OpenAI 兼容协议而不是各家 SDK：DashScope、硅基流动、OpenRouter、
    自建 vLLM 都提供 /chat/completions，写一次就能整体换供应商，
    这在第一阶段"比价 + 比准确率"时非常关键。
    """

    name = "cloud"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        timeout: int = 120,
        max_retries: int = 3,
        price_in: float = 0.0,
        price_out: float = 0.0,
        stitch: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.price_in = price_in
        self.price_out = price_out
        # 是否把多帧拼成一张 2×2 网格图。默认关闭（保细节），
        # 端点不支持多图或成本吃紧时打开（见 stitch_grid 的说明）。
        self.stitch = stitch
        self.tokens_in_total = 0
        self.tokens_out_total = 0
        self.calls = 0

    def infer(self, req: WindowRequest, bs: BehaviorSet) -> WindowResult:
        from .prompts import build_messages, extract_json, sanitize, validate

        if self.stitch and len(req.frames) > 1:
            grid = stitch_grid(req.frames, Path(req.frames[0]).parent / f"{req.window_id}__grid.jpg")
            images = [to_data_url(grid)]
            # 让格子的 #序号 与时间戳对上，模型才描述得准"第几帧手在做什么"
            labels = [
                f"上图是含 {len(req.frames)} 帧的网格，按从左到右、从上到下时序排列："
                + " ".join(f"#{i + 1}={req.labels[i] if i < len(req.labels) else f'帧{i + 1}'}"
                           for i in range(len(req.frames)))
            ]
        else:
            images = [to_data_url(f) for f in req.frames]
            labels = req.labels
        messages = build_messages(bs, images, labels, with_alert=req.with_alert,
                                  with_actions=req.with_actions, with_risks=req.with_risks)
        if self.stitch and len(req.frames) > 1:
            messages[0]["content"] += (
                "\n本次多帧拼在一张网格图内：evidence 必须引用各格左上角的原帧编号"
                f" #1 至 #{len(req.frames)}，不是整张拼图的图片编号；每格是一帧，空格不是帧。"
            )

        t0 = time.time()
        last_err: str | None = None
        retries = 0
        tok_in = tok_out = 0

        for attempt in range(self.max_retries):
            try:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": float(bs.defaults.get("temperature", 0.0)),
                    # 保留原配置预算，按新增输出分别补余量，避免风险 JSON 被截断。
                    "max_tokens": (int(bs.defaults.get("max_new_tokens", 512))
                                   + (180 if req.with_alert else 0)
                                   + (90 if req.with_actions else 0)
                                   + (384 if req.with_risks else 0)),
                    "response_format": {"type": "json_object"},
                }
                if attempt > 0:
                    # 重试时把校验问题回灌给模型，比单纯重发有效得多
                    payload["messages"] = messages + [
                        {"role": "assistant", "content": "(上次输出不合规)"},
                        {
                            "role": "user",
                            "content": f"上次输出未通过校验：{last_err}。"
                            "请严格按 Schema 重新输出完整 JSON，不要任何额外文字。",
                        },
                    ]

                data = self._post(payload)
                usage = data.get("usage") or {}
                tok_in += int(usage.get("prompt_tokens", 0) or 0)
                tok_out += int(usage.get("completion_tokens", 0) or 0)
                self.calls += 1
                self.tokens_in_total += int(usage.get("prompt_tokens", 0) or 0)
                self.tokens_out_total += int(usage.get("completion_tokens", 0) or 0)

                content = data["choices"][0]["message"]["content"]
                if isinstance(content, list):  # 少数端点返回分段内容
                    content = "".join(
                        p.get("text", "") for p in content if isinstance(p, dict)
                    )

                obj = extract_json(content)
                if obj is None:
                    last_err = "输出里找不到合法 JSON"
                    retries += 1
                    continue

                ok, problems = validate(obj, bs, strict=False, with_risks=req.with_risks)
                if not ok:
                    last_err = "; ".join(problems)
                    retries += 1
                    continue

                return WindowResult(
                    window_id=req.window_id,
                    video=req.video,
                    t_start=req.t_start,
                    t_end=req.t_end,
                    parsed=sanitize(obj, bs, frame_count=len(req.frames)),
                    raw=content,
                    retries=retries,
                    latency_sec=time.time() - t0,
                    tokens_in=tok_in,
                    tokens_out=tok_out,
                    backend=self.name,
                )
            except Exception as e:  # 网络类错误：退避重试
                last_err = f"{type(e).__name__}: {e}"
                retries += 1
                time.sleep(min(2 ** attempt, 8))

        return WindowResult(
            window_id=req.window_id,
            video=req.video,
            t_start=req.t_start,
            t_end=req.t_end,
            parsed=None,
            error=last_err,
            retries=retries,
            latency_sec=time.time() - t0,
            tokens_in=tok_in,
            tokens_out=tok_out,
            backend=self.name,
        )

    def _post(self, payload: dict) -> dict:
        import requests

        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        if r.status_code == 429:
            raise RuntimeError(f"429 限流: {r.text[:200]}")
        r.raise_for_status()
        return r.json()

    def cost_cny(self) -> float:
        """按 token 单价估算累计花费（元）。单价以控制台实际账单为准。"""
        return (
            self.tokens_in_total / 1e6 * self.price_in
            + self.tokens_out_total / 1e6 * self.price_out
        )


# ──────────────────────────── Cluster ────────────────────────────


class ClusterBackend:
    """同济 A800 后端：读 sbatch 作业产出的结果，而不是在线调用。

    为什么不做成在线服务：登录节点禁止跑 GPU 计算（会被监控杀），
    计算只能走 sbatch；而计算节点没有公网出口，无法被 PC 直接连。
    第一阶段用"PC 抽帧 → 上传 → sbatch 批推理 → 拉回结果"最稳，
    且成本极低（1 卡 3 小时约 12 元），延迟不敏感的场景完全够用。
    """

    name = "cluster"

    def __init__(self, results_jsonl: str | Path):
        self.results_jsonl = Path(results_jsonl)
        self._cache: dict[str, dict] = {}
        if self.results_jsonl.exists():
            with open(self.results_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # 作业可能正在写最后一行，容忍半行
                    self._cache[rec["window_id"]] = rec

    def infer(self, req: WindowRequest, bs: BehaviorSet) -> WindowResult:
        from .prompts import sanitize

        rec = self._cache.get(req.window_id)
        if rec is None:
            return WindowResult(
                window_id=req.window_id,
                video=req.video,
                t_start=req.t_start,
                t_end=req.t_end,
                parsed=None,
                error=f"集群结果里没有 {req.window_id}（作业未跑完或该窗口失败）",
                backend=self.name,
            )
        return WindowResult(
            window_id=req.window_id,
            video=req.video,
            t_start=req.t_start,
            t_end=req.t_end,
            parsed=(sanitize(rec["parsed"], bs, frame_count=len(req.frames))
                    if isinstance(rec.get("parsed"), dict) else None),
            raw=rec.get("raw", ""),
            error=rec.get("error"),
            latency_sec=float(rec.get("latency_sec", 0.0) or 0.0),
            backend=self.name,
        )


def build_backend(cfg: dict, *, cluster_results: str | Path | None = None) -> Backend:
    """按 settings.yaml 的 backend 字段造后端。"""
    from .config import get_api_key

    kind = (cfg.get("backend") or "mock").lower()

    if kind == "mock":
        m = cfg.get("mock") or {}
        return MockBackend(
            noise=float(m.get("noise", 0.0)),
            seed=int(m.get("seed", 7)),
        )

    if kind == "cloud":
        c = cfg.get("cloud") or {}
        key = get_api_key(cfg)
        if not key:
            raise RuntimeError(
                f"缺少 API Key。请设置环境变量 {c.get('api_key_env')}，"
                "或改回 backend: mock 先跑通链路。"
            )
        return CloudBackend(
            base_url=c.get("base_url", ""),
            model=c.get("model", ""),
            api_key=key,
            timeout=int(c.get("timeout_sec", 120)),
            max_retries=int(c.get("max_retries", 3)),
            price_in=float(c.get("price_in_per_mtok", 0.0)),
            price_out=float(c.get("price_out_per_mtok", 0.0)),
            stitch=bool(c.get("stitch_frames", False)),
        )

    if kind == "cluster":
        if not cluster_results:
            cl = cfg.get("cluster") or {}
            cluster_results = os.path.join(
                cfg["paths"]["results_dir"], "cluster", "results.jsonl"
            )
        return ClusterBackend(cluster_results)

    raise ValueError(f"未知 backend: {kind}")

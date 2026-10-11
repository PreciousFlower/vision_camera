"""时序平滑与告警判定 —— 不要直接把单帧结论当告警。

为什么必须有这一层（这是线上能不能用的分水岭）：
    VLM 对单帧的判断有固有抖动。实测中"某一帧误报"的比例远高于
    "连续多帧都误报"。直接推告警会导致告警疲劳，用户三天就关掉系统。

两层过滤：
    1. 置信度阈值   —— 低于 min_confidence 的结论直接丢
    2. M/N 滑动窗口 —— 连续 N 个窗口里命中 M 次才算告警
       用 M/N 而不是"连续 N 次"：真实行为可能被中间一帧的遮挡打断，
       严格要求连续会把真事件滤掉。

另外提供 stability 指标：同一窗口重复推理 N 次的结果一致率。
它是 Prompt 质量的直接体检项 —— 一致性低于 ~85% 说明 Prompt 有歧义，
此时调阈值、加规则都是治标，必须回去改 Prompt。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import BehaviorSet
from .vlm import WindowResult


@dataclass
class Alert:
    """一条经过判定策略确认的告警。"""

    behavior_id: str
    behavior_name: str
    severity: str
    video: str
    t_start: float
    t_end: float
    hits: int
    span: int
    max_confidence: float
    evidences: list[str] = field(default_factory=list)
    person_desc: str = ""
    window_ids: list[str] = field(default_factory=list)
    # 模型自己给出的告警理由（仅 aggregate.mode: ai 时有值）。
    # 这是"让 AI 判断"最大的附赠价值：人能看到它为什么现在就要提醒。
    reason: str = ""
    # 配置行为与模型主动发现的开放风险分开标识，兼容原有构造调用。
    source: str = "configured"

    def to_dict(self) -> dict:
        return {
            "behavior_id": self.behavior_id,
            "behavior_name": self.behavior_name,
            "severity": self.severity,
            "video": Path(self.video).name,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "hits": self.hits,
            "span": self.span,
            "max_confidence": round(self.max_confidence, 3),
            "person_desc": self.person_desc,
            "evidences": self.evidences,
            "window_ids": self.window_ids,
            "reason": self.reason,
            "source": self.source,
        }


def _conf(rec: dict, min_confidence: float) -> float:
    """取某窗口对某行为的有效置信度；不构成违规或低于阈值都返回 0。"""
    if not rec.get("violation"):
        return 0.0
    try:
        c = float(rec.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return c if c >= min_confidence else 0.0


def derive_events(
    results: list[WindowResult],
    bs: BehaviorSet,
    agg_cfg: dict,
) -> list[Alert]:
    """把逐窗口的模型结论聚合成告警列表。

    三种策略（agg_cfg['mode']）：

      immediate —— 模型判定命中即告警。延迟最低，但单帧抖动直接变误报。
      ai        —— **让模型自己判断要不要告警**（Prompt 里多要求 alert /
                   alert_reason 两个字段），理由一并带回给人看。
      smoothed  —— 时序平滑：连续 N 窗内命中 M 次才告警。误报最少但延迟最高。

    ⚠️ ai 模式的回退策略很重要：模型没给 `alert` 字段时（例如换了模型、
    或 Prompt 没带那段指令），**回落到 violation 判定**而不是当成"不告警"。
    否则一旦字段缺失，告警会静默变成 0 —— 这正是之前踩过的坑
    （sanitize 契约不一致导致告警恒为 0，界面上却看不出来）。
    """
    from .prompts import sanitize

    mode = str(agg_cfg.get("mode") or "smoothed").lower()
    if mode not in ("immediate", "ai", "smoothed"):
        mode = "smoothed"

    min_conf = float(agg_cfg.get("min_confidence", 0.45))
    window_k = max(1, int(agg_cfg.get("window_k", 4)))
    min_hits = max(1, int(agg_cfg.get("min_hits", 2)))
    dedup_sec = float(agg_cfg.get("dedup_sec", 30))

    by_video: dict[str, list[WindowResult]] = {}
    for r in results:
        by_video.setdefault(r.video, []).append(r)

    alerts: list[Alert] = []

    for video, rs in by_video.items():
        rs = sorted(rs, key=lambda x: x.t_start)
        clean = [(r, sanitize(r.parsed, bs) if r.parsed and not r.error else None) for r in rs]

        for b in bs.enabled:
            series: list[tuple[WindowResult, float, dict]] = []
            for r, s in clean:
                if not s:
                    series.append((r, 0.0, {}))
                    continue
                series.append((r, _conf(s["violations"][b.id], min_conf), s["violations"][b.id]))

            last_alert_t = -1e18

            # ── 判定单元：给定命中列表产出一条告警 ──
            # ⚠️ span 必须由调用方显式传入，不能用 series.index(...) 反查 ——
            #    两个元素的元组若内容相同，index() 会返回第一个匹配位置，
            #    算出来的 span 就是错的（而且是静默错，界面看不出来）。
            def emit(hits, t_start, t_end, span: int, reason: str = "") -> None:
                nonlocal last_alert_t
                if not hits:
                    return
                if last_alert_t > -1e17 and t_start - last_alert_t < dedup_sec:
                    return   # 去抖：同一件事短时间内只说一遍
                evidences = [
                    h[2].get("observable", "") for h in hits if h[2].get("observable")
                ]
                alerts.append(
                    Alert(
                        behavior_id=b.id,
                        behavior_name=b.name,
                        severity=b.severity,
                        video=video,
                        t_start=t_start,
                        t_end=t_end,
                        hits=len(hits),
                        span=max(1, span),
                        max_confidence=max(h[1] for h in hits),
                        evidences=evidences[:3],
                        person_desc=next(
                            (h[2].get("person_desc", "") for h in hits
                             if h[2].get("person_desc")), ""
                        ),
                        window_ids=[h[0].window_id for h in hits],
                        reason=reason,
                    )
                )
                last_alert_t = t_start

            if mode in ("immediate", "ai"):
                for r, conf, item in series:
                    if conf <= 0:
                        continue
                    if mode == "ai":
                        # 模型明确表态时听模型的；没表态（None）才回落到 violation。
                        # 这样即使 Prompt 没带告警指令，系统也不会静默不报警。
                        if item.get("alert") is False:
                            continue
                        reason = item.get("alert_reason", "") or ""
                    else:
                        reason = ""
                    emit([(r, conf, item)], r.t_start, r.t_end, 1, reason)
            else:
                # 滑动窗口 M/N：窗口右端逐格前移，统计窗口内命中数
                for i in range(len(series)):
                    if series[i][1] <= 0:
                        continue  # 当前窗已未命中，不把历史命中再包装成新告警
                    lo = max(0, i - window_k + 1)
                    span_slice = series[lo : i + 1]
                    hits = [x for x in span_slice if x[1] > 0]
                    if len(hits) < min_hits:
                        continue
                    emit(hits, span_slice[0][0].t_start, span_slice[-1][0].t_end,
                         len(span_slice))

        if agg_cfg.get("open_risks", True) is True:
            alerts.extend(_derive_risk_events(clean, video, mode, min_conf,
                                              window_k, min_hits, dedup_sec))

    alerts.sort(key=lambda a: (Path(a.video).name, a.t_start))
    return alerts


def _derive_risk_events(clean, video: str, mode: str, min_conf: float,
                        window_k: int, min_hits: int, dedup_sec: float) -> list[Alert]:
    """开放风险按类别聚合；始终尊重 AI 的 alert=false，升级可突破去重。"""
    severity_rank = {"low": 0, "medium": 1, "high": 2}
    ids = sorted({risk["risk_id"] for _, s in clean if s for risk in s["risks"]})
    alerts: list[Alert] = []
    for risk_id in ids:
        series = []
        for result, parsed in clean:
            # 缺失风险、跳过窗口、失败窗口均占一个真实时间窗口，命中记为零。
            item = next((x for x in parsed["risks"] if x["risk_id"] == risk_id), {}) if parsed else {}
            confidence = item.get("confidence", 0.0)
            if item.get("alert") is not True or confidence < min_conf:
                confidence = 0.0
            series.append((result, confidence, item))

        last_alert_t = -1e18
        last_severity = -1
        for index, current in enumerate(series):
            if current[1] <= 0:
                continue
            if mode == "smoothed":
                lo = max(0, index - window_k + 1)
                span_slice = series[lo:index + 1]
                hits = [x for x in span_slice if x[1] > 0]
                if len(hits) < min_hits:
                    continue
                span = len(span_slice)
            else:
                hits = [current]
                span = 1

            # 当前风险升级时保留高等级证据和名称；旧的低等级提醒不能压住它。
            representative = max(hits, key=lambda x: (severity_rank[x[2]["severity"]],
                                                       x[0].t_start, x[1]))[2]
            severity = severity_rank[representative["severity"]]
            trigger_t = current[0].t_start
            within_dedup = trigger_t - last_alert_t < dedup_sec
            if within_dedup and severity <= last_severity:
                continue
            alerts.append(Alert(
                behavior_id="ai_risk:" + risk_id,
                behavior_name=representative["name"],
                severity=representative["severity"],
                video=video,
                t_start=hits[0][0].t_start,
                t_end=current[0].t_end,
                hits=len(hits),
                span=span,
                max_confidence=max(x[1] for x in hits),
                evidences=list(dict.fromkeys(x[2]["observable"] for x in hits))[:3],
                person_desc=representative["person_desc"],
                window_ids=[x[0].window_id for x in hits],
                reason=representative["alert_reason"],
                source="ai_risk",
            ))
            last_alert_t = trigger_t
            last_severity = severity
    return alerts


def stability_report(
    repeated: dict[str, list[dict]],
    bs: BehaviorSet,
) -> dict:
    """一致性体检：同一窗口重复推理多次，结论一致的比例。

    用法见 cli.py 的 `stability` 子命令。
    经验判据：一致率 <85% ⇒ Prompt 有歧义，先改 Prompt，别急着调阈值。
    """
    total = 0
    agree = 0
    per_behavior: dict[str, list[int]] = {b.id: [0, 0] for b in bs.enabled}

    for wid, objs in repeated.items():
        if len(objs) < 2:
            continue
        total += 1
        votes = []
        for obj in objs:
            raw = obj.get("violations") or []
            # Cloud/Cluster 后端返回归一化后的映射；Mock/旧结果仍可为列表。
            # 这里统计模型结论，不再重复应用告警阈值或证据过滤。
            if isinstance(raw, dict):
                v = {bid: bool(item.get("violation"))
                     for bid, item in raw.items() if bid and isinstance(item, dict)}
            else:
                v = {x["behavior_id"]: bool(x.get("violation"))
                     for x in raw if isinstance(x, dict) and x.get("behavior_id")}
            votes.append(v)
            for bid, val in v.items():
                if bid in per_behavior:
                    per_behavior[bid][1] += 1
                    if val:
                        per_behavior[bid][0] += 1

        keys = set().union(*[set(v) for v in votes]) if votes else set()
        stable_all = all(len({v.get(k, False) for v in votes}) == 1 for k in keys)
        if stable_all:
            agree += 1

    return {
        "windows_checked": total,
        "fully_stable": agree,
        "overall_agreement": round(agree / total, 4) if total else None,
        "positive_rate_per_behavior": {
            k: round(v[0] / v[1], 4) if v[1] else None for k, v in per_behavior.items()
        },
        "verdict": (
            "一致率不足，Prompt 存在歧义，优先改 Prompt"
            if total and agree / total < 0.85
            else "一致性可接受"
        ),
    }


def write_alerts(alerts: list[Alert], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for a in alerts:
            f.write(json.dumps(a.to_dict(), ensure_ascii=False) + "\n")
    return path

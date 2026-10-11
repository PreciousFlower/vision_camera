"""评测 —— 对着标注算准确率，而不是"看着像挺准的"。

第一阶段唯一真正的交付物就是**可信的数字**：哪些行为识别得准、
哪些不行、误报和漏报各占多少。没有数字，后面加不改 Prompt、
上不上微调、买不买显卡，全都只能凭感觉决定。

两个层次分开评（这点很重要）：
    window 级：逐窗口对错，反映「模型 + Prompt」本身的能力
    event  级：经时序平滑后的告警，反映「整个系统」的表现

只看 event 级会掩盖模型问题（平滑会把错判抹平），
只看 window 级又无法回答"上线后会不会天天误报"。两个都要。

标注文件格式（annotations.jsonl，UTF-8，每行一条）：
    {"video": "litter_pos.mp4", "behavior_id": "littering",
     "start": 2.0, "end": 10.0}
未标注的行为一律视为该视频中**不存在**（即负例），这与 behaviors.yaml
里"证据不足即判 false"的纪律一致。
开放风险是例外：只有某视频明确标注了 `ai_risk:<类别>`，才评该视频的该类风险；
其他开放风险告警单独记为未评分，不能用配置行为标注推断风险准确率。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .aggregate import Alert
from .config import BehaviorSet
from .vlm import WindowResult


@dataclass
class Label:
    video: str
    behavior_id: str
    start: float
    end: float

    def overlaps(self, t0: float, t1: float) -> bool:
        return not (t1 < self.start or t0 > self.end)


def load_annotations(path: str | Path) -> list[Label]:
    path = Path(path)
    out: list[Label] = []
    if not path.exists():
        raise FileNotFoundError(
            f"标注文件不存在: {path}\n"
            "第一阶段的标注不用多，每个行为 5~10 个片段即可，"
            "但必须有正例和负例，否则算不出误报率。"
        )
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith(("#", "//")):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i} JSON 解析失败: {e}") from e
            out.append(
                Label(
                    video=str(d["video"]),
                    behavior_id=str(d["behavior_id"]),
                    start=float(d.get("start", 0.0)),
                    end=float(d.get("end", 1e9)),
                )
            )
    return out


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and (precision + recall) > 0
        else None
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
    }


# ───────────────────────── window 级 ─────────────────────────


def score_windows(
    results: list[WindowResult],
    labels: list[Label],
    bs: BehaviorSet,
    *,
    min_confidence: float = 0.45,
) -> dict:
    """逐窗口评分：模型说"这一窗有违规" vs 标注是否覆盖该窗。"""
    from .prompts import sanitize

    per: dict[str, dict[str, int]] = {b.id: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for b in bs.enabled}
    unusable = 0

    for r in results:
        if not r.parsed:
            unusable += 1
            continue
        s = sanitize(r.parsed, bs)
        vname = Path(r.video).name

        for b in bs.enabled:
            v = s["violations"][b.id]
            predicted = bool(v["violation"]) and float(v["confidence"]) >= min_confidence
            expected = any(
                lb.video == vname and lb.behavior_id == b.id and lb.overlaps(r.t_start, r.t_end)
                for lb in labels
            )
            c = per[b.id]
            if predicted and expected:
                c["tp"] += 1
            elif predicted and not expected:
                c["fp"] += 1
            elif not predicted and expected:
                c["fn"] += 1
            else:
                c["tn"] += 1

    per_out = {}
    tp = fp = fn = tn = 0
    for bid, c in per.items():
        per_out[bid] = {
            **_prf(c["tp"], c["fp"], c["fn"]),
            "tn": c["tn"],
            "accuracy": round(
                (c["tp"] + c["tn"]) / max(1, c["tp"] + c["tn"] + c["fp"] + c["fn"]), 4
            ),
        }
        tp += c["tp"]; fp += c["fp"]; fn += c["fn"]; tn += c["tn"]

    return {
        "level": "window",
        "min_confidence": min_confidence,
        "windows_scored": len(results) - unusable,
        "windows_unparsed": unusable,
        "per_behavior": per_out,
        "overall": {
            **_prf(tp, fp, fn),
            "tn": tn,
            "accuracy": round((tp + tn) / max(1, tp + tn + fp + fn), 4),
        },
    }


# ───────────────────────── event 级 ─────────────────────────


def score_events(
    alerts: list[Alert],
    labels: list[Label],
    bs: BehaviorSet,
) -> dict:
    """告警级评分；开放风险仅在明确标注的视频与类别范围内评分。

    一个标注区间最多只能被一条告警命中（避免刷召回）。历史配置行为仍按
    未标注即负例评估；未知类别与没有对应风险标注的告警均单独记为未评分。
    """
    per: dict[str, dict[str, int]] = {b.id: {"tp": 0, "fp": 0, "fn": 0} for b in bs.enabled}
    configured_ids = set(per)
    risk_scopes = {(Path(lb.video).name, lb.behavior_id) for lb in labels
                   if re.fullmatch(r"ai_risk:[a-z][a-z0-9_]{0,63}", lb.behavior_id)}
    risk_ids = {bid for _, bid in risk_scopes}
    for bid in sorted(risk_ids):
        per.setdefault(bid, {"tp": 0, "fp": 0, "fn": 0})
    matched: set[int] = set()
    scored_total = open_risk_total = open_risk_scored_total = 0
    unscored: dict[str, int] = {}

    for a in alerts:
        vname = Path(a.video).name
        is_risk = a.behavior_id.startswith("ai_risk:") or a.source == "ai_risk"
        if is_risk:
            open_risk_total += 1
            can_score = (vname, a.behavior_id) in risk_scopes
        else:
            can_score = a.behavior_id in configured_ids
        if not can_score:
            unscored[a.behavior_id] = unscored.get(a.behavior_id, 0) + 1
            continue
        scored_total += 1
        open_risk_scored_total += int(is_risk)
        hit_idx = None
        for i, lb in enumerate(labels):
            if i in matched:
                continue
            if Path(lb.video).name == vname and lb.behavior_id == a.behavior_id and lb.overlaps(a.t_start, a.t_end):
                hit_idx = i
                break
        if hit_idx is None:
            per[a.behavior_id]["fp"] += 1
        else:
            matched.add(hit_idx)
            per[a.behavior_id]["tp"] += 1

    for i, lb in enumerate(labels):
        if i not in matched and lb.behavior_id in per:
            per[lb.behavior_id]["fn"] += 1

    per_out = {bid: _prf(c["tp"], c["fp"], c["fn"]) for bid, c in per.items()}
    tp = sum(c["tp"] for c in per.values())
    fp = sum(c["fp"] for c in per.values())
    fn = sum(c["fn"] for c in per.values())

    def summary(ids: set[str]) -> dict:
        return _prf(*(sum(per[bid][key] for bid in ids) for key in ("tp", "fp", "fn")))

    return {
        "level": "event",
        "alerts_total": len(alerts),
        "alerts_scored_total": scored_total,
        "unscored_alerts_total": len(alerts) - scored_total,
        "unscored_alerts_by_behavior": unscored,
        "open_risk_alerts_total": open_risk_total,
        "open_risk_alerts_scored_total": open_risk_scored_total,
        "open_risk_alerts_unscored_total": open_risk_total - open_risk_scored_total,
        "open_risk_annotation_scopes": len(risk_scopes),
        "per_behavior": per_out,
        "overall": _prf(tp, fp, fn),
        "configured_overall": summary(configured_ids),
        "open_risk_overall": summary(risk_ids),
    }


def render_report(win: dict, evt: dict, bs: BehaviorSet, stats: dict | None = None) -> str:
    """生成人类可读的评测报告（Markdown）。"""
    name_of = {b.id: b.name for b in bs.behaviors}
    lines: list[str] = ["# 第一阶段评测报告", ""]

    if stats:
        lines += [
            "## 运行概况",
            "",
            f"- 后端：`{stats.get('backend')}`",
            f"- 窗口总数：{stats.get('total_windows')}",
            f"- 本次执行：{stats.get('ran_now')}（跳过已完成 {stats.get('skipped_done')}）",
            f"- 失败窗口：{stats.get('errors')}",
            f"- 耗时：{stats.get('elapsed_sec')} s",
            "",
        ]

    def table(d: dict) -> list[str]:
        rows = ["| 行为 | 精确率 P | 召回率 R | F1 | TP | FP | FN |", "|---|---|---|---|---|---|---|"]
        for bid, m in d["per_behavior"].items():
            rows.append(
                f"| {name_of.get(bid, bid)} (`{bid}`) | {_fmt(m['precision'])} | "
                f"{_fmt(m['recall'])} | {_fmt(m['f1'])} | {m['tp']} | {m['fp']} | {m['fn']} |"
            )
        o = d["overall"]
        rows.append(
            f"| **合计** | **{_fmt(o['precision'])}** | **{_fmt(o['recall'])}** | "
            f"**{_fmt(o['f1'])}** | {o['tp']} | {o['fp']} | {o['fn']} |"
        )
        return rows

    lines += ["## 窗口级（配置行为的模型 + Prompt 表现）", ""]
    lines += table(win)
    lines += [
        "",
        f"未解析窗口：{win['windows_unparsed']} ／ 已评分窗口：{win['windows_scored']}",
        "",
        "## 告警级（经时序平滑后的系统表现）", "",
    ]
    lines += table(evt)
    scored_total = evt.get("alerts_scored_total", evt["alerts_total"])
    lines += ["", f"告警总数：{evt['alerts_total']} ／ 已评分：{scored_total}"
              f" ／ 未评分：{evt.get('unscored_alerts_total', 0)}", ""]
    if evt.get("open_risk_alerts_total", 0) or evt.get("open_risk_annotation_scopes", 0):
        lines += [
            f"开放风险告警：{evt.get('open_risk_alerts_total', 0)}"
            f" ／ 已评分：{evt.get('open_risk_alerts_scored_total', 0)}"
            f" ／ 未评分：{evt.get('open_risk_alerts_unscored_total', 0)}",
            "",
            "开放风险只在该视频有明确 `ai_risk:<类别>` 标注时参与评分。"
            "未标注风险告警不计为 TP 或 FP，也不参与精确率、召回率和 F1。"
            "配置行为的窗口级指标不代表开放风险识别能力。",
            "",
        ]
        if not evt.get("open_risk_annotation_scopes", 0):
            lines += ["**尚无开放风险标注，不能据此报告判断风险准确率。**", ""]

    # 给出可执行的判读，而不是只丢一张表
    lines += ["## 判读与下一步", ""]
    for bid, m in win["per_behavior"].items():
        nm = name_of.get(bid, bid)
        if m["fn"] > 0 and (m["recall"] or 0) < 0.7:
            lines.append(
                f"- **{nm}**：召回率偏低（{_fmt(m['recall'])}）。"
                "优先怀疑定义里的可观察线索不够具体，或窗口内帧数不足以体现动作链条。"
                "先改 `behaviors.yaml` 的 `evidence_required` 和 `observable_cues`。"
            )
        if m["fp"] > 0 and (m["precision"] or 1) < 0.7:
            lines.append(
                f"- **{nm}**：误报偏多（精确率 {_fmt(m['precision'])}）。"
                "去看被误报窗口的 `observable` 字段，把模型认错的那个场景写进 `counter_examples`。"
            )
    p = win["overall"]["precision"]
    if p is not None and p < 0.8:
        lines.append(
            "- 整体精确率不足 80%：**先不要调阈值**。阈值只能左右权衡 P/R，"
            "压不出真正的提升。回到 Prompt 加反例是收益最高的动作。"
        )
    lines.append("")
    lines.append(
        "> 提醒：配置行为的窗口级数字反映对应模型能力。比较同一配置行为时，"
        "如果窗口级很高而告警级很低，"
        "问题在平滑参数（`window_k` / `min_hits`），不在 Prompt。"
    )
    return "\n".join(lines)


def _fmt(v) -> str:
    return "—" if v is None else f"{v * 100:.1f}%"

"""Bounded, independent model review of incomplete visual evidence.

Frame references remain model claims, not pixel-level verification. A second
answer must pass the same checks; confidence or prose cannot bypass them.
"""
from __future__ import annotations

import copy
from typing import Callable

from .config import BehaviorSet, behavior_payload
from .prompts import sanitize


def generation_budget(bs: BehaviorSet, *, with_alert: bool, with_actions: bool,
                      with_risks: bool) -> int:
    floor = (160 + 40 * len(bs.enabled)
             + 16 * sum(len(b.evidence_checks) for b in bs.enabled)
             + (96 if with_alert else 0) + (160 if with_actions else 0))
    return max(int(bs.defaults.get("max_new_tokens", 512)), floor) + (384 if with_risks else 0)


def _usable(result: dict) -> bool:
    return (isinstance(result.get("parsed"), dict) and not result.get("error")
            and result.get("valid") is not False)


def _add_request_timings(result: dict, first: dict, second: dict) -> None:
    if first.get("roundtrip_sec") is not None or second.get("roundtrip_sec") is not None:
        result["roundtrip_sec"] = round(sum(float(r.get("roundtrip_sec") or 0)
                                           for r in (first, second)), 3)
    timings = [r.get("timing") or {} for r in (first, second)]
    if any(timings):
        result["timing"] = {key: round(sum(float(t.get(key) or 0) for t in timings), 3)
                            for key in ("queue_sec", "prepare_sec", "generate_sec", "decode_sec")}


def infer_with_review(send: Callable, frames: list, labels: list[str], bs: BehaviorSet,
                      *, max_new_tokens: int, with_alert: bool = True,
                      with_actions: bool = False, with_risks: bool = False,
                      review_enabled: bool = True) -> dict:
    """At most two calls: retry truncated JSON OR review incomplete evidence.

The independent second call gets the same images and a focused definition,
never the previous positive claim. Ordinary positives/negatives make one call.
"""
    kwargs = dict(behaviors=behavior_payload(bs), max_new_tokens=max_new_tokens,
                  few_shot=bool(bs.defaults.get("few_shot", True)),
                  with_alert=with_alert, with_actions=with_actions, with_risks=with_risks)
    first = send(frames, labels, **kwargs)
    if not review_enabled:
        return first
    if first.get("finish_reason") == "length" and not _usable(first):
        second = send(frames, labels, **{**kwargs, "max_new_tokens": max_new_tokens + 384})
        result = dict(second)
        result["generation_retry"] = {"reason": "length", "initial_raw": first.get("raw", ""),
                                      "initial_tokens_out": first.get("tokens_out", 0)}
        result["latency_sec"] = float(first.get("latency_sec") or 0) + float(second.get("latency_sec") or 0)
        _add_request_timings(result, first, second)
        return result
    if not _usable(first):
        return first
    if with_risks and "risks" not in first["parsed"]:
        return first  # Preserve the ingress error for an incompatible old worker.
    clean = sanitize(first["parsed"], bs, frame_count=len(frames))
    targets = [b for b in bs.enabled if clean["violations"][b.id].get("evidence_issues")]
    if not targets:
        return {**first, "parsed": clean}

    focused = BehaviorSet(copy.deepcopy(targets), dict(bs.defaults))
    for b in focused.behaviors:
        b.definition += ("\n本次是独立证据核验：重新按时间观察全部输入帧，逐项定位必要证据。"
                         "允许否定本行为；不可为凑齐检查项猜测帧号。若有完整动作链即可确认，"
                         "缺失、遮挡或符合反例则判false。")
    review_kwargs = {**kwargs, "behaviors": behavior_payload(focused), "with_risks": False,
                     "max_new_tokens": generation_budget(focused, with_alert=with_alert,
                                                         with_actions=with_actions, with_risks=False)}
    try:
        second = send(frames, labels, **review_kwargs)
    except Exception as exc:
        second = {"error": f"{type(exc).__name__}: {exc}"}
    result = {**first, "parsed": copy.deepcopy(clean)}
    statuses = {}
    if _usable(second):
        reviewed = sanitize(second["parsed"], focused, frame_count=len(frames))
        for b in targets:
            item = reviewed["violations"][b.id]
            if item.get("missing") or item.get("evidence_issues"):
                statuses[b.id] = "unresolved"
            else:
                result["parsed"]["violations"][b.id] = item
                statuses[b.id] = "confirmed" if item["violation"] else "rejected"
        # Refresh the visible action description from the independent observation.
        if all(status != "unresolved" for status in statuses.values()):
            result["parsed"]["scene"] = reviewed.get("scene", clean.get("scene", ""))
            if with_actions:
                result["parsed"]["people"] = reviewed.get("people", [])
        result["parsed"] = sanitize(result["parsed"], bs, frame_count=len(frames))
    else:
        statuses = {b.id: "unresolved" for b in targets}
    status = ("unresolved" if "unresolved" in statuses.values() else
              "confirmed" if "confirmed" in statuses.values() else "rejected")
    result["verification"] = {"status": status, "behaviors": statuses,
                              "raw": second.get("raw", ""), "error": second.get("error"),
                              "tokens_out": second.get("tokens_out", 0),
                              "latency_sec": second.get("latency_sec", 0)}
    result["latency_sec"] = float(first.get("latency_sec") or 0) + float(second.get("latency_sec") or 0)
    result["tokens_out"] = int(first.get("tokens_out") or 0) + int(second.get("tokens_out") or 0)
    _add_request_timings(result, first, second)
    return result

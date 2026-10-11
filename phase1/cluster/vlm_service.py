#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""计算节点常驻推理服务 —— 网页实时交互的后端。

架构位置：
    浏览器 → 本机 FastAPI(:8848) → SSH 两级隧道 → 本服务(计算节点:PORT)

为什么用纯标准库 http.server 而不是 FastAPI：
    集群的 conda 环境里**没有** fastapi/uvicorn/flask（已实测），
    而且计算节点上的包安装不应成为依赖。标准库零依赖、启动即用。
    代价是没有异步框架的花哨功能，但对"收帧 → 推理 → 回 JSON"足够。

为什么必须绑 0.0.0.0：
    实测：绑 127.0.0.1 时，登录节点经 SSH 通道连过来会被拒绝；
    绑 0.0.0.0 后可从登录节点直达（已验证 gpu8020:18080 连通）。
    ⚠️ 绑 0.0.0.0 意味着同网段其他节点理论上可达，但这是校园内网
    且作业结束即释放，风险可接受；如需更严可加 token 校验（见 --token）。

核心设计：
    · 模型只在启动时加载一次，之后常驻显存 —— 这正是"实时"的来源
    · 单请求单窗口推理，检测优先的有界队列凑批，避免无界积压与显存争抢
    · 复用 civility/prompts.py 生成 Prompt，与云端/批处理三条路径完全一致
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import io
import json
import os
import socket
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# ── 离线加载已下好的权重（计算节点无稳定公网，走本地缓存）──
MODEL_ROOT = os.environ.get("MODEL_ROOT", "/ssdfs/datahome/u00009/models")
os.environ.setdefault("HF_HOME", f"{MODEL_ROOT}/hf")
os.environ.setdefault("HF_HUB_CACHE", f"{MODEL_ROOT}/hf/hub")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

STATE: dict = {
    "ready": False,
    "loading": False,
    "error": None,
    "model": None,
    "processor": None,
    "model_name": "",
    "device": "",
    "started_at": time.time(),
    "requests": 0,
    "errors": 0,
    "last_latency": 0.0,
    "latencies": [],
    "lock": threading.Lock(),
    "behaviors": None,
    "behaviors_path": "",
    "token": None,
}


# ─────────────────────────── 行为定义加载 ───────────────────────────


def load_behaviors_from(path: str):
    """从 behaviors.json 还原 BehaviorSet（与批处理脚本同一套逻辑）。"""
    from civility.config import Behavior, BehaviorSet

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = [
        Behavior(
            id=b["id"],
            name=b.get("name", b["id"]),
            definition=b.get("definition", ""),
            evidence_required=b.get("evidence_required", ""),
            counter_examples=b.get("counter_examples") or [],
            observable_cues=b.get("observable_cues") or [],
            evidence_checks=b.get("evidence_checks", {}),
            evidence_sequence=b.get("evidence_sequence", []),
            severity=b.get("severity", "medium"),
            enabled=True,
        )
        for b in raw.get("behaviors", [])
        if b.get("id") and b.get("definition")
    ]
    return BehaviorSet(behaviors=items, defaults=raw.get("defaults") or {})


def reload_behaviors(path: str) -> int:
    """热更新行为定义 —— 用户在网页上改了定义，不用重启作业。

    ⚠️ 同时 reload 一次 civility.prompts 模块：仅仅改 behaviors.json 的话，
    进程里已导入的 prompts 模块仍是旧代码，改提示词模板不会生效。
    （踩过：网页上改了定义、效果没变，查了半天发现是模块缓存。）
    """
    import importlib

    from civility import prompts as prompts_mod

    importlib.reload(prompts_mod)

    bs = load_behaviors_from(path)
    STATE["behaviors"] = bs
    STATE["behaviors_path"] = path
    return len(bs.enabled)


# ─────────────────────────── 模型加载 ───────────────────────────


def load_model(model_name: str) -> None:
    STATE["loading"] = True
    t0 = time.time()
    try:
        import torch
        from transformers import AutoProcessor
        torch.set_num_threads(max(1, int(os.environ.get("VLM_CPU_THREADS", "2"))))

        try:
            from transformers import Qwen3VLForConditionalGeneration as Cls
        except ImportError:
            from transformers import AutoModelForVision2Seq as Cls  # type: ignore

        print(f"[load] {model_name} …", flush=True)
        model = Cls.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
            attn_implementation="sdpa",
        ).eval()
        proc = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        vision_cache = None
        inner = getattr(model, "model", None)
        if (os.environ.get("VLM_VISION_CACHE", "1") != "0" and
                callable(getattr(inner, "get_image_features", None))):
            from cluster.vision_cache import VisionCache
            vision_cache = VisionCache(inner.get_image_features)
            inner.get_image_features = vision_cache

        try:
            from qwen_vl_utils import process_vision_info  # noqa: F401

            STATE["has_qvu"] = True
        except ImportError:
            STATE["has_qvu"] = False

        STATE.update(
            model=model,
            processor=proc,
            model_name=model_name,
            device=str(next(model.parameters()).device),
            ready=True,
            loading=False,
            load_sec=round(time.time() - t0, 1),
            vision_cache=vision_cache,
        )
        print(f"[load] 完成，用时 {time.time() - t0:.1f}s  设备 {STATE['device']}", flush=True)
    except Exception as e:
        STATE.update(loading=False, error=f"{type(e).__name__}: {e}")
        print("[load] 失败:\n" + traceback.format_exc(), flush=True)


# ─────────────────────────── 单窗口推理 ───────────────────────────


def decode_images(frames: list) -> list:
    """frames 元素支持两种形式：
        {"b64": "..."}         —— 网页上传的 base64 JPEG
        {"path": "/abs/x.jpg"} —— 集群本地路径（调试用）
    """
    from PIL import Image

    out = []
    for f in frames:
        if not isinstance(f, dict):
            raise ValueError(f"帧格式不对：{type(f)}")
        if f.get("b64"):
            raw = base64.b64decode(f["b64"])
            out.append(Image.open(io.BytesIO(raw)).convert("RGB"))
        elif f.get("path"):
            out.append(Image.open(f["path"]).convert("RGB"))
        else:
            raise ValueError("帧里既没有 b64 也没有 path")
    return out


def infer_one(frames: list, labels: list[str], max_new_tokens: int, few_shot: bool,
              with_alert: bool = False, with_actions: bool = False,
              with_risks: bool = False, behaviors=None) -> dict:
    """单请求推理（保留给 batch=1 的路径与调试用）。"""
    options = {"behaviors": behaviors} if behaviors is not None else {}
    fut = BATCHER.submit(frames, labels, max_new_tokens, few_shot,
                         with_alert, with_actions, with_risks, **options)
    return wait_for_result(fut)


# ───────────────────── 动态批量推理（吞吐的关键）─────────────────────
#
# 为什么要做批量（实测数据，见 tools/run_cluster_job.py 的微基准）：
#     Qwen3-VL-4B 在 A800 上**单序列**解码只有约 30 token/s，
#     而 GPU 显存占用仅 8.3/80 GB —— 说明瓶颈是**每一步的固定开销**
#     （CPU 发射指令、kernel 启动、注意力掩码重算…），不是算力也不是带宽。
#     把 N 个窗口塞进同一次前向，就把这份固定开销摊到 N 个请求上：
#         批量 1：1.96 s/窗   28.5 tok/s
#         批量 2：1.11 s/窗   50.7 tok/s
#         批量 4：0.64 s/窗   87.1 tok/s
#         批量 8：0.49 s/窗  114.2 tok/s   ← 4 倍吞吐，只多花 1 倍时间
#     `torch.compile` 与 device_map 调整都**无效**（真实窗口 1.85s，与基线相同），
#     所以批量是这套栈上唯一有效的提速手段。
#
# ⚠️ 代价必须说清楚：批量意味着"等一凑齐再算"。
#    批量越大越省时间，但单个请求要等更久（等同伴到齐）。
#    设 VLM_BATCH_WAIT_MS=0 就退化成"来一个算一个"（老行为）。

BATCH_MAX = int(os.environ.get("VLM_BATCH_MAX", "8") or 8)
BATCH_WAIT_MS = int(os.environ.get("VLM_BATCH_WAIT_MS", "25") or 0)
BATCH_TIMEOUT_SEC = float(os.environ.get("VLM_BATCH_TIMEOUT", "600") or 600)
# Bound the retained image payloads; reject overload instead of extending lag.
BATCH_QUEUE_MAX = max(1, int(os.environ.get("VLM_QUEUE_MAX", "64") or 64))
BATCH_QA_QUEUE_MAX = max(1, int(os.environ.get("VLM_QA_QUEUE_MAX", "8") or 8))
# Detection normally wins. One aged QA request may run after this wait so a
# continuously active camera cannot permanently starve an accepted question.
BATCH_QA_MAX_WAIT_SEC = max(0.0, float(os.environ.get("VLM_QA_MAX_WAIT_SEC", "20") or 20))


class QueueFullError(RuntimeError):
    """The bounded inference queue cannot accept another image payload."""


def wait_for_result(future):
    """Cancel timed-out work only if model preparation has not started."""
    try:
        return future.result(timeout=BATCH_TIMEOUT_SEC)
    except concurrent.futures.TimeoutError:
        # Future.cancel() returns False once the scheduler has claimed it.
        # Generation that is already running must finish without interruption.
        future.cancel()
        raise


class _Item:
    __slots__ = ("frames", "labels", "max_new_tokens", "few_shot",
                 "with_alert", "with_actions", "with_risks", "future", "behaviors", "queued_at", "qa", "deadline")

    def __init__(self, frames, labels, n, few, alert, actions, future,
                 risks=False, behaviors=None, qa=None):
        self.frames, self.labels = frames, labels
        self.max_new_tokens, self.few_shot = n, few
        self.with_alert, self.with_actions = alert, actions
        self.with_risks = risks
        self.future = future
        self.behaviors = behaviors
        self.qa = qa
        self.queued_at = time.perf_counter()
        self.deadline = self.queued_at + BATCH_TIMEOUT_SEC


class Batcher:
    """收集并发请求 → 凑成一批 → 一次前向算完 → 各自返回。

    这是"没有 vLLM 时的替代品"：vLLM 在本集群装不了（glibc 2.28 太老），
    但它的核心收益（连续批处理）用几十行就能拿到一大部分。
    """

    def __init__(self, max_queue: int | None = None,
                 qa_max_wait_sec: float | None = None,
                 max_qa_queue: int | None = None) -> None:
        self.q: list[_Item] = []
        self.cv = threading.Condition()
        self.thread: threading.Thread | None = None
        self.max_queue = max(1, BATCH_QUEUE_MAX if max_queue is None else int(max_queue))
        self.max_qa_queue = min(self.max_queue, max(1, BATCH_QA_QUEUE_MAX
                               if max_qa_queue is None else int(max_qa_queue)))
        self.qa_max_wait_sec = max(0.0, BATCH_QA_MAX_WAIT_SEC if qa_max_wait_sec is None
                                   else float(qa_max_wait_sec))
        self._last_batch_qa = False
        self.stats = {"batches": 0, "items": 0, "max_batch": 0,
                      "cancelled": 0, "expired": 0, "rejected": 0,
                      "qa_wait_protected": 0}

    def start(self) -> None:
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self._loop, daemon=True,
                                       name="vlm-batcher")
        self.thread.start()
        print(f"[batch] 动态批量已启用：最多 {BATCH_MAX} 个/批，"
              f"凑批等待 {BATCH_WAIT_MS}ms，检测优先，排队上限 {self.max_queue}，"
              f"问答排队上限 {self.max_qa_queue}，问答等待保护 {self.qa_max_wait_sec:g}s", flush=True)

    def submit(self, frames, labels, max_new_tokens, few_shot, with_alert,
               with_actions, with_risks=False, behaviors=None, qa=None):
        fut: "concurrent.futures.Future[dict]" = concurrent.futures.Future()
        it = _Item(frames, labels, max_new_tokens, few_shot, with_alert,
                   with_actions, fut, with_risks, behaviors, qa)
        with self.cv:
            self._prune_queue(time.perf_counter())
            if (len(self.q) >= self.max_queue or
                    (qa is not None and sum(it.qa is not None for it in self.q) >= self.max_qa_queue)):
                self.stats["rejected"] += 1
                raise QueueFullError("推理队列已满，请稍后重试")
            self.q.append(it)
            self.cv.notify()
        return fut

    def _prune_queue(self, now: float) -> None:
        """Drop abandoned payloads before capacity checks and GPU selection."""
        pending = []
        for it in self.q:
            if it.future.cancelled():
                self.stats["cancelled"] += 1
            elif now >= it.deadline:
                # Atomically claim before setting an exception, in case the
                # HTTP waiter concurrently cancels the same Future.
                if it.future.set_running_or_notify_cancel():
                    it.future.set_exception(concurrent.futures.TimeoutError("推理请求排队超时"))
                    self.stats["expired"] += 1
                else:
                    self.stats["cancelled"] += 1
            else:
                pending.append(it)
        self.q = pending

    def _select_mode(self, now: float) -> tuple[bool, bool]:
        detection = any(it.qa is None for it in self.q)
        oldest_qa = next((it for it in self.q if it.qa is not None), None)
        protect_qa = (detection and not self._last_batch_qa and oldest_qa is not None and
                      now - oldest_qa.queued_at >= self.qa_max_wait_sec)
        return not detection or protect_qa, protect_qa

    def snapshot(self) -> dict:
        with self.cv:
            return {**self.stats, "queued": len(self.q),
                    "queued_detection": sum(it.qa is None for it in self.q),
                    "queued_qa": sum(it.qa is not None for it in self.q),
                    "queue_limit": self.max_queue,
                    "qa_queue_limit": self.max_qa_queue,
                    "qa_max_wait_sec": self.qa_max_wait_sec,
                    "priority": "detection", "max_batch": BATCH_MAX,
                    "wait_ms": BATCH_WAIT_MS}

    def _take_batch(self) -> list[_Item]:
        with self.cv:
            self._prune_queue(time.perf_counter())
            while not self.q:
                self.cv.wait(timeout=1.0)
                self._prune_queue(time.perf_counter())
                if not self.q:
                    cache = STATE.get("vision_cache")
                    if cache is not None:
                        cache._expire(time.monotonic())
                    return []
            collect_until = time.perf_counter() + BATCH_WAIT_MS / 1000.0
            while self.q:
                now = time.perf_counter()
                mode, protected = self._select_mode(now)
                # QA answers are intentionally single-request batches: the
                # fairness concession blocks cameras for one answer at most.
                limit = 1 if mode else max(1, BATCH_MAX)
                count = sum((it.qa is not None) == mode for it in self.q)
                remain = collect_until - now
                if count >= limit or remain <= 0:
                    break
                # Keep payloads in q during the short batching wait: capacity
                # stays bounded and new detections can replace the QA choice.
                self.cv.wait(timeout=remain)
                self._prune_queue(time.perf_counter())
            if not self.q:
                return []
            mode, protected = self._select_mode(time.perf_counter())
            limit = 1 if mode else max(1, BATCH_MAX)
            batch, pending = [], []
            for it in self.q:
                if (it.qa is not None) == mode and len(batch) < limit:
                    # Claim before image decoding. A cancelled or timed-out
                    # request must never spend GPU time; running work is kept.
                    if it.future.set_running_or_notify_cancel():
                        batch.append(it)
                    else:
                        self.stats["cancelled"] += 1
                else:
                    pending.append(it)
            self.q = pending
            if batch:
                self._last_batch_qa = mode
            if batch and protected:
                self.stats["qa_wait_protected"] += 1
            return batch

    def _loop(self) -> None:
        while True:
            batch = self._take_batch()
            if not batch:
                continue
            with self.cv:
                self.stats["batches"] += 1
                self.stats["items"] += len(batch)
                self.stats["max_batch"] = max(self.stats["max_batch"], len(batch))
            try:
                results = infer_batch(batch)
            except Exception as e:                      # noqa: BLE001
                print("[batch] 整批失败:\n" + traceback.format_exc(), flush=True)
                results = [{"parsed": None, "raw": "",
                            "error": f"{type(e).__name__}: {e}",
                            "latency_sec": 0.0, "tokens_prompt": 0,
                            "tokens_out": 0}] * len(batch)
            for it, res in zip(batch, results):
                if not it.future.done():
                    it.future.set_result(res)


BATCHER = Batcher()


def _build_inputs(items: list[_Item]):
    """把 N 个请求拼成一次前向的输入。

    ⚠️ 必须用**左填充**：自回归生成时所有序列的"最后一个 token"要对齐在末尾，
    右填充会让短序列的位置错位（生成的 token 会对不上）。
    批量测试里 4 个请求文本完全相同、恰好不需要填充，所以掩盖了这个问题 ——
    真实场景下各窗口的标签长度不同，填充方向错了结果就是错的。
    """
    from civility.prompts import build_messages

    bs = STATE["behaviors"]
    proc = STATE["processor"]
    if getattr(proc, "tokenizer", None) is not None:
        proc.tokenizer.padding_side = "left"

    texts, images, flat_images = [], [], []
    for it in items:
        bs = it.behaviors if it.behaviors is not None else STATE["behaviors"]
        imgs = decode_images(it.frames)
        if it.qa is not None:
            from civility.video_qa import build_qa_messages
            msgs = build_qa_messages(imgs, it.qa)
        else:
            msgs = build_messages(bs, imgs, it.labels, with_few_shot=it.few_shot,
                              with_alert=it.with_alert,
                              with_actions=it.with_actions,
                              with_risks=it.with_risks)
        texts.append(proc.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True))
        images.append(msgs)
        flat_images.extend(imgs)

    if STATE.get("has_qvu"):
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(images)
        return proc(text=texts, images=image_inputs, videos=video_inputs,
                    padding=True, return_tensors="pt")
    return proc(text=texts, images=flat_images, padding=True, return_tensors="pt")


def infer_batch(items: list[_Item]) -> list[dict]:
    """一次前向算完这一批，返回逐个请求的结果。"""
    import torch

    from civility.prompts import extract_json, sanitize, validate

    bs = STATE["behaviors"]
    if not STATE["ready"]:
        raise RuntimeError(f"模型未就绪：{STATE.get('error') or '仍在加载'}")
    model = STATE["model"]

    prepare_started = time.perf_counter()
    inputs = _build_inputs(items).to(model.device)
    prepare_sec = time.perf_counter() - prepare_started
    cache = STATE.get("vision_cache")
    if cache is not None:
        from cluster.vision_cache import frame_key
        cache.keys = [frame_key(f) for it in items for f in it.frames]
    n_prompt = int(inputs.input_ids.shape[1])
    max_new = max(it.max_new_tokens for it in items)

    t0 = time.time()
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    latency = time.time() - t0

    proc = STATE["processor"]
    # 左填充 ⇒ 所有样本的生成部分都从 n_prompt 开始，直接切片即可
    trimmed = gen[:, n_prompt:]
    outs = proc.batch_decode(trimmed, skip_special_tokens=True,
                             clean_up_tokenization_spaces=False)
    decoded_at = time.perf_counter()

    results = []
    for index, (it, raw) in enumerate(zip(items, outs)):
        bs = it.behaviors if it.behaviors is not None else STATE["behaviors"]
        # A batch can finish at different token positions; padding is not output.
        tokenizer = getattr(proc, "tokenizer", None)
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is None:
            eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}
        row = trimmed[index]
        ids = row.tolist() if callable(getattr(row, "tolist", None)) else []
        end = next((i + 1 for i, token in enumerate(ids) if token in eos_ids), None)
        output_tokens = end if end is not None else len(ids) or int(trimmed.shape[1])
        finish_reason = "length" if end is None and output_tokens >= max_new else "stop"
        obj = extract_json(raw)
        if obj is None:
            results.append({"parsed": None, "raw": raw,
                            "error": "输出里找不到合法 JSON",
                            "latency_sec": round(latency, 3),
                            "tokens_prompt": n_prompt,
                            "tokens_out": output_tokens, "finish_reason": finish_reason,
                            "batch": len(items)})
            continue
        if it.qa is not None:
            from civility.video_qa import validate_answer
            try:
                parsed = validate_answer(obj, it.qa["evidence"])
                ok, problems = True, []
            except ValueError as exc:
                parsed, ok, problems = None, False, [str(exc)]
        else:
            ok, problems = validate(obj, bs, strict=False, with_risks=it.with_risks)
            parsed = sanitize(obj, bs, frame_count=len(it.frames))
        results.append({
            "parsed": parsed, "raw": raw,
            # 保留归一化结果方便诊断，但结构失败不能当成一次有效检查。
            "error": None if ok else "Schema 校验失败: " + "; ".join(problems),
            "valid": ok, "problems": problems,
            "latency_sec": round(latency, 3),
            "tokens_prompt": n_prompt,
            "tokens_out": output_tokens, "finish_reason": finish_reason,
            # 批量大小要透出到页面：用户能直接看到"这批算了几个窗口"，
            # 也就知道调度的收益有没有真的发生
            "batch": len(items),
        })
    STATE["requests"] += len(items)
    STATE["last_latency"] = latency
    STATE["latencies"].append(round(latency, 3))
    STATE["latencies"] = STATE["latencies"][-50:]
    for it, result in zip(items, results):
        result["timing"] = {
            "queue_sec": round(max(0.0, prepare_started - it.queued_at), 3),
            "prepare_sec": round(prepare_sec, 3),
            "generate_sec": round(latency, 3),
            "decode_sec": round(max(0.0, decoded_at - prepare_started - prepare_sec - latency), 3),
        }
    return results


# ─────────────────────────── HTTP 服务 ───────────────────────────


class Handler(BaseHTTPRequestHandler):
    server_version = "CivilityVLM/1.0"
    protocol_version = "HTTP/1.1"

    # ── 工具 ──
    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _auth_ok(self) -> bool:
        tok = STATE.get("token")
        if not tok:
            return True
        got = self.headers.get("X-Token") or ""
        return got == tok

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length", 0) or 0)
        if not n:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8"))

    def log_message(self, fmt, *args):  # 静音，避免刷爆作业日志
        pass

    # ── 路由 ──
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/health", "/"):
            bs = STATE["behaviors"]
            return self._json(200, {
                "ok": True,
                "behavior_protocol": 2,
                "video_qa_protocol": 1,
                "hostname": socket.gethostname(),
                "ready": STATE["ready"],
                "loading": STATE["loading"],
                "error": STATE["error"],
                "model": STATE["model_name"],
                "device": STATE["device"],
                "load_sec": STATE.get("load_sec"),
                "uptime_sec": round(time.time() - STATE["started_at"], 1),
                "requests": STATE["requests"],
                "errors": STATE["errors"],
                "last_latency_sec": round(STATE["last_latency"], 3),
                "avg_latency_sec": (
                    round(sum(STATE["latencies"]) / len(STATE["latencies"]), 3)
                    if STATE["latencies"] else None
                ),
                "behaviors": [b.id for b in bs.enabled] if bs else [],
                "pid": os.getpid(),
                "batcher": BATCHER.snapshot(),
                "vision_cache": ({"images": len(STATE["vision_cache"].entries),
                                  "bytes": STATE["vision_cache"].bytes,
                                  "hits": STATE["vision_cache"].hits,
                                  "misses": STATE["vision_cache"].misses,
                                  "supported": STATE["vision_cache"].supported}
                                 if STATE.get("vision_cache") else None),
            })

        if path == "/behaviors":
            bs = STATE["behaviors"]
            if not bs:
                return self._json(503, {"error": "行为定义未加载"})
            return self._json(200, {
                "defaults": bs.defaults,
                "behaviors": [
                    {
                        "id": b.id, "name": b.name, "definition": b.definition,
                        "evidence_required": b.evidence_required,
                        "counter_examples": b.counter_examples,
                        "observable_cues": b.observable_cues,
                        "evidence_checks": b.evidence_checks,
                        "evidence_sequence": b.evidence_sequence,
                        "severity": b.severity,
                    }
                    for b in bs.enabled
                ],
            })
        return self._json(404, {"error": f"未知路径 {path}"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]

        if path == "/reload-behaviors":
            if not self._auth_ok():
                return self._json(403, {"error": "token 不对"})
            try:
                body = self._read_json()
                n = reload_behaviors(body.get("path") or STATE["behaviors_path"])
                print(f"[reload] 行为定义已更新：{n} 项", flush=True)
                return self._json(200, {"ok": True, "behaviors": n})
            except Exception as e:
                return self._json(500, {"error": f"{type(e).__name__}: {e}"})

        if path == "/video-qa":
            if not self._auth_ok():
                return self._json(403, {"error": "token 不对"})
            if not STATE["ready"]:
                return self._json(503, {"error": "模型未就绪"})
            try:
                from civility.video_qa import validate_request
                body = self._read_json()
                qa = validate_request(body)
                tokens = body.get("max_new_tokens", 768)
                if not isinstance(tokens, int) or isinstance(tokens, bool) or not 256 <= tokens <= 1536:
                    raise ValueError("问答输出 token 须为 256 至 1536")
            except (ValueError, TypeError) as exc:
                return self._json(400, {"error": str(exc)})
            try:
                res = wait_for_result(BATCHER.submit(body["frames"], [], tokens, False, False, False, qa=qa))
                res.update({"ok": res.get("error") is None, "model": STATE["model_name"],
                            "hostname": socket.gethostname()})
                if res.get("error"):
                    STATE["errors"] += 1
                return self._json(200, res)
            except QueueFullError as exc:
                return self._json(503, {"error": str(exc), "retryable": True})
            except concurrent.futures.TimeoutError:
                STATE["errors"] += 1
                return self._json(504, {"error": "推理等待超时，尚未开始的请求已取消", "retryable": True})
            except Exception as exc:
                STATE["errors"] += 1
                return self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

        if path == "/infer":
            if not self._auth_ok():
                return self._json(403, {"error": "token 不对"})
            if not STATE["ready"]:
                return self._json(503, {
                    "error": STATE.get("error") or ("模型加载中" if STATE["loading"] else "未就绪"),
                    "loading": STATE["loading"],
                })
            try:
                body = self._read_json()
            except Exception as e:
                return self._json(400, {"error": f"请求体不是合法 JSON: {e}"})

            try:
                frames = body.get("frames") or []
                if not frames:
                    return self._json(400, {"error": "frames 为空"})
                labels = body.get("labels") or [f"帧{i + 1}" for i in range(len(frames))]
                from civility.config import parse_behaviors
                requested_behaviors = (parse_behaviors(body["behaviors"])
                                       if isinstance(body.get("behaviors"), dict) else None)
                res = infer_one(
                    frames,
                    labels,
                    int(body.get("max_new_tokens", 512)),
                    bool(body.get("few_shot", False)),
                    bool(body.get("with_alert", False)),
                    bool(body.get("with_actions", False)),
                    bool(body.get("with_risks", False)),
                    behaviors=requested_behaviors,
                )
                res.update({
                    "ok": res.get("error") is None,
                    "hostname": socket.gethostname(),
                    "model": STATE["model_name"],
                })
                if res.get("error"):
                    STATE["errors"] += 1
                return self._json(200, res)
            except QueueFullError as exc:
                return self._json(503, {"error": str(exc), "retryable": True})
            except concurrent.futures.TimeoutError:
                STATE["errors"] += 1
                return self._json(504, {"error": "推理等待超时，尚未开始的请求已取消", "retryable": True})
            except Exception as e:
                STATE["errors"] += 1
                print("[infer] 异常:\n" + traceback.format_exc(), flush=True)
                return self._json(500, {"error": f"{type(e).__name__}: {e}"})

        return self._json(404, {"error": f"未知路径 {path}"})


def main() -> int:
    ap = argparse.ArgumentParser(description="计算节点常驻 VLM 推理服务")
    ap.add_argument("--pkg", default=".", help="作业包目录（含 civility/ 和 behaviors.json）")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--bind", default="0.0.0.0",
                    help="必须绑 0.0.0.0，否则登录节点的 SSH 通道连不进来（已实测）")
    ap.add_argument("--token", default=os.environ.get("CIVILITY_TOKEN", ""),
                    help="可选的简单鉴权 token")
    ap.add_argument("--behaviors", default="", help="behaviors.json 路径")
    args = ap.parse_args()

    pkg = Path(args.pkg).resolve()
    sys.path.insert(0, str(pkg))

    behaviors_path = args.behaviors or str(pkg / "behaviors.json")
    if not Path(behaviors_path).exists():
        print(f"❌ 找不到行为定义 {behaviors_path}", flush=True)
        return 1

    STATE["token"] = args.token or None
    reload_behaviors(behaviors_path)
    print(f"[init] 行为定义 {len(STATE['behaviors'].enabled)} 项 ← {behaviors_path}", flush=True)

    # 模型在后台线程加载，HTTP 立刻可用 —— 这样网页能马上看到"正在加载模型"
    threading.Thread(target=load_model, args=(args.model,), daemon=True).start()
    # 批量调度线程：与模型加载并行启动，模型就绪前提交的请求会排队等（它们
    # 本来就该等），不会丢。
    BATCHER.start()

    srv = ThreadingHTTPServer((args.bind, args.port), Handler)
    print(f"[serve] http://{args.bind}:{args.port}  host={socket.gethostname()} "
          f"model={args.model}", flush=True)
    print(f"[serve] 健康检查: curl http://127.0.0.1:{args.port}/health", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("[serve] 收到中断，退出", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
    · 单请求单窗口（4 帧）推理，用线程锁串行化，避免并发把显存打满
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

        try:
            from transformers import Qwen3VLForConditionalGeneration as Cls
        except ImportError:
            from transformers import AutoModelForVision2Seq as Cls  # type: ignore

        print(f"[load] {model_name} …", flush=True)
        model = Cls.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        ).eval()
        proc = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

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
              with_alert: bool = False, with_actions: bool = False) -> dict:
    """单请求推理（保留给 batch=1 的路径与调试用）。"""
    fut = BATCHER.submit(frames, labels, max_new_tokens, few_shot,
                         with_alert, with_actions)
    return fut.result(timeout=BATCH_TIMEOUT_SEC)


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
BATCH_WAIT_MS = int(os.environ.get("VLM_BATCH_WAIT_MS", "150") or 0)
BATCH_TIMEOUT_SEC = float(os.environ.get("VLM_BATCH_TIMEOUT", "600") or 600)


class _Item:
    __slots__ = ("frames", "labels", "max_new_tokens", "few_shot",
                 "with_alert", "with_actions", "future")

    def __init__(self, frames, labels, n, few, alert, actions, future):
        self.frames, self.labels = frames, labels
        self.max_new_tokens, self.few_shot = n, few
        self.with_alert, self.with_actions = alert, actions
        self.future = future


class Batcher:
    """收集并发请求 → 凑成一批 → 一次前向算完 → 各自返回。

    这是"没有 vLLM 时的替代品"：vLLM 在本集群装不了（glibc 2.28 太老），
    但它的核心收益（连续批处理）用几十行就能拿到一大部分。
    """

    def __init__(self) -> None:
        self.q: list[_Item] = []
        self.cv = threading.Condition()
        self.thread: threading.Thread | None = None
        self.stats = {"batches": 0, "items": 0, "max_batch": 0}

    def start(self) -> None:
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self._loop, daemon=True,
                                       name="vlm-batcher")
        self.thread.start()
        print(f"[batch] 动态批量已启用：最多 {BATCH_MAX} 个/批，"
              f"凑批等待 {BATCH_WAIT_MS}ms", flush=True)

    def submit(self, frames, labels, max_new_tokens, few_shot, with_alert,
               with_actions):
        fut: "concurrent.futures.Future[dict]" = concurrent.futures.Future()
        it = _Item(frames, labels, max_new_tokens, few_shot, with_alert,
                   with_actions, fut)
        with self.cv:
            self.q.append(it)
            self.cv.notify()
        return fut

    def _take_batch(self) -> list[_Item]:
        with self.cv:
            while not self.q:
                self.cv.wait(timeout=1.0)
                if not self.q:
                    return []
            batch = self.q[:BATCH_MAX]
            del self.q[:len(batch)]
            # 还有余量就再等一小会儿，让"几乎同时到"的请求凑进同一批。
            # 等待时间很短（默认 150ms）：用户看到的延迟增加有限，
            # 但批量带来的是数倍吞吐 —— 这笔交易在任何监控场景都划算。
            if len(batch) < BATCH_MAX and BATCH_WAIT_MS > 0:
                deadline = time.time() + BATCH_WAIT_MS / 1000.0
                while len(batch) < BATCH_MAX:
                    remain = deadline - time.time()
                    if remain <= 0:
                        break
                    self.cv.wait(timeout=remain)
                    if self.q:
                        more = self.q[:BATCH_MAX - len(batch)]
                        del self.q[:len(more)]
                        batch.extend(more)
            return batch

    def _loop(self) -> None:
        while True:
            batch = self._take_batch()
            if not batch:
                continue
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
    import torch

    from civility.prompts import build_messages

    bs = STATE["behaviors"]
    proc = STATE["processor"]
    if getattr(proc, "tokenizer", None) is not None:
        proc.tokenizer.padding_side = "left"

    texts, images = [], []
    for it in items:
        imgs = decode_images(it.frames)
        msgs = build_messages(bs, imgs, it.labels, with_few_shot=it.few_shot,
                              with_alert=it.with_alert,
                              with_actions=it.with_actions)
        texts.append(proc.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True))
        images.append(msgs)

    if STATE.get("has_qvu"):
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(images)
        return proc(text=texts, images=image_inputs, videos=video_inputs,
                    padding=True, return_tensors="pt")
    flat = [im for ims in images for im in ims]
    return proc(text=texts, images=None, padding=True, return_tensors="pt")


def infer_batch(items: list[_Item]) -> list[dict]:
    """一次前向算完这一批，返回逐个请求的结果。"""
    import torch

    from civility.prompts import extract_json, sanitize, validate

    bs = STATE["behaviors"]
    if not STATE["ready"]:
        raise RuntimeError(f"模型未就绪：{STATE.get('error') or '仍在加载'}")
    model = STATE["model"]

    inputs = _build_inputs(items).to(model.device)
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

    results = []
    for raw in outs:
        obj = extract_json(raw)
        if obj is None:
            results.append({"parsed": None, "raw": raw,
                            "error": "输出里找不到合法 JSON",
                            "latency_sec": round(latency, 3),
                            "tokens_prompt": n_prompt,
                            "tokens_out": int(trimmed.shape[1]),
                            "batch": len(items)})
            continue
        ok, problems = validate(obj, bs, strict=False)
        parsed = sanitize(obj, bs)
        results.append({
            "parsed": parsed, "raw": raw, "error": None,
            "valid": ok, "problems": problems,
            "latency_sec": round(latency, 3),
            "tokens_prompt": n_prompt,
            "tokens_out": int(trimmed.shape[1]),
            # 批量大小要透出到页面：用户能直接看到"这批算了几个窗口"，
            # 也就知道调度的收益有没有真的发生
            "batch": len(items),
        })
    STATE["requests"] += len(items)
    STATE["last_latency"] = latency
    STATE["latencies"].append(round(latency, 3))
    return results

    STATE["latencies"] = STATE["latencies"][-50:]
    return {
        "parsed": sanitize(obj, bs) if ok else None,
        "raw": raw,
        "error": None if ok else "Schema 校验失败: " + "; ".join(problems),
        "latency_sec": round(latency, 3),
        # 这两个数决定用户体验：出字太长就是"慢"的主因，要回去压缩输出格式
        "tokens_prompt": n_prompt,
        "tokens_out": n_new,
        "tok_per_sec": round(n_new / latency, 1) if latency > 0 else None,
    }


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
                res = infer_one(
                    frames,
                    labels,
                    int(body.get("max_new_tokens", 512)),
                    bool(body.get("few_shot", False)),
                    bool(body.get("with_alert", False)),
                    bool(body.get("with_actions", False)),
                )
                res.update({
                    "ok": res.get("error") is None,
                    "hostname": socket.gethostname(),
                    "model": STATE["model_name"],
                })
                if res.get("error"):
                    STATE["errors"] += 1
                return self._json(200, res)
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

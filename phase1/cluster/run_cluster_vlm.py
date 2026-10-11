#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""集群侧批推理 —— 在同济 A800 上跑 Qwen3-VL，逐窗口输出 JSON。

⚠️ 这个脚本必须在**计算节点**上跑（sbatch 提交），登录节点跑会被监控杀进程。

设计要点（都踩过坑，见 tju/10-cluster-handover.md）：

1. **不用 vLLM**。glibc 2.28 装不上 vllm>=0.12（要求 manylinux_2_31），
   所以走纯 transformers + PyTorch。代价是没有 continuous batching，
   但我们本来就是单卡串行批处理，实际影响很小。

2. **复用本地的 prompts.py**。同一份 Prompt 生成代码，云端和集群结果才可比。
   如果在这里另写一套 Prompt，两边的评测数字就没有意义了。

3. **断点续跑**。每题结果立刻 append 到 results.jsonl，重跑时跳过已完成。
   配合 sbatch --requeue，超时被杀也能接着跑。

4. **驱动预检**。部分 A800 节点驱动是 550.54.14，而 torch 是 cu130 编译，
   在上面 torch.cuda.is_available() 会是 False。遇到就退出码 3，
   Slurm --requeue 会自动换节点。

退出码约定：
    0   全部跑完
    1   未跑完（部分已落盘）→ 触发 requeue
    3   驱动不兼容 → 触发 requeue 换节点
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# ── 离线加载已下好的权重（计算节点无公网出口，必须离线）──
MODEL_ROOT = os.environ.get("MODEL_ROOT", "/ssdfs/datahome/u00009/models")
os.environ.setdefault("HF_HOME", f"{MODEL_ROOT}/hf")
os.environ.setdefault("HF_HUB_CACHE", f"{MODEL_ROOT}/hf/hub")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def preflight() -> None:
    """驱动与 CUDA 预检。失败直接退出码 3，让 Slurm 换节点。"""
    import subprocess

    try:
        drv = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip().splitlines()
        print(f"[preflight] GPU 驱动: {', '.join(drv) if drv else '未知'}", flush=True)
    except Exception as e:
        print(f"[preflight] nvidia-smi 失败: {e}", flush=True)

    try:
        import torch

        ok = torch.cuda.is_available()
        print(f"[preflight] torch={torch.__version__} cuda_available={ok}", flush=True)
        if ok:
            print(f"[preflight] 卡数={torch.cuda.device_count()} "
                  f"型号={torch.cuda.get_device_name(0)}", flush=True)
        else:
            print("[preflight] ❌ CUDA 不可用（多半是驱动 550.54.14 的老节点）", flush=True)
            sys.exit(3)
    except Exception as e:
        print(f"[preflight] torch 导入失败: {e}", flush=True)
        sys.exit(3)


def load_behaviors_json(path: Path):
    """把 behaviors.json 还原成 BehaviorSet，供 prompts.py 使用。"""
    from civility.config import Behavior, BehaviorSet

    raw = json.loads(path.read_text(encoding="utf-8"))
    items = []
    for b in raw.get("behaviors", []):
        items.append(
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
        )
    return BehaviorSet(behaviors=items, defaults=raw.get("defaults") or {})


def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("window_id") and not rec.get("error"):
                done.add(rec["window_id"])
    return done


def load_request_options(path: Path) -> dict[str, bool]:
    """新版作业包带请求选项，旧包保持原来的固定行为推理。"""
    raw = json.loads(path.read_text(encoding="utf-8"))
    options = raw.get("request_options") or {}
    return {key: bool(options.get(key, False))
            for key in ("with_alert", "with_actions", "with_risks")}


def main() -> int:
    ap = argparse.ArgumentParser(description="A800 上的不文明行为批推理")
    ap.add_argument("--pkg", default=".", help="作业包目录（含 manifest.jsonl / images/）")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个窗口（冒烟测试用）")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--few-shot", action="store_true",
                    help="Prompt 里带 few-shot 示例。默认关闭：示例对生成式模型"
                         "帮助有限却让每次请求都变长；若云端开了这里也要开，两边才可比")
    ap.add_argument("--no-resume", action="store_true", help="忽略已有结果，全部重跑")
    args = ap.parse_args()

    pkg = Path(args.pkg).resolve()
    manifest_path = pkg / "manifest.jsonl"
    behaviors_path = pkg / "behaviors.json"
    images_dir = pkg / "images"
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = pkg / out_path

    for p in (manifest_path, behaviors_path):
        if not p.exists():
            print(f"❌ 缺少 {p}。作业包不完整，重新跑 cluster-prepare 并上传。", flush=True)
            return 1

    preflight()

    from civility.config import ROOT  # noqa: F401  (确认包路径可见)
    from civility.prompts import build_messages, extract_json, sanitize, validate

    bs = load_behaviors_json(behaviors_path)
    request_options = load_request_options(behaviors_path)
    print(f"[cfg] 启用行为 {len(bs.enabled)} 个: {[b.id for b in bs.enabled]}", flush=True)

    windows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                windows.append(json.loads(line))
    if args.limit:
        windows = windows[: args.limit]

    done = set() if args.no_resume else load_done(out_path)
    todo = [w for w in windows if w["window_id"] not in done]
    print(f"[cfg] 窗口总数 {len(windows)}，已完成 {len(done)}，本轮待跑 {len(todo)}", flush=True)
    if not todo:
        print("[done] 没有待跑的窗口，直接退出", flush=True)
        return 0

    # ── 加载模型 ──
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import Qwen3VLForConditionalGeneration as _ModelCls
    except ImportError:  # 兼容不同 transformers 版本的类名
        from transformers import AutoModelForVision2Seq as _ModelCls  # type: ignore

    print(f"[load] 正在加载 {args.model} …", flush=True)
    t0 = time.time()
    model = _ModelCls.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    print(f"[load] 完成，用时 {time.time() - t0:.1f}s", flush=True)

    try:
        from qwen_vl_utils import process_vision_info
        has_qvu = True
    except ImportError:
        has_qvu = False
        print("[warn] 没有 qwen_vl_utils，改用 processor 内置的图片处理", flush=True)

    from PIL import Image

    out_path.parent.mkdir(parents=True, exist_ok=True)
    bar = 50
    n_ok = n_err = 0

    with open(out_path, "a", encoding="utf-8") as fo:
        for i, w in enumerate(todo, 1):
            t_start = time.time()
            rec = {
                "window_id": w["window_id"],
                "video": w.get("video", ""),
                "t_start": w.get("t_start", 0.0),
                "t_end": w.get("t_end", 0.0),
                "parsed": None,
                "raw": "",
                "error": None,
                "retries": 0,
                "latency_sec": 0.0,
                "backend": "cluster",
            }
            try:
                imgs = []
                for f in w["frames"]:
                    p = images_dir / Path(f).name
                    imgs.append(Image.open(p).convert("RGB"))

                # 与云端共用同一个 build_messages：这里传 PIL 对象，
                # 云端传 data URL，Prompt 文本部分逐字相同 → 结果可比。
                messages = build_messages(
                    bs,
                    imgs,
                    w.get("labels") or [f"帧{j + 1}" for j in range(len(imgs))],
                    with_few_shot=args.few_shot,
                    **request_options,
                )

                if has_qvu:
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(
                        text=[text], images=image_inputs, videos=video_inputs,
                        padding=True, return_tensors="pt",
                    ).to(model.device)
                else:
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = processor(text=[text], images=imgs, return_tensors="pt").to(
                        model.device
                    )

                with torch.inference_mode():
                    gen = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,  # 判分类任务必须贪心，否则结果不可复现
                    )
                trimmed = [o[len(i_):] for i_, o in zip(inputs.input_ids, gen)]
                raw = processor.batch_decode(
                    trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

                obj = extract_json(raw)
                if obj is None:
                    rec["error"] = "输出里找不到合法 JSON"
                else:
                    # strict=False：质量 lint 不算错误，避免无谓的失败重试
                    ok, problems = validate(obj, bs, strict=False,
                                            with_risks=request_options["with_risks"])
                    if not ok:
                        rec["error"] = "Schema 校验失败: " + "; ".join(problems)
                    rec["parsed"] = sanitize(obj, bs, frame_count=len(imgs)) if ok else None
                rec["raw"] = raw
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
                print(traceback.format_exc(), flush=True)

            rec["latency_sec"] = round(time.time() - t_start, 3)
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fo.flush()
            os.fsync(fo.fileno())  # 立刻落盘，超时被杀也不丢已完成的题

            if rec["error"]:
                n_err += 1
            else:
                n_ok += 1

            if i % 5 == 0 or i == len(todo):
                el = time.time() - t_start
                rate = i / max(1e-6, time.time() - t_start)
                print(
                    f"[{i}/{len(todo)}] 成功 {n_ok} 失败 {n_err} "
                    f"本窗 {el:.1f}s 均速 {rate:.2f} 窗/s",
                    flush=True,
                )
            if i % bar == 0:
                print("─" * 40, flush=True)

    print(f"[done] 成功 {n_ok} 失败 {n_err} → {out_path}", flush=True)
    # 有失败项就返回非零，配合 --requeue 自动重试（续跑会跳过已成功的）
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

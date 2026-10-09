#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""推理栈微基准：查清"34 token/s 到底慢在哪"，并 A/B 两种加载配置。

背景（为什么值得花一次 GPU 时间做这个）：
    实测 Qwen3-VL-4B 在 A800 上 batch=1 解码只有 **34 token/s**。
    按显存带宽粗算（约 2TB/s ÷ 8GB 权重）理论应到 **200 token/s** 量级 ——
    现在只跑到 14%，很像"配置没生效"而不是"模型天生慢"。可疑点：
      · `from_pretrained(dtype=...)` 在部分 transformers 版本上被静默忽略 → fp32
        （权重翻倍、解码减半）
      · 没显式指定 attn_implementation → 可能退化成 eager 注意力
      · device_map="auto" 走 accelerate，每层多一层 hook
    这三项一旦确认，提速是**免费**的（不改模型、不改流程、不换硬件）。

本脚本把"时间花在哪"拆成四段，逐段测量，并对两种加载配置做对照：
    A. 现状：dtype=bfloat16 + device_map="auto"（无 attn 指定）
    B. 显式：attn_implementation="sdpa" + 直接 .to("cuda")（不用 device_map）
测量项：
    1) 模型信息：真实 dtype、注意力后端、参数量、显存占用
    2) 纯文本解码：无图、生成 N token → **这一项才是 LLM 的真实解码速度**
       （若它也很慢，说明是栈/权重精度问题；若它很快，说明慢在视觉路径）
    3) 视觉 prefill：0/1/4 张图各生成 1 token → 分离"视觉编码 + prefill"成本
    4) 真实窗口：与线上完全相同的 Prompt + 4 帧 + 220 token → 端到端对照

用法（由 cluster/bench-vlm.sbatch 在计算节点上调用，不要在本机跑）：
    python vlm_bench.py --pkg /path/pkg --model Qwen/Qwen3-VL-4B-Instruct
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def human(x: float) -> str:
    return f"{x:.2f}"


def load_model(model_name: str, *, attn: str | None, use_device_map: bool):
    """加载模型。两种配置只差这三个参数，其余完全一致。"""
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import Qwen3VLForConditionalGeneration as Cls
    except ImportError:
        from transformers import AutoModelForVision2Seq as Cls

    kw: dict = {"dtype": torch.bfloat16, "trust_remote_code": True}
    if attn:
        kw["attn_implementation"] = attn
    if use_device_map:
        kw["device_map"] = "auto"

    t0 = time.time()
    model = Cls.from_pretrained(model_name, **kw).eval()
    if not use_device_map:
        model = model.to("cuda")
    proc = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    return model, proc, time.time() - t0


def model_info(model) -> dict:
    import torch

    p = next(model.parameters())
    cfg = getattr(model, "config", None)
    n_param = sum(x.numel() for x in model.parameters())
    info = {
        "dtype": str(p.dtype),
        "param_b": round(n_param / 1e9, 2),
        "attn_implementation": getattr(cfg, "_attn_implementation", "?"),
        "device": str(p.device),
        "vram_used_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
        "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    }
    return info


def bench_text_decode(model, proc, n_tokens: int = 128) -> dict:
    """纯文本解码：没有图像，测 LLM 的真实 token/s。"""
    import torch

    msgs = [{"role": "user",
             "content": "请连续输出 300 个汉字，内容不限，不要换行。"}]
    text = proc.apply_chat_template(msgs, tokenize=False,
                                    add_generation_prompt=True)
    inputs = proc(text=[text], return_tensors="pt").to(model.device)
    n_prompt = int(inputs.input_ids.shape[1])
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=n_tokens, do_sample=False)
    torch.cuda.synchronize()
    dt = time.time() - t0
    n_out = int(gen.shape[1]) - n_prompt
    return {"prompt_tokens": n_prompt, "out_tokens": n_out,
            "sec": round(dt, 3),
            "tok_per_sec": round(n_out / dt, 1) if dt > 0 else 0.0}


def bench_vision(model, proc, images: list, *, n_images: int, labels: list,
                 n_tokens: int = 1) -> dict:
    """视觉 prefill：给 N 张图，只生成少量 token → 得到"看图 + prefill"的成本。"""
    import torch

    imgs = images[:n_images]
    content: list = [{"type": "text", "text": "以下是连续帧："}]
    for i in range(len(imgs)):
        content.append({"type": "text", "text": f"[帧{i}]"})
        content.append({"type": "image", "image": imgs[i]})
    content.append({"type": "text", "text": "用一句话描述画面。"})
    msgs = [{"role": "user", "content": content}]
    text = proc.apply_chat_template(msgs, tokenize=False,
                                    add_generation_prompt=True)
    try:
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = proc(text=[text], images=image_inputs, videos=video_inputs,
                      padding=True, return_tensors="pt").to(model.device)
    except Exception:
        inputs = proc(text=[text], images=imgs, return_tensors="pt").to(model.device)
    n_prompt = int(inputs.input_ids.shape[1])
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=n_tokens, do_sample=False)
    torch.cuda.synchronize()
    dt = time.time() - t0
    return {"n_images": n_images, "prompt_tokens": n_prompt,
            "sec": round(dt, 3)}


def bench_real_window(model, proc, bs, images: list, *, n_tokens: int = 220,
                      few_shot: bool = True) -> dict:
    """与线上**完全同一条路径**：真实 Prompt + 4 帧 + 真实输出长度。"""
    import torch

    from civility.prompts import build_messages

    imgs = images[:4]
    msgs = build_messages(bs, imgs, [f"t={i * 2}.0s" for i in range(len(imgs))],
                          with_few_shot=few_shot, with_alert=True,
                          with_actions=True)
    text = proc.apply_chat_template(msgs, tokenize=False,
                                    add_generation_prompt=True)
    try:
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = proc(text=[text], images=image_inputs, videos=video_inputs,
                      padding=True, return_tensors="pt").to(model.device)
    except Exception:
        inputs = proc(text=[text], images=imgs, return_tensors="pt").to(model.device)
    n_prompt = int(inputs.input_ids.shape[1])
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=n_tokens, do_sample=False)
    torch.cuda.synchronize()
    dt = time.time() - t0
    n_out = int(gen.shape[1]) - n_prompt
    raw = proc.batch_decode(gen[:, n_prompt:], skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)[0]
    return {"prompt_tokens": n_prompt, "out_tokens": n_out, "sec": round(dt, 3),
            "tok_per_sec": round(n_out / dt, 1) if dt > 0 else 0.0,
            "raw_head": raw[:120].replace("\n", "\\n")}


def bench_batch(model, proc, bs, images: list, *, batch: int,
                n_tokens: int = 220) -> dict:
    """批量推理：一次喂 N 个窗口，看总耗时是否接近单窗 —— 这是吞吐的关键。

    为什么值得测：单窗解码只跑到理论带宽的 15%，说明 GPU 有大半时间在**等 CPU**。
    等待时间与批量大小无关，所以把 N 个窗口塞进一次前向，理论上能拿到接近 N 倍的
    吞吐（这正是 vLLM 的核心收益，而本集群装不了 vLLM）。
    """
    import torch

    from civility.prompts import build_messages

    imgs = images[:4]
    msgs = build_messages(bs, imgs, [f"t={i * 2}.0s" for i in range(len(imgs))],
                          with_few_shot=True, with_alert=True, with_actions=True)
    text = proc.apply_chat_template(msgs, tokenize=False,
                                    add_generation_prompt=True)
    try:
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = proc(text=[text] * batch, images=image_inputs * batch,
                      videos=(video_inputs or []) * batch if video_inputs else None,
                      padding=True, return_tensors="pt").to(model.device)
    except Exception:
        inputs = proc(text=[text] * batch, images=imgs * batch,
                      padding=True, return_tensors="pt").to(model.device)
    n_prompt = int(inputs.input_ids.shape[1])
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=n_tokens, do_sample=False)
    torch.cuda.synchronize()
    dt = time.time() - t0
    n_out = int(gen.shape[1]) - n_prompt
    return {"batch": batch, "prompt_tokens": n_prompt, "out_tokens": n_out,
            "sec": round(dt, 3),
            "per_window_sec": round(dt / batch, 3),
            "tok_per_sec": round(n_out * batch / dt, 1) if dt > 0 else 0.0}


def load_compiled(model_name: str):
    """用 torch.compile 加载：目的是消掉每步的 CPU 开销。

    ⚠️ 编译有成本（首次调用要几分钟），所以这里用 reduce-overhead
       （会启用 CUDA Graphs，正是 vLLM 提速的关键手段之一）。
       编译失败不影响结论 —— 会如实报错，不假装成功。
    """
    import torch

    model, proc, load_sec = load_model(model_name, attn="sdpa",
                                       use_device_map=False)
    t0 = time.time()
    model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    return model, proc, load_sec, round(time.time() - t0, 1)


def run_config(tag: str, model_name: str, bs, images: list, *,
               attn: str | None, use_device_map: bool, n_out: int = 220,
               compile_model: bool = False, static_cache: bool = False,
               batch: int = 1) -> dict:
    import torch

    print(f"\n{'=' * 74}\n[{tag}] attn={attn or '(未指定)'} "
          f"device_map={use_device_map} compile={compile_model} "
          f"static_cache={static_cache} batch={batch}\n{'=' * 74}", flush=True)
    model = proc = None
    try:
        if compile_model:
            model, proc, load_sec, comp_sec = load_compiled(model_name)
            info = model_info(getattr(model, "_orig_mod", model))
            info["compile_sec"] = comp_sec
        else:
            model, proc, load_sec = load_model(model_name, attn=attn,
                                               use_device_map=use_device_map)
            info = model_info(model)
        print(f"  加载 {load_sec:.1f}s  {json.dumps(info, ensure_ascii=False)}",
              flush=True)
        r = {"tag": tag, "load_sec": round(load_sec, 1), "info": info}
        if static_cache:
            r["text"] = bench_text_decode(model, proc)
        else:
            r["text"] = bench_text_decode(model, proc)
        print(f"  纯文本解码：{r['text']['tok_per_sec']} tok/s "
              f"（{r['text']['out_tokens']} token / {r['text']['sec']}s）",
              flush=True)

        if batch > 1:
            bench_batch(model, proc, bs, images, batch=2, n_tokens=16)  # 预热
            r["batch"] = bench_batch(model, proc, bs, images, batch=batch,
                                     n_tokens=n_out)
            b = r["batch"]
            print(f"  批量 {batch} 窗：总 {b['sec']}s  "
                  f"均摊 {b['per_window_sec']}s/窗  "
                  f"吞吐 {b['tok_per_sec']} tok/s", flush=True)
            return r

        r["vision"] = [bench_vision(model, proc, images, n_images=n,
                                    labels=[]) for n in (0, 1, 4)]
        for v in r["vision"]:
            print(f"  视觉 prefill {v['n_images']} 图：{v['sec']}s "
                  f"（prompt {v['prompt_tokens']} token）", flush=True)
        bench_real_window(model, proc, bs, images, n_tokens=16)   # 预热
        r["window"] = bench_real_window(model, proc, bs, images, n_tokens=n_out)
        print(f"  真实窗口：{r['window']['sec']}s  "
              f"输出 {r['window']['out_tokens']} token  "
              f"{r['window']['tok_per_sec']} tok/s  "
              f"prompt {r['window']['prompt_tokens']}", flush=True)
        r["window2"] = bench_real_window(model, proc, bs, images, n_tokens=n_out)
        print(f"  真实窗口(第二次)：{r['window2']['sec']}s  "
              f"{r['window2']['tok_per_sec']} tok/s", flush=True)
        return r
    except Exception as e:
        print(f"  ❌ 配置 {tag} 失败：{type(e).__name__}: {e}", flush=True)
        print(traceback.format_exc()[:1200], flush=True)
        return {"tag": tag, "error": f"{type(e).__name__}: {e}"}
    finally:
        try:
            del model
            del proc
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--behaviors", default=None)
    ap.add_argument("--configs", default="A,B",
                    help="要测的配置：A=现状(dtype+device_map) B=显式sdpa+to(cuda)")
    args = ap.parse_args()

    sys.path.insert(0, args.pkg)
    from civility.config import load_behaviors

    bpath = args.behaviors or os.path.join(args.pkg, "behaviors.json")
    bs = load_behaviors(bpath)

    # 用作业包里的真实帧当输入（14 张，挑 4 张不同的）
    cand = sorted(glob.glob(os.path.join(args.pkg, "images", "*.jpg")))
    if len(cand) < 4:
        print("⚠️ 包内图片不足 4 张，视觉测试会退化为重复同一张")
    from PIL import Image

    images = []
    for p in (cand[:4] or cand):
        images.append(Image.open(p).convert("RGB"))
    while len(images) < 4:
        images.append(images[-1])
    print(f"输入图片：{[os.path.basename(p) for p in cand[:4]]}")

    out = {"model": args.model, "results": []}
    # ⚠️ 配置列表用 `+` 或 `,` 分隔都认。
    #   踩过的坑：从 sbatch 用 `--export=ALL,CONFIGS=B,C,D` 传进来时，
    #   **逗号是 SLURM 的变量分隔符** —— 它只会给 CONFIGS 赋 "B"，
    #   另外两个被当成变量名（无效）而丢弃。表现为"只跑了一个配置"，
    #   很容易误以为另外两个配置失败。所以 sbatch 里会把 + 换成 ,。
    cfgs = [c.strip().upper()
            for c in args.configs.replace("+", ",").split(",") if c.strip()]
    for c in cfgs:
        if c == "A":
            out["results"].append(run_config(
                "A-现状", args.model, bs, images, attn=None,
                use_device_map=True))
        elif c == "B":
            out["results"].append(run_config(
                "B-显式sdpa", args.model, bs, images, attn="sdpa",
                use_device_map=False))
        elif c == "C":
            # torch.compile + reduce-overhead：目的是消掉每步的 CPU 开销
            out["results"].append(run_config(
                "C-compile", args.model, bs, images, attn="sdpa",
                use_device_map=False, compile_model=True))
        elif c == "D":
            # 批量 4 窗：测吞吐，不测单流延迟
            out["results"].append(run_config(
                "D-batch4", args.model, bs, images, attn="sdpa",
                use_device_map=False, batch=4))
        elif c == "E":
            # 批量 4 窗 + compile：两者叠加
            out["results"].append(run_config(
                "E-batch4+compile", args.model, bs, images, attn="sdpa",
                use_device_map=False, compile_model=True, batch=4))
        elif c == "F":
            out["results"].append(run_config(
                "F-batch2", args.model, bs, images, attn="sdpa",
                use_device_map=False, batch=2))
        elif c == "G":
            out["results"].append(run_config(
                "G-batch8", args.model, bs, images, attn="sdpa",
                use_device_map=False, batch=8))

    print("\n" + "=" * 74)
    print("对照汇总")
    print("=" * 74)
    print(f"{'配置':<18}{'dtype':<10}{'文本tok/s':>10}{'单窗sec':>9}"
          f"{'均摊/窗':>9}{'吞吐tok/s':>10}{'显存GB':>8}")
    for r in out["results"]:
        if r.get("error"):
            print(f"{r['tag']:<18}失败：{r['error'][:60]}")
            continue
        i = r.get("info", {})
        win = r.get("window") or {}
        bat = r.get("batch") or {}
        print(f"{r['tag']:<18}{i.get('dtype', '?'):<10}"
              f"{r.get('text', {}).get('tok_per_sec', 0):>10}"
              f"{win.get('sec', bat.get('sec', 0)):>9}"
              f"{bat.get('per_window_sec', 0):>9}"
              f"{bat.get('tok_per_sec', win.get('tok_per_sec', 0)):>10}"
              f"{i.get('vram_peak_gb', 0):>8}")

    res_path = os.path.join(os.environ.get("BENCH_OUT_DIR", "."),
                            "bench_result.json")
    try:
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n结果已写入 {res_path}")
    except OSError as e:
        print(f"（结果落盘失败：{e}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

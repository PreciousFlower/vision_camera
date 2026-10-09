#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""查集群推理栈的"该快不快"问题：torch/transformers 版本、注意力后端、模型 dtype。

为什么值得单独查：
    A800 上 4B 模型 batch=1 解码，实测只有 35.8 token/s。
    理论值（显存带宽约 2TB/s ÷ 8GB 权重）应该在 200 token/s 量级 ——
    也就是说现在只跑到约 14%，很可能不是"模型天生慢"，而是**配置没生效**：
      · `from_pretrained(dtype=...)` 在旧版 transformers 上会被**静默忽略**
        → 模型以 fp32 加载 → 权重翻倍、解码速度减半（而且日志里只warning）
      · 没指定 attn_implementation → 可能退化成 eager 注意力（慢好几倍）
    这两项一旦确认，提速是**免费的**（不改模型、不改流程）。
    而"能不能做到 1 秒 1 个结论"完全取决于这个数。

用法（需要 VPN）：
    python tools/probe_infer_stack.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from civility.config import load_settings          # noqa: E402
from webui.cluster_client import ClusterClient     # noqa: E402

PY = r"""
import json, sys
out = {}
try:
    import torch
    out["torch"] = torch.__version__
    out["cuda"] = torch.version.cuda
    out["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
        p = torch.cuda.get_device_properties(0)
        out["vram_gb"] = round(p.total_memory / 1024**3, 1)
        out["capability"] = f"{p.major}.{p.minor}"
        # 显存带宽决定 batch=1 解码速度上限（这里只报算力，带宽靠外部常识）
        out["bf16_tflops_est"] = None
except Exception as e:
    out["torch_err"] = f"{type(e).__name__}: {e}"
try:
    import transformers
    out["transformers"] = transformers.__version__
    # 关键：这个版本认不认 `dtype=` 这个参数名（旧版只认 torch_dtype，dtype 会被忽略）
    import inspect
    sig = inspect.signature(transformers.PreTrainedModel.from_pretrained)
    params = list(sig.parameters)
    out["from_pretrained_has_dtype"] = "dtype" in params
    out["from_pretrained_has_torch_dtype"] = "torch_dtype" in params
    out["from_pretrained_has_attn"] = "attn_implementation" in params
except Exception as e:
    out["tf_err"] = f"{type(e).__name__}: {e}"
for m in ("flash_attn", "qwen_vl_utils", "accelerate", "sentencepiece"):
    try:
        mod = __import__(m)
        out[m] = getattr(mod, "__version__", "ok")
    except Exception:
        out[m] = None
print(json.dumps(out, ensure_ascii=False))
"""


def main() -> int:
    cc = load_settings().get("cluster") or {}
    cl = ClusterClient(workdir=cc["remote_workdir"], python=cc["remote_python"],
                       model_root=cc["remote_model_root"])
    ok, host = cl.check_vpn()
    if not ok:
        print(f"❌ 连不上集群（{host}）")
        return 2
    print(f"✅ {host}")
    rc, out, err = cl.run(f"{cc['remote_python']} -c {chr(39)}{PY}{chr(39)}",
                          timeout=180)
    print(f"rc={rc}")
    print((out or "").strip() or (err or "").strip()[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

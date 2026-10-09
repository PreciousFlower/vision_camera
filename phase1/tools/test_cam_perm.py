#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证"摄像头找不到"的提示语会说真话（用户实测被误导过）。

背景（用户排查了很久才定位）：
    网页服务跑在**受限令牌（沙箱）**下时，摄像头枚举会被系统拒绝，
    返回"一个设备都没有"。而旧提示语写的是
    「没找到可用摄像头；也可能是被其他程序占用」——
    把权限问题说成了硬件/占用问题，让人去查驱动、关微信，方向完全错。
    用户自己启动（非受限）时摄像头就正常（index 0, 640×480, MSMF）。

这个测试同时覆盖两种情况，两边都要说对话：
    · 受限环境 → 提示必须以"权限原因"开头，并给出可执行的解决办法
    · 正常环境 → 不许无端甩锅给权限（那会把用户带到另一个错方向）

用法：
    python tools/test_cam_perm.py                 # 打 127.0.0.1:8848
    python tools/test_cam_perm.py --port 8850
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from webui.winenv import is_restricted, token_info  # noqa: E402

OK, NO = "✅", "❌"
results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    results.append((ok, name, detail))
    print(f"  {OK if ok else NO} {name}" + (f" — {detail}" if detail else ""),
          flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None)
    ap.add_argument("--port", type=int, default=8850)
    args = ap.parse_args()
    base = args.base or f"http://127.0.0.1:{args.port}"

    info = token_info()
    print("=" * 74)
    print(f"摄像头提示语真实性验证   {base}")
    print(f"本进程令牌：受限 SID={info['restricted_sids']} "
          f"完整性={info['integrity'] or '?'} → "
          f"{'受限（沙箱）' if is_restricted() else '正常'}")
    print("=" * 74)

    try:
        with urllib.request.urlopen(base + "/api/live/cameras",
                                    timeout=120) as r:
            d = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        print(f"{NO} 打不开 {base}/api/live/cameras：{type(e).__name__}: {e}")
        print("   （先把网页服务起起来，或用 --port 指定端口）")
        return 2

    cams = d.get("cameras") or []
    hint = str(d.get("hint") or "")
    tried = d.get("tried") or []
    print(f"\n接口返回：cameras={cams}  restricted={d.get('restricted')}")
    print(f"提示语：{hint[:120]}{'…' if len(hint) > 120 else ''}\n")

    check(d.get("ok") is True, "接口本身正常返回")
    check("restricted" in d,
          "接口带上了「是否受限令牌」这个事实（否则前端没法解释）",
          f"restricted={d.get('restricted')}")
    check(bool(tried) or bool(cams),
          "给出了逐次尝试的明细（试了哪些后端/索引）",
          f"{len(tried)} 条：{tried[:2]}{'…' if len(tried) > 2 else ''}")

    if cams:
        # 正常环境（用户启动的那种）：找到了摄像头
        c = cams[0]
        check(all(k in c for k in ("index", "width", "height")),
              "找到的摄像头带索引与分辨率",
              f"#{c.get('index')} {c.get('width')}×{c.get('height')}")
        check(not hint, "找到了就不该再报「没找到」", f"hint={hint[:40]!r}")
        check("权限受限" not in hint, "正常环境下不甩锅给权限")
    else:
        # 没找到：必须按真实性解释
        if d.get("restricted"):
            check("权限" in hint and "受限" in hint,
                  "受限环境下直接点明是权限原因（不是硬件/占用）",
                  "提示语里出现「权限」「受限」")
            check(hint.index("权限") < max(1, hint.find("占用"))
                  if "占用" in hint else True,
                  "权限原因排在「被占用」之前（按可能性排序）")
            check("webui.server" in hint,
                  "给出了可执行的解决办法（用非受限的方式启动）")
            check("硬件" in hint or "不是硬件" in hint,
                  "明确说明这不是硬件问题（避免用户去查驱动）")
        else:
            check("权限" not in hint or "不受限" in hint,
                  "非受限环境下不谎报权限问题")
            check(any(k in hint for k in ("占用", "索引", "驱动", "隐私")),
                  "给出了正常环境下的排查方向（占用/索引/驱动/隐私）")

    print("\n" + "=" * 74)
    bad = [x for x in results if not x[0]]
    if bad:
        print(f"{NO} {len(bad)} 项未通过：")
        for _, n, dd in bad:
            print(f"   · {n} — {dd}")
        return 1
    print(f"{OK} 全部通过 —— 提示语与真实原因一致"
          f"（{'受限环境' if d.get('restricted') else '正常环境'}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

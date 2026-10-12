#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 node 检查单页前端的 JS 语法（浏览器里出语法错是"整页白屏"，很难查）。

为什么要这一步：
    `tools/check_ui.py` 只做结构检查（id、按钮、事件名），**查不出语法错**。
    而前端一旦有语法错，整个 <script> 都不会执行 —— 页面看起来只是"没反应"，
    没有任何报错可见。用 node --check 几秒钟就能排掉这一类问题。

用法：
    python tools/check_js.py
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HTML = ROOT / "webui" / "static" / "index.html"


def main() -> int:
    node = None
    for cand in ("node", "node.exe"):
        try:
            subprocess.run([cand, "--version"], capture_output=True, check=True)
            node = cand
            break
        except Exception:
            continue
    if node is None:
        print("⚠️ 找不到 node，跳过 JS 语法检查（建议安装 node 以便能查这类错）")
        return 0

    html = HTML.read_text(encoding="utf-8")
    blocks = re.findall(r"<script[^>]*>(.*?)</script>", html, re.S)
    if not blocks:
        print("❌ 没找到 <script> 块")
        return 1
    print(f"检查 {HTML.name}：{len(blocks)} 个 script 块，"
          f"共 {sum(len(b) for b in blocks)} 字符")

    bad = 0
    for src in re.findall(r'<script[^>]+src=["\x27](/static/[^"\x27]+\.js)["\x27]', html):
        path=HTML.parent/src.removeprefix('/static/')
        r=subprocess.run([node,'--check',str(path)],capture_output=True,text=True)
        if r.returncode:
            bad+=1;print(f'  ❌ {path.name}: {(r.stderr or "")[:600]}')
        else:print(f'  ✅ {path.name}：语法正确')
    for i, js in enumerate(blocks, 1):
        # 去掉可能的 src 引入的空块
        if not js.strip():
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                         encoding="utf-8") as f:
            f.write(js)
            tmp = f.name
        r = subprocess.run([node, "--check", tmp], capture_output=True, text=True)
        if r.returncode == 0:
            print(f"  ✅ script #{i}：语法正确（{len(js)} 字符）")
        else:
            bad += 1
            print(f"  ❌ script #{i}：语法错误")
            print("     " + (r.stderr or "").strip()[:600])
        Path(tmp).unlink(missing_ok=True)

    if bad:
        print(f"\n❌ {bad} 个 script 块有语法错 —— 页面会整段不执行")
        return 1
    print("\n✅ JS 语法检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

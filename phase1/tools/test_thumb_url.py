#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""回归：页面缩略图 URL —— 直接跑页面里**真实的那个函数**。

用户反馈："ai动作识别中的图片始终是同一张，并不是人做对应动作的那一帧。"

服务端侧已由 tools/test_frame_index.py 覆盖（帧号不再靠时间戳推算）。
这一支测**前端那一半**，判据是 URL 本身：

    A. 不同窗口 → URL 必须不同
       这是"始终是同一张"的直接成因：URL 相同 + 服务端不返回缓存头
       → 浏览器启发式缓存复用第一张，后面所有卡片都是它。
    B. 同一窗口重复渲染 → URL 必须相同
       否则浏览器缓存完全失效，每张卡片都重新下载几十 KB。
       （"修好缓存"不等于"关掉缓存"—— 要对症。）
    C. 取的是**中间帧**（4 帧 → 第 2 帧），不是第 0 帧。
       第 0 帧是窗口开头那一帧，动作往往在后面几帧里。
    D. 后端没给真实帧号（frame0 = -1，旧记录）时不能返回 undefined/NaN。

⚠️ 为什么要抠页面里的源码来跑，而不是在测试里重写一遍：
    重写一遍只能证明"我以为的公式是对的"，证明不了**页面上真在用的那个函数**
    是对的。这类"测试和实现各写一份、实现改坏了测试还是绿的"是本项目踩过的坑。
"""
from __future__ import annotations

import json
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


def extract_function(src: str, name: str) -> str:
    """从页面源码里抠出一个顶层 function 的完整定义（按花括号配平）。"""
    m = re.search(rf"^function\s+{re.escape(name)}\s*\(", src, re.M)
    if not m:
        raise SystemExit(f"❌ 页面里找不到 function {name}()")
    i = src.index("{", m.end() - 1)
    depth = 0
    for j in range(i, len(src)):
        c = src[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[m.start(): j + 1]
    raise SystemExit(f"❌ function {name}() 的花括号不配平")


HARNESS = """
// 下面这段由 tools/test_thumb_url.py 生成：把页面里真实的函数抠出来跑。
%(fn)s

const cases = %(cases)s;
const out = cases.map(c => ({
  label: c.label,
  url: frameThumbUrl(c.jid, c.d),
}));
console.log(JSON.stringify(out));
"""


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
        print("⚠️ 找不到 node，跳过（建议安装 node）")
        return 0

    src = HTML.read_text(encoding="utf-8")
    fn = extract_function(src, "frameThumbUrl")
    print(f"从页面的 {HTML.name} 抠出 frameThumbUrl（{len(fn)} 字符）\n")

    cases = [
        {"label": "窗口A（frame0=0, 4帧）",
         "jid": "job1", "d": {"window_id": "job1__w00000", "t_start": 0.0,
                              "frames": 4, "frame0": 0}},
        {"label": "窗口B（frame0=8, 4帧）",
         "jid": "job1", "d": {"window_id": "job1__w00004", "t_start": 16.0,
                              "frames": 4, "frame0": 8}},
        {"label": "窗口C（frame0=40, 4帧）",
         "jid": "job1", "d": {"window_id": "job1__w00020", "t_start": 80.0,
                              "frames": 4, "frame0": 40}},
        {"label": "同窗口A再来一次",
         "jid": "job1", "d": {"window_id": "job1__w00000", "t_start": 0.0,
                              "frames": 4, "frame0": 0}},
        {"label": "旧记录（无 frame0 字段）",
         "jid": "job1", "d": {"window_id": "job1__w00007", "t_start": 14.0,
                              "frames": 4}},
        {"label": "3 帧窗口（frame0=6）",
         "jid": "job1", "d": {"window_id": "job1__w00003", "t_start": 12.0,
                              "frames": 3, "frame0": 6}},
    ]
    js = HARNESS % {"fn": fn, "cases": json.dumps(cases, ensure_ascii=False)}
    tmp = Path(tempfile.gettempdir()) / "dsh_thumb_url_test.js"
    tmp.write_text(js, encoding="utf-8")
    r = subprocess.run([node, str(tmp)], capture_output=True, text=True,
                       encoding="utf-8")
    if r.returncode != 0:
        print("❌ node 执行失败：\n" + (r.stderr or "")[:1500])
        return 1
    rows = json.loads(r.stdout.strip().splitlines()[-1])

    failed: list[str] = []

    def check(cond: bool, label: str, extra: str = "") -> None:
        print(f"  {'✅' if cond else '❌'} {label}" + (f"　{extra}" if extra else ""))
        if not cond:
            failed.append(label)

    by_label = {r["label"]: r["url"] for r in rows}
    print("生成的 URL：")
    for r in rows:
        print(f"  {r['label']}: {r['url']}")
    print()

    ua = by_label["窗口A（frame0=0, 4帧）"]
    ub = by_label["窗口B（frame0=8, 4帧）"]
    uc = by_label["窗口C（frame0=40, 4帧）"]
    ua2 = by_label["同窗口A再来一次"]

    print("[A] 不同窗口必须给出不同 URL（这是缓存复用的直接成因）")
    check(len({ua, ub, uc}) == 3, "三个窗口的 URL 互不相同")

    print("\n[B] 同一窗口必须给出相同 URL（否则缓存完全失效、每张都重下）")
    check(ua == ua2, "同一窗口重复渲染 → URL 相同")

    print("\n[C] 取中间帧，而不是第 0 帧")
    # 窗口A：frames=4, frame0=0 → mid=1 → /1?f=1
    m = re.match(r"^/api/job/([^/]+)/frames/([^/]+)/(\d+)\?f=(\d+)$", ua)
    check(bool(m), f"URL 形状符合预期（含缓存破坏参数 f=）", ua)
    if m:
        check(m.group(3) == "1",
              f"4 帧窗口取的是第 {m.group(3)} 帧（= 中间帧 1，不是第 0 帧）")
        check(m.group(4) == "1",
              f"缓存破坏参数 f={m.group(4)}（= frame0 + mid = 0 + 1）")
    m3 = re.match(r"^/api/job/([^/]+)/frames/([^/]+)/(\d+)\?f=(\d+)$",
                  by_label["3 帧窗口（frame0=6）"])
    if m3:
        check(m3.group(3) == "1", f"3 帧窗口取第 {m3.group(3)} 帧（中间帧 1）")
        check(m3.group(4) == "7", f"3 帧窗口 f={m3.group(4)}（= 6 + 1）")

    print("\n[D] 旧记录（没有 frame0）不能产出 undefined / NaN")
    uold = by_label["旧记录（无 frame0 字段）"]
    check("undefined" not in uold and "NaN" not in uold,
          "URL 里没有 undefined / NaN", uold)
    check(bool(re.match(r"^/api/job/job1/frames/job1__w00007/\d+\?f=\d+$", uold)),
          "回退路径也给出了合法的缓存破坏参数", uold)

    print("\n" + "=" * 62)
    if failed:
        print(f"❌ {len(failed)} 项失败：")
        for f in failed:
            print(f"   · {f}")
        return 1
    print("✅ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

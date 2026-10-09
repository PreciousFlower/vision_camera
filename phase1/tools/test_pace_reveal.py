#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证"逐条显示"真的是一条一条、且不会越积越多（用 node 跑真实的页面代码）。

用户的要求是明确的："不要一批 6~7 条一起出，就要一条一条出。"
这件事只靠肉眼看页面很难判断（要盯着等几十秒），而如果节奏算法写错，
表现是"一阵一阵"或"越积越多"，用户只会觉得"还是卡"。
所以把页面里**真正的**那两个函数（`paceDelayFor` / `updatePace`）抠出来，
用假时钟喂给它们真实的到达序列，量一下"显示时刻"的间隔。

关键判据：
    1. 一批 8 条到达后，显示时刻应当是**均匀铺开**的，不是挤在一起
    2. 铺开的总时长应当≈批间隔（8 条 / 11.5 秒 → 1.44 秒一条），
       这样稳态下显示速度=产出速度，积压不会增长
    3. 真的落后（积压超两批）时必须加速追，不能无限滞后

用法：
    python tools/test_pace_reveal.py
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
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HTML = ROOT / "webui" / "static" / "index.html"

results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    results.append((ok, name, detail))
    print(f"  {'✅' if ok else '❌'} {name}" + (f" — {detail}" if detail else ""),
          flush=True)
    return ok


def extract_functions(src: str, names: list[str]) -> str:
    """从页面源码里抠出指定函数（按大括号配平），保证测的是真实代码。"""
    out = []
    for name in names:
        m = re.search(r"^function\s+" + re.escape(name) + r"\s*\([^)]*\)\s*\{",
                      src, re.M)
        if not m:
            raise SystemExit(f"❌ 页面里找不到函数 {name}（改名了？）")
        i = m.end() - 1
        depth = 0
        for j in range(i, len(src)):
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
                if depth == 0:
                    out.append(src[m.start():j + 1])
                    break
    return "\n".join(out)


def main() -> int:
    html = HTML.read_text(encoding="utf-8")
    js = extract_functions(html, ["paceDelayFor", "updatePace"])

    harness = js + r"""
// ── 假时钟 + 假队列：驱动页面里**真实的**节奏函数 ──
let now = 0;
const reveals = [];
let monQueue = [], monPaceMs = 1400, monPaceTimer = null,
    monBatchSize = 8, monPrevN = 0, monPrevSpan = 0;
let injected = 0;                         // 实际注入了多少条（不靠手算）
const BATCH = 8, INTERVAL = 11500, NBATCH = 5;
// 跑 5 批 + 一段"排空期"（最后一批到达后要留够时间把积压显示完）
const TOTAL = NBATCH * INTERVAL + 3 * INTERVAL;

function schedule(){
  if(monPaceTimer === null && monQueue.length)
    monPaceTimer = now + paceDelayFor(monQueue.length, monPaceMs, monBatchSize);
}

for(let t = 0; t < TOTAL; t += 50){
  now = t;
  if(t % INTERVAL === 0){                 // 新一批到达（页面里由 SSE 触发）
    monPrevN = BATCH; monPrevSpan = INTERVAL / 1000; updatePace();
    for(let i = 0; i < BATCH; i++){ monQueue.push(1); injected++; }
    schedule();
  }
  if(monPaceTimer !== null && now >= monPaceTimer){   // 到点显示一条
    monPaceTimer = null;
    if(monQueue.length){ monQueue.shift(); reveals.push(now); }
    schedule();
  }
}

const gaps = [];
for(let i = 1; i < reveals.length; i++) gaps.push(reveals[i] - reveals[i-1]);
const sortedGaps = gaps.slice().sort((a, b) => a - b);
console.log(JSON.stringify({
  reveals: reveals.length,
  expected: injected,
  pace_ms: monPaceMs,
  max_gap_ms: gaps.length ? Math.max(...gaps) : 0,
  median_gap_ms: sortedGaps.length
    ? sortedGaps[Math.floor(sortedGaps.length / 2)] : 0,
  first8_span_ms: reveals.length >= 8 ? reveals[7] - reveals[0] : 0,
  backlog_end: monQueue.length,
}));
"""
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                     encoding="utf-8") as f:
        f.write(harness)
        tmp = f.name
    r = subprocess.run(["node", tmp], capture_output=True, text=True)
    Path(tmp).unlink(missing_ok=True)
    if r.returncode != 0:
        print("❌ node 执行失败：\n" + (r.stderr or "")[:800])
        return 1
    d = json.loads((r.stdout or "{}").strip().splitlines()[-1])

    print("=" * 74)
    print("逐条显示节奏验证（用页面里真实的 paceDelayFor/updatePace）")
    print(f"场景：每批 {8} 条、每 {11500/1000:.1f} 秒来一批，共 5 批")
    print("=" * 74)
    print(f"  显示条数 {d['reveals']}（预期 {d['expected']}）"
          f"　每条间隔：中位 {d['median_gap_ms']}ms / 最大 {d['max_gap_ms']}ms")
    print(f"  首批 8 条的显示时长（第1条→第8条）= {d['first8_span_ms']}ms")
    print(f"  结束时待显示积压 = {d['backlog_end']} 条\n")

    # 不变量：显示出来的 + 还排队的 = 收到的（一条都不能丢）
    check(d["reveals"] + d["backlog_end"] == d["expected"],
          "每条结论都被显示或被排队（没有丢）",
          f"已显示 {d['reveals']} + 待显示 {d['backlog_end']} = "
          f"{d['reveals'] + d['backlog_end']} / 收到 {d['expected']}")
    check(d["max_gap_ms"] < 6000,
          "不存在「长时间空等」（不再是一阵一阵）",
          f"最大间隔 {d['max_gap_ms']}ms（批间隔是 11500ms）")
    check(d["first8_span_ms"] > 6000,
          "一批 8 条被**铺开**显示，而不是挤在一瞬间",
          f"跨 {d['first8_span_ms']}ms（挤在一起会是几十毫秒）")
    # ⚠️ 稳态不变量是"积压**不超过一批**"（显示滞后有界），不是"必须为 0"：
    #    只要产出与显示速率相同，积压就会稳定在 0~1 批之间波动。
    check(d["backlog_end"] <= 8,
          "积压有界（显示滞后不超过一批）",
          f"结束时积压 {d['backlog_end']} 条（≤8 即合格）")
    check(d["median_gap_ms"] and abs(d["median_gap_ms"] - d["pace_ms"]) < 300,
          "逐条间隔与计算出的节奏一致（匀速，不是忽快忽慢）",
          f"中位 {d['median_gap_ms']}ms ≈ 目标 {d['pace_ms']}ms")

    print("\n" + "=" * 74)
    bad = [x for x in results if not x[0]]
    if bad:
        print(f"❌ {len(bad)} 项未通过")
        return 1
    print("✅ 全部通过 —— 页面会把批量结果匀速铺成「一条一条」")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

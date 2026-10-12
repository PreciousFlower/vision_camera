#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查前端页面的结构性健康 —— 不需要浏览器，也不需要跑模型。

为什么值得做：
    界面重写后很容易出现"某个按钮的处理器丢了""某个容器 id 打错了"，
    浏览器里只表现为"点了没反应"，而这在服务端日志里毫无痕迹。
    本脚本用静态检查抓住这类问题：每个按钮都有处理器、每个 getElementById
    目标都真实存在、每个 fetch 的路径后端都有实现。

用法：
    python tools/check_ui.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
HTML = PROJECT / "webui" / "static" / "index.html"
SERVER = PROJECT / "webui" / "server.py"

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'✅' if ok else '❌'} {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def main() -> int:
    if not HTML.exists():
        print(f"❌ 找不到 {HTML}")
        return 1
    html = HTML.read_text(encoding="utf-8")
    server = SERVER.read_text(encoding="utf-8") if SERVER.exists() else ""

    print("=" * 74)
    print("前端结构健康检查")
    print("=" * 74)

    # ── 1. 页面区块 ──
    tabs = re.findall(r'data-tab="([^"]+)"', html)
    sections = re.findall(r'<section id="tab-([^"]+)"', html)
    check("导航项与页面区块一一对应",
          set(tabs) == set(sections),
          f"导航 {tabs} / 区块 {sections}")

    # ── 2. 每个 id="x" 在选择器里引用时都存在 ──
    ids = set(re.findall(r'id="([^"]+)"', html))
    used = set(re.findall(r"\$\('#([A-Za-z0-9_-]+)'\)", html))
    missing = sorted(used - ids)
    check("所有 $('#id') 选择器都能找到元素",
          not missing,
          f"缺失：{missing}" if missing else f"引用 {len(used)} 个 id 全部存在")

    # ── 3. 每个按钮都有 on_click 处理器 ──
    btn_ids = re.findall(r'<button[^>]*id="([^"]+)"', html)
    # 两种绑定写法都算：$('#id').onclick = ... 和 $('#id')?.addEventListener(...)
    # （成本横幅里的按钮是动态生成的，只能用 addEventListener —— 不能因此误报）
    handled = set(re.findall(r"\$\('#([A-Za-z0-9_-]+)'\)\s*[?]?\s*\.\s*onclick", html))
    handled |= set(re.findall(
        r"\$\('#([A-Za-z0-9_-]+)'\)\s*[?]?\s*\.\s*addEventListener", html))
    no_handler = sorted(set(btn_ids) - handled)
    check("所有按钮都绑定了点击处理器",
          not no_handler,
          f"没绑的：{no_handler}" if no_handler else f"{len(btn_ids)} 个按钮都有处理器")

    # ── 4. fetch 的接口路径后端都有实现 ──
    # ⚠️ 要处理四种写法，否则会误报/漏报（实测踩到）：
    #     api('/api/state')                  —— 纯字面量
    #     api(`/api/job/${jid}/stream`)      —— 模板串
    #     api('/api/detect/' + jid)          —— 字符串拼接
    #     const u = `/api/job/${jid}/latest` —— 先赋给变量再 fetch（实时画面轮询就这么写的）
    #    前三种要抠静态前缀；第四种不在 api()/fetch() 里，
    #    所以这里干脆**凡是 '/api/...' 形式的字符串都算"前端会调的接口"**。
    #    代价是可能多算几个（比如注释里提到的路径），但宁多勿漏：
    #    漏掉一个就会让"接口对不上"这种低级问题溜到线上。
    called = set(re.findall(r"""['"`](/api/[^'"`]*)""", html))

    def shape(path: str) -> str:
        """把路径归一成"形状"：变量段和数字段都算 `*`。

        ⚠️ 为什么不能只做字符串比对（实测踩到）：
            前端 `/api/job/${jid}/frames/${wid}/0`
            路由 `/api/job/{job_id}/frames/{window_id}/{idx}`
            —— 变量名不同（jid vs job_id）、末段是字面量 `0` 而路由是 `{idx}`，
            逐字符比会误报"接口没实现"。比"形状"才对得上。
        """
        segs = []
        p = path.split("?")[0]
        # 末尾是 / 说明后面用 + 拼了参数（api('/api/stop/' + jid)）→ 补一个变量段
        if p.endswith("/"):
            p = p + "*"
        for s in p.split("/"):
            if re.fullmatch(r"\{[^}]*\}", s):        # 路由里的 {job_id}
                segs.append("*")
            elif re.fullmatch(r"(?:f_)?\$\{[^}]*\}", s):  # ${jid} or f_${index}
                segs.append("*")
            elif re.fullmatch(r"(?:\d+|f_\d+)", s):  # 帧序号 0 或证据 ID f_0
                segs.append("*")
            else:
                segs.append(s)
        return "/".join(segs)

    norm = {shape(c) for c in called}
    routes = {shape(r) for r in
              re.findall(r'@app\.(?:get|post|delete)\("(/api/[^"]+)"', server)}
    missing_routes = sorted(norm - routes)
    check("前端调用的接口后端都已实现",
          not missing_routes,
          f"缺失：{missing_routes}" if missing_routes
          else f"{len(norm)} 个接口全部有实现")

    # ── 5. 关键功能元素齐全 ──
    need = {
        "摄像头监测页": 'id="tab-monitor"',
        "监测源输入": 'id="m-src"',
        "开始监测按钮": 'id="m-start"',
        # 按钮按不动时必须说明原因 —— 实测有用户对着置灰按钮猜了很久
        "不可用原因提示": 'id="m-why"',
        "动作流容器": 'id="m-feed"',
        "实时指标": 'id="mm-people"',
        "告警策略选择": 'id="agg-mode"',
        "行为定义编辑器": 'id="yaml-box"',
        "推理服务控制": 'id="btn-start"',
        "状态徽标": 'id="c-svc"',
    }
    miss = [k for k, v in need.items() if v not in html]
    check("关键功能元素齐全", not miss, f"缺：{miss}" if miss else f"{len(need)} 项都在")

    # ── 6. SSE 事件类型前后端一致 ──
    # 后端用 Python dict 字面量（单引号或双引号都可能），前端用 === 比较。
    # 只在前端"多出"时才算问题；后端发了前端没处理只是浪费，不算错。
    be_events = set(re.findall(r"['\"]type['\"]:\s*['\"]([a-z]+)['\"]", server))
    fe_events = set(re.findall(r"d\.type\s*===\s*'([a-z]+)'", html))
    fe_events |= set(re.findall(r'd\.type\s*===\s*"([a-z]+)"', html))
    unknown = sorted(fe_events - be_events)
    check("前端处理的 SSE 事件类型后端都会发",
          not unknown,
          f"前端多出：{unknown}" if unknown
          else f"后端 {sorted(be_events)} / 前端处理 {sorted(fe_events)}")

    # ── 7. 大厂风格的视觉要素 ──
    style_bits = {
        "CSS 变量主题": "--accent",
        "侧边导航": "aside",
        "卡片布局": ".card",
        "指标卡": ".metric",
        "状态徽标": ".chip",
        "动作卡": ".person",
        "动效": "@keyframes",
    }
    miss2 = [k for k, v in style_bits.items() if v not in html]
    check("视觉体系完整", not miss2, f"缺：{miss2}" if miss2 else f"{len(style_bits)} 项视觉要素都在")

    print()
    if FAILS:
        print(f"❌ {len(FAILS)} 项失败：{FAILS}")
        return 1
    print("✅ 前端结构健康")
    return 0


if __name__ == "__main__":
    sys.exit(main())

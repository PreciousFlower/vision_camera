#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对 sbatch 脚本做一次粗粒度语法体检（本机没有 bash 时的兜底）。

⚠️ 为什么需要：sbatch 脚本只在集群上执行，本机 Windows 没有 bash，
   `bash -n` 跑不了。而脚本一旦有语法错，作业会**秒退**——
   在集群上排查这个要占卡、要排队，代价很高。
   这里检查最容易出错、且一错就整段失效的几件事：
     引号是否成对、if/for/case 是否闭合、变量是否在赋值前被引用、
     /dev/tcp 探针写法、后台进程与 trap 是否配套。

用法：
    python tools/check_sbatch.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SBATCH = ROOT / "cluster" / "serve-vlm.sbatch"
results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    results.append((ok, name, detail))
    print(f"  {'✅' if ok else '❌'} {name}" + (f" — {detail}" if detail else ""),
          flush=True)
    return ok


def main() -> int:
    if not SBATCH.exists():
        print(f"❌ 找不到 {SBATCH}")
        return 2
    src = SBATCH.read_text(encoding="utf-8")
    print("=" * 72)
    print(f"sbatch 脚本体检：{SBATCH.name}")
    print("=" * 72)

    # 去掉注释行再查引号（注释里的中文引号会干扰计数）
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))

    for q, label in (('"', "双引号"), ("'", "单引号")):
        n = code.count(q)
        check(n % 2 == 0, f"{label}成对", f"出现 {n} 次")

    for kw, cl in (("if", "fi"), ("for", "done"), ("case", "esac"),
                   ("while", "done")):
        a = len(re.findall(r"^\s*" + kw + r"\b", code, re.M))
        b = len(re.findall(r"^\s*" + cl + r"\b", code, re.M))
        if a:
            check(a <= b, f"{kw} … {cl} 数量合理", f"{kw} {a} 个 / {cl} {b} 个")

    # 关键片段
    check("WORKERS=" in src, "worker 数可配置（WORKERS=）")
    check(re.search(r"trap\s+\w+\s+EXIT", src) is not None,
          "有 EXIT trap（scancel 时收掉 worker，不留孤儿占显存）")
    check("kill 0" in src, "trap 里 kill 0（连带清理所有子进程）")
    # ★ 必须等"模型就绪"而不是"端口能连"（实测踩到）：
    #   vlm_service 是先起 HTTP、再在后台线程加载模型，
    #   只等端口会让网关把请求送给还在加载的 worker，
    #   表现为"第一批请求全部 RemoteDisconnected 失败"。
    check("wait_ready.py" in src,
          "等 worker 的**模型**真正就绪（不是只等端口监听）")
    check("/dev/tcp" not in src,
          "不再用 /dev/tcp 只探端口（那种判断会放过未就绪的 worker）")
    check(re.search(r"GW=\"pkg/cluster/", src) is not None,
          "网关路径优先取 pkg/cluster 下那份（作业包一定会更新它）")
    check(src.count("vlm_service.py") >= 2,
          "worker 启动命令齐全（多 worker 分支）",
          f"出现 {src.count('vlm_service.py')} 次")
    check("--bind 0.0.0.0" in src, "服务绑 0.0.0.0（否则隧道连不上）")
    check("exec " in src, "单 worker 分支用 exec（少一层进程）")
    # 变量引用顺序：BACKENDS 在拼接前必须先初始化
    i_init = src.find('BACKENDS=""')
    i_use = src.find("BACKENDS=${BACKENDS:+")
    check(i_init != -1 and (i_use == -1 or i_init < i_use),
          "BACKENDS 先初始化再拼接", f"初始化@{i_init} 拼接@{i_use}")
    # PIDS 同理
    i_pids = src.find('PIDS=""')
    i_puse = src.find('PIDS="$PIDS')
    check(i_pids != -1 and (i_puse == -1 or i_pids < i_puse),
          "PIDS 先初始化再拼接")

    print("\n" + "=" * 72)
    bad = [x for x in results if not x[0]]
    if bad:
        print(f"❌ {len(bad)} 项未通过：")
        for _, n, d in bad:
            print(f"   · {n} — {d}")
        return 1
    print("✅ 体检通过（注意：这只查结构性错误，真正的执行验证仍在集群上）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

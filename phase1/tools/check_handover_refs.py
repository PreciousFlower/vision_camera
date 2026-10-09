"""核查交接文档里提到的每个工具/文件是否真实存在。

为什么值得写这个：
    交接文档最要命的错误不是"写得不好"，而是**指向不存在的东西** ——
    接手方照着敲命令，得到 "No such file or directory"，然后开始怀疑整个文档。
    我自己在这次交接里就凭记忆写错过两次（`tools/test_capture.py` 其实在 `tests/`、
    `tools/smoke_cluster_import.py` 同理），所以把核对做成机械检查。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# ⚠️ Windows 控制台默认 GBK，直接 print emoji/中文会 UnicodeEncodeError 崩掉。
#    本项目所有工具都要做这一步（这里曾经漏掉过）。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

PROJECT = Path(__file__).resolve().parents[1]      # phase1/
ROOT = PROJECT.parent                              # 仓库根（文档在这里）
# ⚠️ 跨仓库依赖：有文档引用另一个仓库里的文件（`aipc` 仓库）。
#    这类引用**不算坏链接**，但必须显式登记 —— 否则接手方在本地找不到会以为文档错了。
OTHER_REPOS = [Path(r"E:\vibecoding\aipc")]

# 检查这些文档里出现的 tools\xxx.py / tests\xxx.py / cluster\xxx / xxx.md
DOCS = ["HANDOVER.md", "README.md", "WEBUI-可视化界面.md", "PHASE1-开发流程.md"]

# 这些是**运行产物**（跑起来才有），不是仓库里的源文件 —— 引用它们不算错。
# 例如 runs/default/report.md 是 `civility.cli all` 跑完才生成的。
OUTPUT_NAMES = {"report.md", "manifest.jsonl", "last_port.txt"}

PAT = re.compile(
    r"(?:phase1[\\/])?(?:tools|tests)[\\/][A-Za-z0-9_]+\.py"
    r"|(?:phase1[\\/])?cluster[\\/][A-Za-z0-9_.\-]+"
    r"|(?:phase1[\\/])?(?:config|webui|civility)[\\/][A-Za-z0-9_./\\-]+\.(?:py|yaml|html)"
    r"|[A-Za-z0-9_\-]+\.md"
)

missing: list[tuple[str, str]] = []
checked = 0
for doc in DOCS:
    p = ROOT / doc
    if not p.exists():
        print(f"⚠️ 文档本身不存在：{doc}（检查 DOCS 列表与 ROOT 是否指对）")
        continue
    text = p.read_text(encoding="utf-8")
    for m in set(PAT.findall(text)):
        ref = m.replace("\\", "/").lstrip("/")
        if Path(ref).name in OUTPUT_NAMES:
            continue                    # 运行产物，跳过
        # 文档里写路径的方式不统一（有的带 phase1\，有的只写文件名，
        # 而 tju/ 和 runs/ 下的东西经常只写文件名）—— 都试一遍。
        name = Path(ref).name
        cands = [ROOT / ref, PROJECT / ref,
                 PROJECT / "tools" / name, PROJECT / "tests" / name,
                 ROOT / "tju" / name, PROJECT / "runs" / ref,
                 PROJECT / "runs" / "cluster" / name]
        for other in OTHER_REPOS:
            cands.append(other / ref)
            cands.append(other / "docs" / name)
        checked += 1
        if not any(c.exists() for c in cands):
            missing.append((doc, m))

print(f"共检查 {checked} 处引用（去重后），覆盖 {len(DOCS)} 份文档")
print(f"（跨仓库引用的候选位置也查了：{[str(o) for o in OTHER_REPOS]}）")
if missing:
    print(f"\n❌ {len(missing)} 处指向不存在的文件：")
    for doc, ref in sorted(missing):
        print(f"   {doc}: {ref}")
    sys.exit(1)
print("\n✅ 所有引用都指向真实存在的文件")

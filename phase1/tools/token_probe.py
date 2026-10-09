#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探测当前进程的 Windows 令牌：是不是"受限令牌"（沙箱）。

为什么需要它（用户实测踩到的坑）：
    在受限令牌下跑网页服务时，摄像头枚举会**静默失败**，
    报出来的却是我们自己写的"没找到可用摄像头；也可能是被其他程序占用"
    —— 这句话把"权限不足"误导成了"硬件/占用问题"，
    让人去查驱动、查微信占用，方向完全错。
    程序自己能判断令牌是否受限，就应该直接说出来。

用法：
    python tools/token_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 判据实现放在 webui/winenv.py —— 服务端要用同一套判断，
# 两处各写一份迟早会不一致（这个工具的早期版本就漏了 argtypes，
# 结果"读不到令牌"被误判成"没有问题"）。
from webui.winenv import is_restricted, token_info  # noqa: E402


def main() -> int:
    info = token_info()
    print("=" * 70)
    print("当前进程令牌探测")
    print("=" * 70)
    n = info["restricted_sids"]
    print(f"  受限 SID 数量 : {n}"
          + ("  ← 受限令牌（沙箱）" if n > 0 else
             "  ← 读不到（见下面备注）" if n < 0 else ""))
    print(f"  完整性级别    : {info['integrity'] or '（读不到）'}")
    print(f"  AppContainer  : {info['appcontainer']}")
    if info.get("note"):
        print(f"  备注          : {info['note']}")
    print()
    if is_restricted():
        print("❌ 这是**受限令牌**：摄像头枚举/打开很可能被拒绝，")
        print("   而失败信息看起来像'没找到设备'—— 不要据此判断硬件好坏。")
        print("   解决办法：用你自己的 PowerShell 启动网页服务：")
        print("     cd E:\\vibecoding\\camera\\phase1")
        print("     python -m webui.server --port 8848")
        return 1
    print("✅ 不是受限令牌：摄像头枚举结果可信。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

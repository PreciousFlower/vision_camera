#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RTSP 地址发现工具 —— 接真实摄像头时的一次性诊断。

**它解决什么问题**：
    拿到一台网络摄像头，最大的摩擦不是"代码能不能拉流"，而是
    **"我该填什么 RTSP 地址"**。各厂商的 URL 格式完全不同，
    手册常常找不到，网上搜到的又常是别的型号。结果就是对着
    "探测失败"发呆，不知道是地址错、密码错、还是网络不通。

    这个脚本把你手头的 IP 和账号密码，套进常见厂商的 URL 模板里
    逐个试，直接告诉你**哪个地址能用**，并顺带测出分辨率与帧率。

**安全与合规**（重要）：
    · 只在你自己的设备上使用。扫描别人的摄像头是入侵行为。
    · 脚本只做**你显式提供凭据**的登录尝试，不做密码爆破。
    · 不联网、不上传任何数据，全程在你本机跑。

用法：
    python tools/find_rtsp_url.py --ip 192.168.1.64 --user admin --password 你的密码
    python tools/find_rtsp_url.py --ip 192.168.1.64 --user admin --password xxx --channel 1
    # 不知道密码就先用空密码试常见路径（部分设备默认无密码）
    python tools/find_rtsp_url.py --ip 192.168.1.64 --user admin --password ""

    # 只列出候选地址、不实际连接（先看看有哪些可能）
    python tools/find_rtsp_url.py --ip 192.168.1.64 --user admin --password xxx --dry-run
"""
from __future__ import annotations

import argparse
import socket
import sys
import urllib.parse
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

# FFmpeg 在试错过程中会往 stderr 刷大量解码告警（"error count: 69" 之类），
# 把真正的进度输出淹掉。这里把它压到只报致命错误。
# 必须在 import cv2 / 打开任何流之前设。
import os  # noqa: E402

os.environ.setdefault("OPENCV_LOG_LEVEL", "FATAL")
os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")

# 常见厂商的 RTSP 路径模板。{ch} = 通道号（主码流），{sub} = 子码流。
# 顺序按"国内监控设备出现频率"排 —— 先试最可能的，省时间。
TEMPLATES: list[tuple[str, str]] = [
    ("海康 Hikvision 主码流", "/Streaming/Channels/{ch}01"),
    ("海康 Hikvision 子码流", "/Streaming/Channels/{ch}02"),
    ("海康 旧固件", "/cam/realmonitor?channel={ch}&subtype=0"),
    ("大华 Dahua 主码流", "/cam/realmonitor?channel={ch}&subtype=0"),
    ("大华 Dahua 子码流", "/cam/realmonitor?channel={ch}&subtype=1"),
    ("宇视 Uniview", "/media/video{ch}"),
    ("雄迈/天视通 XM", "/user={user}&password={pwd}&channel={ch}&stream=0.sdp"),
    ("雄迈 XM 备用", "/user={user}&password={pwd}&channel={ch}&stream=1.sdp"),
    ("普联 TP-Link", "/stream{ch}"),
    ("普联 TP-Link 备用", "/stream{ch}/"),
    ("华为 Huawei", "/LiveMedia/ch{ch}/Media1"),
    ("天地伟业 Tiandy", "/profile{ch}"),
    ("乐橙/萤石 兼容", "/cam/realmonitor?channel={ch}&subtype=0"),
    ("通用 onvif 风格", "/onvif{ch}"),
    ("通用 /live", "/live"),
    ("通用备用 1", "/live/ch{ch}"),
    ("通用备用 2", "/live{ch}"),
    ("通用备用 3", "/h264/ch{ch}"),
    ("通用备用 4", "/ch{ch}"),
    ("通用备用 5", "/video{ch}"),
    ("通用备用 6", "/stream"),
    ("通用备用 7", "/11"),
    ("通用备用 8", "/12"),
    ("RTSP 标准", "/"),
    ("RTSP 标准 2", ""),
]

# 常见 RTSP 端口
PORTS = [554, 8554]


def build_urls(ip: str, user: str, pwd: str, channel: int,
               port: int) -> list[tuple[str, str]]:
    """把模板展开成候选 URL 列表。"""
    cred = ""
    if user:
        u = urllib.parse.quote(user, safe="")
        p = urllib.parse.quote(pwd, safe="")
        cred = f"{u}:{p}@" if pwd else f"{u}@"
    out = []
    seen = set()
    for label, path in TEMPLATES:
        p = (path.replace("{ch}", str(channel))
                 .replace("{user}", urllib.parse.quote(user, safe=""))
                 .replace("{pwd}", urllib.parse.quote(pwd, safe="")))
        url = f"rtsp://{cred}{ip}:{port}{p}"
        if url in seen:
            continue
        seen.add(url)
        out.append((f"{label}  [端口{port}]", url))
    return out


def tcp_reachable(ip: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(
        description="RTSP 地址发现（只用于你自己的设备）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", required=True, help="摄像头 IP，例如 192.168.1.64")
    ap.add_argument("--user", default="admin", help="用户名（默认 admin）")
    ap.add_argument("--password", default="", help="密码（不知道就先留空）")
    ap.add_argument("--port", type=int, default=0,
                    help="指定端口；不给就依次试 554 / 8554")
    ap.add_argument("--channel", type=int, default=1, help="通道号，默认 1")
    ap.add_argument("--timeout", type=float, default=6.0,
                    help="每个地址的探测超时秒数，默认 6")
    ap.add_argument("--dry-run", action="store_true", help="只列候选地址，不连接")
    args = ap.parse_args()

    print("=" * 76)
    print("RTSP 地址发现")
    print("=" * 76)
    print(f"  目标     {args.ip}   用户 {args.user or '(无)'}   通道 {args.channel}")
    print("  ⚠️ 请只对你自己的设备使用本工具。扫描他人设备是入侵行为。")
    print()

    ports = [args.port] if args.port else PORTS

    if args.dry_run:
        for port in ports:
            print(f"── 端口 {port} 的候选地址 ──")
            for label, url in build_urls(args.ip, args.user, args.password,
                                         args.channel, port):
                print(f"  {label:28s} {url}")
            print()
        return 0

    # ── 先看端口通不通，避免把"网络不通"误判成"地址不对" ──
    print("[1] 端口可达性")
    reachable = []
    for port in ports:
        ok = tcp_reachable(args.ip, port)
        print(f"    {args.ip}:{port}  {'✅ 通' if ok else '❌ 不通'}")
        if ok:
            reachable.append(port)
    if not reachable:
        print()
        print("❌ 所有常见 RTSP 端口都不通。这**不是地址问题**，先解决：")
        print("   · 摄像头和本机在同一网段吗？（ping 一下）")
        print("   · 电脑和摄像头在同一局域网吗？（很多摄像头只能内网访问）")
        print("   · 摄像头是 PoE/独立供电吗？指示灯正常吗？")
        print("   · 设备管理页面能打开吗？（浏览器试 http://" + args.ip + "）")
        print("   · RTSP 功能被关掉了吗？（有些要在设置里手动开启）")
        return 2

    # ── 逐个试 URL ──
    print()
    print(f"[2] 逐个尝试 {len(reachable)} 个端口上的候选地址"
          f"（每个最多 {args.timeout:.0f} 秒）")
    from webui.capture import probe_source

    # ★ 先做一次"路径敏感性"探测：故意用一个绝不可能存在的路径。
    # 如果它也能返回画面，说明这台设备**根本不看路径** ——
    # 那么后面每个候选地址都会"成功"，结果毫无区分力。
    # （实测踩到：自建的测试服务器就忽略路径，工具一度把 23 个地址全报成可用。）
    # 这一步能把"真找到了"和"设备不看路径"区分开，避免给出误导性的建议。
    print()
    print("[2a] 路径敏感性探测（用一个不存在的路径试）")
    bogus = probe_source(f"rtsp://{[i for i in [args.ip]][0]}:{reachable[0]}/"
                         f"__nonexistent_path_{os.getpid()}__",
                         timeout_sec=int(args.timeout))
    path_agnostic = bogus.ok
    if path_agnostic:
        print(f"    ⚠️ 不存在的路径也能读到画面（{bogus.width}×{bogus.height}）")
        print("       —— 这台设备/网关**忽略 URL 路径**，任何地址都会'成功'。")
        print("       下面找到的地址仅供参考，建议直接用最标准的那条。")
    else:
        print("    ✅ 不存在的路径读不到画面 —— 该设备按路径分发，结果有区分力")

    found: list[tuple[str, str, object]] = []
    tried = 0
    print()
    print("[2b] 尝试候选地址")
    for port in reachable:
        for label, url in build_urls(args.ip, args.user, args.password,
                                     args.channel, port):
            tried += 1
            info = probe_source(url, timeout_sec=int(args.timeout))
            if info.ok:
                found.append((label, url, info))
                print(f"    ✅ {label:28s} {info.width}×{info.height} "
                      f"{info.measured_fps} fps  传输 {info.transport}")
            else:
                short = (info.error or "")[:44]
                print(f"    ·  {label:28s} {short}")

    # 按分辨率去重：同分辨率的多个路径其实是同一个流，只保留第一个
    # （路径不敏感的网关会制造大量重复）
    if path_agnostic and found:
        uniq: list[tuple[str, str, object]] = []
        seen_res = set()
        for label, url, info in found:
            key = (info.width, info.height)
            if key in seen_res:
                continue
            seen_res.add(key)
            uniq.append((label, url, info))
        if len(uniq) < len(found):
            print()
            print(f"    （按分辨率去重：{len(found)} → {len(uniq)}，"
                  f"其余是同一路流的重复）")
        found = uniq

    print()
    print("=" * 76)
    print("结果")
    print("=" * 76)
    print(f"  尝试了 {tried} 个地址，找到 {len(found)} 个可用")
    print()
    if found:
        label, url, info = found[0]
        safe = url  # 打印时保留密码，方便直接复制；这是用户自己的设备
        print("  🎉 可用的地址（复制到网页的「摄像头监测」页即可）：")
        print()
        print(f"     {safe}")
        print()
        print(f"  设备：{label}")
        print(f"  分辨率 {info.width}×{info.height}   实测 {info.measured_fps} fps"
              f"   传输 {info.transport}")
        print()
        if len(found) > 1:
            print("  其他也可用的地址：")
            for l2, u2, i2 in found[1:]:
                print(f"     {u2}")
                print(f"        （{l2}  {i2.width}×{i2.height}）")
        print()
        print("  下一步：打开网页 →「摄像头监测」→ 粘贴上面的地址 → 点「探测源」")
        print("         确认无误后点「开始监测」。窗口步长建议 6–8 秒。")
    else:
        print("  ❌ 端口通，但没有一个候选地址能用。按可能性排查：")
        print()
        print("   1. 用户名/密码错 —— 这是最常见的原因。")
        print("      在摄像头自己的管理页面上（浏览器打开 http://" + args.ip + "）")
        print("      确认账号密码；有些设备要先在设置里创建 RTSP 用户。")
        print()
        print("   2. 地址格式不在候选表里 —— 用管理页面或以下方式找真实地址：")
        print("      · 管理页面里的「网络 → 高级 → RTSP」通常直接写着")
        print("      · 用 ONVIF 工具（如 ONVIF Device Manager）扫描出媒体地址")
        print("      · 若本机装了 ffmpeg：")
        print(f"          ffprobe -rtsp_transport tcp rtsp://{args.ip}/<路径>")
        print()
        print("   3. RTSP 被禁用 —— 部分设备默认关闭，需在设置里开启。")
        print()
        print("   4. 设备只支持 ONVIF 不支持裸 RTSP —— 那就需要走 ONVIF 取流，")
        print("      或在管理页面里开启 RTSP。")
    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main())

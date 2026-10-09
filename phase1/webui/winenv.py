#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Windows 进程环境探测：这个进程是不是跑在"受限令牌"（沙箱）里。

为什么需要它（用户实测踩到的坑，花了很久才定位）：
    在受限令牌下启动网页服务时，**摄像头枚举会静默失败** ——
    返回"一个设备都没有"。而我们的提示语写的是
    「没找到可用摄像头；也可能是被其他程序占用」，
    把"权限不足"误导成了"硬件坏了/被微信占用"，
    让人去查驱动、关微信，方向完全错。
    程序自己完全能判断令牌是否受限，就应该直说。

判据（两个都要看，缺一不可）：
    1. 令牌里的"受限 SID"数量 > 0 → 受限令牌
    2. 连 OpenProcessToken(自己) 都被拒绝（WinError 5）→ 也是受限环境
       注意：读不到令牌**不等于**没问题，必须区分"0 个"和"读不到"。

用法（作为库）：
    from webui.winenv import token_info, is_restricted, camera_diagnosis
"""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

TOKEN_QUERY = 0x0008
_TOKEN_RESTRICTED_SIDS = 11
_TOKEN_INTEGRITY_LEVEL = 25
_TOKEN_IS_APP_CONTAINER = 29

_IS_WINDOWS = sys.platform.startswith("win")


def token_info() -> dict:
    """探测当前进程令牌。

    返回 {"restricted_sids": int, "integrity": str, "appcontainer": bool,
          "note": str}
    · restricted_sids > 0  → 受限令牌
    · restricted_sids == 0 → 正常
    · restricted_sids < 0  → **读不到**（看 note，别当成正常）
    """
    out: dict = {"restricted_sids": -1, "integrity": "", "appcontainer": False,
                 "note": ""}
    if not _IS_WINDOWS:
        out["note"] = "非 Windows，跳过"
        out["restricted_sids"] = 0
        return out
    try:
        advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        # ⚠️ 必须先声明类型。不声明的话 GetCurrentProcess 返回的伪句柄 (-1)
        #    在 64 位上会被当成 32 位整数传递，OpenProcessToken 必然失败 ——
        #    第一版就是这样，表现为"读不到令牌"，
        #    于是差点又给出"没有问题"的错误结论。
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetCurrentProcess.argtypes = []
        advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                              ctypes.POINTER(wintypes.HANDLE)]
        advapi32.OpenProcessToken.restype = wintypes.BOOL
        advapi32.GetTokenInformation.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD)]
        advapi32.GetTokenInformation.restype = wintypes.BOOL

        tok = wintypes.HANDLE()
        if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(),
                                         TOKEN_QUERY, ctypes.byref(tok)):
            err = ctypes.get_last_error()
            out["note"] = f"OpenProcessToken 失败（WinError {err}）"
            if err == 5:
                out["note"] += "：连自己进程的令牌都被拒绝 → 基本可确定进程受限"
            return out

        size = wintypes.DWORD(0)
        advapi32.GetTokenInformation(tok, _TOKEN_RESTRICTED_SIDS, None, 0,
                                     ctypes.byref(size))
        if size.value:
            buf = ctypes.create_string_buffer(size.value)
            if advapi32.GetTokenInformation(tok, _TOKEN_RESTRICTED_SIDS, buf,
                                            size.value, ctypes.byref(size)):
                # TOKEN_GROUPS = DWORD GroupCount + SID_AND_ATTRIBUTES[]
                out["restricted_sids"] = int(ctypes.cast(
                    buf, ctypes.POINTER(wintypes.DWORD)).contents.value)
        else:
            out["restricted_sids"] = 0

        size2 = wintypes.DWORD(0)
        advapi32.GetTokenInformation(tok, _TOKEN_INTEGRITY_LEVEL, None, 0,
                                     ctypes.byref(size2))
        if size2.value:
            buf2 = ctypes.create_string_buffer(size2.value)
            if advapi32.GetTokenInformation(tok, _TOKEN_INTEGRITY_LEVEL, buf2,
                                            size2.value, ctypes.byref(size2)):
                advapi32.GetSidSubAuthorityCount.argtypes = [ctypes.c_void_p]
                advapi32.GetSidSubAuthorityCount.restype = ctypes.POINTER(
                    ctypes.c_ubyte)
                advapi32.GetSidSubAuthority.argtypes = [ctypes.c_void_p,
                                                        wintypes.DWORD]
                advapi32.GetSidSubAuthority.restype = ctypes.POINTER(
                    wintypes.DWORD)
                sid_ptr = ctypes.cast(buf2, ctypes.POINTER(ctypes.c_void_p))[0]
                cnt = advapi32.GetSidSubAuthorityCount(sid_ptr)
                if cnt:
                    n = cnt.contents.value
                    rid = advapi32.GetSidSubAuthority(sid_ptr, n - 1)
                    if rid:
                        lvl = rid.contents.value
                        out["integrity"] = {
                            0: "Untrusted", 4096: "Low", 8192: "Medium",
                            12288: "High", 16384: "System"}.get(lvl, str(lvl))

        appc = wintypes.DWORD(0)
        sz = wintypes.DWORD(ctypes.sizeof(appc))
        if advapi32.GetTokenInformation(tok, _TOKEN_IS_APP_CONTAINER,
                                        ctypes.byref(appc), sz,
                                        ctypes.byref(sz)):
            out["appcontainer"] = bool(appc.value)
    except Exception as e:                      # noqa: BLE001
        out["note"] = f"{type(e).__name__}: {e}"
    return out


def is_restricted() -> bool:
    """当前进程是否受限（受限 SID / AppContainer / 令牌读不到）。"""
    info = token_info()
    if info.get("restricted_sids", -1) > 0 or info.get("appcontainer"):
        return True
    return "拒绝" in (info.get("note") or "")


def restriction_reason() -> str:
    """给用户看的一句话说明（正常时返回空串）。"""
    info = token_info()
    n = info.get("restricted_sids", -1)
    if n > 0:
        return f"本进程是受限令牌（{n} 个受限 SID）"
    if info.get("appcontainer"):
        return "本进程跑在 AppContainer 里"
    if "拒绝" in (info.get("note") or ""):
        return f"读不到本进程令牌（{info.get('note')}）"
    return ""


def camera_diagnosis(found: list, tried: str = "") -> str:
    """摄像头一个都没找到时，给出**按可能性排序**的真实原因。

    ⚠️ 顺序很重要：受限令牌下"枚举不到设备"是最容易被误判成硬件故障的一种，
       所以只要进程受限就把它放在第一位，并给出可执行的解决办法。
       以前这里写死的是「也可能是被其他程序占用」—— 那句话把用户带偏了。
    """
    if found:
        return ""
    tips: list[str] = []
    if is_restricted():
        tips.append(
            f"**本服务进程权限受限**（{restriction_reason()}）："
            "受限令牌下系统会拒绝摄像头枚举，看起来就像'一个设备都没有'。"
            "这不是硬件问题 —— 请用**你自己的 PowerShell** 启动本服务："
            "cd E:\\vibecoding\\camera\\phase1 然后 python -m webui.server --port 8848"
        )
    tips.append("摄像头确实被其他程序占用（微信 / 腾讯会议 / 相机应用 / 浏览器标签页）")
    tips.append("索引不对：本机摄像头一般是 0，也可能是 1 或 2（可逐个试）")
    tips.append("驱动或隐私设置：Windows「设置 → 隐私和安全性 → 相机」里"
                "要允许桌面应用访问相机")
    head = ("没找到可用摄像头。按可能性排序的原因："
            if not is_restricted() else "没找到可用摄像头（**最可能是权限原因**）：")
    body = "；".join(f"{i + 1}) {t}" for i, t in enumerate(tips))
    return f"{head}{body}" + (f"（已尝试：{tried}）" if tried else "")

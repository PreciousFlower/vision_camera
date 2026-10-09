"""判断某个进程是否运行在"受限令牌"下（DSH 的 Windows 沙箱会用到），
并打印它的父进程与命令行 —— 用来确认"项目服务器是不是被沙箱启动的"。

用法：
    python tools/proc_probe.py            # 只看自己
    python tools/proc_probe.py 61824 ...  # 看指定 pid
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import sys

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
adv = ctypes.WinDLL("advapi32", use_last_error=True)
ntdll = ctypes.WinDLL("ntdll", use_last_error=True)

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
PROCESS_VM_READ = 0x0010
TOKEN_QUERY = 0x0008
TokenGroups = 2
TokenRestrictedSids = 11


class PROCESS_BASIC_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("ExitStatus", ctypes.c_long),
        ("PebBaseAddress", ctypes.c_void_p),
        ("AffinityMask", ctypes.c_void_p),
        ("BasePriority", ctypes.c_long),
        ("UniqueProcessId", ctypes.c_void_p),
        ("InheritedFromUniqueProcessId", ctypes.c_void_p),
    ]


class SID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wt.DWORD)]


class TOKEN_GROUPS(ctypes.Structure):
    _fields_ = [("GroupCount", wt.DWORD), ("Groups", SID_AND_ATTRIBUTES * 1)]


k32.OpenProcess.restype = wt.HANDLE
k32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
k32.ReadProcessMemory.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
k32.CloseHandle.argtypes = [wt.HANDLE]
adv.OpenProcessToken.argtypes = [wt.HANDLE, wt.DWORD, ctypes.POINTER(wt.HANDLE)]
adv.GetTokenInformation.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                    wt.DWORD, ctypes.POINTER(wt.DWORD)]
adv.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_wchar_p)]


def sid_to_str(sid: int) -> str:
    out = ctypes.c_wchar_p()
    if adv.ConvertSidToStringSidW(ctypes.c_void_p(sid), ctypes.byref(out)):
        return out.value or "?"
    return "?"


def token_sids(hproc: int, cls: int) -> list[str]:
    tok = wt.HANDLE()
    if not adv.OpenProcessToken(hproc, TOKEN_QUERY, ctypes.byref(tok)):
        return [f"<OpenProcessToken 失败 err={ctypes.get_last_error()}>"]
    try:
        size = wt.DWORD(0)
        adv.GetTokenInformation(tok, cls, None, 0, ctypes.byref(size))
        if not size.value:
            return []
        buf = ctypes.create_string_buffer(size.value)
        if not adv.GetTokenInformation(tok, cls, buf, size.value, ctypes.byref(size)):
            return [f"<GetTokenInformation 失败 err={ctypes.get_last_error()}>"]
        groups = ctypes.cast(buf, ctypes.POINTER(TOKEN_GROUPS)).contents
        arr = ctypes.cast(
            ctypes.addressof(buf) + SID_AND_ATTRIBUTES.Sid.offset + 4 + 4,
            ctypes.POINTER(SID_AND_ATTRIBUTES))
        return [sid_to_str(arr[i].Sid) for i in range(groups.GroupCount)]
    finally:
        k32.CloseHandle(tok)


def probe(pid: int) -> None:
    print(f"\n===== PID {pid} =====")
    h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ,
                        False, pid)
    if not h:
        print(f"  OpenProcess 失败 err={ctypes.get_last_error()}")
        return
    try:
        pbi = PROCESS_BASIC_INFORMATION()
        ntdll.NtQueryInformationProcess.argtypes = [wt.HANDLE, ctypes.c_int,
                                                    ctypes.c_void_p, wt.ULONG,
                                                    ctypes.POINTER(wt.ULONG)]
        st = ntdll.NtQueryInformationProcess(h, 0, ctypes.byref(pbi),
                                             ctypes.sizeof(pbi), None)
        if st != 0:
            print(f"  NtQueryInformationProcess 失败 status={st:#x}")
        else:
            print(f"  父进程 PID : {pbi.InheritedFromUniqueProcessId}")

        def rd(addr: int, n: int) -> bytes:
            buf = ctypes.create_string_buffer(n)
            got = ctypes.c_size_t()
            if not k32.ReadProcessMemory(h, ctypes.c_void_p(addr), buf, n,
                                         ctypes.byref(got)):
                raise OSError(f"ReadProcessMemory err={ctypes.get_last_error()}")
            return buf.raw[:got.value]

        if pbi.PebBaseAddress:
            pp = int.from_bytes(rd(pbi.PebBaseAddress + 0x20, 8), "little")
            if pp:
                us = rd(pp + 0x70, 16)            # CommandLine (UNICODE_STRING)
                length = int.from_bytes(us[0:2], "little")
                buffer = int.from_bytes(us[8:16], "little")
                if length and buffer:
                    print("  命令行     :",
                          rd(buffer, length).decode("utf-16-le", "replace"))
    except Exception as e:
        print(f"  读 PEB 失败: {e}")

    # ⚠️ 必须在 CloseHandle **之前**查令牌 —— 之前把 CloseHandle 放在 finally 里，
    #    导致下面两个查询拿到的是已关闭的句柄，报 err=6（ERROR_INVALID_HANDLE），
    #    看上去像"没有权限"，其实是自己写错了。
    normal = token_sids(h, TokenGroups)
    restricted = token_sids(h, TokenRestrictedSids)
    k32.CloseHandle(h)
    print(f"  令牌组数量 : {len(normal)}")
    print(f"  受限 SID   : {restricted if restricted else '（无 → 不是受限令牌）'}")


pids = [int(a) for a in sys.argv[1:]] or [0]
for p in pids:
    probe(p if p else k32.GetCurrentProcessId())

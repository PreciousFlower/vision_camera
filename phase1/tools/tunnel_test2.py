#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SSH 隧道真实验证 —— 决定 Web 界面用哪种架构。

流程：
    1. sbatch 在计算节点起一个最小 HTTP 服务（纯标准库）
    2. 从 squeue 查出作业落在哪个节点
    3. 用 paramiko 的 direct-tcpip 通道做两级跳：
           本机 → 登录节点 → 计算节点:PORT
    4. 在计算节点上发一个真实 HTTP 请求，看服务是否应答

为什么用 paramiko 而不是 `ssh -L`：
    Windows 的 ssh.exe 走 `-J`/ProxyJump 也可以，但需要密钥或交互输密码，
    而密码含 `}` `&` 特殊字符在命令行里极难处理（见 tju/11b-credentials.md）。
    paramiko 的 `transport.open_channel("direct-tcpip", ...)` 直接覆盖
    "经登录节点跳到计算节点"这一场景，且密码走参数不走 shell，零转义风险。
    这也正是本地 Web 后端将来要用的连接方式。

用法：
    set VLM_SSH_PASSWORD=...
    python tools/tunnel_test2.py
"""
from __future__ import annotations

import json
import os
import re
import socket
import sys
import threading
import time

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

TOOLS_DIR = os.environ.get("VLM_TOOLS_DIR", r"E:\vibecoding\aipc\tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

REMOTE_DIR = "/share/home/u00009/apps/civility-kit"
PORT = int(os.environ.get("TUNNEL_PORT", "18080"))

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)


def log(msg: str = "") -> None:
    print(msg, flush=True)


def main() -> int:
    import cluster  # type: ignore

    pw = os.environ.get("VLM_SSH_PASSWORD")
    kw = {"password": pw, "allow_agent": False, "look_for_keys": False} if pw else {}

    # ── 1. 上传并提交 ──
    log("=" * 64)
    log("步骤 1：提交计算节点服务作业")
    log("=" * 64)
    local_sbatch = os.path.join(PROJECT, "cluster", "tunnel_probe.sbatch")
    try:
        cluster.run(f"mkdir -p {REMOTE_DIR}/cluster {REMOTE_DIR}/logs", timeout=60, **kw)
        c, sftp = cluster.sftp(timeout=25, **kw)
        try:
            sftp.put(local_sbatch, f"{REMOTE_DIR}/cluster/tunnel_probe.sbatch")
        finally:
            sftp.close()
            c.close()
    except Exception as e:
        log(f"❌ 上传失败：{type(e).__name__}: {e}")
        return 1

    rc, out, err = cluster.run(
        f"cd {REMOTE_DIR} && sbatch --export=PORT={PORT} cluster/tunnel_probe.sbatch",
        timeout=60, **kw,
    )
    m = re.search(r"(\d+)", out or "")
    if not m:
        log(f"❌ 提交失败：{out} {err}")
        return 1
    jobid = m.group(1)
    log(f"✅ 作业 {jobid} 已提交（监听 127.0.0.1:{PORT}）")

    # ── 2. 等它跑起来，查出节点名 ──
    log()
    log("=" * 64)
    log("步骤 2：等待作业启动并定位计算节点")
    log("=" * 64)
    node = None
    t0 = time.time()
    while time.time() - t0 < 900:
        rc, out, _ = cluster.run(
            f'squeue -j {jobid} -h -o "%T %N" 2>/dev/null', timeout=60, **kw
        )
        s = (out or "").strip()
        if s:
            parts = s.split()
            state = parts[0]
            nodelist = parts[1] if len(parts) > 1 else ""
            log(f"  状态={state}  节点={nodelist or '-'}  ({time.time() - t0:.0f}s)")
            if state == "RUNNING" and nodelist:
                # 节点名可能是 gpu8001 或压缩形式
                node = nodelist.split(",")[0]
                break
        else:
            rc2, out2, _ = cluster.run(
                f"grep -h '^NODE=' {REMOTE_DIR}/logs/tunnel-{jobid}.out 2>/dev/null | head -1",
                timeout=60, **kw,
            )
            log(f"  作业已不在队列；日志: {(out2 or '').strip() or '(还没写)'}")
            if out2.strip():
                node = out2.strip().split("=", 1)[1]
                break
        time.sleep(10)

    if not node:
        log("❌ 没等到作业启动（可能一直在排队）。可稍后重跑本脚本。")
        return 1
    log(f"✅ 作业运行在 {node}")

    # 等作业日志出现 "serving on" —— 这是服务真正在监听的证据。
    # 第一版没等这一步，隧道建早了拿到 Connection refused，白判了一次。
    log("  等待服务真正开始监听…")
    serving = False
    t1 = time.time()
    while time.time() - t1 < 180:
        rc, out, _ = cluster.run(
            f"grep -h 'serving on' {REMOTE_DIR}/logs/tunnel-{jobid}.out 2>/dev/null | head -1",
            timeout=60, **kw,
        )
        if (out or "").strip():
            log(f"    ✅ {(out or '').strip()}")
            serving = True
            break
        time.sleep(6)
    if not serving:
        log("    ⚠️ 没看到 serving 日志，仍继续测试（可能是缓冲）")
        time.sleep(10)

    # ── 关键判据：登录节点能不能连到计算节点的这个端口 ──
    log()
    log("=" * 64)
    log("步骤 2.5：登录节点 → 计算节点:端口 是否可达（决定性判据）")
    log("=" * 64)
    probe = (
        f"hostname; "
        f"timeout 6 bash -c 'echo > /dev/tcp/{node}/{PORT}' 2>&1 "
        f"&& echo REACHABLE || echo UNREACHABLE; "
        f"echo '--- nc 对照 ---'; "
        f"(which nc && timeout 6 nc -z -v {node} {PORT}) 2>&1 | tail -3"
    )
    rc, out, err = cluster.run(probe, timeout=120, **kw)
    log(out or "")
    # ⚠️ 别用 `"REACHABLE" in out` —— "UNREACHABLE" 里也含 REACHABLE，
    #    第一版就因为这个子串陷阱把不可达误判成可达。逐行精确比对。
    lines = [(out or "").strip().splitlines() or [""]]
    markers = [l.strip() for l in (out or "").splitlines() if l.strip() in ("REACHABLE", "UNREACHABLE")]
    login_reach = markers[:1] == ["REACHABLE"]
    log(f"  登录节点直达计算节点: {'✅ 可达' if login_reach else '❌ 不可达'}")

    if not login_reach:
        log()
        log("⚠️ 登录节点连不到计算节点的自定义端口。再验证一次：是不是所有端口都不通。")
        rc, out, _ = cluster.run(
            f"for p in 22 80 18080; do timeout 4 bash -c 'echo > /dev/tcp/{node}/$p' "
            f"2>/dev/null && echo \"  port $p ✅\" || echo \"  port $p ❌\"; done",
            timeout=90, **kw,
        )
        log(out or "")
        log()
        log("⇒ 若 22 也不通，说明计算节点与登录节点之间没有可用的 TCP 通道，")
        log("   方案 B（常驻服务 + 隧道）不成立。但请注意：SSH 会话本身是")
        log("   login→compute 的反向（compute 连 login:10022 实测可达），")
        log("   仍可考虑「计算节点主动回连」的变体，见文档结论。")

    # ── 3. 建两级隧道并打请求 ──
    log()
    log("=" * 64)
    log("步骤 3：两级隧道 本机 → 登录节点 → 计算节点")
    log("=" * 64)
    try:
        c = cluster.connect(timeout=25, **kw)
    except Exception as e:
        log(f"❌ 连接登录节点失败：{e}")
        return 1

    login_host = getattr(c, "_vlm_host", "?")
    log(f"登录节点 {login_host}；目标 {node}:{PORT}")

    try:
        t = c.get_transport()

        # 第一步：登录节点 → 计算节点:PORT 是否可达
        log(f"  ① 登录节点 → {node}:{PORT} …")
        try:
            ch_probe = t.open_channel("direct-tcpip", (node, PORT), ("127.0.0.1", 0))
            ch_probe.close()
            log("     ✅ 可达")
        except Exception as e:
            log(f"     ❌ 不可达：{type(e).__name__}: {e}")
            log("     ⇒ 结论：不能做「常驻服务 + 隧道」，必须用方案 A（传帧取结果）")
            return 2

        # 第二步：本机起本地监听，转发到上面那条通道（等价于 ssh -L）
        log(f"  ② 本机 127.0.0.1:{PORT} → 经登录节点 → {node}:{PORT} …")
        lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            lsock.bind(("127.0.0.1", PORT))
        except OSError as e:
            log(f"     ❌ 本地端口 {PORT} 被占用：{e}")
            return 1
        lsock.listen(8)
        lsock.settimeout(60)
        log(f"     监听中 127.0.0.1:{PORT}")

        def handle():
            """接受一条本地连接，开一条到计算节点的通道，双向转发。"""
            import select

            conn, addr = lsock.accept()
            try:
                chan = t.open_channel("direct-tcpip", (node, PORT), addr)
            except Exception as e:
                log(f"     ❌ 打开通道失败：{e}")
                conn.close()
                return
            try:
                while True:
                    r, _, _ = select.select([conn, chan], [], [], 30)
                    if not r:
                        break
                    if conn in r:
                        d = conn.recv(65536)
                        if not d:
                            break
                        chan.sendall(d)
                    if chan in r:
                        d = chan.recv(65536)
                        if not d:
                            break
                        conn.sendall(d)
            finally:
                try:
                    chan.close()
                except Exception:
                    pass
                conn.close()

        threading.Thread(target=handle, daemon=True).start()

        # 第三步：真的发一个 HTTP 请求
        log("  ③ 经隧道发 HTTP 请求 …")
        time.sleep(0.5)
        body = (
            f"GET /tunnel-ok HTTP/1.1\r\nHost: 127.0.0.1:{PORT}\r\n"
            "Connection: close\r\n\r\n"
        ).encode()
        s = socket.create_connection(("127.0.0.1", PORT), timeout=30)
        s.sendall(body)
        resp = b""
        while True:
            d = s.recv(65536)
            if not d:
                break
            resp += d
        s.close()
        text = resp.decode("utf-8", "replace")
        log("     收到响应：")
        for line in text.splitlines()[:12]:
            log(f"       {line}")

        ok = '"ok": true' in text.lower() or '"ok":true' in text.lower()
        log()
        log("=" * 64)
        if ok:
            payload = text.split("\r\n\r\n", 1)[-1]
            try:
                info = json.loads(payload)
                log(f"🎉 隧道打通！计算节点 {info.get('hostname')} 已通过 "
                    f"127.0.0.1:{PORT} 直接应答。")
            except Exception:
                log("🎉 隧道打通！")
            log()
            log("⇒ 结论：可以做「计算节点常驻推理服务 + 本机隧道」，")
            log("   网页能做到接近本地的响应速度（省掉每次 SSH 往返）。")
            return 0
        log("⚠️ 隧道建立但响应异常，看上面的原始输出。")
        return 1
    finally:
        try:
            c.close()
        except Exception:
            pass
        try:
            cluster.run(f"scancel {jobid}", timeout=60, **kw)
            log(f"\n[清理] 已 scancel {jobid}")
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最小 RTSP 服务器（H.264 版）—— 用真实摄像头同款编码验证拉流。

**为什么从 MJPEG 换成 H.264**：
    前面几轮死磕 MJPEG 的 RTP 分片，把所有 offset 写法和包大小都穷举过了
    （见 tools/sweep_rtsp.py），单包能通、分片全败，始终过不了 ffmpeg 的
    RTP/JPEG 解包器。
    但**真实 IP 摄像头走的是 H.264，不是 MJPEG** —— 也就是说我一直在一个
    跟真实场景无关的编码上耗时间。H.264 的 RTP 打包（RFC 6184）规则完全不同：
        · 单个 NAL 装得下 → 单个 RTP 包，载荷直接是 NAL
        · 装不下        → FU-A 分片：一个 FU indicator + FU header + 片数据
    而且**分片偏移这件事在 H.264 里根本不存在**（靠 FU header 的起始/结束位
    排序，不靠字节偏移）—— 这正是 MJPEG 方案卡住的原因所在。

**前置**：需要先用 ffmpeg 生成 Annex-B 裸流：
    ffmpeg -y -i 视频.mp4 -t 6 -an -c:v libx264 -preset ultrafast \\
           -tune zerolatency -g 10 -bf 0 -bsf:v h264_mp4toannexb -f h264 out.h264

用法：
    python tools/mini_rtsp_h264.py --h264 runs/_ref.h264 --port 8554
    # 另一个窗口：python -c "import cv2;cap=cv2.VideoCapture('rtsp://127.0.0.1:8554/live');print(cap.isOpened())"
"""
from __future__ import annotations

import argparse
import re
import socket
import struct
import sys
import threading
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

PT_H264 = 96            # 动态载荷类型（RFC 6184 常用值）
SESSION_ID = "24681357"
CLOCK = 90000
MAX_PKT = 1400          # 含 RTP 头的包大小上限


class _ClientGone(Exception):
    """某个客户端断开 —— 只终止该客户端的推流线程，服务继续待命。"""


def split_annexb(data: bytes) -> list[bytes]:
    """把 Annex-B 流拆成 NAL 单元列表（去掉起始码）。

    Annex-B 用 00 00 01 或 00 00 00 01 分隔 NAL。这里保留原始分隔信息，
    在打包时再统一加回 4 字节起始码（RTP 载荷里的单个 NAL 不带起始码，
    但 SDP 里的 sprop-parameter-sets 需要；为简化，我们在 SDP 里用
    profile-level-id + 参数集单独发，见 _sdp）。
    """
    nals = []
    i = 0
    n = len(data)
    starts = []
    while i < n - 3:
        if data[i] == 0 and data[i + 1] == 0:
            if data[i + 2] == 1:
                starts.append((i, 3))
                i += 3
                continue
            if data[i + 2] == 0 and i + 3 < n and data[i + 3] == 1:
                starts.append((i, 4))
                i += 4
                continue
        i += 1
    for k, (off, sc) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else n
        payload = data[off + sc:end]
        if payload:
            nals.append(payload)
    return nals


class MiniRtspH264:
    """单流 RTSP + H.264（RTP/AVP/TCP 交织）服务器。"""

    def __init__(self, nals: list[bytes], fps: float, host: str = "0.0.0.0",
                 port: int = 8554, path: str = "live"):
        self.nals = nals
        self.fps = max(1.0, fps)
        self.host = host
        self.port = port
        self.path = path.strip("/")
        self.sock: socket.socket | None = None
        self.running = False
        # 参数集（SPS/PPS）要在 SDP 里给，客户端才知道怎么解码
        self.sps = next((n for n in nals if (n[0] & 0x1F) == 7), None)
        self.pps = next((n for n in nals if (n[0] & 0x1F) == 8), None)

    def start(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(8)
        self.running = True
        print(f"[rtsp] 服务已启动 rtsp://{self.host}:{self.port}/{self.path}"
              f"（{len(self.nals)} 个 NAL，SPS={'有' if self.sps else '无'} "
              f"PPS={'有' if self.pps else '无'}）", flush=True)
        while self.running:
            try:
                conn, addr = self.sock.accept()
            except OSError:
                break
            threading.Thread(target=self._serve, args=(conn, addr),
                             daemon=True).start()

    def stop(self) -> None:
        self.running = False
        try:
            if self.sock:
                self.sock.close()
        except OSError:
            pass

    # ── 单客户端 ──
    def _serve(self, conn: socket.socket, addr) -> None:
        print(f"[rtsp] 客户端接入 {addr}", flush=True)
        conn.settimeout(30)
        cseq = 1
        try:
            while self.running:
                req = self._read_request(conn)
                if req is None:
                    break
                method, url, headers = req
                cseq = headers.get("cseq", str(cseq))
                hdr = {"CSeq": cseq, "Server": "mini-rtsp-h264/1.0"}

                # ★ 路径校验：真实摄像头对错误的路径会返回 404，
                #   而"任何路径都给流"会让客户端误以为随便填的地址是对的
                #   （网页的探测按钮有专门的路径敏感性检查来抓这种情况）。
                #   这里照着真实设备的行为做，测试才有意义。
                if not self._path_ok(url):
                    self._send(conn, 404, "Not Found", hdr)
                    print(f"[rtsp]   {method} → 404（路径不匹配：{url}）", flush=True)
                    continue

                if method == "OPTIONS":
                    hdr["Public"] = ("OPTIONS, DESCRIBE, SETUP, PLAY, "
                                     "TEARDOWN, GET_PARAMETER")
                    self._send(conn, 200, "OK", hdr)
                    print("[rtsp]   OPTIONS → 200", flush=True)
                elif method == "DESCRIBE":
                    sdp = self._sdp()
                    hdr["Content-Type"] = "application/sdp"
                    hdr["Content-Base"] = f"rtsp://{addr[0]}:{self.port}/{self.path}/"
                    self._send(conn, 200, "OK", hdr, sdp)
                    print(f"[rtsp]   DESCRIBE → 200（SDP {len(sdp)} 字节）", flush=True)
                elif method == "SETUP":
                    tr = headers.get("transport", "")
                    m = re.search(r"interleaved=(\d+)-(\d+)", tr)
                    ch = (int(m.group(1)), int(m.group(2))) if m else (0, 1)
                    self._rtp_chan = ch[0]
                    hdr["Transport"] = (f"RTP/AVP/TCP;unicast;"
                                        f"interleaved={ch[0]}-{ch[1]};ssrc=00000002")
                    hdr["Session"] = SESSION_ID
                    self._send(conn, 200, "OK", hdr)
                    print(f"[rtsp]   SETUP → 200（interleaved {ch[0]}-{ch[1]}）", flush=True)
                elif method == "PLAY":
                    hdr["Session"] = SESSION_ID
                    hdr["Range"] = "npt=0.000-"
                    self._send(conn, 200, "OK", hdr)
                    print("[rtsp]   PLAY → 200，开始推流", flush=True)
                    break
                elif method in ("TEARDOWN", "GET_PARAMETER"):
                    hdr["Session"] = SESSION_ID
                    self._send(conn, 200, "OK", hdr)
                    if method == "TEARDOWN":
                        return
                else:
                    self._send(conn, 501, "Not Implemented", hdr)
            self._stream(conn)
        except _ClientGone:
            print("[rtsp] 客户端已断开（服务继续待命）", flush=True)
        except (socket.timeout, ConnectionError, OSError) as e:
            print(f"[rtsp] 客户端断开：{type(e).__name__}", flush=True)
        finally:
            try:
                conn.close()
            except OSError:
                pass

    # ── 推流 ──
    def _stream(self, conn: socket.socket) -> None:
        """按帧推送。一"帧"= 从 SPS/IDR 开始到下一个 IDR 之前的所有 NAL。"""
        groups = self._group_by_frame()
        if not groups:
            print("[rtsp] ⚠️ 没有可推的帧（缺 IDR？）", flush=True)
            return
        seq = 0
        ts = 0
        interval = 1.0 / self.fps
        next_t = time.time()
        i = 0
        while self.running:
            for nal in groups[i % len(groups)]:
                seq = self._send_nal(conn, nal, ts, seq, last=False)
            # 一个 RTP 时间戳对应一帧；把该帧最后一个包标 marker 更方便，
            # 但客户端按时间戳切帧也认，为简化统一不标（H.264 允许）。
            seq = self._send_marker(conn, ts, seq)
            ts = (ts + int(CLOCK / self.fps)) & 0xFFFFFFFF
            i += 1
            next_t += interval
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.time()

    def _group_by_frame(self) -> list[list[bytes]]:
        """把 NAL 分组，每组 = 一帧（含它前面的 SPS/PPS/SEI）。

        ⚠️ 这里必须按 **VCL（切片）边界** 分组，不能按 IDR 边界分。
           踩过的坑：第一版是"一个 IDR 到下一个 IDR 之间的所有 NAL 算一帧"，
           只有在**每帧都是关键帧**（`-g 1`）时才成立。
           而 ffmpeg 默认每 1 秒才一个 IDR（`-g 20`），于是 20 帧被当成 1 帧、
           按 --fps 推出去 —— 实测投递速率 **232 帧/秒**，
           14 秒的素材 0.06 秒就播完一遍、每秒循环十几次。
           后果非常隐蔽：
             · 网页上"实时画面"看着一直在乱跳（用户反馈"明显卡顿"）
             · 窗口里采到的 4 帧来自素材的不同位置、根本不是连续动作，
               所以模型看不出"丢垃圾"（用户反馈"AI 完全没识别出来"）
           两个真实反馈其实是同一个 bug 引起的。

        一帧 = 若干 SPS/PPS/SEI（可选）+ 一个 VCL NAL（type 1 非关键帧 / 5 关键帧）。
        x264 默认每个切片就是一帧（单切片），所以这样切是准的。
        """
        groups: list[list[bytes]] = []
        cur: list[bytes] = []
        pending_params: list[bytes] = []
        for nal in self.nals:
            t = nal[0] & 0x1F
            if t in (7, 8, 6):          # SPS/PPS/SEI：跟着下一帧走
                pending_params.append(nal)
                continue
            if t in (1, 5):              # VCL：新的一帧开始
                if cur:
                    groups.append(cur)
                cur = pending_params + [nal]
                pending_params = []
            elif cur:
                cur.append(nal)          # 其它非 VCL NAL 并进当前帧
            else:
                pending_params.append(nal)
        if cur:
            groups.append(cur)
        return groups

    def _send_nal(self, conn: socket.socket, nal: bytes, ts: int, seq: int,
                  last: bool) -> int:
        """发一个 NAL：装得下就单包，装不下就 FU-A 分片（RFC 6184）。"""
        hdr_len = 12
        if hdr_len + len(nal) <= MAX_PKT:
            return self._raw(conn, nal, ts, seq, marker=last)

        # FU-A：头 1 字节 FU indicator + 1 字节 FU header，然后分片数据。
        # ⚠️ 关键区别：H.264 的 FU-A **不用字节偏移**，靠 FU header 里的
        #    S(起始)/E(结束) 位让客户端按顺序拼 —— MJPEG 卡住的"offset 怎么写"
        #    这个问题在这里根本不存在。
        nri = nal[0] & 0x60
        nal_type = nal[0] & 0x1F
        body = nal[1:]
        chunk_max = MAX_PKT - hdr_len - 2
        n = max(1, (len(body) + chunk_max - 1) // chunk_max)
        for k in range(n):
            piece = body[k * chunk_max:(k + 1) * chunk_max]
            fu_ind = bytes([nri | 28])                 # 28 = FU-A
            fu_hdr = bytes([(0x80 if k == 0 else 0)    # S
                            | (0x40 if k == n - 1 else 0)  # E
                            | nal_type])
            seq = self._raw(conn, fu_ind + fu_hdr + piece, ts, seq,
                            marker=(last and k == n - 1))
        return seq

    def _send_marker(self, conn: socket.socket, ts: int, seq: int) -> int:
        """发一个空的 marker 包，显式标记"这一帧到此结束"。

        有些客户端（含 ffmpeg）靠 marker 位切帧；给一个空载荷的 marker 包
        是最省事又不会干扰解码的做法。
        """
        return self._raw(conn, b"", ts, seq, marker=True)

    def _raw(self, conn: socket.socket, payload: bytes, ts: int, seq: int,
             marker: bool) -> int:
        b2 = (0x80 | PT_H264) if marker else PT_H264
        rtp = struct.pack("!BBHII", 0x80, b2, seq & 0xFFFF,
                          ts & 0xFFFFFFFF, 2)
        pkt = rtp + payload
        try:
            conn.sendall(b"$" + bytes([getattr(self, "_rtp_chan", 0)])
                         + struct.pack("!H", len(pkt)) + pkt)
        except (ConnectionError, OSError):
            # ⚠️ 这里**只结束这个客户端**，绝不能置 self.running = False。
            #    踩过的坑：一开始设了，结果第一个客户端断开（比如探测工具
            #    读完几帧就释放）之后整个服务器跟着退出，下一次探测就变成
            #    "Connection refused"，看起来像地址错，其实是服务自己停了。
            #    真实 RTSP 服务器必须能持续服务多个客户端。
            self._client_gone = True
            raise _ClientGone()
        return (seq + 1) & 0xFFFF

    # ── 协议工具 ──
    def _path_ok(self, url: str) -> bool:
        """校验请求 URL 的路径部分是否匹配本服务。

        真实摄像头对错误路径返回 404；照做才能让测试有意义 ——
        否则"路径敏感性"检查永远会说忽略路径，掩盖真实行为。
        允许 OPTIONS 用 `*`（部分客户端这么探活）。
        """
        if url.strip() in ("*", ""):
            return True
        m = re.match(r"rtsp://[^/]+/(.*)$", url)
        path = (m.group(1) if m else url).strip("/")
        # 允许 trackID=0 这类子路径（SETUP 常指向流的 track）
        return path == self.path or path.startswith(self.path + "/")

    def _read_request(self, conn: socket.socket):
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = conn.recv(4096)
            if not chunk:
                return None
            data += chunk
        head, _, rest = data.partition(b"\r\n\r\n")
        lines = head.decode("utf-8", "replace").split("\r\n")
        parts = lines[0].split()
        if len(parts) < 2:
            return None
        headers = {}
        for line in lines[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        clen = int(headers.get("content-length", "0") or 0)
        got = len(rest)
        while got < clen:
            chunk = conn.recv(4096)
            if not chunk:
                break
            got += len(chunk)
        return parts[0], parts[1], headers

    def _send(self, conn: socket.socket, code: int, reason: str,
              headers: dict, body: str = "") -> None:
        payload = body.encode("utf-8") if body else b""
        lines = [f"RTSP/1.0 {code} {reason}"]
        for k, v in headers.items():
            lines.append(f"{k}: {v}")
        if payload:
            lines.append(f"Content-Length: {len(payload)}")
        conn.sendall("\r\n".join(lines).encode("utf-8") + b"\r\n\r\n" + payload)

    def _sdp(self) -> str:
        import base64

        sps = base64.b64encode(self.sps).decode() if self.sps else ""
        pps = base64.b64encode(self.pps).decode() if self.pps else ""
        params = ""
        if sps and pps:
            params = f"; sprop-parameter-sets={sps},{pps}"
        # profile-level-id 从 SPS 取（字节 1..3）
        plid = "42e01e"
        if self.sps and len(self.sps) >= 4:
            plid = self.sps[1:4].hex()
        return (
            "v=0\r\n"
            "o=- 0 0 IN IP4 127.0.0.1\r\n"
            "s=mini-rtsp-h264\r\n"
            "c=IN IP4 0.0.0.0\r\n"
            "t=0 0\r\n"
            f"m=video 0 RTP/AVP {PT_H264}\r\n"
            f"a=rtpmap:{PT_H264} H264/90000\r\n"
            f"a=fmtp:{PT_H264} packetization-mode=1; profile-level-id={plid}"
            f"{params}\r\n"
            "a=control:trackID=0\r\n"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="最小 RTSP 服务器（H.264）")
    ap.add_argument("--h264", default="runs/_ref.h264",
                    help="Annex-B 裸流文件（用 ffmpeg -f h264 生成）")
    ap.add_argument("--port", type=int, default=8554)
    ap.add_argument("--path", default="live")
    ap.add_argument("--fps", type=float, default=20.0,
                    help="推流帧率，必须与素材编码帧率一致，否则画面快放/慢放"
                         "（tools/make_demo_stream.py 默认按 20fps 生成）")
    args = ap.parse_args()

    p = Path(args.h264)
    if not p.is_absolute():
        p = PROJECT / p
    if not p.exists():
        print(f"❌ 找不到 {p}")
        print("   先生成 Annex-B 裸流：")
        print("   ffmpeg -y -i data/videos/littering_pos_vis.mp4 -t 6 -an \\")
        print("     -c:v libx264 -preset ultrafast -tune zerolatency -g 10 -bf 0 \\")
        print("     -bsf:v h264_mp4toannexb -f h264 runs/_ref.h264")
        return 1

    nals = split_annexb(p.read_bytes())
    if not nals:
        print("❌ 没解析出 NAL 单元（文件不是 Annex-B 格式？）")
        return 1

    kinds: dict[int, int] = {}
    for n in nals:
        t = n[0] & 0x1F
        kinds[t] = kinds.get(t, 0) + 1
    names = {1: "非IDR片", 5: "IDR片", 6: "SEI", 7: "SPS", 8: "PPS"}
    print(f"[rtsp] 解析出 {len(nals)} 个 NAL："
          + "，".join(f"{names.get(t, t)}×{c}" for t, c in sorted(kinds.items())),
          flush=True)
    if 5 not in kinds:
        print("[rtsp] ⚠️ 没有 IDR 帧 —— 客户端可能解不出画面。"
              "重新生成时加 -g 10 强制关键帧间隔。", flush=True)

    srv = MiniRtspH264(nals, fps=args.fps, port=args.port, path=args.path)
    print(f"[rtsp] 拉流地址 rtsp://127.0.0.1:{args.port}/{args.path}", flush=True)
    try:
        srv.start()
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

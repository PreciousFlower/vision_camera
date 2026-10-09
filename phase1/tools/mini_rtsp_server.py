#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最小 RTSP 服务器（标准库实现）—— 为验证"拉真实 RTSP 流"而造。

**为什么需要它**：
    要验证 webui/capture.py 能不能真的从 rtsp:// 拉流，就得有一个真的
    RTSP 服务器。本机无法访问外网（github / gitee / hf-mirror 全被阻断），
    下载不了 MediaMTX；ffmpeg 的 `-rtsp_flags listen` 模式又和 OpenCV 的
    客户端握手不兼容（实测 opened=False）。所以自己写一个。

**它实现什么**（够 OpenCV/FFmpeg 正常拉流即可）：
    OPTIONS / DESCRIBE / SETUP / PLAY / TEARDOWN
    RTP/AVP/TCP 交织传输（interleaved），负载为 JPEG（MJPEG, 载荷类型 26）
    单播、单会话、循环推送指定视频的帧

**它不实现什么**（真实摄像头有、这里没有）：Digest 认证、UDP 传输、
    RTCP 完整交互、H.264 等编码。所以它验证的是"我们的客户端能不能拉流"，
    不是"我们的服务器有多完整"。

用法：
    python tools/mini_rtsp_server.py --video data/videos/clean_neg_vis.mp4 --port 8554
    # 另一个窗口：python tools/test_rtsp_real.py
"""
from __future__ import annotations

import argparse
import os
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

SESSION_ID = "12345678"
PT_MJPEG = 26          # RTP payload type for JPEG

# 标准 JPEG 亮度量化表（ISO/IEC 10918-1 Annex K）。
# 只作为"提取失败"时的兜底 —— 正常情况下从实际帧里取真表（见 extract_qtables）。
STD_LUMA_Q = bytes([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99])


def extract_qtables(frame_jpeg: bytes) -> list[bytes]:
    """从一帧 JPEG 里取出量化表（DQT 段）。

    为什么必须取真表而不是写死常量：
        RTP/JPEG 的 Q=255 模式要求服务器把量化表随首片发给客户端，
        客户端用它 + 帧数据拼出可解析的 JPEG。表必须与编码时用的**一致**，
        否则解出来的图像是花的（甚至解不出来）。
        与其手抄 ISO 表（我第一版就抄错了色度表），不如直接从帧里读。
    """
    import io

    from PIL import Image

    try:
        im = Image.open(io.BytesIO(frame_jpeg))
        q = getattr(im, "quantization", None)
        if q:
            tables = [bytes(t) for _k, t in sorted(q.items())]
            # RFC 2435 的表必须是 64 项
            if tables and all(len(t) == 64 for t in tables):
                return tables
    except Exception:
        pass
    return [STD_LUMA_Q]


class MiniRtspServer:
    """单流 RTSP 服务器。每个客户端连接一个线程。"""

    def __init__(self, frames: list[bytes], fps: float, host: str = "0.0.0.0",
                 port: int = 8554, path: str = "live",
                 width: int = 640, height: int = 360,
                 qtables: list[bytes] | None = None,
                 single_packet: bool = False,
                 offset_mode: int = 0,
                 pkt_size: int = 1400):
        self.frames = frames
        self.fps = max(1.0, fps)
        self.host = host
        self.port = port
        self.path = path.strip("/")
        self.width = width
        self.height = height
        # 每帧只发一个 RTP 包（图很小）。用于把"分片"这个变量排除掉。
        self.single_packet = single_packet
        # 分片 offset 的写法（0=纯图像偏移 1=含8字节头 2=含头+量化表）
        self.offset_mode = offset_mode
        # 每个 RTP 包的总大小上限（含 RTP 头）。做成可调是为了排查
        # "是不是包太大被中间层重组出问题"这个可能性。
        self.pkt_size = max(200, int(pkt_size))
        # 量化表：随首片发送（RFC 2435 的 Q=255 模式）。
        # 缺省用两张标准表，保证任何情况下都有表可发。
        self.qtables = qtables or [STD_LUMA_Q, STD_CHROMA_Q]
        self.sock: socket.socket | None = None
        self.running = False
        self.clients = 0

    # ── 生命周期 ──
    def start(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(8)
        self.running = True
        print(f"[rtsp] 服务已在 rtsp://{self.host}:{self.port}/{self.path} 启动"
              f"（{len(self.frames)} 帧 @ {self.fps:.0f}fps）", flush=True)
        while self.running:
            try:
                conn, addr = self.sock.accept()
            except OSError:
                break
            threading.Thread(target=self._serve, args=(conn, addr), daemon=True).start()

    def stop(self) -> None:
        self.running = False
        try:
            if self.sock:
                self.sock.close()
        except OSError:
            pass

    # ── 单个客户端 ──
    def _serve(self, conn: socket.socket, addr) -> None:
        self.clients += 1
        cid = self.clients
        print(f"[rtsp] 客户端 #{cid} 接入 {addr}", flush=True)
        conn.settimeout(30)
        playing = False
        cseq = 0
        session = SESSION_ID
        try:
            while self.running:
                if not playing:
                    req = self._read_request(conn)
                    if req is None:
                        break
                    method, url, headers = req
                    cseq = headers.get("cseq", "1")
                    resp_hdr = {
                        "CSeq": cseq,
                        "Server": "mini-rtsp/1.0",
                    }
                    if method == "OPTIONS":
                        resp_hdr["Public"] = ("OPTIONS, DESCRIBE, SETUP, PLAY, "
                                              "TEARDOWN, GET_PARAMETER")
                        self._send(conn, 200, "OK", resp_hdr)
                        print(f"[rtsp]   OPTIONS → 200", flush=True)

                    elif method == "DESCRIBE":
                        sdp = self._sdp()
                        resp_hdr["Content-Type"] = "application/sdp"
                        resp_hdr["Content-Base"] = f"rtsp://{addr[0]}:{self.port}/{self.path}/"
                        self._send(conn, 200, "OK", resp_hdr, sdp)
                        print(f"[rtsp]   DESCRIBE → 200（SDP {len(sdp)} 字节）", flush=True)

                    elif method == "SETUP":
                        # 记录客户端选的交织通道
                        tr = headers.get("transport", "")
                        m = re.search(r"interleaved=(\d+)-(\d+)", tr)
                        if m:
                            self._rtp_chan = int(m.group(1))
                            self._rtcp_chan = int(m.group(2))
                        else:
                            self._rtp_chan, self._rtcp_chan = 0, 1
                        resp_hdr["Transport"] = (
                            f"RTP/AVP/TCP;unicast;interleaved="
                            f"{self._rtp_chan}-{self._rtcp_chan};ssrc=00000001"
                        )
                        resp_hdr["Session"] = session
                        self._send(conn, 200, "OK", resp_hdr)
                        print(f"[rtsp]   SETUP → 200（interleaved "
                              f"{self._rtp_chan}-{self._rtcp_chan}，客户端要求：{tr[:60]}）",
                              flush=True)

                    elif method == "PLAY":
                        resp_hdr["Session"] = session
                        resp_hdr["Range"] = "npt=0.000-"
                        self._send(conn, 200, "OK", resp_hdr)
                        print("[rtsp]   PLAY → 200，开始推流", flush=True)
                        playing = True
                        break

                    elif method in ("TEARDOWN", "GET_PARAMETER"):
                        resp_hdr["Session"] = session
                        self._send(conn, 200, "OK", resp_hdr)
                        if method == "TEARDOWN":
                            break
                    else:
                        self._send(conn, 501, "Not Implemented", resp_hdr)

            if playing:
                self._stream(conn)
        except (socket.timeout, ConnectionError, OSError) as e:
            print(f"[rtsp] 客户端 #{cid} 断开：{type(e).__name__}", flush=True)
        finally:
            try:
                conn.close()
            except OSError:
                pass
            print(f"[rtsp] 客户端 #{cid} 结束", flush=True)

    def _stream(self, conn: socket.socket) -> None:
        """按目标帧率循环推送 RTP/JPEG（RFC 2435），交织在同一条 TCP 连接里。

        ⚠️ 这里踩过一个坑：**必须带 RFC 2435 的载荷头，JPEG 数据不能裸发**。
        第一版直接把 JPEG 字节塞进 RTP 负载，FFmpeg 报了几百行
        "Received packet without a start chunk; dropping frame" ——
        它靠载荷头里的 type 字节识别 JPEG/SOF，没有头就永远拼不出完整帧。

        载荷头 8 字节：
            0        type-specific（0 = 完整帧用 Y 量化表）
            1        fragment offset，高 8 位
            2-3      fragment offset，低 16 位（大端）
            4        type（0 = YCbCr 4:2:2，1 = 4:2:0 …）
            5        Q（量化表，255 = 动态）
            6-7      width/height，各 8 位，单位 8 像素
        再加上**分片**：RTP 包不能超过 MTU，一帧 JPEG 往往几 KB 到几十 KB，
        必须切成多个包，只有最后一片置 marker 位。

        Q 字段用 **128** 而不是 255：255 表示"量化表随帧发送"，
        接收端还得额外等一个量化表包才能解出图；128 是 RFC 2435 里的
        "默认量化表"约定值，接收端可以直接用内置表解码，少一个来回。
        """
        # ⚠️ 序列号与时间戳必须**跨帧持续递增**，不能在每帧开头重置。
        #
        # 踩坑经过：最初把 seq/ts 放在 while 循环里每帧从 0 开始。
        # 结果是帧内分片 seq 连续，但一进入下一帧又跳回 0，
        # ffmpeg 看到序列号倒退就判定"Missing packets; dropping frame"，
        # 一帧都解不出来。RTP 序列号是**整条流**的计数器，
        # 不是每帧独立编号 —— 这是 RTP 与"帧内分片"最容易搞混的地方。
        seq = getattr(self, "_seq", 0)
        ts = getattr(self, "_ts", 0)
        interval = 1.0 / self.fps
        next_t = time.time()
        i = 0
        max_payload = 1400 - 12 - 8
        while self.running:
            jpg = self.frames[i % len(self.frames)]
            i += 1

            # 量化表随首片发送（Q=255）。表取自待发送的 JPEG 本身
            # （PIL 的 quantization 属性），保证与编码时用的表一致。
            tables = b"".join(bytes(t) for t in self.qtables)   # 每表 64 字节
            lqt = 2 + 1 + 64 * len(self.qtables)   # 长度字段 + 精度/表号 + 表数据
            dqt = struct.pack("!H", lqt) + bytes([0]) + tables

            if self.single_packet:
                # 单包模式：整帧一个包，offset=0、marker=1、Q=255 带表。
                # ★ 实测这档能被 OpenCV 正常解码（12/12 帧）→ **载荷格式没问题**，
                #   问题只出在分片。这是定位问题最关键的一次实验。
                self._send_frame(conn, jpg, dqt, [(0, jpg)], ts, seq, per_frag_dqt=False)
                seq = (seq + 1) & 0xFFFF
            elif self.offset_mode == 3:
                # ★★ 思路 3：**每个分片都带量化表**。
                #
                # 为什么试这个：前三种 offset 写法都失败，根源是"表只在首片、
                # 但 offset 又要体现表长度"这个矛盾。如果每片都带表，
                # 那么"片内数据起点"与"片在帧内的图像偏移"就始终一致 ——
                # offset 直接用纯图像偏移即可，不再需要引入表长度。
                # 代价是每片多 133 字节，但换来规则自洽。
                body_max = max(64, self.pkt_size - 12 - 8 - len(dqt))
                n_frag = max(1, (len(jpg) + body_max - 1) // body_max)
                frags = [(k * body_max, jpg[k * body_max:(k + 1) * body_max])
                         for k in range(n_frag)]
                self._send_frame(conn, jpg, dqt, frags, ts, seq, per_frag_dqt=True)
                seq = (seq + n_frag) & 0xFFFF
            else:
                # 分片：offset 的写法由 offset_mode 决定
                body_max = self.pkt_size - 12 - 8 - len(dqt)   # 留给图像数据的空间
                body_max = max(64, body_max)
                n_frag = max(1, (len(jpg) + body_max - 1) // body_max)
                frags = []
                for k in range(n_frag):
                    chunk = jpg[k * body_max:(k + 1) * body_max]
                    if k == 0:
                        off = 0                       # 首片必须是 0（识别起始块）
                    elif self.offset_mode == 0:
                        # 纯图像偏移：首片之后的片，offset 直接接着图像数据编号
                        off = k * body_max
                    elif self.offset_mode == 1:
                        off = 8 + k * body_max
                    else:
                        off = 8 + len(dqt) + k * body_max
                    frags.append((off, chunk))
                seq = self._send_frame(conn, jpg, dqt, frags, ts, seq,
                                       per_frag_dqt=False)

            ts = (ts + int(90000 / self.fps)) & 0xFFFFFFFF
            next_t += interval
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.time()   # 落后了就追上，不累积延迟
        # 退出前存回，便于同一进程内重连时继续递增（不重置才不会让客户端误判丢包）
        self._seq, self._ts = seq, ts

    def _send_frame(self, conn: socket.socket, jpg: bytes, dqt: bytes,
                    frags: list[tuple[int, bytes]], ts: int, seq: int,
                    per_frag_dqt: bool = False) -> int:
        """把一帧按给定的分片列表打成 RTP/JPEG 发出去，返回更新后的序列号。

        frags 是 [(offset, 图像数据块), ...]。
        offset 的写法差异很大，是这次调试的核心变量 ——
        ffmpeg 用 `offset == 0` 判断"起始块"（带 JPEG 头/量化表的那一片），
        而后续分片又必须让接收端能算出正确的数据位置，两者互相牵制。

        per_frag_dqt=True 时**每个分片都带量化表**，让"片内数据起点"
        与"帧内图像偏移"始终一致，绕开表长度带来的矛盾。
        """
        n = len(frags)
        for k, (offset, chunk) in enumerate(frags):
            if not self.running:
                return seq
            last = (k == n - 1)
            jpeg_hdr = struct.pack(
                "!BBHBBBB",
                0,                              # type-specific
                (offset >> 16) & 0xFF,          # fragment offset 高 8 位
                offset & 0xFFFF,                # fragment offset 低 16 位
                0,                              # type：0 = YCbCr 4:2:2
                255,                            # Q：255 = 表随本片发送
                min(255, self.width // 8),      # width / 8
                min(255, self.height // 8),     # height / 8
            )
            carry_dqt = per_frag_dqt or k == 0
            payload = jpeg_hdr + (dqt if carry_dqt else b"") + chunk
            b2 = (0x80 | PT_MJPEG) if last else PT_MJPEG   # marker 只在末片置 1
            rtp = struct.pack("!BBHII", 0x80, b2, seq & 0xFFFF,
                              ts & 0xFFFFFFFF, 1)
            pkt = rtp + payload
            try:
                conn.sendall(b"$" + bytes([getattr(self, "_rtp_chan", 0)])
                             + struct.pack("!H", len(pkt)) + pkt)
            except (ConnectionError, OSError):
                return seq
            seq = (seq + 1) & 0xFFFF
            if not last:
                time.sleep(0.0005)
        return seq

    # ── 协议工具 ──
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
        method, url = parts[0], parts[1]
        headers = {}
        for line in lines[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        # 有 body 就吃掉（本服务不解析 body）
        clen = int(headers.get("content-length", "0") or 0)
        got = len(rest)
        while got < clen:
            chunk = conn.recv(4096)
            if not chunk:
                break
            got += len(chunk)
        return method, url, headers

    def _send(self, conn: socket.socket, code: int, reason: str,
              headers: dict, body: str = "") -> None:
        payload = body.encode("utf-8") if body else b""
        lines = [f"RTSP/1.0 {code} {reason}"]
        for k, v in headers.items():
            lines.append(f"{k}: {v}")
        if payload:
            lines.append(f"Content-Length: {len(payload)}")
        msg = "\r\n".join(lines).encode("utf-8") + b"\r\n\r\n" + payload
        conn.sendall(msg)

    def _sdp(self) -> str:
        return (
            "v=0\r\n"
            f"o=- 0 0 IN IP4 127.0.0.1\r\n"
            f"s=mini-rtsp\r\n"
            f"c=IN IP4 0.0.0.0\r\n"
            f"t=0 0\r\n"
            f"m=video 0 RTP/AVP {PT_MJPEG}\r\n"
            f"a=rtpmap:{PT_MJPEG} JPEG/90000\r\n"
            f"a=control:trackID=0\r\n"
        )


def load_frames(video: Path, max_side: int, quality: int,
                limit: int) -> tuple[list[bytes], float, int, int]:
    """把视频抽成 JPEG 帧序列（拉流时循环推送）。返回 (帧, 帧率, 宽, 高)。"""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频 {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
    frames: list[bytes] = []
    w = h = 0
    while len(frames) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if max(h, w) > max_side:
            s = max_side / max(h, w)
            frame = cv2.resize(frame, (int(w * s), int(h * s)),
                               interpolation=cv2.INTER_AREA)
            h, w = frame.shape[:2]
        ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok2:
            frames.append(buf.tobytes())
    cap.release()
    return frames, fps, w, h


def main() -> int:
    ap = argparse.ArgumentParser(description="最小 RTSP 服务器（用于验证拉流）")
    ap.add_argument("--video", default="data/videos/littering_pos_vis.mp4")
    ap.add_argument("--port", type=int, default=8554)
    ap.add_argument("--path", default="live")
    ap.add_argument("--fps", type=float, default=10.0, help="推流帧率")
    ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument("--quality", type=int, default=82)
    ap.add_argument("--limit", type=int, default=120, help="最多用多少帧")
    ap.add_argument("--single-packet", action="store_true",
                    help="强制每帧一个 RTP 包（把图缩小以放得下）。"
                         "用来把'分片'这个变量排除掉，定位问题在分片还是载荷格式")
    ap.add_argument("--offset-mode", type=int, default=0, choices=[0, 1, 2, 3],
                    help="分片 offset 写法：0=纯图像偏移 1=含8字节头 "
                         "2=含头+量化表 3=每片都带量化表（规则自洽）")
    ap.add_argument("--pkt-size", type=int, default=1400,
                    help="每个 RTP 包的总大小上限（含 RTP 头），默认 1400。"
                         "调小可排查'包太大被中间层重组出问题'这类怀疑")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.is_absolute():
        video = PROJECT / video
    if not video.exists():
        print(f"❌ 找不到视频 {video}")
        return 1

    frames, src_fps, w, h = load_frames(video, args.max_side, args.quality, args.limit)
    if not frames:
        print("❌ 没抽到任何帧")
        return 1

    # --single-packet：把图缩到很小（保证一帧塞进一个 RTP 包），
    # 用来把"分片"这个变量从问题里排除掉。
    if args.single_packet:
        import cv2
        import numpy as np

        small = []
        for jpg in frames:
            arr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            small_img = cv2.resize(arr, (160, 90), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", small_img,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            if ok:
                small.append(buf.tobytes())
        frames = small
        w, h = 160, 90
        sizes = [len(f) for f in frames]
        print(f"[rtsp] 单包模式：{w}×{h}，帧大小 "
              f"{min(sizes)}–{max(sizes)} 字节（上限约 1380）", flush=True)

    qtables = extract_qtables(frames[0])
    print(f"[rtsp] 量化表 {len(qtables)} 张，每张 {len(qtables[0])} 项"
          f"（随首片发送，Q=255 模式）", flush=True)

    srv = MiniRtspServer(frames, fps=args.fps, port=args.port, path=args.path,
                         width=w, height=h, qtables=qtables,
                         single_packet=args.single_packet,
                         offset_mode=args.offset_mode,
                         pkt_size=args.pkt_size)
    print(f"[rtsp] 源视频 {video.name}（原始 {src_fps:.1f}fps，{w}×{h}）", flush=True)
    print(f"[rtsp] 拉流地址 rtsp://127.0.0.1:{args.port}/{args.path}", flush=True)
    print("[rtsp] Ctrl+C 停止", flush=True)
    try:
        srv.start()
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

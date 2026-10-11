"""实时视频源采集 —— USB 摄像头 / 笔记本内置摄像头 / RTSP 网络摄像头 / 视频文件。

用途有二：
    1. **不用买摄像头就能拿到真实画面**：笔记本内置摄像头就是真实场景，
       可以立刻验证"真视频能不能识别"，不必先采购设备。
    2. **为 RTSP 做准备**：RTSP 与本地摄像头的代码路径完全一样
       （都是 OpenCV VideoCapture），先用 index=0 跑通，换成
       `rtsp://user:pass@ip:554/stream` 即可，无需改任何其他代码。

为什么不用 ffmpeg 而用 OpenCV：
    抽帧用 ffmpeg 是因为要精确控制 fps 且一次性抽完整段视频。
    实时流是"边读边丢"——读到的帧有新有旧，必须取最新帧而不是排队消费，
    否则会越读越滞后。OpenCV 的 grab/retrieve 正好能做这件事。

⚠️ 一个必须知道的坑：VideoCapture 打开失败时**不报错**，
   isOpened() 返回 False 要主动检查；RTSP 连不上时 read() 会阻塞很久，
   所以打开前要设 OPENCV_FFMPEG_CAPTURE_OPTIONS 里的超时。
"""
from __future__ import annotations

import os
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse, urlunparse


@dataclass
class SourceInfo:
    kind: str            # local | rtsp | file
    label: str
    ok: bool
    width: int = 0
    height: int = 0
    measured_fps: float = 0.0
    error: str = ""
    backend: str = ""
    # 实际连通的传输方式（仅 rtsp 有意义）：tcp / udp。
    # 记录下来很有用 —— 同型号的下一台设备可以直接用这个设置。
    transport: str = ""


def parse_source(spec: str) -> tuple[str, str | int]:
    """把用户输入解析成 OpenCV 能吃的 source。

    支持：
        "0" / "1"                        本地摄像头索引
        "rtsp://..." / "http://..."      网络流
        "E:\\path\\to\\video.mp4"        视频文件
    """
    spec = (spec or "").strip()
    if not spec:
        raise ValueError("视频源不能为空")
    if spec.isdigit():
        return "local", int(spec)
    low = spec.lower()
    if low.startswith(("rtsp://", "rtmp://", "http://", "https://")):
        return "rtsp", spec
    return "file", spec


def mask_credentials(url: str) -> str:
    """把 URL 里的密码打码，便于打印/返回给前端而不泄露凭据。"""
    try:
        u = urlparse(url)
        if u.password:
            netloc = f"{u.username}:***@{u.hostname}"
            if u.port:
                netloc += f":{u.port}"
            return urlunparse((u.scheme, netloc, u.path, u.params, u.query, u.fragment))
    except Exception:
        pass
    return url


def _set_rtsp_timeout(sec: int = 8, transport: str = "tcp") -> None:
    """给 OpenCV 的 FFmpeg 后端设超时与传输方式。

    必须在 VideoCapture 构造**之前**设置才生效。

    ⚠️ 选项名踩过坑：旧版 FFmpeg 用 `stimeout`（微秒），**新版已不认这个名字**，
    改叫 `timeout`。实测只设 stimeout 时，连一个不可达的 RTSP 地址要干等
    30 秒才返回，用户会以为网页卡死。新旧名字都写上，各自认得一个，互不影响。

    ⚠️ transport 也踩过：**有些摄像头只接受 TCP 交织**（UDP 的 RTP 端口被
    防火墙挡掉），有些反过来只稳在 UDP 上。默认 TCP（穿透性更好、更常用于
    跨网段），失败时调用方会用 udp 再试一次。这个参数对"换真实设备能不能
    一次连上"影响很大，所以做成可切换而不是写死。
    """
    us = sec * 1_000_000
    parts = [
        f"rtsp_transport;{transport}",
        f"timeout;{us}",          # 新版 FFmpeg
        f"stimeout;{us}",         # 旧版 FFmpeg（保留兼容）
        f"rw_timeout;{us}",       # 读写超时，兜底
        "max_delay;500000",
        "reorder_queue_size;0",
    ]
    if transport == "udp":
        # UDP 下缓冲更不可控，显式压小
        parts.append("buffer_size;1048576")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(parts)


# 常见网络摄像头的 RTSP 地址模板。
# 不是为了自动猜密码，而是给出"可以照着改"的清单 ——
# 用户手上有一台设备却不知道 URL 格式时，这个列表能省掉一次搜索。
RTSP_URL_HINTS = [
    ("海康 Hikvision", "rtsp://用户名:密码@IP:554/Streaming/Channels/101"),
    ("大华 Dahua", "rtsp://用户名:密码@IP:554/cam/realmonitor?channel=1&subtype=0"),
    ("宇视 Uniview", "rtsp://用户名:密码@IP:554/media/video1"),
    ("雄迈/通用 onvif", "rtsp://用户名:密码@IP:554/user=admin&password=xxx&channel=1&stream=0.sdp"),
    ("带子码流(更省带宽)", "把主码流的 101/0 改成 102/1 之类"),
    ("通用探测", "用 ONVIF Device Manager 或 ffprobe 读出真实 URL"),
]


def open_source(spec: str, *, timeout_sec: int = 8, transport: str = "tcp"):
    """打开视频源，返回 (cap, kind, source, error)。失败时 cap 为 None。

    transport 只对 rtsp 生效：先按指定的试，失败由 probe_source 换另一种重试。
    """
    import cv2

    kind, source = parse_source(spec)
    if kind == "rtsp":
        _set_rtsp_timeout(timeout_sec, transport)
        # ⚠️ 0.0.0.0 / :: 是"监听地址"，不是"连接地址"（用户实测踩到）。
        #    自建演示服务器会打印 `0.0.0.0:8554`（那是它绑定的地址），
        #    照抄进网页就会连不上，而且报错是通用的"RTSP 连不上"，
        #    根本看不出是地址写错了 —— 所以这里直接点破。
        bad_host = _unspecified_host(source)
        if bad_host:
            return None, kind, source, (
                f"地址里的主机是 {bad_host}，那是**服务器用来监听的地址**，"
                f"不能用来连接。本机演示请填 127.0.0.1（或 localhost）；"
                f"连真实摄像头请填它自己的 IP"
            )
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        try:
            cap.release()
        except Exception:
            pass
        hint = {
            "local": "本地摄像头打不开：可能被其他程序占用（微信/腾讯会议/相机应用），"
                     "或索引不对（试 0 / 1 / 2）",
            "rtsp": "RTSP 连不上：检查 URL、用户名密码、端口（常见 554）、"
                    "以及摄像头是否允许该网段访问",
            "file": "文件打不开：检查路径是否正确、编码是否被 OpenCV 支持",
        }[kind]
        if kind == "local":
            # 受限令牌下"打不开"的默认解释是错的（用户实测踩到），优先说权限
            from .winenv import restriction_reason

            why = restriction_reason()
            if why:
                hint = (f"本地摄像头打不开，且{why} —— "
                        f"受限令牌下系统会拒绝访问摄像头，这不一定是硬件问题。"
                        f"请用你自己的 PowerShell 启动网页服务再试")
        return None, kind, source, hint
    return cap, kind, source, ""


def _unspecified_host(url: str) -> str:
    """URL 里的主机是不是"未指定地址"（0.0.0.0 / :: / *），是就返回它。"""
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        return ""
    return host if host in ("0.0.0.0", "::", "[::]", "*") else ""


def probe_source(spec: str, *, timeout_sec: int = 15) -> SourceInfo:
    """快速探测一个源是否可用，并测出实际帧率。

    ⚠️ timeout_sec 默认 15 秒（曾用 8 秒，太紧）：
        RTSP 探测要跑完整的 OPTIONS/DESCRIBE/SETUP/PLAY 握手，
        真实网络摄像头（尤其是走 WiFi 或需要认证的）常要 5–10 秒；
        再加上我们要试 TCP 和 UDP 两种传输方式，每种的超时都要留足。
        8 秒时实测把一台本地可达的 RTSP 服务器都判成了超时。
        对不可达地址，15 秒 ×2 种传输 = 最多 30 秒，仍可接受
        （FFmpeg 自己的硬编码超时就是 30 秒，我们并没有更差）。

    为什么要用线程 + join 做硬超时，而不是靠 OpenCV 自己的超时选项：
        实测设置 `OPENCV_FFMPEG_CAPTURE_OPTIONS` 里的
        timeout / stimeout / rw_timeout 全都**不生效** —— 探测一个不可达的
        RTSP 地址仍然要干等 **30 秒**。所以这里在子线程里打开，
        主线程超时就返回 —— 这是我们能控制的唯一办法。

    帧率为什么要实测而不是读 CAP_PROP_FPS：USB 摄像头常报 30 实际给 10，
    RTSP 协商帧率也常与真实不符。实测值才能用来决定抽帧策略。
    """
    import cv2
    import threading

    try:
        kind, _ = parse_source(spec)
    except ValueError as e:
        return SourceInfo(kind="?", label=spec, ok=False, error=str(e))

    label = mask_credentials(spec) if kind == "rtsp" else spec

    # RTSP 要试两种传输方式：有些摄像头只接受 TCP 交织（UDP 的 RTP 端口被
    # 防火墙挡掉），有些反过来只稳在 UDP 上。真实设备连不上时，
    # "换 transport 再试一次"是最常见的解法，所以这里自动做掉。
    transports = ["tcp", "udp"] if kind == "rtsp" else [""]
    last_info: SourceInfo | None = None

    for tr in transports:
        box: dict = {}

        # ⚠️ 这里用闭包捕获 tr，不要写成 def worker(_tr=tr) 再在异常分支里
        #    引用 _tr —— 默认参数只在函数签名作用域里存在，
        #    外层 except 里引用会 NameError（实测踩到）。
        # 超时后的线程仍可能完成，必须绑定本次参数和结果箱，避免写到下一次
        # UDP 尝试；下面的绑定参数仅在 worker 内使用，外层仍使用 tr/box。
        def worker(_transport=tr, _result=box) -> None:
            try:
                cap, k, _s, err = open_source(spec, timeout_sec=timeout_sec,
                                              transport=_transport or "tcp")
                if cap is None:
                    _result["info"] = SourceInfo(kind=k, label=label, ok=False, error=err)
                    return
                try:
                    backend = cap.getBackendName() if hasattr(cap, "getBackendName") else ""
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                    fps_reported = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

                    got = 0
                    t0 = time.time()
                    for _ in range(12):
                        okf, _f = cap.read()
                        if not okf:
                            break
                        got += 1
                    dt = max(1e-6, time.time() - t0)
                    measured = got / dt if got >= 2 else 0.0

                    if got == 0:
                        _result["info"] = SourceInfo(
                            kind=k, label=label, ok=False, backend=str(backend),
                            width=w, height=h,
                            error="能打开但读不到画面（RTSP 认证后无流 / 设备被占用）",
                        )
                        return
                    _result["info"] = SourceInfo(
                        kind=k, label=label, ok=True, backend=str(backend),
                        width=w, height=h,
                        measured_fps=round(measured or fps_reported, 2),
                        transport=_transport,
                    )
                finally:
                    try:
                        cap.release()
                    except Exception:
                        pass
            except Exception as e:
                _result["info"] = SourceInfo(kind=kind, label=label, ok=False,
                                         error=f"{type(e).__name__}: {e}")

        th = threading.Thread(target=worker, daemon=True)
        th.start()
        th.join(timeout_sec)

        if th.is_alive():
            last_info = SourceInfo(
                kind=kind, label=label, ok=False, transport=tr,
                error=f"探测超时（超过 {timeout_sec} 秒未响应，transport={tr}）",
            )
            continue

        info = box.get("info", SourceInfo(kind=kind, label=label, ok=False,
                                          error="探测未返回结果"))
        if info.ok:
            return info
        last_info = info
        # 只有在"打开失败"时换 transport 才有意义；
        # "能打开但读不到画面"说明协议通了、是流或认证的问题，换也没用。
        if kind != "rtsp" or "打不开" not in (info.error or ""):
            break

    if last_info is not None:
        if kind == "rtsp" and transports and "tcp" in transports and "udp" in transports:
            last_info.error = (last_info.error or "") + "（已尝试 TCP 与 UDP 两种传输方式）"
        return last_info
    return SourceInfo(kind=kind, label=label, ok=False, error="探测未返回结果")


# ─────────────────────── 专职读帧线程（治"实时画面卡顿"）───────────────────────
#
# 为什么必须有它（用户实测反馈"摄像头实时的很卡顿"）：
#     原本采集和推理在**同一个线程**里顺序执行：
#         采 6 秒（期间每 0.3 秒写一张预览帧）→ 推理 10~12 秒 → 再采下一窗
#     推理那 10~12 秒里**没有任何代码在读摄像头**，于是：
#         · 页面上的"实时画面"整整冻住十几秒（一个周期 17 秒里冻 11 秒）
#         · 推理结束后才开始读，摄像头缓冲里压着这十几秒的旧画面 →
#           采样拿到的帧比现实晚十几秒（就是页面提示的"跟不上"）
#     根因不是摄像头慢，而是"没人读"。
#
# 设计要点：
#     · **只有这个线程能碰 VideoCapture**（OpenCV 的 VideoCapture 不是线程安全的，
#       多线程同时 read() 会崩或给错帧）
#     · 它持续读、只保留最新一帧，并顺手做三件事：
#         ① 缩放 + JPEG 编码缓存（给 MJPEG 流用，约 10~15fps）
#         ② 按较低频率写预览帧文件（磁盘开销可控，兼容 /preview 接口）
#         ③ 记录真实读取帧率（用来判断摄像头到底给多少帧）
#     · 取样方（capture_windows）到点只做一次"取最新帧"，**不再自己 read()**，
#       所以推理跑多久都不会挡住画面。


class StreamReader:
    """一个实时源对应一个读帧线程。由 LiveStream 持有，跨批次复用。"""

    def __init__(self, cap, kind: str, *, max_side: int = 640,
                 jpeg_quality: int = 82, preview_dir: Path | None = None,
                 preview_fps: float = 3.0, name: str = "") -> None:
        self.cap = cap
        self.kind = kind
        self.max_side = max_side
        self.jpeg_quality = jpeg_quality
        self.preview_dir = Path(preview_dir) if preview_dir else None
        self.preview_interval = 1.0 / max(0.5, preview_fps)
        self.name = name

        self.lock = threading.Lock()
        self.frame = None            # 最新一帧（ndarray，已缩放）
        self.jpeg: bytes | None = None   # 最新一帧的 JPEG 字节（给 MJPEG 流）
        self.seq = 0                 # 帧序号（单调递增，取样方据此判断"有没有更新"）
        self.t = 0.0                 # 最新帧的时间戳
        self.frames_read = 0
        self.read_fail = 0
        self.dead = False            # 连续读失败过多 → 判定流已断
        self.started_at = time.time()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_preview = 0.0
        self._last_encoded = 0.0
        self.history_jpegs = deque(maxlen=256)  # <=32s at 8fps; RAM only.

    # ── 生命周期 ──
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name=f"reader-{self.name or 'live'}")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        th = self._thread
        if th is not None:
            th.join(timeout=2.0)
        self._thread = None
        with self.lock:
            self.history_jpegs.clear()
            self.frame = None
            self.jpeg = None

    # ── 读帧主循环 ──
    def _loop(self) -> None:
        import cv2

        fail_streak = 0
        while not self._stop.is_set():
            try:
                ok, frame = self.cap.read()
            except Exception:                      # noqa: BLE001
                ok, frame = False, None
            if self._stop.is_set():
                break
            if not ok or frame is None:
                self.read_fail += 1
                fail_streak += 1
                # 连续读不到 ≈ 5 秒 → 判定流死了（由上层决定重连）
                if fail_streak > 50:
                    self.dead = True
                    break
                time.sleep(0.05)
                continue
            fail_streak = 0

            h, w = frame.shape[:2]
            if max(h, w) > self.max_side:
                s = self.max_side / max(h, w)
                frame = cv2.resize(frame, (int(w * s), int(h * s)),
                                   interpolation=cv2.INTER_AREA)

            now = time.time()
            jpeg = None
            try:
                if now - self._last_encoded >= 0.125:
                    ok2, buf = cv2.imencode(
                        ".jpg", frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)])
                    if ok2:
                        jpeg = buf.tobytes()
                        self._last_encoded = now
            except Exception:                      # noqa: BLE001
                pass

            now = time.time()
            with self.lock:
                self.frame = frame
                if jpeg is not None:
                    self.jpeg = jpeg
                    self.history_jpegs.append((now, jpeg))
                self.seq += 1
                self.t = now
                self.frames_read += 1

            # 预览帧文件：单独一路、较低频率（给 /preview 与回归测试用）。
            # ⚠️ 文件名必须全局单调（毫秒时间戳），不能用"本批第几张" ——
            #    否则每批都覆盖同一批名字，"按名字取最新"永远拿到同一张，
            #    页面上就是冻住的画面（这个坑前面踩过一次）。
            if self.preview_dir is not None and (now - self._last_preview
                                                 >= self.preview_interval):
                self._last_preview = now
                try:
                    if self.jpeg is not None:
                        pvf = self.preview_dir / f"preview_{int(now * 1000):013d}.jpg"
                        pvf.write_bytes(self.jpeg)
                        olds = sorted(self.preview_dir.glob("preview_*.jpg"))[:-3]
                        for old in olds:
                            try:
                                old.unlink()
                            except OSError:
                                pass
                except OSError:
                    pass

    def snapshot_history(self):
        with self.lock:
            return list(self.history_jpegs)

    # ── 给取样方：取"此刻最新"的一帧 ──
    def take_newer(self, last_seq: int, timeout: float = 2.0):
        """等到出现比 last_seq 更新的帧，返回 (frame, seq, t)；超时返回 (None, last_seq, 0)。

        为什么要等"更新的帧"而不是直接拿当前帧：读帧线程如果卡住/流断了，
        直接拿会反复取到同一帧 —— 一个窗口里 4 张一模一样的图送给模型，
        看起来"有结果"，其实是垃圾输入。等不到就明确返回 None。
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.lock:
                if self.seq > last_seq and self.frame is not None:
                    return self.frame.copy(), self.seq, self.t
                if self.dead:
                    return None, last_seq, 0.0
            time.sleep(0.02)
        with self.lock:
            if self.seq > last_seq and self.frame is not None:
                return self.frame.copy(), self.seq, self.t
        return None, last_seq, 0.0

    def snapshot_jpeg(self) -> bytes | None:
        with self.lock:
            return self.jpeg

    def measured_fps(self) -> float:
        dt = time.time() - self.started_at
        return round(self.frames_read / dt, 1) if dt > 0.5 else 0.0

    def status(self) -> dict:
        return {"frames_read": self.frames_read, "read_fail": self.read_fail,
                "fps": self.measured_fps(), "dead": self.dead,
                "seq": self.seq}


def capture_windows(
    spec: str,
    out_dir: str | Path,
    *,
    window_stride_sec: float = 6.0,
    frames_per_window: int = 4,
    max_windows: int = 10,
    max_side: int = 640,
    jpeg_quality: int = 82,
    on_progress=None,
    should_stop=None,
    keep_frames: int = 0,
    t_offset: float = 0.0,
    idx_offset: int = 0,
    cap=None,
    kind: str | None = None,
    clean_dir: bool = True,
    loop_file: bool = False,
    reader: "StreamReader | None" = None,
) -> list[list[tuple[float, str]]]:
    """从实时源采集若干"判断窗口"的帧，落盘成 JPEG。

    返回 [[(时间戳, 路径), ...], ...]，每个子列表是一个窗口。

    关键设计——**取最新帧而不是排队消费**：
        实时流每读一次都会拿到"当前最新"帧，中间那些帧直接丢弃。
        窗口内按固定间隔取样（例如 6 秒窗口取 4 帧 → 每 2 秒取一帧）。
        这样即使设备给的是 30fps，我们也不会积压，
        判断延迟只取决于窗口长度本身。

    另一个要点：**连续读取与取样解耦**。
        如果不读就一直等，很多 RTSP 源会因缓冲积压而给出越来越旧的画面。
        所以即使还没到取样时刻，也要持续 read() 把旧帧丢掉。

    ⚠️ 文件源与实时源的取样方式**必须不同**（实测踩到）：
        实时源按"墙钟时间"取样是对的（画面本身就在按真实时间前进）。
        但视频文件是"能读多快就读多快"——14 秒的视频 0.1 秒就读完了，
        按墙钟取样会在第一个窗口之后全部读到 EOF，结果一个窗口都拿不到。
        所以文件源改用**按帧号定位**：算出目标帧序号，cap.set() 跳过去再读。
        两种情况都覆盖到，网页上"用视频文件测实时流链路"才成立。

    ⚠️ t_offset / idx_offset 是给**持续监测**用的（踩过的坑）：
        持续模式会反复调用本函数（每批一次），而时间戳与帧序号是按本次调用
        从 0 开始的。视频文件源跑得极快（实测一秒能采上百个窗口），
        于是每批的时间戳都落在 0.0–0.6s、窗口 ID 都从 w0000 开始 ——
        结果几百个窗口的时间段互相重叠、ID 重复，"回看历史"完全乱套。

        **t_offset 应当是"本批采集开始时刻距任务开始的秒数"**，
        而不是把上一批的末值累加上去 —— 后者会复利式膨胀
        （实测 450 个窗口后时间戳变成 1e+33，完全不可用）。
        用挂钟算偏移，对"文件源（比实时快）"和"摄像头（实时）"都成立。

    ⚠️ keep_frames 在"源比实时快"时不要开：
        裁剪会把**即将要读**的帧删掉，结果一个窗口都取不到
        （实测抛"没有采集到任何窗口"）。它只适合实时摄像头那种
        帧率受设备限制、不会瞬间堆几千帧的场景。
        ★ 持续监测现在**不再**用这个参数：裁剪统一挪到"一批推理完之后"
        （见 webui/server.py 的 prune_frames）。原因是采集循环里剪，
        剪掉的正是这一批马上要送去推理的帧，而且"文件源一秒钟采上百个窗口"
        时目录能被剪空 —— 那类 bug 表现为"窗口数在涨、结果全是空的"。

    ★ cap / kind / clean_dir 三个参数是给**持续监测**用的（2025 实测踩到）：
        持续模式每批都调用本函数。如果每批都重新 open_source，
        单单 RTSP 握手就要 2.9 秒（实测），再叠上采集与推理，
        节拍会明显慢于窗口步长；
        如果每批都清空输出目录，上一批刚写好的帧会被删掉，
        页面里"回看历史"的图全变碎图。
        所以持续模式应当是：**开一次流、复用到任务结束**，
        目录只在任务第一批清一次。

    ★ loop_file：用**视频文件**冒充实时源做持续监测时，读到末尾要绕回开头。
        不绕回的话，任务会以"没有采集到任何窗口"报错结束 ——
        而"用视频文件验证整条实时链路"正是没有摄像头时的主要验证手段
        （开发机没有摄像头，实测就是这个场景）。
    """
    import cv2

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ⚠️ 必须先清空输出目录。
    #    生产路径下每个任务用独立的 job_id 目录，不会冲突；
    #    但重跑同一个 job_id（网页上"重新检测"）或复用目录时，
    #    旧帧会和新帧混在一起，而帧是按 f_00000.jpg 递增命名的 ——
    #    结果是窗口里混进上一次的画面，模型看到的是前后不一致的内容，
    #    这种错误极难从结果上看出来。所以这里强制清干净。
    #    （持续模式从第二批起 clean_dir=False，否则会删掉刚采的帧。）
    if clean_dir:
        stale = sorted(out_dir.glob("f_*.jpg"))
        if stale:
            for p in stale:
                try:
                    p.unlink()
                except OSError:
                    pass

    # ⚠️ interval 必须在 frame_step 之前算 —— 文件源要用它换算跳帧步长。
    #    （这里出过一次 UnboundLocalError：删除重复定义时删错了位置，
    #      而 py_compile 只查语法、查不出这种顺序问题。）
    interval = window_stride_sec / max(1, frames_per_window)

    # ★ 实时源的取样间隔**不能**用 interval（实测踩到）：
    #   4 帧、6 秒窗口若按 6/4=1.5 秒取样，取样时刻是 0/1.5/3.0/4.5 ——
    #   末帧落在 4.5 秒，而窗口循环立刻开始下一个窗口，
    #   于是**实际节拍变成 4.5 秒**（实测时间戳 0.01→4.52→9.05），
    #   比声明的 6 秒快 1/4：既多花推理算力，窗口时间段还互相重叠
    #   （前一个窗口声明 [0,6]，后一个从 4.5 开始）。
    #   改成 stride/(帧数-1)，末帧正好落在窗口末尾，窗口首尾相接。
    #   文件源不受影响：它按帧号跳，整窗推进量必须还是 window_stride_sec。
    live_interval = window_stride_sec / max(1, frames_per_window - 1)

    # 复用调用方传进来的流（持续监测），否则自己开一条
    owns_cap = cap is None
    if owns_cap:
        cap, kind, source, err = open_source(spec)
        if cap is None:
            raise RuntimeError(err)

    # ⚠️ 实时源要把缓冲压到最小，否则会"看着的其实是几秒前的画面"。
    #    RTSP/USB 摄像头的解码缓冲默认可能攒好几帧，
    #    我们每次都读"最新帧"的策略就失效了。
    #    能设的都设一遍，设不上也不影响（不同后端支持度不一）。
    #    kind 为 None（调用方复用流但没说是哪种）时也照样设，设不上就算了。
    if kind in ("rtsp", "local", None):
        for prop, val in (
            (getattr(cv2, "CAP_PROP_BUFFERSIZE", None), 1),
        ):
            if prop is not None:
                try:
                    cap.set(prop, val)
                except Exception:
                    pass
    self_lag_t0 = time.time()

    # 文件源：按帧号跳转；实时源：按墙钟时间取样
    is_file = kind == "file"
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    src_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    # ⚠️ 裸 .h264（没有容器）读不出帧数，OpenCV 会给一个**荒谬的值**
    #    —— 实测 demo_cam.h264 返回 -153722867280913。
    #    文件源的取样是"按帧号 cap.set(POS_FRAMES, target) 跳过去读"，
    #    帧数一错就整条链断掉：target 算成天文数字 → set 失败 → read 出坏帧
    #    → **一个窗口都取不到** → 外层持续监测把它当成"流断了"无限重连
    #    （实测日志：`实时流已重新连接（第 4 次）` / `采集完成：0 个窗口`，
    #      最后以 ClusterError 结束，而且没人知道真正原因是帧数读成了负数）。
    #    所以这里必须设防：帧数不可信时**不要**指望跳转，改用顺序读。
    #    判据：<=0，或大得超出常理（按 1000fps 算都不可能有那么多帧）。
    src_total_bad = src_total <= 0 or src_total > 1000 * 60 * 60 * 24
    if src_total_bad:
        src_total = 0
    # 顺序读回退模式：不支持按帧号跳的文件（裸 h264）用 grab/retrieve 顺序推进，
    # 读到末尾再 seek(0) 回绕（持续监测要能循环播放）。
    seq_mode = is_file and src_total_bad
    frame_step = max(1, int(round(interval * src_fps))) if is_file else 0
    # 文件源在持续监测下要能"循环播放"，见 loop_file 参数说明
    file_windows = 0
    if is_file and src_total and frame_step:
        file_windows = max(
            1, (src_total // frame_step) // max(1, frames_per_window))
    if seq_mode:
        # 只能"读到一个窗口算一个"，无法预知总窗口数 —— 让回绕判据退化成
        # "读不动了就从头开始"，见下面 win_frames 全空的处理。
        file_windows = 0

    windows: list[list[tuple[float, str]]] = []
    t_start = time.time()
    idx = 0
    # 读帧线程模式下用来判断"这一帧是不是新的"
    last_reader_seq = 0
    # ⚠️ 流死了必须能退出：cap.read() 失败在循环里是"记空继续"，
    #    如果对端静默掉线，就会变成"一帧都读不到但也不报错"的死转，
    #    页面上表现为"一直在采集、进度不动"。
    #    连续几个窗口一帧都没取到，就判定流已不可用，直接退出。
    # 顺序读回退模式下用来判断"源已经读完"（见 seq_mode）。
    # 为什么需要单独一个标志：这个模式下 src_total=0，末尾判断不能靠帧号，
    # 只能靠"read 失败且回绕也失败"这件事本身。没有它，持续监测会一直空转
    # （实测：1 分 50 秒里重连 4 次、每次都是 0 个窗口，最后报错退出）。
    eof = False
    empty_streak = 0

    try:
        while len(windows) < max_windows:
            if should_stop and should_stop():
                break

            win_frames: list[tuple[float, str]] = []
            win_t0 = time.time()
            # ★ 全局窗口序号：`idx+idx_offset` 是本任务累计写出的帧数。
            #   持续模式每批都调用本函数、len(windows) 每批从 0 开始 ——
            #   如果照着 len(windows) 定位，每批都会**重读视频开头那一段**
            #   （实测：页面上时间戳在涨，画面却一直是同一段）。
            gwin = (idx + idx_offset) // max(1, frames_per_window)
            if is_file and loop_file and file_windows:
                gwin %= file_windows      # 循环播放：读到末尾回到开头

            for k in range(frames_per_window):
                if should_stop and should_stop():
                    break

                if is_file:
                    if seq_mode:
                        # ── 顺序读回退（裸 h264：帧数读不出来，不能按帧号跳）──
                        # 按 frame_step 抓一帧、丢 frame_step-1 帧，等价于原来的
                        # "每隔 frame_step 帧取一帧"，只是用顺序推进代替 seek。
                        okf, frame = True, None
                        for _ in range(frame_step):
                            if should_stop and should_stop():
                                break
                            if frame is not None:
                                cap.grab()          # 丢掉中间帧（只 grab 不解码）
                            else:
                                okf, frame = cap.read()
                                if not okf:
                                    break
                        if not okf or frame is None:
                            # 读到末尾：持续监测下回绕到开头重来（否则任务会以
                            # "没有采集到任何窗口"结束，见 loop_file 的说明）
                            if loop_file:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                okf, frame = cap.read()
                            if not okf or frame is None:
                                eof = True
                                break
                    else:
                        # 文件源：直接跳到目标帧，不去等真实时间
                        target_idx = (gwin * frames_per_window + k) * frame_step
                        if src_total and target_idx >= src_total:
                            break
                        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                        okf, frame = cap.read()
                        if not okf or frame is None:
                            continue
                elif reader is not None:
                    # ★ 有专职读帧线程：这里**不自己 read()**，只等取样时刻到了
                    #   取"此刻最新"的那一帧。
                    #   这样推理跑多久都不会挡住画面：读帧线程一直在读、
                    #   一直在更新预览与 MJPEG 缓存（用户反馈的"卡顿"就是这么来的）。
                    target = win_t0 + k * live_interval
                    while time.time() < target:
                        if should_stop and should_stop():
                            break
                        time.sleep(0.02)
                    frame, rseq, _rt = reader.take_newer(
                        last_reader_seq, timeout=max(1.0, live_interval))
                    if frame is None:
                        # 等不到更新的帧 → 流断了或卡住；别把重复帧当输入送模型
                        if reader.dead:
                            break
                        continue
                    last_reader_seq = rseq
                else:
                    # 回退路径（没有读帧线程，例如一次性采集）：老办法，
                    # 等取样时刻期间持续 read() 丢帧，保证拿到最新画面
                    target = win_t0 + k * live_interval
                    while time.time() < target:
                        if should_stop and should_stop():
                            break
                        cap.grab()
                        remaining = target - time.time()
                        if remaining > 0.02:
                            time.sleep(min(0.05, remaining))
                    okf, frame = cap.read()
                    if not okf or frame is None:
                        continue

                h, w = frame.shape[:2]
                if max(h, w) > max_side:
                    scale = max_side / max(h, w)
                    frame = cv2.resize(
                        frame, (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                rel_t = round(time.time() - t_start + t_offset, 2)
                p = out_dir / f"f_{idx + idx_offset:05d}.jpg"
                cv2.imwrite(str(p), frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
                win_frames.append((rel_t, str(p)))
                idx += 1

            if len(win_frames) >= 2:
                windows.append(win_frames)
                empty_streak = 0
                # ⚠️ 长期监测必须裁掉旧帧文件，否则磁盘一直涨
                #    （按 5.5 秒/窗口、4 帧/窗口算，一小时约 2600 张图）。
                #    只保留最近 keep_frames 张：页面能回看最近一段，老的自动画掉。
                if keep_frames and (idx + idx_offset) > keep_frames:
                    olds = sorted(out_dir.glob("f_*.jpg"))[:-keep_frames]
                    for old in olds:
                        try:
                            old.unlink()
                        except OSError:
                            pass

            if on_progress:
                on_progress(len(windows), max_windows)

            if is_file:
                # 文件是有限长度的：跳到超过总帧数就停
                # （loop_file=True 时不停，上面用取模绕回开头继续）
                if not win_frames:
                    eof = True
                if eof:
                    break   # 源已读完（顺序读模式或帧号超界），本批到此为止
                nxt = ((gwin + 1) * frames_per_window) * frame_step
                if src_total and nxt >= src_total and not loop_file:
                    break
            elif not win_frames:
                empty_streak += 1
                if empty_streak >= 3:
                    break   # 连续 3 个窗口全空 → 流已不可用，别再死转
    finally:
        # ★ 只释放自己开的流：复用模式（持续监测）下把流留给下一批，
        #   否则等于每批都重新握手，白白多花 2.9 秒且画面会跳。
        if owns_cap:
            try:
                cap.release()
            except Exception:
                pass

    return windows


def frame_motion(paths: list[str], *, size: int = 48) -> float:
    """窗口内相邻帧的"动作量"：缩到 48×48 灰度后算相邻帧的平均绝对差，取最大值。

    用途：**画面基本没动就别去调模型**（省 GPU 时间、也省用户的等待）。
    实测每窗口推理要 10~12 秒，而空走廊里的窗口占了大多数 ——
    这些窗口送模型纯属浪费。

    ⚠️ 为什么用"相邻帧最大差"而不是"首末帧差"：
        一个人走过去又走回来，首末帧几乎一样但中间动得很厉害；
        用最大值才不会把这种明显有动作的窗口误判成静止。
    ⚠️ 为什么缩到 48×48：既要抗噪（保留一点空间平均），又要几乎不花时间
        （4 张图约 3~5 毫秒，相对于十秒的推理可以忽略）。

    返回值量纲是灰度差（0~255）：静止画面通常在 1 以下，有人走动通常 > 5。
    可读帧少于两张时返回 -1，表示无法测量；不能把损坏输入视为静止。
    """
    import cv2

    prev = None
    worst = 0.0
    readable = 0
    # 当前窗口包含历史上下文和新帧，必须检查整窗，不能只检查最早四帧。
    for p in paths:
        try:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        except Exception:                       # noqa: BLE001
            continue
        if img is None:
            continue
        small = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        readable += 1
        if prev is not None:
            worst = max(worst, float(cv2.absdiff(small, prev).mean()))
        prev = small
    return worst if readable >= 2 else -1.0


def list_local_cameras(max_index: int = 4, *, with_detail: bool = False):
    """探测本地摄像头索引。

    ⚠️ 这个操作会真的打开设备，Windows 上可能触发摄像头指示灯并短暂占用。
    探测完立刻释放。返回可用的索引列表（with_detail=True 时返回
    (found, detail)，detail 说明每个索引用哪些后端试过、结果如何）。

    ⚠️ 为什么要把"试过什么"返回给上层（用户实测踩到）：
        受限令牌（沙箱）下这里会**一个都找不到**，而上层提示语写的是
        "没找到可用摄像头；也可能是被其他程序占用" ——
        把权限问题说成了硬件/占用问题，严重误导。
        现在把逐次尝试的真实结果与进程权限一起交给上层去解释。
    """
    import cv2      # ⚠️ 必须在函数内导入：本模块其它地方都是延迟导入 cv2，
                    #    重写这个函数时漏掉过一次，结果是 NameError 被 except
                    #    吞掉、接口返回"0 个摄像头"—— 又变回那个误导性结论。

    found: list[dict] = []
    tried: list[str] = []
    for i in range(max_index):
        for backend_name, backend in (("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF),
                                      ("ANY", cv2.CAP_ANY)):
            cap = None
            err = ""
            try:
                cap = cv2.VideoCapture(i, backend) if backend != cv2.CAP_ANY \
                    else cv2.VideoCapture(i)
                if not cap.isOpened():
                    err = "打不开"
                else:
                    okf, frame = cap.read()
                    if okf and frame is not None:
                        h, w = frame.shape[:2]
                        found.append({"index": i, "width": w, "height": h,
                                      "backend": backend_name})
                        tried.append(f"#{i}/{backend_name}=成功 {w}×{h}")
                        break                      # 这个索引已经成功，换下一个索引
                    err = "打开了但读不到画面"
            except Exception as e:                  # noqa: BLE001
                err = f"{type(e).__name__}"
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
            tried.append(f"#{i}/{backend_name}={err}")
    return (found, tried) if with_detail else found


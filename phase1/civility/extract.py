"""抽帧 —— 把视频切成"判断窗口"，每个窗口是若干张连续帧。

窗口（window）是整个系统的判断单位：
    一段上下文及其中的多帧 → 一次 VLM 调用 → 一个是否违规的结论。
    当前文件分析默认6秒、最多18帧，每2秒前移一次。

为什么是"按固定速率整段抽帧 + 事后组窗"，而不是"边播边抽"：
    第一阶段要反复调 Prompt。抽帧是一次性成本（几秒），
    之后改 Prompt 只重跑推理，不用重新解码视频 —— 迭代速度差一个数量级。

分辨率策略：
    当前配置长边960px，为烟支和手中小物体保留细节。
    低分辨率可能丢失关键物体，高分辨率会增加上传量和推理开销。
"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from bisect import bisect_left
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .vlm import WindowRequest


@dataclass
class Window:
    window_id: str
    video: str
    t_start: float
    t_end: float
    frames: list[str]
    labels: list[str]
    # 本窗口**第一帧的全局帧序号**（对应文件名 f_00023.jpg 里的 23）。
    #
    # 为什么需要它（实测踩到的坑）：
    #     页面缩略图原本是拿 `int(t_start) + idx` 去推算帧号的 ——
    #     这只在"窗口步长 == 帧间隔，且一帧都没丢"时才成立。
    #     步长 2 秒、帧间隔 2 秒时它**碰巧**对；一旦改步长或丢帧，
    #     取到的就是别的帧（画面与结论对不上，而结果看起来完全正常）。
    #     直接把帧号带在窗口上，就与时间戳的取值方式彻底解耦。
    # 为什么不在 extract.py 里算：
    #     视频文件路径走 ffmpeg 抽帧，序号由文件名决定；实时流由采集循环
    #     决定。两边各自知道自己的序号，所以由**构造窗口的人**填。
    #     -1 = 未提供（旧调用方/序列化回来的记录），调用方回退旧算法。
    idx0: int = -1
    # Optional live sampling provenance; timestamps remain actual sampled frames.
    sampling: dict = field(default_factory=dict)

    @property
    def frame_indices(self) -> list[int]:
        return [frame_index_of(p) for p in self.frames]

    def to_request(self) -> WindowRequest:
        return WindowRequest(
            window_id=self.window_id,
            video=self.video,
            t_start=self.t_start,
            t_end=self.t_end,
            frames=self.frames,
            labels=self.labels,
        )


def find_ffmpeg() -> tuple[str, str]:
    """定位 ffmpeg / ffprobe。找不到就给出可执行的修复建议。"""
    ff = shutil.which("ffmpeg")
    fp = shutil.which("ffprobe")
    if not ff or not fp:
        raise RuntimeError(
            "找不到 ffmpeg/ffprobe。Windows 上可用：winget install Gyan.FFmpeg\n"
            "或从 https://www.gyan.dev/ffmpeg/builds/ 下载后把 bin 目录加入 PATH。"
        )
    return ff, fp


def probe_duration(video: str | Path) -> float:
    """返回视频时长（秒）。"""
    _, fp = find_ffmpeg()
    out = subprocess.run(
        [
            fp,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def extract_frames(
    video: str | Path,
    out_dir: str | Path,
    *,
    fps: float = 1.0,
    max_side: int = 640,
    jpeg_quality: int = 82,
    overwrite: bool = False,
) -> list[tuple[float, str]]:
    """按固定 fps 抽帧并缩放，返回 [(时间戳, 图片路径)]。

    单条 ffmpeg 命令同时完成 fps 采样的缩放与编码，比逐帧调用快得多。
    """
    video = Path(video)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 已抽过就直接复用（抽帧是"一次性成本"的关键前提）
    existing = sorted(out_dir.glob("f_*.jpg"))
    cache_file = out_dir / "sampling.json"
    signature = {"video": str(video.resolve()), "size": video.stat().st_size,
                 "mtime_ns": video.stat().st_mtime_ns, "fps": fps,
                 "max_side": max_side, "jpeg_quality": jpeg_quality}
    try:
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cached = None
    if existing and not overwrite and cached == signature:
        return [(i / fps, str(p)) for i, p in enumerate(existing)]

    if existing:
        for p in existing:
            p.unlink()

    ff, _ = find_ffmpeg()
    pattern = str(out_dir / "f_%05d.jpg")
    vf = f"fps={fps},scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'"

    subprocess.run(
        [
            ff,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            vf,
            "-q:v",
            str(max(2, min(31, int(31 - jpeg_quality / 100 * 31)))),
            "-start_number",
            "0",
            pattern,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    files = sorted(out_dir.glob("f_*.jpg"))
    cache_file.write_text(json.dumps(signature), encoding="utf-8")
    return [(i / fps, str(p)) for i, p in enumerate(files)]


def build_windows(
    video: str | Path,
    frames: list[tuple[float, str]],
    *,
    window_stride_sec: float = 4.0,
    max_frames_per_window: int = 4,
    window_duration_sec: float | None = None,
) -> list[Window]:
    """把帧序列切成判断窗口。

    window_duration_sec 决定上下文时长，stride 决定向前移动的间隔。
    超过上限时均匀取样，保留首尾帧；较长上下文允许相邻窗口重叠。
    帧太少（<2）时不做时序判断 —— 单帧无法区分"手持"和"已丢弃"，
    这时宁可不判，也不要给模型一个它必然答错的题。
    """
    video = str(video)
    stem = Path(video).stem
    windows: list[Window] = []

    if not math.isfinite(window_stride_sec) or window_stride_sec <= 0:
        raise ValueError("window_stride_sec 必须为正数")
    if not frames:
        return windows

    duration = window_duration_sec if window_duration_sec is not None else window_stride_sec
    if not math.isfinite(duration) or duration < window_stride_sec:
        raise ValueError("窗口时长必须不小于窗口步长")
    if max_frames_per_window < 2:
        raise ValueError("每窗至少需要2帧")
    ordered = sorted(frames)
    times = [f[0] for f in ordered]
    interval = ordered[-1][0] - ordered[-2][0] if len(ordered) > 1 else 0
    end = ordered[-1][0] + interval
    # The last window already covers the tail. Do not re-analyze shorter subsets
    # that omit the same action's earlier evidence (e.g. cigarette vs later smoke).
    n_windows = max(1, math.ceil((end - duration) / window_stride_sec) + 1)
    for bi in range(n_windows):
        t_start = bi * window_stride_sec
        block = ordered[bisect_left(times, t_start):bisect_left(times, t_start + duration)]
        block = sample_evenly(block, max_frames_per_window)
        if len(block) < 2:
            continue  # 帧不足，跳过（见函数文档）

        wid = f"{stem}__w{len(windows):04d}"
        windows.append(
            Window(
                window_id=wid,
                video=video,
                t_start=round(t_start, 2),
                t_end=round(min(t_start + duration, end), 2),
                frames=[p for _, p in block],
                labels=[f"t={t:.1f}s" for t, _ in block],
                # 视频文件路径下帧名是 f_%05d.jpg（ffmpeg 按 -start_number 0
                # 生成），所以"文件里的数字"就是全局帧序号 —— 从文件名取，
                # 比用桶号 × 每窗帧数可靠（桶会因帧稀疏而拿不满）。
                idx0=frame_index_of(block[0][1]),
            )
        )
    return windows


def sample_evenly(items: list, limit: int) -> list:
    """Keep both endpoints and distribute the frame budget over the whole interval."""
    if len(items) <= limit:
        return items
    if limit < 2:
        raise ValueError("采样上限至少为2")
    return [items[round(i * (len(items) - 1) / (limit - 1))] for i in range(limit)]


# ⚠️ 帧文件名 → 全局帧序号。**只认文件名里的数字**，不 stat 文件、不猜时间。
#    踩过的坑：页面上"第几帧"原本是拿 `int(t_start)` 推算的，那只在
#    "窗口步长 == 帧间隔、且一帧没丢"时才成立；改步长就会指到别的帧上。
#    文件名的数字是采集时写死的，永远准。
#
#    模式**故意收得很紧**（只有 f_数字.jpg 一种）：宽松的匹配会把
#    preview_*.jpg 之类也算成帧，而"算错了"比"认不出"危险得多 ——
#    认不出会返回 -1 让调用方回退，算错则是静默显示错误的画面。
_FRAME_RE = re.compile(r"^f_(\d+)\.jpg$")


def frame_index_of(path: str | Path) -> int:
    """从 f_00023.jpg 里取出 23；认不出返回 -1。

    认不出不算错误（旧目录、别的命名），调用方自己回退。
    """
    m = _FRAME_RE.match(Path(path).name)
    return int(m.group(1)) if m else -1


def write_manifest(windows: list[Window], path: str | Path) -> Path:
    """落盘 manifest.jsonl —— 这是本地与集群之间唯一的接口契约。

    集群脚本只认这个格式，不认识 behaviors.yaml 之外任何东西。
    好处：以后把后端换成真摄像头、换成别的模型，这个契约都不用动。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for w in windows:
            f.write(json.dumps(asdict(w), ensure_ascii=False) + "\n")
    return path


def read_manifest(path: str | Path) -> list[Window]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(Window(**d))
    return out

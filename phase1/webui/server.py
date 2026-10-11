#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本机 Web 后端 —— 可视化界面的中枢。

它做四件事：
    1. 管集群连接：提交常驻推理作业、发现节点、维护两级 SSH 隧道
    2. 管视频：接收上传的视频，解码切窗，打包成帧
    3. 管推理：把窗口抽帧发给计算节点上的模型，回收结构化结果
    4. 管告警：时序平滑 + 去抖，把逐窗口结论变成告警事件

为什么放在本机而不是服务器：
    · 登录节点没装 web 框架、且禁止长期占用（已实测）
    · 计算节点是作业，会随作业结束消失
    · 本机能跑 FastAPI，浏览器直连 localhost 最省事
    服务器负责"重"的部分（模型），本机负责"黏合"和"呈现"。

启动：
    python -m webui.server            # 默认 http://127.0.0.1:8848
"""
from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import functools
import hashlib
import io
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# ⚠️ 必须在导入 civility 之前放宽 Windows 控制台编码，
#    否则中文日志会 UnicodeEncodeError 直接崩掉服务。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent                      # phase1/
sys.path.insert(0, str(PROJECT))

from civility.aggregate import derive_events          # noqa: E402
from civility.config import load_behaviors, load_settings  # noqa: E402
from civility.extract import build_windows, extract_frames, frame_index_of  # noqa: E402
from civility.prompts import sanitize, build_system_prompt  # noqa: E402
from civility.recognition import generation_budget, infer_with_review  # noqa: E402

# ⚠️ FastAPI 的类型必须import在模块顶层。
#    踩过的坑：把 `from fastapi import UploadFile` 写在 create_app() 内部，
#    路由签名里的 UploadFile 就成了无法解析的前向引用，Pydantic 报
#    "TypeAdapter[ForwardRef('UploadFile')] is not fully defined"，
#    表现为上传接口 500 —— 而且只有上传接口会挂，其他都正常，很隐蔽。
from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import (  # noqa: E402
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)

from .cluster_client import ClusterClient, ClusterError, ServiceSession  # noqa: E402
from .video_library import VideoLibrary  # noqa: E402

QA_BUSY = threading.Lock()


@functools.lru_cache(maxsize=8)
def _library_for(results: str, frames: str, data: str) -> VideoLibrary:
    return VideoLibrary(Path(results) / "video_analyses", Path(frames), Path(data) / "uploads")


def video_library() -> VideoLibrary:
    paths = STATE.cfg["paths"]
    return _library_for(str(paths["results_dir"]), str(paths["frames_dir"]), str(paths["data_dir"]))


def archive_window(job, rec: dict) -> None:
    if not getattr(job, "_archive_enabled", False):
        return
    try:
        # A looping file repeats the same source; archive one complete pass.
        if job.loop and getattr(job, "_archived_windows", 0) >= job.windows_total:
            return
        fps = float((STATE.cfg.get("extract") or {}).get("fps", 3))
        saved = dict(rec)
        saved["frame_times"] = [i / fps for i in rec.get("frame_indices", [])]
        video_library().record(job, saved)
        job._archived_windows = getattr(job, "_archived_windows", 0) + 1
    except Exception as exc:
        job._archive_error = f"保存视频分析窗口失败：{type(exc).__name__}: {exc}"
        STATE.log_line(job._archive_error)


def runtime_source_signature() -> str:
    files = sorted((PROJECT / "civility").glob("*.py")) + sorted(
        (PROJECT / "cluster").glob("*.py")) + [
        Path(__file__), HERE / "cluster_client.py", HERE / "capture.py", HERE / "video_library.py",
        HERE / "motion_highlights.py",
        PROJECT / "cluster/serve-vlm.sbatch"]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.read_bytes())
    return digest.hexdigest()


LOADED_SOURCE_SIGNATURE = runtime_source_signature()


def runtime_status() -> dict:
    settings = Path(os.environ.get("CIVILITY_SETTINGS") or PROJECT / "config/settings.yaml")
    changed = hashlib.sha256(settings.read_bytes()).hexdigest() != STATE.settings_signature
    return {"protocol": 3, "restart_required": changed or runtime_source_signature() != LOADED_SOURCE_SIGNATURE,
            "settings_changed": changed}


def ensure_current_runtime() -> None:
    if runtime_status()["restart_required"]:
        raise HTTPException(409, "本机服务仍在使用旧代码或旧配置。请在原终端停止并重新启动网页服务，再刷新页面。")
    if STATE.behaviors_changed():
        STATE.reload_behaviors()

# ─────────────────────────── 全局状态 ───────────────────────────


@dataclass
class JobRecord:
    """一次检测任务。来源可以是上传的视频，也可以是实时流。"""

    job_id: str
    filename: str
    state: str = "queued"       # queued|capturing|extracting|running|done|error|stopped
    progress: float = 0.0
    message: str = ""
    created: float = field(default_factory=time.time)
    windows_total: int = 0
    windows_done: int = 0
    alerts: list = field(default_factory=list)
    window_results: list = field(default_factory=list)
    error: str = ""
    events: "queue.Queue" = field(default_factory=queue.Queue)
    stop_flag: bool = False
    avg_latency: float = 0.0
    # ── 来源 ──
    source_kind: str = "file"   # file | local | rtsp
    source_spec: str = ""       # "0" / "rtsp://..." / 上传文件路径
    is_live: bool = False
    # 持续监测模式：窗口采完不结束，继续跑直到用户停止。
    # 这是"摄像头监测"页要的行为 —— 系统一直看着画面。
    continuous: bool = False
    # 是否要求模型描述"画面里每个人在做什么"（监测模式必开）
    with_actions: bool = False
    # 用本地 mock 替代集群模型。
    # 用途：没有 VPN / 集群排队时，仍能完整验证监测链路（采集→SSE→渲染）。
    # 页面会明确标注这是模拟数据，避免把假结果当成真实识别效果。
    use_mock: bool = False
    # 演示场景（见 civility.vlm.MOCK_SCENES）：让演示模式的
    # 场景描述 / 人物动作 / 判定依据 / 告警**互相自洽**，
    # 否则用户看到的就是一堆和画面对不上的占位文字（实测被反馈过）。
    mock_scene: str = ""
    # ── 文件任务 ──
    # 循环播放：跑完一遍后是否从头再来。**默认关闭** ——
    # 踩过的坑：以前外层 while 没有出口，文件任务会无限重跑同一段视频
    # （实测 3 个窗口被分析了 4 遍），页面上同一个窗口反复出现、
    # 告警却要等到整遍跑完才出。现在改成显式开关，默认跑一遍就结束。
    loop: bool = False
    # 切窗结果缓存：一次任务只抽一次帧（ffmpeg 很贵，而且它会清空帧目录，
    # 把上一轮的缩略图删成碎图）。重跑检测时才清空。
    cached_windows: list | None = None
    # 当前任务的读帧线程（供 MJPEG 实时流接口取最新帧；由 run_job 挂上）
    reader: object | None = None
    # ⚠️ 长期运行必须有上限：窗口结果与帧文件都会持续增长。
    #    按 5.5 秒/窗口算，一小时约 650 个窗口；不限就吃满磁盘和内存。
    max_kept_windows: int = 200
    windows_seen: int = 0
    # 持续监测的累计偏移：保证跨批次的时间戳与窗口 ID 全局单调、不重复
    t_offset: float = 0.0
    frame_idx_offset: int = 0
    # 窗口全局序号：持续模式每批从 0 编号会与上一批 ID 撞车（实测踩到）
    window_seq: int = 0
    # 任务开始采集的时刻，用于把各批时间戳换算成"距开始的秒数"
    capture_started_at: float | None = None
    # 上一批的总耗时（推理+采集），用来判断系统是否跟不上实时
    last_batch_sec: float = 0.0
    behind: bool = False
    result_age_sec: float = 0.0
    # 门控以视频时间为准；跳过窗口不是一次新的模型检查。
    last_model_t_end: float | None = None
    # 窗口会被裁剪，告警去重状态必须独立保留。
    alert_checked_until: float = -1.0
    alert_last_by_type: dict = field(default_factory=dict)
    max_kept_alerts: int = 500
    context_frames: list = field(default_factory=list)
    recent_frame_files: dict = field(default_factory=dict)
    last_recent_wall: float | None = None
    last_model_frame_at: float | None = None
    live_metrics: dict = field(default_factory=dict)

    def summary(self) -> dict:
        age = self.result_age_sec
        if (self.is_live and self.source_kind != "file" and self.last_model_frame_at is not None
                and self.state in ("capturing", "running")):
            age = max(0.0, time.time() - self.last_model_frame_at)
        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "state": self.state,
            "progress": round(self.progress, 4),
            "message": self.message,
            "windows_total": self.windows_total,
            "windows_done": self.windows_done,
            "alerts": self.alerts,
            "error": self.error,
            "created": self.created,
            "avg_latency": round(self.avg_latency, 2),
            "result_age_sec": round(age, 2),
            "behind": self.behind,
            "live_metrics": dict(self.live_metrics),
            "source_kind": self.source_kind,
            "is_live": self.is_live,
            "continuous": self.continuous,
            "with_actions": self.with_actions,
            "use_mock": self.use_mock,
            "mock_scene": self.mock_scene,
            "loop": self.loop,
            "windows_seen": self.windows_seen,
        }


class AppState:
    def __init__(self) -> None:
        # ★ CIVILITY_SETTINGS：让测试/诊断能指到一份**隔离的** settings.yaml。
        #   为什么需要：想拿真实 HTTP 链路做端到端验证（例如"两个窗口的图是不是
        #   真的不一样"），就必须能换一个 frames_dir，否则会写进用户正在使用的
        #   data/frames/<job_id>，把页面上的历史画面搅乱。
        #   （tools/test_frame_cache.py 就是这么做的。）
        self.cfg = load_settings(os.environ.get("CIVILITY_SETTINGS") or None)
        settings = Path(os.environ.get("CIVILITY_SETTINGS") or PROJECT / "config/settings.yaml")
        self.settings_signature = hashlib.sha256(settings.read_bytes()).hexdigest()
        self.behaviors = load_behaviors(self.cfg["paths"]["behaviors"])
        self.session: ServiceSession | None = None
        self.jobs: dict[str, JobRecord] = {}
        self.uploads: dict[str, Path] = {}
        self.lock = threading.Lock()
        self.log: list[str] = []
        self.behaviors_mtime = self._behaviors_mtime()

        # ── 成本守卫相关的状态（用户的原话："我不用的时候也在花钱"）──
        # ⚠️ 这里为什么要分三个时间戳（实测发现的漏洞）：
        #    最初只有一个 last_ui_seen，而心跳是固定每 30 秒发一次的 ——
        #    **标签页切到后台时定时器照样在跑**，心跳一直更新，
        #    于是"人已经走了"永远判不出来，节点永远不会被释放。
        #    所以分开记：visible=true 才更新 last_visible_at；
        #    人在页面上的真实操作（鼠标/键盘）才更新 last_interact。
        self.page_visible: bool = False
        self.last_visible_at: float = time.time()   # 最后一次"页面可见"
        self.last_interact: float = time.time()     # 最后一次用户操作
        self.last_ui_seen: float = time.time()      # 最后一次收到心跳（诊断用）
        self.auto_released: str = ""      # 非空 = 最近一次是被自动释放的，附原因
        self.orphans: list[dict] = []     # 启动时扫到的、不在我们管理下的作业
        self.cap_extended_at: float = 0.0   # 最长运行上限的最近一次延长时刻
        self.cost_note: str = ""

    # ── 行为定义 ──
    def _behaviors_mtime(self) -> float:
        p = Path(self.cfg["paths"]["behaviors"])
        return p.stat().st_mtime if p.exists() else 0.0

    def reload_behaviors(self) -> None:
        self.behaviors = load_behaviors(self.cfg["paths"]["behaviors"])
        self.behaviors_mtime = self._behaviors_mtime()

    def behaviors_changed(self) -> bool:
        return self._behaviors_mtime() != self.behaviors_mtime

    # ── 集群 ──
    def client(self) -> ClusterClient:
        cl = self.cfg.get("cluster") or {}
        return ClusterClient(
            workdir=cl.get("remote_workdir", "/share/home/u00009/apps/civility-kit"),
            python=cl.get("remote_python",
                          "/share/home/u00009/apps/conda/envs/vlm/bin/python"),
            model_root=cl.get("remote_model_root", "/ssdfs/datahome/u00009/models"),
        )

    def log_line(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        self.log.append(line)
        self.log = self.log[-300:]
        print(line, flush=True)


STATE = AppState()


# ─────────────────────── 成本守卫（别让 A800 空转烧钱）───────────────────────
#
# 为什么必须有这个（真实账单）：
#     用户反馈"我不用的时候也在花钱"。对账后发现 14 天里
#     一个 civility-serve 作业跑满 12 小时被 TIMEOUT，
#     单它就占了约 12 GPU·小时 ≈ 48 元 —— 占这期间全部花费的 85%。
#     原因是孤儿作业：网页重启后内存里的作业号丢了，
#     集群上的作业继续跑到 sbatch 的 12 小时上限，没人管。
#     所以"点了停止"不足以省心，必须有**不依赖人**的兜底。


def _price_per_hour() -> float:
    svc = STATE.cfg.get("service") or {}
    try:
        return float(svc.get("gpu_price_per_hour", 4.0))
    except (TypeError, ValueError):
        return 4.0


def service_cost() -> dict:
    """服务已运行时长与估算花费 —— 让"正在烧钱"变成看得见的数字。"""
    sess = STATE.session
    up = 0.0
    if sess is not None and sess.started_at:
        up = time.time() - sess.started_at
    price = _price_per_hour()
    return {
        "uptime_sec": round(up, 1),
        "uptime_text": (f"{int(up // 3600)} 小时 {int(up % 3600 // 60)} 分"
                        if up else "—"),
        "est_cost": round(up / 3600 * price, 2),
        "price_per_hour": price,
    }


def describe_orphans(jobs: list[dict]) -> list[dict]:
    """挑出"在集群上跑着、但不归当前网页管"的作业（孤儿）。"""
    mine = STATE.session.jobid if STATE.session else None
    out = []
    for j in jobs or []:
        if j.get("jobid") == mine:
            continue
        out.append(j)
    return out


def check_orphans() -> list[dict]:
    """问集群：有没有我名下、但当前网页不知道的 civility 作业。"""
    try:
        cl = STATE.client()
        jobs = cl.list_my_jobs()
    except Exception:
        return STATE.orphans          # 连不上就沿用上次结果，别把 UI 搞崩
    ours = [j for j in jobs
            if str(j.get("name", "")).startswith("civility-")]
    STATE.orphans = describe_orphans(ours)
    return STATE.orphans


def _active_job() -> bool:
    return QA_BUSY.locked() or any(j.state in ("capturing", "extracting", "running", "queued")
               for j in STATE.jobs.copy().values())


def watchdog_tick() -> str:
    """成本守卫的一拍。返回做了什么的说明（空 = 没动作）。

    三条规则：
        1. 空闲释放：页面关掉或切到后台超过 idle_release_min 分钟 → 停服务。
        2. 最长运行上限：服务连续跑超过 max_run_hours → 停服务；
           但如果**此刻正在分析画面**，不打断干活，改成每 30 分钟延长一次
           并大声记日志（人还在用，硬切掉比省钱更糟）。
        3. 孤儿提示：启动时扫到就记日志并在页面顶部挂横幅（不自动杀，
           避免误杀你从别处提交的作业）。
    """
    svc = STATE.cfg.get("service") or {}
    sess = STATE.session
    if sess is None or sess.state not in ("ready", "running", "pending"):
        return ""

    idle_min = float(svc.get("idle_release_min", 15) or 0)
    max_hours = float(svc.get("max_run_hours", 3) or 0)
    ups = (time.time() - sess.started_at) if sess.started_at else 0.0

    # ── 1. 空闲释放 ──
    if idle_min > 0 and not _active_job():
        # 人在看页面 → 按"最后一次真实操作"算；页面不可见/已关闭 → 按
        # "最后一次可见"算。两条路都能走到释放，因为定时器不会因为
        # 切后台就停（这个漏洞害得守卫永远不触发，测试里抓到过）。
        ref = STATE.last_interact if STATE.page_visible else STATE.last_visible_at
        idle = time.time() - ref
        # 服务刚起来时给一个宽限期，否则"启动完还没打开页面"就会立刻被退掉
        if ups > idle_min * 60 and idle > idle_min * 60:
            if STATE.page_visible:
                why = (f"页面开着但已 {idle / 60:.0f} 分钟没有任何操作"
                       f"（鼠标/键盘都没动）")
            else:
                why = (f"页面已关闭或切到后台 {idle / 60:.0f} 分钟")
            return _auto_release(f"{why}（空闲自动释放，避免空转计费）")

    # ── 2. 最长运行上限 ──
    if max_hours > 0 and ups > max_hours * 3600:
        if _active_job():
            if (time.time() - STATE.cap_extended_at) > 1800:
                STATE.cap_extended_at = time.time()
                STATE.log_line(
                    f"⚠️ 服务已连续运行 {ups / 3600:.1f} 小时（超过上限 "
                    f"{max_hours:.1f} 小时），但**正在分析画面**，先不打断；"
                    f"30 分钟后再判断。若长期不用请点「停止并释放节点」。"
                )
            return ""
        return _auto_release(
            f"服务已连续运行 {ups / 3600:.1f} 小时，达到自设上限 "
            f"{max_hours:.1f} 小时（防止忘记关闭一直在计费）")

    return ""


def _auto_release(why: str) -> str:
    """停服务 + 扫孤儿 + 把原因写到页面能看见的地方。"""
    sess = STATE.session
    cost = service_cost()
    try:
        cancelled = sess.stop(sweep_orphans=True) if sess else []
    except Exception as e:
        STATE.auto_released = (
            f"{why}。自动释放失败：{type(e).__name__}: {e}。"
            "请重试「停止并释放节点」。"
        )
        STATE.log_line(f"⚠️ {STATE.auto_released}")
        return STATE.auto_released
    STATE.auto_released = (
        f"{why}。已释放节点，共运行 {cost['uptime_text']}，"
        f"约省下 {cost['est_cost']:.2f} 元"
        + (f"；另外清理了 {len(cancelled)} 个遗留作业" if cancelled else "")
    )
    STATE.log_line(f"🛑 {STATE.auto_released}")
    return STATE.auto_released


# ─────────────────────────── 视频 → 窗口 ───────────────────────────


def _last_window_had_hit(job: JobRecord) -> bool:
    """上一窗有没有命中违规。

    门控（画面没动就跳过模型）必须在"上一窗有命中"时**失效** ——
    刚报过违规说明场面正有事，这时候省一次调用可能就漏掉了后续动作。
    """
    if not job.window_results:
        return False
    latest = job.window_results[-1]
    v = latest.get("violations") or {}
    for item in v.values() if isinstance(v, dict) else []:
        if isinstance(item, dict) and item.get("violation"):
            return True
    return any(isinstance(r, dict) and r.get("alert") is True
               for r in (latest.get("risks") or []))


def _should_skip_window(job: JobRecord, motion: float, threshold: float,
                        t_end: float, recheck_sec: float,
                        last_checked: float | None) -> bool:
    """静止可省调用，但首窗、检查失败、风险后续和到期复检必须送模型。"""
    if not job.continuous or not job.window_results or last_checked is None:
        return False
    if (job.window_results[-1].get("error")
            or job.window_results[-1].get("evidence_warnings") or _last_window_had_hit(job)):
        return False
    return (0 <= motion < threshold
            and t_end - last_checked < recheck_sec)


def _refresh_alerts(job: JobRecord, video_label: str, agg_cfg: dict) -> None:
    """按当前累积的窗口结果重算告警，写回 job.alerts。

    抽成函数是因为它现在要在**每个窗口之后**都被调用（增量告警），
    以前只在整批之后调用一次 —— 那次改动正是"告警要等中止才出现"的根因。

    聚合是按"视频"分组的（derive_events 内部按 video 归并），
    所以实时流和文件都必须传一个稳定的名字，否则不同来源的窗口会互相串。
    """
    from civility.vlm import WindowResult

    results = [
        WindowResult(
            window_id=r["window_id"], video=video_label,
            t_start=r["t_start"], t_end=r["t_end"],
            parsed={"scene": r["scene"], "violations": r["violations"],
                    "people": r.get("people") or [],
                    "risks": r.get("risks") or []},
            raw=r["raw"], error=r["error"], backend="cluster",
        )
        for r in job.window_results
    ]
    # 持续任务用独立状态去重。若先在裁剪后的窗口内去重，旧起点会压制
    # 当前候选，导致达到 dedup_sec 后也无法产生新的提醒。
    local_cfg = {**agg_cfg, "dedup_sec": 0} if job.continuous else agg_cfg
    alerts = derive_events(results, STATE.behaviors, local_cfg)
    if not job.continuous:
        job.alerts = [a.to_dict() for a in alerts]
        return
    # 只接收本次新增窗口产生的告警，避免裁剪历史后重算的起点移动。
    severity_order = {"low": 0, "medium": 1, "high": 2}
    for alert in alerts:
        if alert.t_end <= job.alert_checked_until:
            continue
        key = alert.behavior_id
        last = job.alert_last_by_type.get(key)
        timing = _live_alert_timing(job, alert)
        # Motion-preserved windows can start long before the new violation.
        # Only the new live sampling protocol may change dedup's time basis;
        # file analyses and historical/legacy records retain their semantics.
        real_sampling = (job.is_live and job.source_kind != "file"
            and any(r.get("sampling") and r.get("frame_times")
                    for r in job.window_results if r.get("window_id") in alert.window_ids))
        dedup_time = (timing.get("evidence_time_sec") if real_sampling else alert.t_start)
        if dedup_time is None:
            dedup_time = alert.t_end
        upgraded = (last is not None and key.startswith("ai_risk:")
                    and severity_order.get(alert.severity, 0)
                    > severity_order.get(last[1], 0))
        if last is not None and not upgraded:
            if dedup_time - last[0] < float(agg_cfg.get("dedup_sec", 30)):
                continue
        payload = alert.to_dict()
        payload.update(timing)
        job.alerts.append(payload)
        job.alert_last_by_type[key] = (dedup_time, alert.severity)
        if timing:
            age = timing["evidence_age_sec"]
            job.live_metrics["last_alert_age_sec"] = age
            if age is not None:
                job.live_metrics["max_alert_age_sec"] = max(age, job.live_metrics.get("max_alert_age_sec") or 0)
    if results:
        job.alert_checked_until = max(r.t_end for r in results)
    job.alerts = job.alerts[-job.max_kept_alerts:]
    if len(job.alert_last_by_type) > job.max_kept_alerts:
        # 模型即使不断生成新类别，去重状态也不能无限增长。
        newest = sorted(job.alert_last_by_type.items(), key=lambda kv: kv[1][0],
                        reverse=True)[:job.max_kept_alerts]
        job.alert_last_by_type = dict(newest)


def _live_alert_timing(job: JobRecord, alert) -> dict:
    """Use the model's actual evidence references, not the newer window tail.

    Open risks may have no numbered evidence; leave its age unknown and expose
    the window image age separately. Neither age claims an exact event onset.
    """
    published = time.time()
    if not job.is_live or job.source_kind == "file" or job.capture_started_at is None:
        return {}
    records = [r for r in job.window_results if r.get("window_id") in alert.window_ids]
    evidence_times = []
    for rec in records:
        item = (rec.get("violations") or {}).get(alert.behavior_id) or {}
        refs = item.get("evidence") or {}
        times = rec.get("frame_times") or []
        if not isinstance(refs, dict):
            continue
        evidence_times.extend(times[ref - 1] for ref in refs.values()
                              if type(ref) is int and 1 <= ref <= len(times))
    age = max(0.0, published - job.capture_started_at - max(evidence_times)) if evidence_times else None
    window_age = max(0.0, published - job.capture_started_at - alert.t_end)
    return {"published_at": published, "notification_age_sec": round(age, 3) if age is not None else None,
            "evidence_age_sec": round(age, 3) if age is not None else None,
            "window_age_sec": round(window_age, 3),
            "evidence_time_sec": max(evidence_times) if evidence_times else None}


def slice_video_to_windows(video: Path, job: JobRecord) -> list:
    """把上传的视频解码成判断窗口。用 ffmpeg 抽帧（走 civility.extract）。"""
    ex = STATE.cfg.get("extract") or {}
    dflt = STATE.behaviors.defaults or {}
    frames_root = Path(STATE.cfg["paths"]["frames_dir"]) / job.job_id

    job.state = "extracting"
    job.message = "ffmpeg 抽帧中…"
    frames = extract_frames(
        video, frames_root,
        fps=float(ex.get("fps", 1.0)),
        max_side=int(ex.get("max_side", 640)),
        jpeg_quality=int(ex.get("jpeg_quality", 82)),
    )
    windows = build_windows(
        video, frames,
        window_stride_sec=float(dflt.get("window_stride_sec", 4.0)),
        max_frames_per_window=int(ex.get("max_frames_per_window", 4)),
        window_duration_sec=float(ex.get("window_duration_sec", dflt.get("window_stride_sec", 4.0))),
    )
    job.windows_total = len(windows)
    job.message = f"抽帧完成：{len(frames)} 帧 → {len(windows)} 个窗口"
    STATE.log_line(f"[{job.job_id[:8]}] {job.message}")
    return windows


def _port_owner(port: int) -> str:
    """找出占用某端口的进程 PID（用于提示用户杀掉残留的旧服务）。

    踩过的坑：不能用 Get-NetTCPConnection 判断端口占用（嵌套环境里不准），
    netstat -ano 才可靠。
    """
    import subprocess

    try:
        out = subprocess.run(["netstat", "-ano"], capture_output=True,
                             text=True, timeout=15).stdout
    except Exception:
        return ""
    for line in out.splitlines():
        if "LISTENING" not in line:
            continue
        parts = line.split()
        if len(parts) >= 5 and parts[1].endswith(f":{port}"):
            return parts[-1]
    return ""


def prune_frames(frames_root: Path, keep: int) -> int:
    """只保留最近 keep 个帧文件，返回删除数量。

    为什么放在"一批推理完之后"，而不是采集循环里（实测踩到）：
        采集还在往同一个目录里写 f_00012.jpg…，这时候剪"最旧的"看着没问题，
        但视频文件源一秒钟能采上百个窗口、写上千张图，
        一次采集调用里前面的窗口还没送去推理就被剪掉了 ——
        那些窗口推理时读不到帧，被当成"没采到窗口"整窗丢弃，
        页面上表现为"窗口数在涨、结果一条都没有"。
        等这一批推理读完帧再剪，剪掉的一定是已经用过的旧帧。

    帧名是 f_{全局帧号:05d}.jpg，按名字排序 = 按新旧排序
    （五位数补零，所以字符串序和数字序一致）。
    """
    if keep <= 0:
        return 0
    try:
        files = sorted(frames_root.glob("f_*.jpg"))
    except OSError:
        return 0
    if len(files) <= keep:
        return 0
    n = 0
    for p in files[:-keep]:
        try:
            p.unlink()
            n += 1
        except OSError:
            pass
    return n


class LiveStream:
    """持续监测期间**复用同一条**已经打开的实时流。

    为什么需要它（实测踩到的坑）：
        每批都重新 open_source 时，RTSP 要重新走一遍
        OPTIONS/DESCRIBE/SETUP/PLAY 握手 —— 本机实测约 2.9 秒，
        而窗口步长是 6 秒：握手就吃掉一半预算，
        再叠上采集与推理，节拍必然慢于实时，画面越看越旧。
        开一次、用到任务结束，节拍才由"采集+推理"决定。

    失败处理：流坏了（网络抖动、摄像头重启）不能一直用，
        drop() 之后下一批会重新打开，页面顶部会打日志说明。
    """

    def __init__(self, spec: str, *, log_prefix: str = "") -> None:
        self.spec = spec
        self.cap = None
        self.kind: str | None = None
        self.opened_at: float = 0.0
        self.reopen_count: int = 0
        self.log_prefix = log_prefix
        # ★ 专职读帧线程：由它独占 VideoCapture，持续读最新帧并更新
        #   预览帧与 MJPEG 缓存。这样**推理期间画面照走**（治"实时画面卡顿"），
        #   采样也永远拿到此刻最新的帧（不再有十几秒滞后）。
        self.reader = None

    def ensure_reader(self, *, max_side: int, jpeg_quality: int,
                      preview_dir) -> None:
        """确保读帧线程在跑（在流已打开之后调用）。"""
        from .capture import StreamReader

        self.ensure()
        if self.reader is not None and not self.reader.dead:
            return
        if self.reader is not None:
            self.reader.stop()
            self.reader = None
            STATE.log_line(f"[{self.log_prefix}] 读帧线程已停止（流被判定为断），重建")
        self.reader = StreamReader(
            self.cap, self.kind or "?", max_side=max_side,
            jpeg_quality=jpeg_quality, preview_dir=preview_dir,
            preview_fps=3.0, name=self.log_prefix)
        self.reader.start()

    def ensure(self) -> None:
        """需要时打开流；已打开就直接复用。"""
        if self.cap is not None:
            return
        from .capture import mask_credentials, open_source

        t0 = time.time()
        cap, kind, source, err = open_source(self.spec)
        if cap is None:
            raise ClusterError(
                f"打不开实时流 {mask_credentials(self.spec)}：{err}"
            )
        self.cap, self.kind, self.opened_at = cap, kind, time.time()
        if self.reopen_count:
            STATE.log_line(
                f"[{self.log_prefix}] 实时流已重新连接（第 {self.reopen_count} 次），"
                f"用时 {time.time() - t0:.1f}s"
            )
        else:
            STATE.log_line(
                f"[{self.log_prefix}] 实时流已连接（{kind}，"
                f"握手 {time.time() - t0:.1f}s），后续批次复用它"
            )

    def drop(self, why: str = "") -> None:
        """丢掉当前流，下一批 ensure() 会重开。"""
        had = self.cap is not None
        if self.reader is not None:
            self.reader.stop()
            self.reader = None
        if had:
            try:
                self.cap.release()
            except Exception:
                pass
            if why:
                STATE.log_line(f"[{self.log_prefix}] 实时流断开（{why}），将自动重连")
        self.cap = None
        self.kind = None
        if had:
            self.reopen_count += 1

    def close(self) -> None:
        if self.reader is not None:
            self.reader.stop()
            self.reader = None
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None


def capture_recent_windows(job: JobRecord, reader) -> list:
    """Select fresh windows from continuous capture, avoiding stale prefetched batches."""
    import math
    from civility.extract import Window, sample_evenly
    from .motion_highlights import action_frames, merge_sampling
    live = STATE.cfg.get("live") or {}
    ex = STATE.cfg.get("extract") or {}
    stride = float(live.get("window_stride_sec", 2))
    count = int(live.get("continuous_batch_windows", 2))
    n_frames = int(live.get("frames_per_window", 4))
    context = max(0.0, float(live.get("context_sec", 4)))
    # The reader retains at most 256 encoded images at 8fps. Oversized live
    # settings are a configuration error, not a broken camera to reconnect.
    ring = getattr(reader, "history_jpegs", None)
    capacity = (getattr(ring, "maxlen", None) or 256) / 8.0
    requested_span = stride * count + context
    if requested_span > capacity:
        raise ValueError(f"实时采样配置需保留 {requested_span:g} 秒画面，超过当前约 {capacity:g} 秒缓存；"
                         "请缩小采样范围或增加实时帧缓存容量")
    if job.capture_started_at is None:
        job.capture_started_at = time.time()
    earliest = (job.last_recent_wall if job.last_recent_wall is not None
                else job.capture_started_at) + stride * count
    deadline = time.monotonic() + max(10.0, stride * count + 5)
    while not job.stop_flag:
        history = reader.snapshot_history()
        if history and history[-1][0] >= earliest:
            if time.time()-history[-1][0] > max(2.0, stride):
                raise ClusterError("摄像头没有更新画面，正在重新连接")
            break
        if reader.dead or time.monotonic() >= deadline:
            raise ClusterError("摄像头近期画面不足，正在重新连接")
        time.sleep(0.02)
    if job.stop_flag:
        return []
    latest = history[-1][0]
    snapshot_highlights = getattr(reader, "snapshot_highlights", None)
    highlights = snapshot_highlights() if callable(snapshot_highlights) else []
    previous_end = job.last_recent_wall
    directory = Path(STATE.cfg["paths"]["frames_dir"]) / job.job_id
    directory.mkdir(parents=True, exist_ok=True)
    windows = []
    for index in range(count):
        end = latest - stride * (count - index - 1)
        start = end - stride
        limit = max(2, int(ex.get("max_frames_per_window", 18)))
        fresh = sample_evenly([(t, data) for t, data in history if start < t <= end], min(n_frames, limit))
        prior = sample_evenly([(t, data) for t, data in history if start-context <= t <= start],
                             math.ceil(context/stride) * n_frames) if context else []
        normal_start = start - context
        gap_sec = max(0.0, normal_start - previous_end) if index == 0 and previous_end is not None else 0.0
        lost_sec = min(gap_sec, max(0.0, history[0][0] - previous_end)) if gap_sec else 0.0
        expected_start = max(job.capture_started_at, normal_start)
        lost_sec += max(0.0, min(end, history[0][0]) - expected_start)
        # The first window may preserve an action from the skipped timeline;
        # the last window always remains the most recent normal window.
        budget = min(limit, len(prior) + len(fresh))
        anchors = min(len(prior), 2)
        spare = max(0, budget - len(fresh) - anchors)
        retained = action_frames(history, highlights,
            start=max(history[0][0], previous_end if gap_sec else normal_start),
            end=start, max_frames=min(6, spare)) if index == 0 else []
        selected = merge_sampling(prior, fresh, retained, budget=budget)
        if len(fresh) < 2:
            raise ClusterError("近期画面缺少连续帧，正在重新连接")
        source = []
        for timestamp, data in selected:
            path = job.recent_frame_files.get(timestamp)
            if path is None:
                path = directory / f"f_{job.frame_idx_offset:05d}.jpg"
                path.write_bytes(data)
                job.frame_idx_offset += 1
                job.recent_frame_files[timestamp] = path
            source.append((timestamp-job.capture_started_at, str(path)))
        retained_count = sum(t in {ts for ts, _ in retained} for t, _ in selected)
        sampling = {"mode": "recent_with_motion" if retained_count else "recent",
                    "gap_sec": round(gap_sec, 3), "retained_action_frames": retained_count,
                    "unexamined_gap_sec": round(lost_sec, 3),
                    "capture_age_sec": round(max(0.0, time.time() - end), 3),
                    "latest_frame_at": selected[-1][0],
                    "frame_times": [t - job.capture_started_at for t, _ in selected]}
        windows.append(Window(window_id=f"{job.job_id}__w{job.window_seq:05d}",
            video=job.filename,t_start=round(source[0][0],2),t_end=round(source[-1][0],2),
            frames=[p for _,p in source],labels=[f"t={t:.1f}s" for t,_ in source],
            idx0=frame_index_of(source[0][1]), sampling=sampling))
        job.window_seq += 1
    job.last_recent_wall = latest
    first = windows[0].sampling
    job.live_metrics.update(capture_age_sec=round(max(0.0, time.time() - latest), 3),
        sampled_gap_sec=first["gap_sec"], retained_action_frames=first["retained_action_frames"],
        unexamined_gap_sec=first["unexamined_gap_sec"],
        batch_processing_sec=round(job.last_batch_sec, 3))
    cutoff = latest-context-stride*count
    job.recent_frame_files = {t:p for t,p in job.recent_frame_files.items() if t>=cutoff}
    job.windows_total = len(windows)
    job.windows_done = 0
    job.state = "running"
    coverage = stride * count
    job.behind = job.last_batch_sec > coverage * 1.05
    STATE.log_line(f"[{job.job_id[:8]}] 取最近 {len(windows)} 个窗口，末帧距现场 "
                   f"{max(0.0,time.time()-latest):.2f}s，采集持续运行")
    return windows


def capture_live_windows(job: JobRecord, spec: str,
                         stream: "LiveStream | None" = None) -> list:
    """从实时源（本地摄像头 / RTSP）采集判断窗口。

    与文件的区别：实时流不知道"总共有多少窗口"，所以进度是
    "已采集 N / 目标 M"，而不是百分比。采集批次与推理流水线配合；
    近期帧跨批次保留，让模型观察动作开始、变化与结果。

    stream 不为空时复用同一条已打开的流（持续监测用），
    这样每批只花"采集时间"而不必重复 RTSP 握手。
    """
    from .capture import capture_windows, mask_credentials, parse_source
    if (job.continuous and job.source_kind != "file" and stream is not None and
            callable(getattr(stream.reader, "snapshot_history", None))):
        return capture_recent_windows(job, stream.reader)

    try:
        # kind 决定几个分支：文件源取样方式不同、也不能裁剪帧文件
        kind, _ = parse_source(spec)
    except ValueError:
        kind = "?"
    if stream is not None and stream.kind:
        kind = stream.kind

    ex = STATE.cfg.get("extract") or {}
    dflt = STATE.behaviors.defaults or {}
    agg = STATE.cfg.get("aggregate") or {}
    live_cfg = STATE.cfg.get("live") or {}

    stride = float(live_cfg.get("window_stride_sec",
                                dflt.get("window_stride_sec", 6.0)))
    n_frames = int(live_cfg.get("frames_per_window", ex.get("max_frames_per_window", 4)))
    # ⚠️ 持续监测模式**不能**把 max_windows 当成"总共要采多少"。
    #    踩过的坑：这里原本写 continuous_max_windows=100000，
    #    而 capture_windows 的循环条件是"采满 max_windows 才返回"——
    #    于是第一批永远不返回、永远进入不了推理，
    #    现象就是"页面上窗口数一直在涨，但一条结果、一条告警都没有"。
    #    正确做法：一批只采几个窗口（continuous_batch_windows），
    #    采完就交给推理，再回到采集 —— 这就是 run_job 里的外层循环。
    if job.continuous:
        max_windows = int(live_cfg.get("continuous_batch_windows", 1))
    else:
        max_windows = int(live_cfg.get("max_windows", 10))
    frames_root = Path(STATE.cfg["paths"]["frames_dir"]) / job.job_id

    job.state = "capturing"
    est_total = max(1, max_windows)
    job.windows_total = est_total
    job.windows_done = 0
    job.message = (f"正在从 {mask_credentials(spec)} 采集"
                   f"（每 {stride:.0f}s 一个窗口"
                   f"{'，持续监测中' if job.continuous else ''}）…")

    def progress(done: int, total: int) -> None:
        job.windows_done = done
        job.windows_total = total
        # 持续模式下 total 是个很大的数，进度条意义不大，用"已监测 N 个窗口"更直观
        if job.continuous:
            job.progress = min(0.99, done / 100.0)
            job.message = f"持续监测中，已完成 {done} 个窗口…"
        else:
            job.progress = done / max(1, total)
            job.message = f"采集中 {done}/{total} 个窗口…"

    # ⚠️ 时间基准必须在**采集之前**确定。
    #    踩过的坑：原本写在 capture_windows 之后，于是第 1 批 t_offset=0，
    #    第 2 批的基准变成了"第 1 批采完那一刻"→ 第 2 批时间戳又回到 0.0，
    #    第 3 批才跳到 4.5。表现是"前两个窗口时间段完全重叠"，
    #    页面上两条记录看着像同一段时间（实测 t=0.01 与 t=0.02）。
    if job.capture_started_at is None:
        job.capture_started_at = time.time()

    raw = capture_windows(
        spec, frames_root,
        window_stride_sec=stride,
        frames_per_window=n_frames,
        max_windows=max_windows,
        max_side=int(ex.get("max_side", 640)),
        jpeg_quality=int(ex.get("jpeg_quality", 82)),
        on_progress=progress,
        should_stop=lambda: job.stop_flag,
        # ⚠️ 这里**不**做帧文件裁剪：裁剪要等这一批推理读完帧再做，
        #    否则剪掉的就是马上要送模型的帧（见下面 prune_frames 的说明）。
        keep_frames=0,
        # ★ 时间偏移 = 距任务开始的挂钟秒数。
        #   不能累加（会复利膨胀成 1e+33），也不能用局部时间戳（会重叠）。
        t_offset=(time.time() - job.capture_started_at) if job.capture_started_at else 0.0,
        idx_offset=job.frame_idx_offset,
        # ★ 复用已开的流 / 只在第一批清目录（见 capture.py 的参数说明）
        cap=(stream.cap if stream is not None else None),
        kind=(stream.kind if stream is not None and stream.kind else None),
        clean_dir=(job.frame_idx_offset == 0),
        # 视频文件冒充实时源做持续监测：读完绕回开头（否则末尾一到就报错）
        loop_file=(job.continuous and kind == "file"),
        # ★ 读帧线程：推理期间画面照走（治"实时画面卡顿"）
        reader=(stream.reader if stream is not None else None),
    )
    if job.capture_started_at is None:      # 理论上不会走到（上面已设）
        job.capture_started_at = time.time()
    job.frame_idx_offset += sum(len(w) for w in raw)

    # 转成 build_windows 所需的 (时间戳, 路径) 形式再切窗
    from civility.extract import Window

    windows = []
    for i, wf in enumerate(raw):
        if len(wf) < 2:
            continue
        # ⚠️ 窗口序号必须**全局递增**，不能每批从 0 开始。
        #    持续模式每批都调用本函数，若用 i 编号，
        #    第二批又会给出 w0000、w0001… → ID 与上一批重复，
        #    页面里"回看/放大"取到的会是另一批的画面（实测踩到）。
        seq = job.window_seq + i
        # ⚠️ t_end 要说真话（实测踩到）：
        #    实时源的取样间隔是 stride/(帧数-1)，末帧正好落在窗口末尾，
        #    所以窗口区间就是 [末帧时刻- stride, 末帧时刻]；
        #    如果还按"末帧 + interval"算，声明区间会比实际多出 1.5 秒，
        #    和下一个窗口重叠（页面上看就是两条记录时间段咬在一起）。
        #    文件源仍按"末帧 + interval"（跳帧取样，窗口尾部还有间隔）。
        t_end = (round(wf[-1][0], 2) if kind != "file"
                 else round(wf[-1][0] + stride / max(1, len(wf)), 2))
        # Include recent source frames so short live windows do not sever an action chain.
        from civility.extract import sample_evenly
        context_sec = max(0.0, float(live_cfg.get("context_sec", 0)))
        if job.context_frames and wf[0][0] <= job.context_frames[-1][0]:
            job.context_frames = []  # Source loop/reset: never join different timeline passes.
        context = [f for f in job.context_frames
                   if wf[0][0] - context_sec <= f[0] < wf[0][0]] if context_sec else []
        source_frames = sample_evenly(context + wf, int(ex.get("max_frames_per_window", 4)))
        job.context_frames = [f for f in job.context_frames + wf
                              if f[0] >= wf[-1][0] - context_sec][-int(ex.get("max_frames_per_window", 4)):]
        windows.append(
            Window(
                window_id=f"{job.job_id}__w{seq:05d}",
                video=job.filename,
                t_start=round(source_frames[0][0], 2),
                t_end=t_end,
                frames=[p for _, p in source_frames],
                labels=[f"t={t:.1f}s" for t, _ in source_frames],
                # ★ 真实帧号取自文件名（f_00023.jpg → 23），不再靠 t_start 推算。
                #   踩过的坑：/frames 接口原本算的是 int(t_start)+idx，只在
                #   "步长 == 帧间隔、且一帧没丢"时才成立；改步长就会指到别的
                #   帧上，而页面看起来完全正常（静默显示错误的画面）。
                #   实测：步长 6 秒时老算法取第 6 帧，本窗口第一帧其实是第 3 帧。
                idx0=frame_index_of(source_frames[0][1]),
            )
        )
    job.window_seq += len(windows)
    job.windows_done = 0
    job.windows_total = len(windows)
    job.message = f"采集完成：{len(windows)} 个窗口"

    # ⚠️ 跟不上就明确警告，而不是让用户对着一堆越来越旧的画面猜。
    #    判据：这一批的采集耗时 + 上一轮推理耗时 超过窗口步长×窗口数，
    #    说明系统处理不过来，画面会持续滞后。
    if job.continuous and windows:
        need = stride * len(windows)
        if job.last_batch_sec and job.last_batch_sec > need * 1.05:
            STATE.log_line(
                f"[{job.job_id[:8]}] ⚠️ 检测吞吐不足：上一批耗时 "
                f"{job.last_batch_sec:.1f}s，而 {len(windows)} 个窗口只覆盖 "
                f"{need:.1f}s。预览保持实时，检测结果可能晚于现场；"
                f"最近结果距现场约 {job.result_age_sec:.1f}s。"
            )
            job.behind = True
        else:
            job.behind = False

    STATE.log_line(f"[{job.job_id[:8]}] {job.message}")
    if not windows:
        raise ClusterError(
            "没有采集到任何窗口。常见原因：摄像头被其他程序占用、"
            "RTSP 地址/密码不对、或采集过程中被中止。"
        )
    return windows


def b64_of(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def run_job(job_id: str) -> None:
    """后台线程：逐窗口推理 → 推送进度 → 时序聚合出告警。

    持续监测模式（job.continuous）下，处理完一批窗口后**回到采集**继续跑，
    直到用户点停止。这样"系统一直看着画面"才成立，
    而不是采一批就结束（那种只能叫"录一段分析"）。

    job.use_mock 为真时用本地假模型，不需要集群 —— 用于在没有 VPN/排队时
    验证整条监测链路（采集 → 推理 → SSE → 页面渲染）。
    """
    job = STATE.jobs[job_id]
    video_label = job.filename
    # 持续监测复用一条实时流（见 LiveStream：每批重开要白花 2.9 秒握手）
    stream: LiveStream | None = (
        LiveStream(job.source_spec, log_prefix=job.job_id[:8])
        if (job.is_live and job.continuous) else None
    )
    cap_pool = None
    infer_pool = None
    next_cap = None
    try:
        sess = STATE.session if not job.use_mock else None
        if not job.use_mock and (sess is None or sess.state != "ready"):
            raise ClusterError("推理服务未就绪，请先在「服务」页启动")

        job._archive_enabled = False
        job._archive_error = ""
        job._archived_windows = 0
        if not job.is_live and job.source_kind == "file":
            try:
                job._archive_source = STATE.uploads[job.job_id]
                library = video_library()
                if Path(job._archive_source).resolve().is_relative_to(library.uploads_root):
                    library.begin(job)
                    job._archive_enabled = True
            except Exception as exc:
                job._archive_error = f"视频问答档案未能创建：{type(exc).__name__}: {exc}"
                STATE.log_line(job._archive_error)

        # ── 推理入口：集群 or 本地 mock ──
        def infer(frames_b64: list[str], labels: list[str], w, **kw) -> dict:
            if job.use_mock:
                from civility.vlm import MockBackend, WindowRequest

                req = WindowRequest(
                    window_id=w.window_id, video=job.source_spec or job.filename,
                    t_start=w.t_start, t_end=w.t_end, frames=list(w.frames), labels=labels,
                    with_alert=kw.get("with_alert", False),
                    with_actions=kw.get("with_actions", False),
                    with_risks=kw.get("with_risks", False),
                )
                # 演示场景：让场景描述/动作/判定依据/告警互相自洽（见 MOCK_SCENES）
                # CIVILITY_MOCK_LATENCY：给本地假模型加人为延迟，**专供诊断**。
                # 用途：不用连集群就能复现"推理要好几秒"的真实节奏，
                # 从而验证"推理期间实时画面还在不在动"（见 tools/test_live_smooth.py）。
                _lat = float(os.environ.get("CIVILITY_MOCK_LATENCY", "0.02") or 0.02)
                res = MockBackend(noise=0.0, scene=job.mock_scene or None,
                                  latency=_lat).infer(req, STATE.behaviors)
                from civility.prompts import sanitize

                parsed = sanitize(res.parsed, STATE.behaviors) if res.parsed else None
                # 让 mock 的结果结构与真实后端一致（violations 用 dict）
                if res.parsed:
                    parsed = dict(parsed)
                    parsed["scene"] = (res.parsed or {}).get("scene", "")
                    parsed["people"] = (res.parsed or {}).get("people", [])
                return {"ok": True, "parsed": parsed, "raw": res.raw,
                        "error": None, "latency_sec": res.latency_sec,
                        "backend": "mock"}
            from civility.config import behavior_payload
            kw.setdefault("few_shot", bool(STATE.behaviors.defaults.get("few_shot", True)))
            definitions = kw.pop("behaviors", None)
            return sess.infer(frames_b64, labels, behaviors=definitions or behavior_payload(STATE.behaviors), **kw)

        lat: list[float] = []
        t0 = time.time()
        round_no = 0
        total_done = 0
        cap_fail = 0        # 连续采集失败次数（>5 次就放弃，避免无限重连）
        # 每个窗口落盘几张图 —— 裁剪时按"窗口数 × 每窗帧数"算保留量
        keep_files_per_window = int(
            (STATE.cfg.get("extract") or {}).get("max_frames_per_window", 4))
        # 动作量门控（见窗口循环里的说明）
        _live = STATE.cfg.get("live") or {}
        motion_gate = bool(_live.get("motion_gate", True))
        motion_thr = float(_live.get("motion_threshold", 0.5))
        recheck_sec = max(1.0, float(_live.get("motion_recheck_sec", 30)))
        # 同批请求并发抵达同一个 GPU worker，避免被轮询拆散成多个单窗任务。
        infer_workers = max(1, int(_live.get("infer_workers", 6) or 6))
        if not job.use_mock:
            infer_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=infer_workers, thread_name_prefix="infer")
        # 采集与推理流水线：推理这一批的同时，后台线程已经在采下一批。
        # 不做的话一个周期 = 采集 + 推理（串行），做了就是两者取较长者。
        # 步长 2 秒 × 8 窗时：串行 16+14=30s/批，流水线 16s/批 —— 差近两倍。
        pipeline = bool(_live.get("pipeline", True)) and job.is_live and job.continuous
        cap_pool = (concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="capture") if pipeline else None)
        next_cap = None

        while True:
            # 每批读取一个一致快照；页面调整策略/复检无需中断持续监测。
            agg_cfg = dict(STATE.cfg.get("aggregate") or {})
            with_alert = str(agg_cfg.get("mode") or "").lower() == "ai"
            with_risks = bool(agg_cfg.get("open_risks", True))
            batch_behaviors = STATE.behaviors
            max_tokens = generation_budget(batch_behaviors, with_alert=with_alert,
                                           with_actions=job.with_actions, with_risks=with_risks)
            _live = STATE.cfg.get("live") or {}
            motion_gate = bool(_live.get("motion_gate", True))
            motion_thr = float(_live.get("motion_threshold", 0.5))
            recheck_sec = max(1.0, float(_live.get("motion_recheck_sec", 30)))
            round_no += 1
            batch_t0 = time.time()
            if job.is_live:
                if stream is not None:
                    stream.ensure()
                    # ★ 启动/复用读帧线程：它独占 VideoCapture 持续读帧，
                    #   于是"推理期间画面冻住十几秒"这件事不再发生。
                    #   CIVILITY_NO_READER=1 是为了做 A/B 对比诊断而留的开关
                    #   （见 tools/test_live_smooth.py），正常运行不要设。
                    if job.source_kind != "file" and not os.environ.get("CIVILITY_NO_READER"):
                        stream.ensure_reader(
                            max_side=int((STATE.cfg.get("extract") or {})
                                         .get("max_side", 640)),
                            jpeg_quality=int((STATE.cfg.get("extract") or {})
                                             .get("jpeg_quality", 82)),
                            preview_dir=Path(STATE.cfg["paths"]["frames_dir"])
                            / job.job_id)
                    job.reader = stream.reader
                try:
                    if next_cap is not None:
                        # ★ 流水线：这一批的采集**上一轮就已经在后台跑**了，
                        #   走到这里通常已经采完 —— 采集与推理真正重叠。
                        windows = next_cap.result()
                        next_cap = None
                    else:
                        windows = capture_live_windows(job, job.source_spec, stream)
                except Exception as e:
                    next_cap = None
                    # 流坏了（拔线/摄像头重启）→ 丢掉重连，持续模式下不要
                    # 因为一次抖动就整场监测结束；重连也救不回来才报错。
                    if stream is None or not isinstance(e, ClusterError):
                        raise
                    stream.drop(str(e)[:50])
                    cap_fail += 1
                    if cap_fail >= 5:
                        raise
                    job.message = (f"实时流断开，正在重连（第 {cap_fail} 次）…")
                    job.events.put({"type": "error",
                                    "error": f"实时流断开，正在重连（第 {cap_fail} 次）",
                                    "recovering": True})
                    STATE.log_line(
                        f"[{job.job_id[:8]}] ⚠️ 采集失败，2 秒后重连"
                        f"（{cap_fail}/5）：{e}"
                    )
                    time.sleep(2.0)
                    continue
                cap_fail = 0
                video_label = job.filename
                # 把"下一批的采集"提前扔到后台，与随后的推理重叠执行
                if (cap_pool is not None and not job.stop_flag and
                        not (stream is not None and job.source_kind != "file" and
                             callable(getattr(stream.reader, "snapshot_history", None)))):
                    next_cap = cap_pool.submit(capture_live_windows, job,
                                               job.source_spec, stream)
            else:
                video = STATE.uploads[job.job_id]
                # ★ 一个任务只抽一次帧。
                #   踩过的坑：以前每轮都调用 slice_video_to_windows，
                #   而 extract_frames 会**清空帧目录重建**，于是
                #   ① 白跑一遍 ffmpeg（14 秒视频要几秒）
                #   ② 上一轮窗口的缩略图全被删掉 → 页面上变成碎图
                if job.cached_windows is None:
                    job.cached_windows = slice_video_to_windows(video, job)
                windows = job.cached_windows
                video_label = str(video)
                # 循环播放时轮次要体现在窗口 ID 上，否则第二轮的 ID 与第一轮
                # 完全相同（build_windows 的 ID 是确定性的），
                # 页面/接口按 ID 取帧会串到另一轮的画面。
                if job.loop and round_no > 1:
                    for w in windows:
                        w.window_id = f"{w.window_id}__r{round_no:02d}"

            job.state = "running"
            inference_t0 = time.time()

            # ── 预提交：把本批能算的窗口**并发**发出去 ──
            # 为什么必须并发（实测结论）：服务端做了动态批量，
            # 但批量只有在"多个请求几乎同时到达"时才会形成 ——
            # 一个一个顺序发的话，服务端永远只收到 1 个请求，批量形同虚设。
            # 当前小批量实测：同卡批量 2 比两个 worker 各算 1 窗更快。
            # 服务端凑批最多等待 25ms，保留全部证据和动作描述。
            skip_flags: list[bool] = []
            motions: list[float] = []
            gate_last_checked = job.last_model_t_end
            for w in windows:
                sc = -1.0
                skip = False
                has_retained_action = bool(w.sampling.get("retained_action_frames"))
                if motion_gate and job.continuous and job.window_results and not has_retained_action:
                    try:
                        from .capture import frame_motion

                        sc = frame_motion(list(w.frames))
                        skip = _should_skip_window(
                            job, sc, motion_thr, w.t_end, recheck_sec,
                            gate_last_checked)
                    except Exception as e:              # noqa: BLE001
                        STATE.log_line(f"（动作量计算失败，照常推理：{e}）")
                motions.append(sc)
                skip_flags.append(skip)
                if not skip:
                    gate_last_checked = w.t_end

            inflight: dict[int, object] = {}
            ex = None

            def _infer_window(_w):
                try:
                    fb = [b64_of(p) for p in _w.frames]
                    if job.use_mock:
                        return infer(fb, _w.labels, _w, max_new_tokens=max_tokens,
                                     with_alert=with_alert, with_actions=job.with_actions, with_risks=with_risks)
                    return infer_with_review(
                        lambda frames, labels, **kw: infer(frames, labels, _w, **kw),
                        fb, _w.labels, batch_behaviors, max_new_tokens=max_tokens,
                        with_alert=with_alert, with_actions=job.with_actions, with_risks=with_risks,
                        review_enabled=bool(batch_behaviors.defaults.get("evidence_review", True)))
                except Exception as e:
                    return {"ok": False, "error": f"{type(e).__name__}: {e}", "parsed": None, "latency_sec": 0}

            if not job.use_mock:
                ex = infer_pool
                for i, w in enumerate(windows):
                    if skip_flags[i]:
                        continue

                    inflight[i] = ex.submit(_infer_window, w)
                STATE.log_line(
                    f"[{job.job_id[:8]}] 本批 {len(windows)} 窗："
                    f"跳过 {sum(skip_flags)}、并发提交 {len(inflight)}"
                    f"（并发度 {infer_workers}）"
                )

            for i, w in enumerate(windows):
                if job.stop_flag:
                    job.state = "stopped"
                    job.message = f"已停止（共完成 {total_done} 个窗口）"
                    job.progress = 1.0
                    job.events.put({"type": "done", "summary": job.summary()})
                    STATE.log_line(f"[{job.job_id[:8]}] {job.message}")
                    return
                job.message = (f"推理中 {i + 1}/{len(windows)}"
                               + (f"（第 {round_no} 批，累计 {total_done}）"
                                  if job.continuous else "…"))

                # ── 动作量门控（判定在预提交时已算好，见上）──
                # 前一个已送模型的窗口可能刚发现风险/失败；整批预判后还要
                # 再检查一次，不能用原 skip_flags 抹掉新发现的风险和重试保护。
                if skip_flags[i]:
                    skip_flags[i] = _should_skip_window(
                        job, motions[i], motion_thr, w.t_end, recheck_sec,
                        job.last_model_t_end)
                if skip_flags[i]:
                    score = motions[i]
                    rec = {
                        "window_id": w.window_id,
                        "t_start": w.t_start, "t_end": w.t_end,
                        "frames": len(w.frames),
                        # 跳过窗口也有自己的帧（只是没送模型），照样记帧号，
                        # 否则页面缩略图会指着上一窗的帧。
                        "frame0": int(getattr(w, "idx0", -1)),
                        "frame_indices": w.frame_indices,
                        "frame_labels": list(w.labels),
                        "frame_times": list(w.sampling.get("frame_times", [])),
                        "sampling": dict(w.sampling),
                        "latency": 0.0,
                        "skipped": True,
                        "motion": round(score, 2),
                        "people": [], "violations": {}, "risks": [],
                        "raw": "", "error": None,
                    }
                    rec["scene"] = (f"[未调模型] 画面无明显变化"
                                    f"（动作量 {score:.2f} < {motion_thr}），"
                                    f"静止画面每 {recheck_sec:g} 秒复检")
                    job.window_results.append(rec)
                    archive_window(job, rec)
                    if len(job.window_results) > job.max_kept_windows:
                        del job.window_results[:-job.max_kept_windows]
                    total_done += 1
                    job.windows_seen = total_done
                    job.windows_done = i + 1
                    job.progress = (i + 1) / len(windows)
                    job.events.put({
                        "type": "window", "index": i + 1,
                        "total": len(windows), "cumulative": total_done,
                        "continuous": True, "window_id": w.window_id,
                        "t_start": w.t_start, "t_end": w.t_end,
                        "scene": rec["scene"],
                        "people": rec.get("people") or [],
                        "hits": [], "risks": [], "error": None, "latency": 0.0,
                        "frame0": rec.get("frame0", -1),
                        "frame_indices": rec.get("frame_indices", []),
                        "frame_labels": rec.get("frame_labels", []),
                        "sampling": rec.get("sampling", {}),
                        "live_metrics": dict(job.live_metrics),
                        "frames": rec.get("frames", 0),
                        "skipped": True, "motion": round(score, 2),
                        "tokens_prompt": 0,
                        "tokens_out": 0,
                        "progress": round(job.progress, 4),
                    })
                    _refresh_alerts(job, video_label, agg_cfg)
                    job.events.put({
                        "type": "alerts", "alerts": job.alerts,
                        "cumulative": total_done,
                        "behind": job.behind,
                        "live_metrics": dict(job.live_metrics),
                        "index": i + 1, "total": len(windows),
                    })
                    STATE.log_line(
                        f"[{job.job_id[:8]}] 第 {total_done} 窗画面无变化"
                        f"（动作量 {score:.2f}），跳过模型调用"
                    )
                    continue
                if 0 <= motions[i] < motion_thr * 3:
                    STATE.log_line(
                        f"[{job.job_id[:8]}] 动作量 {motions[i]:.2f}"
                        f"（阈值 {motion_thr}）—— 接近阈值，仍送模型"
                    )

                if i in inflight:
                    res = inflight.pop(i).result()      # 并发提交过 → 等它
                else:
                    res = _infer_window(w)


                if res.get("latency_sec"):
                    lat.append(res["latency_sec"])
                    job.avg_latency = sum(lat) / len(lat)

                parsed = res.get("parsed")
                if res.get("valid") is False and not res.get("error"):
                    res["error"] = "模型输出格式校验失败：" + "; ".join(
                        str(p) for p in (res.get("problems") or [])[:3])
                if with_risks and isinstance(parsed, dict) and "risks" not in parsed:
                    res["error"] = ("模型未返回开放风险字段；请确认已部署并重启新模型服务，"
                                    "或检查原始输出")
                if parsed is None and not res.get("error"):
                    res["error"] = "模型未返回可用的结构化结论"
                if res.get("error"):
                    parsed = None  # 不把失败输出渲染为正常识别或用于静止门控。
                elif isinstance(parsed, dict):
                    # Local ingress must enforce the same evidence gate even for an old worker.
                    parsed = sanitize(parsed, batch_behaviors, frame_count=len(w.frames))
                if parsed is not None and not res.get("error"):
                    job.last_model_t_end = w.t_end
                    if job.is_live and job.source_kind != "file" and job.capture_started_at is not None:
                        job.last_model_frame_at = w.sampling.get("latest_frame_at", job.capture_started_at + w.t_end)
                if job.is_live and job.source_kind != "file" and job.capture_started_at is not None:
                    latest_at = w.sampling.get("latest_frame_at", job.capture_started_at + w.t_end)
                    job.result_age_sec = max(0.0, time.time() - latest_at)
                evidence_warnings = []
                for bid, value in (parsed or {}).get("violations", {}).items():
                    if value.get("evidence_issues"):
                        behavior = STATE.behaviors.by_id(bid)
                        evidence_warnings.append({"behavior_id": bid,
                                                  "name": behavior.name if behavior else bid,
                                                  "issues": value["evidence_issues"]})
                for value in (parsed or {}).get("risks", []):
                    if value.get("evidence_issues"):
                        evidence_warnings.append({"behavior_id": value["risk_id"],
                                                  "name": value["name"],
                                                  "issues": value["evidence_issues"]})
                rec = {
                    "window_id": w.window_id,
                    "t_start": w.t_start,
                    "t_end": w.t_end,
                    "frames": len(w.frames),
                    # ★ 本窗口第一帧的真实帧号（见 api_frame 的说明）。
                    #   前端缩略图要拿它当缓存破坏参数，也要用它定位"中间帧"。
                    "frame0": int(getattr(w, "idx0", -1)),
                    "frame_indices": w.frame_indices,
                    "frame_labels": list(w.labels),
                    "frame_times": list(w.sampling.get("frame_times", [])),
                    "sampling": dict(w.sampling),
                    "latency": res.get("latency_sec"),
                    "roundtrip_sec": res.get("roundtrip_sec"),
                    "timing": res.get("timing"),
                    "result_age_sec": round(job.result_age_sec, 2) if job.is_live else None,
                    # 这三个数用来判断延迟花在哪：prompt token 多 → 视觉编码是瓶颈
                    # （降帧数/降分辨率有效）；输出 token 多 → 生成是瓶颈
                    # （精简输出格式或减行为数有效）。见 tools/bench_latency.py
                    "tokens_prompt": res.get("tokens_prompt"),
                    "tokens_out": res.get("tokens_out"),
                    "tok_per_sec": res.get("tok_per_sec"),
                    # 服务端把这批算的窗口数带回来 —— 用户能直接看到
                    # "一次前向算了几个窗口"，也就知道批量的收益发生了没有
                    "batch": res.get("batch"),
                    "error": res.get("error"),
                    "scene": (parsed or {}).get("scene", ""),
                    # 画面里每个人在做什么（监测模式的核心内容）
                    "people": (parsed or {}).get("people", []),
                    "violations": (parsed or {}).get("violations", {}),
                    "risks": (parsed or {}).get("risks", []) if with_risks else [],
                    "evidence_warnings": evidence_warnings,
                    "raw": (res.get("raw") or "")[:8000],
                    "verification": res.get("verification"),
                    "generation_retry": res.get("generation_retry"),
                }
                job.window_results.append(rec)
                archive_window(job, rec)
                job.windows_done = i + 1
                total_done += 1
                job.windows_seen = total_done
                job.progress = (i + 1) / len(windows)

                # ⚠️ 长期运行必须裁剪历史，否则内存线性增长。
                #    裁剪只影响"页面能往回看多少"，不影响计数与告警。
                if len(job.window_results) > job.max_kept_windows:
                    drop = len(job.window_results) - job.max_kept_windows
                    del job.window_results[:drop]

                hits = [
                    {
                        "behavior_id": bid,
                        "name": STATE.behaviors.by_id(bid).name if STATE.behaviors.by_id(bid) else bid,
                        "confidence": v.get("confidence", 0),
                        "observable": v.get("observable", ""),
                        # mode: ai 时模型自己给的告警决定与理由，透给前端展示
                        "alert": v.get("alert"),
                        "alert_reason": v.get("alert_reason", ""),
                    }
                    for bid, v in (rec["violations"] or {}).items()
                    if v.get("violation")
                    and float(v.get("confidence", 0)) >= float(agg_cfg.get("min_confidence", 0.45))
                ]

                # 每个窗口推一条 SSE，前端就能实时看到画面级的进展
                job.events.put({
                    "type": "window",
                    "index": i + 1,
                    "total": len(windows),
                    "cumulative": total_done,
                    "continuous": job.continuous,
                    "window_id": w.window_id,
                    "t_start": w.t_start,
                    "t_end": w.t_end,
                    "scene": rec["scene"],
                    "people": rec["people"],
                    "hits": hits,
                    "risks": rec["risks"],
                    "evidence_warnings": rec["evidence_warnings"],
                    "verification": rec.get("verification"),
                    "error": rec["error"],
                    "latency": rec["latency"],
                    "roundtrip_sec": rec.get("roundtrip_sec"),
                    "timing": rec.get("timing"),
                    "result_age_sec": rec.get("result_age_sec"),
                    "sampling": rec.get("sampling", {}),
                    "live_metrics": dict(job.live_metrics),
                    # ★ 帧号：前端缩略图用它①做缓存破坏 ②定位"中间帧"
                    "frame0": rec.get("frame0", -1),
                    "frame_indices": rec.get("frame_indices", []),
                    "frame_labels": rec.get("frame_labels", []),
                    "frames": rec.get("frames", 0),
                    # ★ 延迟分解：让"时间花在哪"看得见。
                    #   tokens_out 是决定延迟的那个数（实测 35.8 tok/s ≈
                    #   每 token 28 毫秒），输出瘦身是否生效就看它。
                    "tokens_prompt": rec.get("tokens_prompt"),
                    "tokens_out": rec.get("tokens_out"),
                    "batch": rec.get("batch"),
                    "skipped": bool(rec.get("skipped")),
                    "motion": rec.get("motion"),
                    "progress": round(job.progress, 4),
                })

                # ★ 告警**逐窗口增量计算**，不等整遍跑完。
                #   踩过的坑：聚合原本只在"一批窗口全部推理完"之后做一次，
                #   而文件任务只有一批 —— 于是页面上窗口一条条滚出来，
                #   告警区却一直空白，用户以为告警坏了，点了「中止」才看到告警。
                #   现在每出一个窗口就算一次（纯 Python，几十个窗口的开销可忽略），
                #   命中后 1-2 秒内就能看到告警。
                _refresh_alerts(job, video_label, agg_cfg)
                job.events.put({
                    "type": "alerts",
                    "alerts": job.alerts,
                    "cumulative": total_done,
                    "behind": job.behind,
                    "live_metrics": dict(job.live_metrics),
                    "index": i + 1,
                    "total": len(windows),
                })

            # 时序平滑 → 告警（每一批都重算，这样监测过程中告警会持续更新）
            job.message = "时序聚合…"
            _refresh_alerts(job, video_label, agg_cfg)

            # 持续监测：告警推给前端后回到采集继续；用户点停止才算结束。
            #
            # 这段与 `for w in windows` 同级缩进，`continue` 作用于 `while True`
            # —— 这样才叫"持续监测"，否则采一批就结束了。
            # ⚠️ 这里**不能**重置 job.window_results：聚合按累积窗口算，
            #    一重置告警就会周期性归零，看起来像"系统忘了刚才发生过什么"。
            if job.continuous:
                if job.stop_flag:
                    job.state = "stopped"
                    job.progress = 1.0
                    job.message = f"已停止（共完成 {total_done} 个窗口）"
                    job.events.put({"type": "done", "summary": job.summary()})
                    STATE.log_line(f"[{job.job_id[:8]}] {job.message}")
                    return
                # 这一批推理已经把帧读完了 → 现在才剪旧帧。
                # （在采集过程中剪会剪掉还没送模型的帧，见 prune_frames 说明）
                dropped = prune_frames(
                    Path(STATE.cfg["paths"]["frames_dir"]) / job.job_id,
                    job.max_kept_windows * keep_files_per_window,
                )
                if dropped:
                    STATE.log_line(
                        f"[{job.job_id[:8]}] 已清理 {dropped} 张旧帧"
                        f"（页面保留最近 {job.max_kept_windows} 个窗口的画面）"
                    )
                job.events.put({
                    "type": "alerts",
                    "alerts": job.alerts,
                    "cumulative": total_done,
                    "behind": job.behind,
                })
                STATE.log_line(
                    f"[{job.job_id[:8]}] 第 {round_no} 批完成，累计 {total_done} 窗口 / "
                    f"{len(job.alerts)} 条告警 / {time.time() - t0:.0f}s，继续监测…"
                )
                recent_source = (stream is not None and job.source_kind != "file" and
                                 callable(getattr(stream.reader, "snapshot_history", None)))
                job.last_batch_sec = time.time() - (inference_t0 if recent_source else batch_t0)
                job.live_metrics["batch_processing_sec"] = round(job.last_batch_sec, 3)
                if recent_source:
                    coverage = len(windows) * float(_live.get("window_stride_sec", 2))
                    job.behind = job.last_batch_sec > coverage * 1.05
                    if job.behind:
                        STATE.log_line(f"[{job.job_id[:8]}] 检测本批用时 {job.last_batch_sec:.1f}s / "
                            f"覆盖 {coverage:.1f}s，最新判定画面距现场约 {job.result_age_sec:.1f}s；"
                            "下一批直接取近期画面，预览持续刷新")
                continue

            # ★ 非持续任务的出口（这里以前什么都没有 —— 是一个真 bug）。
            #
            #   踩过的坑：文件任务（continuous=False）跑完一遍后，
            #   代码直接走到 while 末尾又从头再来一遍，**永远不会结束**：
            #   实测一个 3 窗口的视频被完整分析了 4 遍（windows_seen=12/total=3），
            #   页面上同一个窗口反复出现，告警却要等到用户点「中止」才显示
            #   （因为中止的提前 return 把上一遍算出的 alerts 一起推了过去）。
            #
            #   现在的语义：
            #     loop=False（默认）→ 跑一遍就结束
            #     loop=True         → 每遍算一轮，窗口 ID 带轮次后缀，用户点停止才结束
            if job.loop:
                # ⚠️ 循环模式必须**按视频时长配速**（实测踩到）：
                #    视频文件是"能读多快就读多快"，14 秒的视频加本地演示模型
                #    一秒能跑几十轮 —— 实测 75 秒里产出 3447 个窗口，
                #    SSE 把页面刷爆、告警区一直在跳，完全没法看。
                #    真实使用场景是"反复观察同一段画面"，所以让它按视频长度播。
                round_sec = time.time() - batch_t0
                video_sec = windows[-1].t_end if windows else 0.0
                wait = video_sec - round_sec
                if wait > 0.5:
                    job.message = (f"第 {round_no} 轮完成，"
                                   f"按视频时长等待 {wait:.0f}s 后重播…")
                    STATE.log_line(
                        f"[{job.job_id[:8]}] 第 {round_no} 轮完成（{round_sec:.1f}s / "
                        f"视频 {video_sec:.1f}s），等待 {wait:.0f}s 后重播…"
                    )
                    t_wait = time.time() + wait
                    while time.time() < t_wait and not job.stop_flag:
                        time.sleep(0.2)
                job.message = f"第 {round_no} 轮完成，循环播放中…"
                if job.stop_flag:
                    job.state = "stopped"
                    job.progress = 1.0
                    job.message = f"已停止（共完成 {total_done} 个窗口）"
                    job.events.put({"type": "done", "summary": job.summary()})
                    STATE.log_line(f"[{job.job_id[:8]}] {job.message}")
                    return
                STATE.log_line(
                    f"[{job.job_id[:8]}] 累计 {total_done} 窗口 / "
                    f"{len(job.alerts)} 条告警，从头再跑一轮…"
                )
                continue
            break

        job.state = "done"
        job.progress = 1.0
        elapsed = time.time() - t0
        job.message = (
            f"完成：{len(windows)} 窗口 / {len(job.alerts)} 条告警 / "
            f"用时 {elapsed:.0f}s / 平均 {job.avg_latency:.1f}s 每窗口"
        )
        job.events.put({"type": "done", "summary": job.summary()})
        STATE.log_line(f"[{job.job_id[:8]}] {job.message}")

    except Exception as e:
        job.state = "error"
        job.error = f"{type(e).__name__}: {e}"
        job.message = job.error
        job.events.put({"type": "error", "error": job.error})
        STATE.log_line(f"[{job.job_id[:8]}] ❌ {job.error}")
        print(traceback.format_exc(), flush=True)

    finally:
        if getattr(job, "_archive_enabled", False):
            try:
                video_library().save(job)
            except Exception as exc:
                STATE.log_line(f"视频问答档案保存失败：{type(exc).__name__}: {exc}")
        if infer_pool is not None:
            infer_pool.shutdown(wait=False, cancel_futures=True)
        # 任务结束/停止/出错都要把复用的实时流还回去，
        # 否则 RTSP 会话会一直挂在摄像头上（很多设备并发数有限）。
        if next_cap is not None:
            next_cap.cancel()
        if cap_pool is not None:
            cap_pool.shutdown(wait=False)
        if stream is not None:
            stream.close()


# ─────────────────────────── FastAPI ───────────────────────────


def create_app():
    app = FastAPI(title="不文明行为识别 · 本地控制台")

    # ── 页面 ──
    @app.get("/", response_class=HTMLResponse)
    def index():
        return (HERE / "static" / "index.html").read_text(encoding="utf-8")

    @app.get("/static/alert_speech.js")
    def speech_script():
        return FileResponse(str(HERE / "static" / "alert_speech.js"), media_type="text/javascript")

    # ── 已分析视频与可验证的视觉问答 ──
    def archived_video(video_id: str) -> tuple[VideoLibrary, dict]:
        library = video_library()
        job = STATE.jobs.get(video_id)
        if job is not None and (job.state in ("queued", "extracting", "running", "capturing")
                                or (getattr(job, "_detect_worker", None) is not None and job._detect_worker.is_alive())):
            raise HTTPException(409, "视频仍在分析，请等本轮结束后提问")
        try:
            metadata = library.load(video_id)
            if metadata.get("state") not in ("done", "stopped", "error"):
                raise HTTPException(409, "该视频没有已结束的分析记录")
            return library, metadata
        except (OSError, ValueError):
            # Materialize analyses completed before this feature was loaded.
            if job is None or job.is_live or not job.window_results:
                raise HTTPException(404, "未找到已分析视频，请先上传并分析")
            try:
                job._archive_source = STATE.uploads.get(video_id)
                library.begin(job)
                if job.windows_seen > len(job.window_results):
                    job._archive_error = "这是功能升级前的分析，仅能恢复页面仍保留的窗口；更早记录未保存，请重新分析以获得完整档案"
                for row in job.window_results:
                    library.record(job, row)
                library.save(job)
                return library, library.load(video_id)
            except (OSError, ValueError) as exc:
                raise HTTPException(404, "分析记录无法读取") from exc

    @app.get("/api/videos")
    def api_videos():
        for job in list(STATE.jobs.copy().values()):
            if not job.is_live and job.window_results and job.state in ("done", "stopped", "error"):
                try:
                    archived_video(job.job_id)
                except HTTPException:
                    pass
        return {"videos": video_library().list()}

    @app.get("/api/videos/{video_id}")
    def api_video(video_id: str):
        library, metadata = archived_video(video_id)
        return library.public(metadata)

    @app.get("/api/videos/{video_id}/source")
    def api_video_source(video_id: str):
        library, metadata = archived_video(video_id)
        path = library.source(metadata)
        if path is None:
            raise HTTPException(404, "原视频已清理或不可用")
        return FileResponse(str(path))

    @app.get("/api/videos/{video_id}/frames/{frame_id}")
    def api_video_frame(video_id: str, frame_id: str, revision: str = ""):
        library, metadata = archived_video(video_id)
        if revision and revision != metadata.get("analysis_revision"):
            raise HTTPException(409, "视频已重新分析，请重新提问以更新证据")
        try:
            path, _ = library.frame(video_id, frame_id)
            return FileResponse(str(path), media_type="image/jpeg", headers={"Cache-Control": "no-store"})
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise HTTPException(404, "证据帧不可用：" + str(exc)) from exc

    @app.post("/api/videos/{video_id}/ask")
    def api_video_ask(video_id: str, body: dict):
        ensure_current_runtime()
        question = body.get("question")
        if not isinstance(question, str) or not question.strip() or len(question) > 2000:
            raise HTTPException(400, "请输入 1 至 2000 字的问题")
        sess = STATE.session
        if sess is None or sess.state != "ready":
            raise HTTPException(503, "请先启动推理服务；视频问答使用部署的大模型")
        if not QA_BUSY.acquire(blocking=False):
            raise HTTPException(429, "已有视频问答正在处理，请等回答完成后再提问")
        try:
            library, metadata = archived_video(video_id)
            evidence, context, missing = library.retrieve(video_id, question.strip())
            if metadata.get("use_mock"):
                context = "历史检测为模拟数据，不是事实；只以本次真实画面为准。\n" + context
            prompt_evidence = [{k: v for k, v in e.items() if k not in ("path", "recovered")} for e in evidence]
            result = sess.ask_video([b64_of(e["path"]) for e in evidence], question.strip(), prompt_evidence, context)
            from civility.video_qa import validate_answer
            answer = validate_answer(result.get("parsed"), prompt_evidence)
            _, latest = archived_video(video_id)
            if latest.get("analysis_revision") != metadata.get("analysis_revision"):
                raise HTTPException(409, "提问期间视频已重新分析，请重新提问")
            by_id = {e["frame_id"]: e for e in evidence}
            refs = []
            for citation in answer["citations"]:
                e = by_id[citation["frame_id"]]
                refs.append({**{k: v for k, v in e.items() if k != "path"}, **citation,
                             "image_url": f'/api/videos/{video_id}/frames/{e["frame_id"]}?revision={metadata["analysis_revision"]}'})
            STATE.last_interact = time.time()
            return {**answer, "citations": refs, "video_id": video_id, "question": question.strip(),
                    "model": result.get("model", sess.model), "basis": "deployed_vlm_visual_qa",
                    "sampled_frames": len(evidence), "total_windows": metadata.get("archived_windows", metadata.get("windows_seen", 0)),
                    "roundtrip_sec": result.get("roundtrip_sec"),
                    "warnings": missing + ([metadata["archive_error"]] if metadata.get("archive_error") else [])}
        except HTTPException:
            raise
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(422, str(exc)) from exc
        except ClusterError as exc:
            raise HTTPException(502, str(exc)) from exc
        except OSError as exc:
            raise HTTPException(422, "读取视频证据失败：" + str(exc)) from exc
        finally:
            QA_BUSY.release()

    # ── 系统 ──
    @app.get("/api/state")
    def api_state():
        s = STATE.session.status() if STATE.session else {"state": "idle"}
        return {
            "backend": "cluster",
            "runtime": runtime_status(),
            "service": s,
            "jobs": [j.summary() for j in sorted(STATE.jobs.values(),
                                                 key=lambda x: -x.created)],
            "behaviors": {
                "enabled": [{"id": b.id, "name": b.name, "severity": b.severity}
                            for b in STATE.behaviors.enabled],
                "total": len(STATE.behaviors.behaviors),
                "file": STATE.cfg["paths"]["behaviors"],
                "changed_on_disk": STATE.behaviors_changed(),
            },
            "aggregate": STATE.cfg.get("aggregate") or {},
            "extract": STATE.cfg.get("extract") or {},
            # 实时流配置也暴露出来：页面要显示"当前步长是多少"，
            # 工具（tools/real_verify.py）要能读到原值并在跑完后还原。
            "live": STATE.cfg.get("live") or {},
            "service_cfg": STATE.cfg.get("service") or {},
            "cost": service_cost(),
            "auto_released": STATE.auto_released,
            "orphans": STATE.orphans,
            "log": STATE.log[-60:],
        }

    # ── 集群连接 ──
    @app.get("/api/cluster/check")
    def api_cluster_check():
        try:
            cl = STATE.client()
        except ClusterError as e:
            return {"ok": False, "error": str(e)}
        ok, info = cl.check_vpn()
        return {"ok": ok, "host": info if ok else None,
                "error": None if ok else info}

    @app.get("/api/cluster/resources")
    def api_cluster_resources():
        try:
            cl = STATE.client()
            return {
                "ok": True,
                "gpu": cl.gpu_summary(),
                "jobs": cl.list_my_jobs(),
                "models": cl.run(
                    f"ls -1 {cl.model_root}/hf/hub/ 2>/dev/null | grep '^models--'",
                    timeout=60,
                )[1].splitlines(),
            }
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # ── 常驻服务 ──
    @app.post("/api/service/start")
    def api_service_start(body: dict | None = None):
        ensure_current_runtime()
        with STATE.lock:
            previous = STATE.session
            previous_worker = getattr(previous, "_start_worker", None)
            if previous and (previous.state in ("ready", "running", "pending", "submitting", "stopping")
                             or (previous_worker is not None and previous_worker.is_alive())):
                raise HTTPException(400, "服务已在启动或运行中，请先停止")
            cl_cfg = STATE.cfg.get("cluster") or {}
            model = (body or {}).get("model") or cl_cfg.get(
                "remote_model", "Qwen/Qwen3-VL-4B-Instruct"
            )
            cl = STATE.client()
            sess = ServiceSession(
                cl, model=model,
                port=int((STATE.cfg.get("cluster") or {}).get("service_port", 18080)),
                exclude=cl_cfg.get("exclude_nodes", "gpu8002,gpu8003,gpu8012"),
                extra_sbatch=f"--time={cl_cfg.get('time_limit', '0-03:00:00')}",
            )
            # Reserve before dispatch: concurrent requests must see a startup,
            # rather than a still-idle session that can be replaced.
            sess.state = "submitting"
            STATE.session = sess

        def worker():
            try:
                from .cluster_client import deploy_service_package
                from civility.config import behavior_payload
                sess._raise_if_cancelled()
                sess.message = "同步当前推理代码与行为定义…"
                deploy_service_package(cl, Path(__file__).resolve().parents[1],
                                       behavior_payload(STATE.behaviors))
                sess.start(progress=lambda m: STATE.log_line(f"[服务] {m}"))
            except ServiceSession.StartCancelled:
                STATE.log_line("[服务] 启动已取消")
            except Exception as e:
                was_cancelled = sess.cancelled
                if sess.jobid or was_cancelled:
                    try:
                        sess.stop(sweep_orphans=not was_cancelled)
                    except Exception as cleanup_error:
                        # Keep the job id and failure visible when cancellation
                        # could not actually release the allocated resources.
                        e = ClusterError(f"{e}；释放计算节点失败：{cleanup_error}")
                        was_cancelled = False
                if was_cancelled:
                    STATE.log_line("[服务] 启动已取消")
                    return
                sess.state = "error"
                sess.message = f"{type(e).__name__}: {e}"
                STATE.log_line(f"[服务] ❌ {sess.message}")

        thread = threading.Thread(target=worker, daemon=True)
        sess._start_worker = thread
        try:
            thread.start()
        except Exception as e:
            sess.state = "error"
            sess.message = f"{type(e).__name__}: {e}"
            raise HTTPException(500, sess.message) from e
        return {"ok": True, "message": "已提交，正在启动（排队 + 加载模型可能要几分钟）"}

    @app.post("/api/service/stop")
    def api_service_stop():
        with STATE.lock:
            session = STATE.session
        if not session:
            # 没有会话也要扫一遍：网页重启过的话，旧作业会变成孤儿还在计费
            try:
                cl = STATE.client()
                orphans = [j for j in cl.list_my_jobs() if j.get("name") == "civility-serve"]
                for j in orphans:
                    cl.cancel(j["jobid"])
                if orphans:
                    STATE.log_line(f"[服务] 清理了 {len(orphans)} 个遗留作业")
                    return {"ok": True, "message": f"没有活动会话，已清理 {len(orphans)} 个遗留作业"}
            except Exception as e:
                STATE.log_line(f"[服务] 遗留作业清理失败：{e}")
            return {"ok": True, "message": "没有在运行的服务"}
        cancelled = session.stop()
        STATE.log_line(f"[服务] 已停止，取消作业 {cancelled}")
        return {"ok": True, "message": session.message}

    @app.get("/api/service/health")
    def api_service_health():
        if not STATE.session or STATE.session.state != "ready":
            return {"ok": False, "state": STATE.session.state if STATE.session else "idle"}
        try:
            return {"ok": True, **STATE.session.health()}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # ── 成本守卫 ──

    @app.post("/api/heartbeat")
    def api_heartbeat(body: dict | None = None):
        """页面上报"我还开着/还看着/还在操作"。成本守卫靠它判断人走没走。

        body:
            visible   页面是否在前台（document.visibilityState === 'visible'）
            idle_sec  距上次鼠标/键盘操作过了多少秒

        ⚠️ 必须区分"页面存在"和"人在用"：
            切到后台时 JS 定时器仍然会发心跳，只看"有没有请求"会永远
            以为有人在用，节点永远不退（实测踩到这个漏洞，测试里也复现过）。
        """
        body = body or {}
        now = time.time()
        vis = bool(body.get("visible", False))
        STATE.page_visible = vis
        STATE.last_ui_seen = now
        if vis:
            STATE.last_visible_at = now
            try:
                idle_sec = max(0.0, float(body.get("idle_sec") or 0.0))
            except (TypeError, ValueError):
                idle_sec = 0.0
            # 用"距今多久没操作"反推最后操作时刻，后端不必自己跟踪事件
            STATE.last_interact = now - idle_sec
        return {"ok": True}

    @app.get("/api/service/orphans")
    def api_service_orphans():
        """查集群上有没有本项目名下的"孤儿"作业（页面不知道、却还在计费）。"""
        try:
            found = check_orphans()
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}", "orphans": []}
        return {"ok": True, "orphans": found, "count": len(found)}

    @app.post("/api/service/sweep")
    def api_service_sweep():
        """一键停掉孤儿作业。只动 civility-* 名字的作业，不碰其他。"""
        try:
            cl = STATE.client()
            jobs = [j for j in cl.list_my_jobs()
                    if str(j.get("name", "")).startswith("civility-")
                    and j.get("jobid") != (STATE.session.jobid if STATE.session else None)]
            for j in jobs:
                cl.cancel(j["jobid"])
            STATE.orphans = []
            if jobs:
                msg = f"已停掉 {len(jobs)} 个遗留作业（{', '.join(j['jobid'] for j in jobs)}）"
            else:
                msg = "没有发现遗留作业"
            STATE.log_line(f"[服务] {msg}")
            return {"ok": True, "message": msg, "cancelled": [j["jobid"] for j in jobs]}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    @app.get("/api/service/cost")
    def api_service_cost():
        """当前服务的运行时长与估算花费（页面上的"正在烧钱"提示）。"""
        return {"ok": True, **service_cost(), "auto_released": STATE.auto_released,
                "orphans": STATE.orphans}

    @app.post("/api/aggregate/config")
    def api_aggregate_config(body: dict | None = None):
        """切换告警判定策略：immediate / ai / smoothed。

        改完**不需要重启推理服务** —— 聚合在本机做，只影响本机行为。
        但 ai 模式会在 Prompt 里多要两个字段，所以要等下一次推理才生效。
        """
        body = body or {}
        agg = dict(STATE.cfg.get("aggregate") or {})
        if body.get("mode") is not None:
            mode = str(body["mode"]).lower()
            if mode not in ("immediate", "ai", "smoothed"):
                raise HTTPException(400, "mode 必须是 immediate / ai / smoothed 之一")
            agg["mode"] = mode
        if body.get("dedup_sec") is not None:
            try:
                v = float(body["dedup_sec"])
            except (TypeError, ValueError):
                raise HTTPException(400, "dedup_sec 必须是有限数字")
            if not math.isfinite(v) or v < 0:
                raise HTTPException(400, "dedup_sec 必须是非负有限数字")
            agg["dedup_sec"] = v
        if body.get("min_confidence") is not None:
            try:
                v = float(body["min_confidence"])
            except (TypeError, ValueError):
                raise HTTPException(400, "min_confidence 需在 0-1 之间")
            if not (0 <= v <= 1):
                raise HTTPException(400, "min_confidence 需在 0-1 之间")
            agg["min_confidence"] = v
        if "open_risks" in body:
            if not isinstance(body["open_risks"], bool):
                raise HTTPException(400, "open_risks 必须为 true 或 false")
            agg["open_risks"] = body["open_risks"]
        if not agg.get("open_risks", True) and not STATE.behaviors.enabled:
            raise HTTPException(400, "请至少启用一条行为定义，或保持 AI 开放风险开启")
        STATE.cfg["aggregate"] = agg

        # 记住上次的配置，重跑时才能正确评估（否则改了模式但用旧结果评分会误导）
        STATE.log_line(f"[告警] 策略更新 → {agg}")
        return {"ok": True, "aggregate": agg}

    # ── 行为定义 ──
    @app.get("/api/behaviors/prompt")
    def api_prompt():
        # 预览必须和实际发给模型的 Prompt 一致 —— 包括 ai 模式下多出来的
        # 告警判定段落。否则用户在这里看不出模型被要求了什么。
        mode = str((STATE.cfg.get("aggregate") or {}).get("mode") or "").lower()
        open_risks = bool((STATE.cfg.get("aggregate") or {}).get("open_risks", True))
        return {"prompt": build_system_prompt(STATE.behaviors, with_alert=(mode == "ai"),
                                               with_risks=open_risks),
                "mode": mode, "open_risks": open_risks}

    @app.get("/api/behaviors/raw")
    def api_behaviors_raw():
        p = Path(STATE.cfg["paths"]["behaviors"])
        return {"path": str(p), "text": p.read_text(encoding="utf-8")}

    @app.post("/api/behaviors/raw")
    def api_behaviors_save(body: dict):
        """网页上直接编辑 YAML 并保存。

        刻意用"整份 YAML 文本"而不是结构化表单：
        行为定义里最关键的字段（定义/证据/反例）都是长文本，
        表单输入框反而难用；而且 YAML 让用户能直接复制粘贴复用。
        保存前必须校验，避免写坏配置导致整个服务起不来。
        """
        text = (body or {}).get("text") or ""
        p = Path(STATE.cfg["paths"]["behaviors"])
        backup = p.with_suffix(".yaml.bak")
        try:
            import yaml

            parsed = yaml.safe_load(text)
            if not isinstance(parsed, dict) or "behaviors" not in parsed:
                raise ValueError("YAML 顶层必须有 behaviors 列表")
            from civility.config import parse_behaviors
            parse_behaviors(parsed)  # Reject malformed evidence rules before writing the user's file.
            enabled = [b for b in parsed["behaviors"]
                       if b.get("enabled", True) and b.get("definition")]
            if not enabled and not (STATE.cfg.get("aggregate") or {}).get("open_risks", True):
                raise ValueError("至少要有一条 enabled: true 且填了 definition 的行为")
        except Exception as e:
            raise HTTPException(400, f"校验失败，未保存：{e}")

        if p.exists():
            backup.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
        p.write_text(text, encoding="utf-8")
        STATE.reload_behaviors()
        STATE.log_line(f"[行为] 已保存并重载（启用 {len(STATE.behaviors.enabled)} 项）")

        # Every new request carries its current definition to whichever worker handles it.
        pushed = bool(STATE.session and STATE.session.state == "ready")
        return {"ok": True, "enabled": len(STATE.behaviors.enabled), "pushed": pushed}

    # ── 上传视频 ──
    @app.post("/api/upload")
    async def api_upload(file: UploadFile = File(...)):
        data = await file.read()
        if not data:
            raise HTTPException(400, "空文件")
        up_dir = Path(STATE.cfg["paths"]["data_dir"]) / "uploads"
        up_dir.mkdir(parents=True, exist_ok=True)
        jid = uuid.uuid4().hex[:12]
        suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
        dst = up_dir / f"{jid}{suffix}"
        dst.write_bytes(data)
        STATE.uploads[jid] = dst
        job = JobRecord(job_id=jid, filename=file.filename or dst.name)
        STATE.jobs[jid] = job
        STATE.log_line(f"[上传] {job.filename}  {len(data) / 1e6:.1f} MB")
        return {"ok": True, "job_id": jid, "size_mb": round(len(data) / 1e6, 2)}

    # ── 启动检测 ──
    @app.post("/api/detect/{job_id}")
    def api_detect(job_id: str, body: dict | None = None):
        """开始分析一个已上传的视频。

        body:
            use_mock    用本地演示模型（不需要 VPN/集群），供离线看整条链路
            mock_scene  演示场景（normal / littering / smoking / queue_jumping），
                        让演示时的画面描述、动作、判定依据、告警**互相自洽**
            loop        循环播放：跑完一遍从头再来（默认 False = 跑一遍就结束）
        """
        body = body or {}
        ensure_current_runtime()
        with STATE.lock:
            job = STATE.jobs.get(job_id)
            if not job:
                raise HTTPException(404, "任务不存在")
            previous_worker = getattr(job, "_detect_worker", None)
            # An uploaded file is initially queued without a worker. Reserve the
            # first start atomically so a second request cannot launch another.
            if (job.state in ("running", "extracting", "capturing")
                    or (job.state == "queued" and getattr(job, "_detect_start_reserved", False))
                    or (previous_worker is not None and previous_worker.is_alive())):
                raise HTTPException(400, "该任务已在运行")
            use_mock = bool(body.get("use_mock", job.use_mock))
            if not use_mock and (not STATE.session or STATE.session.state != "ready"):
                raise HTTPException(400, "推理服务未就绪。可先点「启动服务」，"
                                         "或勾选「演示模式」在本地跑通链路")
            # Rejected starts must not alter the current backend or loop mode.
            job.use_mock = use_mock
            job.with_actions = bool(body.get("with_actions", True))
            if "mock_scene" in body:
                job.mock_scene = str(body.get("mock_scene") or "")
            job.loop = bool(body.get("loop", job.loop))
            if job.state in ("done", "error", "stopped"):
                # 允许重跑：清掉旧结果，但保留同一个 job_id 方便前端复用
                job.window_results.clear()
                job.alerts.clear()
            job.progress = 0.0
            job.error = ""
            job.message = ""
            job.avg_latency = 0.0
            job.windows_done = 0
            job.windows_total = 0
            # Old terminal SSE events must not close the new run's connection.
            # Keep the queue: a client may subscribe before starting detection.
            while True:
                try:
                    job.events.get_nowait()
                except queue.Empty:
                    break
            # ★ 重跑必须把累计量一起归零，否则窗口 ID 会从上一轮的序号继续，
            #   而帧文件名、时间偏移也会跟着串（实测重跑后 ID 混乱）
            job.cached_windows = None
            job.window_seq = 0
            job.frame_idx_offset = 0
            job.capture_started_at = None
            job.last_recent_wall = None
            job.last_model_frame_at = None
            job.recent_frame_files.clear()
            job.context_frames.clear()
            job.live_metrics.clear()
            job.result_age_sec = 0.0
            job.last_batch_sec = 0.0
            job.windows_seen = 0
            job.last_model_t_end = None
            job.alert_checked_until = -1.0
            job.alert_last_by_type.clear()
            job.t_offset = 0.0
            job.behind = False
            job.stop_flag = False
            job.state = "queued"
            job._detect_start_reserved = True
            worker = threading.Thread(target=run_job, args=(job_id,), daemon=True)
            job._detect_worker = worker
        try:
            worker.start()
        except Exception as exc:
            with STATE.lock:
                job._detect_start_reserved = False
                job.state = "error"
                job.error = f"启动检测失败：{type(exc).__name__}: {exc}"
                job.message = job.error
            raise HTTPException(500, job.error) from exc
        return {"ok": True, "message": "已开始检测"}

    @app.post("/api/stop/{job_id}")
    def api_stop(job_id: str):
        job = STATE.jobs.get(job_id)
        if not job:
            raise HTTPException(404, "任务不存在")
        job.stop_flag = True
        return {"ok": True}

    @app.get("/api/job/{job_id}")
    def api_job(job_id: str):
        job = STATE.jobs.get(job_id)
        if not job:
            raise HTTPException(404, "任务不存在")
        return {**job.summary(), "window_results": job.window_results}

    @app.get("/api/job/{job_id}/stream")
    async def api_stream(job_id: str):
        """SSE 实时推送 —— 每跑完一个窗口立刻推给页面，不用轮询。"""
        job = STATE.jobs.get(job_id)
        if not job:
            raise HTTPException(404, "任务不存在")

        async def gen():
            yield f"data: {json.dumps({'type': 'hello', 'summary': job.summary()}, ensure_ascii=False)}\n\n"
            while True:
                try:
                    ev = await asyncio.to_thread(job.events.get, True, 25)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    if job.state in ("done", "error", "stopped"):
                        # 状态已终态且队列空了，再推一次汇总后收尾
                        yield f"data: {json.dumps({'type': 'final', 'summary': job.summary()}, ensure_ascii=False)}\n\n"
                        break
                    continue
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                if ev.get("type") == "done" or (ev.get("type") == "error"
                                                and not ev.get("recovering")):
                    break

        return StreamingResponse(
            gen(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── 其他 ──
    @app.get("/api/log")
    def api_log():
        return {"log": STATE.log[-200:]}

    # ── 实时流 ──
    @app.post("/api/live/config")
    def api_live_config(body: dict | None = None):
        """临时调整实时采集参数（窗口步长/窗口数/攒批大小），免去改 YAML 重启。

        ⚠️ 只改内存里的配置，不写回 settings.yaml —— 页面上的临时试验
        不应该污染配置文件。要长期生效请改 YAML。

        ★ continuous_batch_windows（攒批大小）是本项目**实时性的主旋钮**：
          滞后 ≈ 批内窗口数 × 窗口步长。它同时决定成本（每窗均摊 GPU 秒），
          所以只能取一个折中。做成可调是为了让用户自己感受这条曲线，
          不必每换一档就改 YAML 重启（重启要等约 2 分钟模型加载）。
        """
        body = body or {}
        live = dict(STATE.cfg.get("live") or {})
        if body.get("window_stride_sec") is not None:
            try:
                v = float(body["window_stride_sec"])
            except (TypeError, ValueError):
                raise HTTPException(400, "窗口步长必须是有限数字")
            if not math.isfinite(v) or v < 1:
                raise HTTPException(400, "窗口步长至少 1 秒")
            live["window_stride_sec"] = v
        if body.get("max_windows") is not None:
            n = int(body["max_windows"])
            if not (1 <= n <= 200):
                raise HTTPException(400, "窗口数需在 1-200 之间")
            live["max_windows"] = n
        if body.get("continuous_batch_windows") is not None:
            n = int(body["continuous_batch_windows"])
            # 上限 16：超过服务端 VLM_BATCH_MAX（默认 8）后，超出的窗口会排到
            # 下一批，攒批时间白白翻倍而吞吐不变 —— 属于纯亏，直接拦住。
            if not (1 <= n <= 16):
                raise HTTPException(400, "攒批窗口数需在 1-16 之间")
            live["continuous_batch_windows"] = n
        if "motion_recheck_sec" in body:
            try:
                v = float(body["motion_recheck_sec"])
            except (TypeError, ValueError):
                raise HTTPException(400, "静止复检间隔需在 1-300 秒之间")
            if not math.isfinite(v) or not 1 <= v <= 300:
                raise HTTPException(400, "静止复检间隔需在 1-300 秒之间")
            live["motion_recheck_sec"] = v
        STATE.cfg["live"] = live
        STATE.log_line(f"[实时] 参数更新 {live}")
        return {"ok": True, "live": live}

    @app.get("/api/live/cameras")
    def api_live_cameras():
        """枚举本机摄像头。

        ⚠️ 会真的打开设备（Windows 上可能亮摄像头灯），探测完立即释放。
        """
        try:
            from .capture import list_local_cameras
            from .winenv import camera_diagnosis, is_restricted

            cams, tried = list_local_cameras(with_detail=True)
            # ⚠️ 提示语必须说真话：受限令牌下"一个都找不到"的最常见原因
            #    是**权限**，不是硬件 —— 旧文案写死"可能是被其他程序占用"，
            #    把用户带去查驱动、关微信（实测就是这么被误导的）。
            return {"ok": True, "cameras": cams,
                    "restricted": is_restricted(),
                    "tried": tried,
                    "hint": camera_diagnosis(cams, "、".join(tried))}
        except Exception as e:
            return {"ok": False, "cameras": [],
                    "error": f"{type(e).__name__}: {e}"}

    @app.post("/api/live/probe")
    def api_live_probe(body: dict | None = None):
        """探测一个视频源是否可用，并测出真实帧率。

        先探测再采集 —— RTSP 地址/密码错、摄像头被占用这类问题，
        在探测阶段几秒内就能暴露，不必等采集完才发现没画面。
        """
        spec = ((body or {}).get("source") or "").strip()
        if not spec:
            raise HTTPException(400, "请填写视频源（0 表示本机摄像头，或 rtsp://...）")
        try:
            from .capture import RTSP_URL_HINTS, probe_source

            info = probe_source(spec)
            out = {"ok": info.ok, "kind": info.kind, "label": info.label,
                   "width": info.width, "height": info.height,
                   "measured_fps": info.measured_fps, "backend": info.backend,
                   "transport": info.transport, "error": info.error,
                   # 连不上时给出常见摄像头 URL 模板，省掉用户去搜索
                   "hints": RTSP_URL_HINTS if not info.ok else []}

            # ★ 路径敏感性探测（仅 RTSP 且已连通时做）：
            #   用一个绝不可能存在的路径再探一次。如果它也能读到画面，
            #   说明设备/网关**忽略 URL 路径**，那么这个地址"能用"并不能
            #   说明地址填对了 —— 这种情况在真实设备（尤其是 NVR 转发、
            #   某些路由器插件）上并不罕见，不提示会让人误以为万事大吉。
            if info.ok and info.kind == "rtsp":
                try:
                    import os as _os
                    import re as _re

                    bogus = _re.sub(r"(rtsp://[^/]+)/.*$", r"\1",
                                    spec.split("?", 1)[0])
                    bogus = f"{bogus}/__nonexistent_{_os.getpid()}__"
                    b = probe_source(bogus, timeout_sec=6)
                    if b.ok:
                        out["path_agnostic"] = True
                        out["note"] = ("设备忽略 URL 路径：随便填什么路径都能出画面。"
                                       "地址可用，但请优先用设备手册里的标准路径。")
                except Exception:
                    pass
            return out
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    @app.post("/api/live/start")
    def api_live_start(body: dict | None = None):
        """从实时源开始检测。

        body.continuous = true 时进入**持续监测**：一直跑直到用户停止，
        并要求模型描述画面里每个人在做什么（对应「摄像头监测」页）。
        """
        body = body or {}
        ensure_current_runtime()
        spec = (body.get("source") or "").strip()
        if not spec:
            raise HTTPException(400, "请填写视频源")

        # use_mock：没有 VPN / 集群排队时也能验证整条监测链路。
        # 生产用词统一叫「演示模式」，并且必须让用户一眼看出结论是假的。
        use_mock = bool(body.get("use_mock"))
        if not use_mock and (not STATE.session or STATE.session.state != "ready"):
            raise HTTPException(
                400,
                "推理服务未就绪。请先在「服务」页启动；"
                "或勾选「演示模式」在不启动集群的情况下跑通链路。",
            )

        try:
            from .capture import mask_credentials, parse_source

            kind, _ = parse_source(spec)
        except Exception as e:
            raise HTTPException(400, str(e))

        continuous = bool(body.get("continuous"))
        with_actions = bool(body.get("with_actions", continuous))

        # 同一时间只保留一个持续监测任务：否则多开几个会把推理服务排满队
        if continuous:
            for j in STATE.jobs.values():
                if j.continuous and j.state in ("capturing", "running", "queued"):
                    j.stop_flag = True
                    STATE.log_line(f"[监测] 停掉上一个监测任务 {j.job_id[:8]}")

        jid = uuid.uuid4().hex[:12]
        label = (mask_credentials(spec) if kind == "rtsp"
                 else (f"本机摄像头 #{spec}" if kind == "local" else Path(spec).name))
        job = JobRecord(
            job_id=jid, filename=label,
            source_kind=kind, source_spec=spec, is_live=True,
            continuous=continuous, with_actions=with_actions, use_mock=use_mock,
            mock_scene=str(body.get("mock_scene") or ""),
        )
        STATE.jobs[jid] = job
        STATE.log_line(f"[实时] {'持续监测' if continuous else '采集'} {label}"
                       + (f"（演示模式·{job.mock_scene or '默认'}）"
                          if use_mock else ""))
        threading.Thread(target=run_job, args=(jid,), daemon=True).start()
        return {"ok": True, "job_id": jid, "label": label, "kind": kind,
                "continuous": continuous, "with_actions": with_actions,
                "use_mock": use_mock}

    # ── 帧图片 ──
    @app.get("/api/job/{job_id}/frames/{window_id}/{idx}")
    def api_frame(job_id: str, window_id: str, idx: int):
        """窗口里的第 idx 帧（页面缩略图靠它）。

        ★ 帧号优先用**采集时记下的真实帧号**（rec["frame0"]），
          而不是拿 t_start 推算。

        ⚠️ 为什么必须改（用户实测反馈"ai动作识别中的图片始终是同一张，
           并不是人做对应动作的那一帧"）—— 这里是三重缺陷里的两个：
             ① 老算法 `i = int(t_start) + idx` 只在"窗口步长 == 帧间隔、
                且一帧都没丢"时才成立。步长 2 秒/帧间隔 2 秒时它**碰巧**对，
                一旦改步长就静默取到别的帧。实测（tools/test_frame_index.py）：
                步长 6 秒时它取第 6 帧，而本窗口第一帧其实是第 3 帧。
             ② 本接口**不返回任何缓存头**（Cache-Control/ETag/Last-Modified
                全缺）。响应没有验证器也没有显式过期时间时，浏览器可以按
                启发式规则自行缓存复用 —— 而前端缩略图 URL 又没有缓存破坏
                参数（对比：实时预览那两处都带了 `?t=Date.now()`）。
                两者叠加就是"所有卡片显示同一张老图"。
                修法分两半：**前端 URL 带帧号**（真正的修复）+ 这里不发
                缓存头（明知故犯是为了让"同一帧"还能走缓存，别让它变成
                每次都重传几十 KB）。见 index.html 的 addAction。
        """
        job = STATE.jobs.get(job_id)
        if not job:
            raise HTTPException(404, "任务不存在")
        rec = next((r for r in job.window_results if r["window_id"] == window_id), None)
        if not rec:
            raise HTTPException(404, "窗口不存在")
        d = Path(STATE.cfg["paths"]["frames_dir"]) / job_id
        files = sorted(d.glob("f_*.jpg"))
        if not files:
            raise HTTPException(404, "帧目录为空")
        base = int(rec.get("frame0", -1))
        if idx < 0 or idx >= int(rec.get("frames", 1)):
            raise HTTPException(404, "窗口内帧序号越界")
        if base >= 0:
            # 全局帧号不是裁剪后文件列表的下标；旧帧删掉后仍要定位同一张图。
            indices = rec.get("frame_indices")
            wanted = indices[idx] if isinstance(indices, list) and len(indices) > idx else base + idx
            target = next((p for p in files if frame_index_of(p) == wanted), None)
            if target is None:
                raise HTTPException(404, "该帧已清理或不存在")
            return FileResponse(str(target), media_type="image/jpeg")
        if base < 0:
            # 回退旧算法：只为兼容"改动前就已经记下的窗口记录"。
            # 新记录都会带 frame0，所以这条路径在正常使用中不会走到。
            base = int(rec["t_start"])
        i = max(0, min(base + idx, len(files) - 1))
        return FileResponse(str(files[i]), media_type="image/jpeg")

    @app.get("/api/job/{job_id}/latest")
    def api_job_latest(job_id: str):
        """返回该任务**最新的一帧**（监测页"实时画面"轮询这个接口）。

        为什么取"帧号最大的那张 jpg"而不是另开一路取流：
            它就是模型最近真正看到的那一帧 —— 所以页面上显示的画面
            和 AI 的判断依据**完全一致**。
            如果另开一路拉流做预览，页面会比 AI 的判断"更新"，
            用户看到"画面里明明有人在扔垃圾，怎么没告警"时无法分辨
            是模型漏了还是预览流与判定帧根本不是同一时刻。
            代价是画面最多滞后一个取样间隔（默认 2 秒），
            对"看摄像头那边在干什么"足够，而且这个滞后是**真实的**。
        """
        d = Path(STATE.cfg["paths"]["frames_dir"]) / job_id
        try:
            files = sorted(d.glob("f_*.jpg"))
        except OSError:
            files = []
        if not files:
            raise HTTPException(404, "暂无画面（可能还在采集第一个窗口）")
        return FileResponse(str(files[-1]), media_type="image/jpeg",
                            headers={"Cache-Control": "no-store"})

    @app.get("/api/job/{job_id}/stream.mjpg")
    async def api_job_mjpg(job_id: str):
        """MJPEG 实时画面（`multipart/x-mixed-replace`）。

        为什么不用"每秒轮询一张 JPEG"（原来那样）：
            轮询最多 3 张/秒，而且每次都要新建 HTTP 请求；用户看到的就是"卡"。
            浏览器对 multipart/x-mixed-replace 是**原生当视频渲染**的，
            可以稳定到摄像头本身的帧率（本地摄像头实测约 10fps），
            而且没有轮询开销。
        帧来源：读帧线程缓存的最新 JPEG（它在推理期间也一直在更新）——
            所以**推理再慢，画面也不会冻**。
        """
        job = STATE.jobs.get(job_id)
        if not job:
            raise HTTPException(404, "任务不存在")
        boundary = "civilityframe"

        async def gen():
            last_seq = -1
            idle = 0.0
            while True:
                reader = getattr(job, "reader", None)
                data = reader.snapshot_jpeg() if reader is not None else None
                seq = getattr(reader, "seq", -1) if reader is not None else -1
                if data is not None and seq != last_seq:
                    last_seq = seq
                    yield (b"--" + boundary.encode() + b"\r\n"
                           b"Content-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(data)).encode()
                           + b"\r\n\r\n" + data + b"\r\n")
                    idle = 0.0
                else:
                    idle += 0.06
                    # 任务已结束且没帧可发 → 收尾（浏览器会保留最后一帧）
                    if job.state in ("stopped", "done", "error") and idle > 1.0:
                        break
                    # 长时间没帧（流断了）→ 也收尾，别挂着连接
                    if idle > 20.0:
                        break
                await asyncio.sleep(0.06)
            yield b"--" + boundary.encode() + b"--\r\n"

        return StreamingResponse(
            gen(),
            media_type=f"multipart/x-mixed-replace; boundary={boundary}",
            headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
        )

    @app.get("/api/job/{job_id}/preview")
    def api_job_preview(job_id: str):
        """实时**预览**帧（约 3 fps），与"模型判定帧"分开。

        为什么要跟 /latest 分开（用户实测反馈"模拟的 RTSP 明显卡顿"）：
            /latest 给的是**模型判定帧** —— 只在取样时刻落盘（6 秒窗口 4 帧
            → 每 2 秒才换一张）。它诚实，但看起来就是一跳一跳的。
            预览帧由采集线程在"等下一个取样时刻"的空档里顺手写（约 0.3 秒一张），
            于是画面是顺的；而页面会明确标注两者的区别：
            上面那块是"看得顺的预览"，缩略图那块是"AI 真正看到的帧"。
        """
        d = Path(STATE.cfg["paths"]["frames_dir"]) / job_id
        # 预览帧是 preview_000012.jpg 这样的**一串**文件（每帧一个新名字），
        # 取最新的一张。
        # ⚠️ 为什么不是固定一个 preview.jpg：Windows 上"覆盖正被 HTTP 读着的文件"
        #    会 PermissionError，写入被静默跳过，页面上的实时画面就永远不动了
        #    （实测，见 tools/diag_preview.py）。
        cands = list(d.glob("preview_*.jpg"))
        if not cands:
            # 兼容旧路径 / 尚无预览时：退回最新的判定帧
            latest = sorted(d.glob("f_*.jpg"))
            if not latest:
                raise HTTPException(404, "暂无画面")
            return FileResponse(str(latest[-1]), media_type="image/jpeg",
                                headers={"Cache-Control": "no-store"})
        # 先按文件名（零填充序号，字典序=时间序），异常情况再按修改时间兜底
        cands.sort(key=lambda p: (p.name, p.stat().st_mtime))
        return FileResponse(str(cands[-1]), media_type="image/jpeg",
                            headers={"Cache-Control": "no-store"})

    return app


def start_watchdog() -> None:
    """起成本守卫线程：每 45 秒看一眼该不该把 A800 退掉。

    为什么是 45 秒：判据是"分钟级"的空闲，45 秒的检查间隔足够灵敏，
    又不会因为频繁 SSH 去打扰登录节点。
    看门狗本身不占 GPU、不花钱，只在真的要释放时才连集群。
    """
    svc = STATE.cfg.get("service") or {}
    idle_min = float(svc.get("idle_release_min", 15) or 0)
    max_hours = float(svc.get("max_run_hours", 3) or 0)
    if idle_min <= 0 and max_hours <= 0:
        STATE.log_line("成本守卫：已关闭（settings.yaml 里 idle_release_min "
                       "和 max_run_hours 都是 0）")
        return
    STATE.log_line(
        f"成本守卫已开启：页面关闭/切后台 {idle_min:.0f} 分钟自动释放节点；"
        f"服务最长连续运行 {max_hours:.1f} 小时"
        f"（正在分析画面时不会被硬切）"
    )

    def loop() -> None:
        while True:
            time.sleep(45)
            try:
                why = watchdog_tick()
                if why:
                    # 释放完顺手扫一遍孤儿，避免"停了自己、留着别人"
                    check_orphans()
            except Exception as e:
                STATE.log_line(f"（成本守卫异常，已跳过本轮：{type(e).__name__}: {e}）")

    threading.Thread(target=loop, daemon=True, name="cost-watchdog").start()


def find_free_port(preferred: int, tries: int = 20) -> int:
    """探测可用端口；默认启动拒绝占用，仅显式允许时才顺延。"""
    import socket

    for p in range(preferred, preferred + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError(f"{preferred}–{preferred + tries - 1} 全部被占用")


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="不文明行为识别 · 本地 Web 控制台")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8848)
    ap.add_argument("--no-auto-port", action="store_true",
                    help="兼容旧脚本；默认已不自动换端口")
    ap.add_argument("--allow-another-port", action="store_true",
                    help="明确允许同时启动另一实例并换端口；仍须访问新地址")
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()
    # Check before cleaning frame history or starting another cost watchdog.
    try:
        port = find_free_port(args.port, tries=20 if args.allow_another_port and not args.no_auto_port else 1)
    except RuntimeError:
        STATE.log_line(f"端口 {args.port} 已被旧服务占用。本次未启动。")
        STATE.log_line("请回到原启动终端按 Ctrl+C，停止旧服务后重新执行启动命令。")
        STATE.log_line("仅在新终端重复启动不会更新旧进程，浏览器仍会连接旧规则。")
        return 2
    if port != args.port:
        STATE.log_line(f"已按明确指定另开服务：http://{args.host}:{port}，请访问这个新地址。")

    # Model startup uploads the current code; no prebuilt package is required.
    STATE.log_line("═" * 56)
    STATE.log_line("不文明行为识别 · 本地控制台")
    STATE.log_line(f"行为定义：{STATE.cfg['paths']['behaviors']}")
    STATE.log_line(f"启用行为：{[b.id for b in STATE.behaviors.enabled]}")
    STATE.log_line(f"告警策略：{(STATE.cfg.get('aggregate') or {}).get('mode')}")
    STATE.log_line("集群代码与行为定义：启动模型时自动上传当前版本")

    # 启动时把"进程权限"和"摄像头能不能用"打出来。
    # ⚠️ 这条日志很重要：受限令牌（沙箱）下摄像头会静默枚举不到，
    #    而失败现象看起来像硬件坏了（实测被误导过很久）。
    try:
        from .winenv import restriction_reason

        why = restriction_reason()
        if why:
            STATE.log_line(f"⚠️ 进程权限受限：{why}")
            STATE.log_line("   → 摄像头枚举会被系统拒绝（看起来像'没找到设备'）。"
                           "要使用本机摄像头，请用你自己的 PowerShell 启动本服务")
        else:
            STATE.log_line("进程权限：正常（摄像头枚举结果可信）")
    except Exception as e:                      # noqa: BLE001
        STATE.log_line(f"（进程权限探测跳过：{type(e).__name__}）")

    # 启动时清理旧任务的帧目录：持续监测一小时能产几千张图，    # 攒着不管最后会把盘写满（实测一次 25 秒的持续监测就写了 2200+ 张）。
    # 只保留最近 N 个任务的目录 —— 启动时还没有任何任务在跑，删了不会影响页面。
    try:
        froot = Path(STATE.cfg["paths"]["frames_dir"])
        keep_jobs = int((STATE.cfg.get("paths") or {}).get("frames_keep_jobs", 20))
        if froot.is_dir() and keep_jobs > 0:
            dirs = [d for d in froot.iterdir() if d.is_dir()]
            if len(dirs) > keep_jobs:
                dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
                gone = freed = 0
                for d in dirs[keep_jobs:]:
                    for f in d.rglob("*"):
                        try:
                            freed += f.stat().st_size
                        except OSError:
                            pass
                    try:
                        import shutil as _sh

                        _sh.rmtree(d, ignore_errors=True)
                        gone += 1
                    except OSError:
                        pass
                STATE.log_line(
                    f"清理旧帧目录：{gone} 个任务 / {freed / 1048576:.1f} MB"
                    f"（保留最近 {keep_jobs} 个）"
                )
    except Exception as e:                      # 清理失败不能影响启动
        STATE.log_line(f"（旧帧清理跳过：{type(e).__name__}: {e}）")

    # 把实际端口写进文件：脚本、浏览器、其他工具都靠它找服务，
    # 免得出现"我明明启动了，怎么访问不到"。
    try:
        (HERE / "last_port.txt").write_text(str(port), encoding="utf-8")
    except OSError:
        pass

    STATE.log_line(f"打开浏览器访问：http://{args.host}:{port}")
    STATE.log_line("═" * 56)

    # ── 成本守卫：启动即开 ──
    start_watchdog()

    # 启动时扫一遍孤儿作业：网页重启后旧作业号会丢，它还在集群上按 12 小时
    # 上限一路烧钱（实测一个孤儿 ≈ 12 GPU·小时 ≈ 48 元）。
    # ⚠️ 这里**只提示不自动杀**：万一是你自己从别处提交的实验，
    #    自动杀会毁掉你的工作。页面顶部会挂横幅，一键即可停。
    if (STATE.cfg.get("service") or {}).get("check_orphans_on_start", True):
        def _orphan_probe() -> None:
            try:
                found = check_orphans()
                if found:
                    names = "、".join(f"{j.get('jobid')}({j.get('name')})"
                                      for j in found[:5])
                    STATE.log_line(
                        f"⚠️ 集群上还有 {len(found)} 个本项目名下的作业在跑：{names}"
                    )
                    STATE.log_line(
                        "   → 如果不是你正在用的，请点网页顶部横幅的"
                        "「立即停掉」，或跑：python tools/cluster_cost.py --stop"
                    )
                else:
                    STATE.log_line("集群检查：没有在跑的本项目作业 ✅")
            except Exception as e:
                STATE.log_line(f"集群检查跳过（{type(e).__name__}，多半是没连 VPN）")

        threading.Thread(target=_orphan_probe, daemon=True,
                         name="orphan-probe").start()

    import uvicorn

    try:
        uvicorn.run(create_app(), host=args.host, port=port, log_level="info")
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

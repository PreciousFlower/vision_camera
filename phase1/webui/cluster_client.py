"""本机 → 同济 A800 的客户端：作业提交、节点发现、两级隧道。

架构角色（整个网页方案的连接层）：

    浏览器(localhost:8848)
        │  HTTP
        ▼
    本机 FastAPI  ← 本模块
        │  ① paramiko SSH 到登录节点（需校园 VPN）
        │  ② direct-tcpip 通道：登录节点 → 计算节点
        ▼
    计算节点上的 vlm_service.py（常驻 Qwen3-VL）

为什么用 paramiko 的 direct-tcpip 而不是 `ssh -L`：
    密码含 `}` `&`，在 shell 里转义极易出错（见 tju/11b-credentials.md）；
    paramiko 走参数传递，零转义风险。且 `open_channel("direct-tcpip", ...)`
    天然覆盖"经登录节点跳到计算节点"这一场景，不需要 ssh_config。

实测记录（2026-10-05）：
    · 计算节点服务必须绑 0.0.0.0；绑 127.0.0.1 时登录节点连过来被拒
    · `srun -w <node>` 必须写成 `-n1 -c7`，写 `--cpus-per-task=7` 会被拒
      （平台按 ntasks × cpus 算核数，srun 默认 ntasks=2 → 等于要 14 核/卡）
"""
from __future__ import annotations

import os
import json
import re
import select
import socket
import sys
import threading
import time
import tempfile
import shutil
from pathlib import Path

# Linux 上的 select 没有 EPOLL 之外的问题；Windows 上 select 只支持 socket，
# 而我们只 select socket，所以两端都可用。
_POLL_TIMEOUT = 30.0


class ClusterError(RuntimeError):
    pass


def deploy_service_package(client, local_root: Path, behaviors: dict) -> None:
    """Upload the current service source and definitions before allocating a GPU."""
    with tempfile.TemporaryDirectory(prefix="civility-service-") as directory:
        package = Path(directory) / "pkg"
        package.mkdir()
        for name in ("civility", "cluster"):
            shutil.copytree(local_root / name, package / name,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (package / "behaviors.json").write_text(json.dumps(behaviors, ensure_ascii=False), encoding="utf-8")
        errors = client.upload_tree(package, client.workdir + "/pkg")
        if errors:
            raise ClusterError("推理代码上传失败：" + "; ".join(errors[:3]))
        # The submission entry point lives outside pkg; update it from the same source.
        conn, sftp = client.sftp()
        try:
            client.mkdirs(client.workdir + "/cluster", client.workdir + "/logs")
            sftp.put(str(local_root / "cluster/serve-vlm.sbatch"),
                     client.workdir + "/cluster/serve-vlm.sbatch")
        finally:
            sftp.close()
            conn.close()


def _tools_dir() -> str:
    return os.environ.get("VLM_TOOLS_DIR", r"E:\vibecoding\aipc\tools")


def _load_cluster_module():
    """复用 aipc 仓库里的 cluster.py（已修好密码认证，自动绕 DNS 问题）。"""
    d = _tools_dir()
    if d not in sys.path:
        sys.path.insert(0, d)
    try:
        import cluster  # type: ignore

        return cluster
    except ImportError as e:
        raise ClusterError(
            f"无法从 {d} 导入 cluster.py。"
            "请确认 aipc 仓库存在，或用环境变量 VLM_TOOLS_DIR 指定其 tools 目录。"
        ) from e


def auth_kwargs() -> dict:
    """认证参数：优先免密，其次环境变量里的密码。

    刻意不把密码写进配置或代码 —— 密码含特殊字符，落盘是安全隐患。
    """
    pw = os.environ.get("VLM_SSH_PASSWORD")
    if pw:
        return {"password": pw, "allow_agent": False, "look_for_keys": False}
    return {}


# ─────────────────────────── 集群操作 ───────────────────────────


class ClusterClient:
    """登录节点上的只读/提交操作。每次调用建一次连接，够用且简单。"""

    def __init__(self, workdir: str, python: str, model_root: str):
        self.workdir = workdir
        self.python = python
        self.model_root = model_root
        self._cluster = _load_cluster_module()

    # ── 基础 ──
    def run(self, cmd: str, timeout: int = 120) -> tuple[int, str, str]:
        return self._cluster.run(cmd, timeout=timeout, **auth_kwargs())

    def connect(self, timeout: int = 25):
        return self._cluster.connect(timeout=timeout, **auth_kwargs())

    def sftp(self, timeout: int = 25):
        return self._cluster.sftp(timeout=timeout, **auth_kwargs())

    def check_vpn(self) -> tuple[bool, str]:
        """快速判断能不能连上（VPN 是否已连）。"""
        try:
            rc, out, _ = self.run("hostname", timeout=30)
            return (rc == 0), (out or "").strip()
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    # ── 文件 ──
    def mkdirs(self, *dirs: str) -> None:
        self.run("mkdir -p " + " ".join(dirs), timeout=60)

    def upload_tree(self, local_dir: str | Path, remote_dir: str) -> list[str]:
        """递归上传目录（只传文件，自动建子目录）。返回失败清单。"""
        local_dir = Path(local_dir)
        c, sftp = self.sftp()
        failed: list[str] = []
        directories = {remote_dir}
        try:
            self.run(f"mkdir -p {remote_dir}", timeout=60)
            for p in sorted(local_dir.rglob("*")):
                if p.is_dir() or "__pycache__" in p.parts or p.suffix == ".pyc":
                    continue
                rel = p.relative_to(local_dir).as_posix()
                target = f"{remote_dir}/{rel}"
                parent = target.rsplit("/", 1)[0]
                try:
                    relative_parent = p.parent.relative_to(local_dir)
                    current = remote_dir
                    for component in relative_parent.parts:
                        current += "/" + component
                        if current not in directories:
                            try:
                                sftp.mkdir(current)
                            except OSError:
                                sftp.stat(current)  # Existing directories are fine; access failures are not.
                            directories.add(current)
                    sftp.put(str(p), target)
                except Exception as e:
                    failed.append(f"{rel}: {e}")
                    # 重连一次（长传容易掉线）
                    try:
                        sftp.close()
                        c.close()
                    except Exception:
                        pass
                    c, sftp = self.sftp()
                    directories = {remote_dir}
        finally:
            try:
                sftp.close()
                c.close()
            except Exception:
                pass
        return failed

    # ── Slurm ──
    def submit(self, sbatch_rel: str, extra: str = "") -> str:
        cmd = f"cd {self.workdir} && sbatch {extra} {sbatch_rel}"
        rc, out, err = self.run(cmd, timeout=90)
        m = re.search(r"(\d+)", out or "")
        if not m:
            raise ClusterError(f"提交失败：{out} {err}")
        return m.group(1)

    def job_state(self, jobid: str) -> tuple[str, str]:
        """返回 (状态, 节点)。作业已结束则状态为 COMPLETED/FAILED/'' 。"""
        rc, out, _ = self.run(
            f'squeue -j {jobid} -h -o "%T|%N" 2>/dev/null', timeout=60
        )
        s = (out or "").strip()
        if s and "|" in s:
            state, node = s.split("|", 1)
            return state.strip(), node.strip()
        # 不在队列里了 —— 查历史
        rc, out, _ = self.run(
            f'sacct -j {jobid} -n -X -o "State,NodeList" 2>/dev/null | head -1', timeout=60
        )
        parts = (out or "").split()
        if parts:
            return parts[0].strip(), (parts[1] if len(parts) > 1 else "")
        return "", ""

    def tail_log(self, jobid: str, kind: str = "out", n: int = 60) -> str:
        rc, out, _ = self.run(
            f"tail -n {n} {self.workdir}/logs/serve-{jobid}.{kind} 2>/dev/null", timeout=60
        )
        return out or ""

    def cancel(self, jobid: str) -> None:
        if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", str(jobid)):
            raise ClusterError("作业号无效，未执行释放命令")
        try:
            rc, _, _ = self.run(f"scancel {jobid}", timeout=20)
            if rc == 0:
                return
            # An already-ended allocation is safe only if the scheduler
            # independently confirms the job is absent.
            check_rc, current, _ = self.run('squeue -h -u u00009 -o "%i"', timeout=20)
            if check_rc == 0 and str(jobid) not in {line.strip() for line in current.splitlines()}:
                return
        except Exception as exc:
            raise ClusterError("释放作业失败，请检查集群连接后重试") from exc
        raise ClusterError(f"作业 {jobid} 尚未确认释放，请重试停止")

    def list_my_jobs(self) -> list[dict]:
        rc, out, _ = self.run(
            'squeue -u u00009 -h -o "%i|%j|%T|%M|%N|%R" 2>/dev/null', timeout=20
        )
        if rc != 0:
            raise ClusterError("集群作业查询失败，不能确认作业已释放")
        jobs = []
        for line in (out or "").splitlines():
            parts = line.split("|")
            if len(parts) >= 4:
                jobs.append({
                    "jobid": parts[0].strip(), "name": parts[1].strip(),
                    "state": parts[2].strip(), "elapsed": parts[3].strip(),
                    "node": (parts[4].strip() if len(parts) > 4 else ""),
                    "reason": (parts[5].strip() if len(parts) > 5 else ""),
                })
        return jobs

    def gpu_summary(self) -> str:
        rc, out, _ = self.run(
            'sinfo -p A800 -o "%12N %8t %14C %26G" 2>/dev/null | head -24', timeout=60
        )
        return out or ""

    def remote_py_compile(self, rel_path: str) -> tuple[bool, str]:
        """在集群上做语法检查 —— 上传后立刻验证，避免浪费一次排队。"""
        rc, out, err = self.run(
            f"cd {self.workdir} && {self.python} -m py_compile {rel_path} && echo OK",
            timeout=120,
        )
        return ("OK" in (out or "")), (out or "") + (err or "")


# ─────────────────────────── 两级隧道 ───────────────────────────


class Tunnel:
    """本机 127.0.0.1:<local> → 登录节点 → <node>:<remote> 的 TCP 转发。

    每条本地连接对应一条 paramiko 的 direct-tcpip 通道。
    用非阻塞 select 做双向 pump，超时 30 秒无数据即关闭，
    避免浏览器 keep-alive 把通道堆死。
    """

    def __init__(self, client: ClusterClient, node: str, remote_port: int,
                 local_port: int = 0):
        self.client = client
        self.node = node
        self.remote_port = remote_port
        self.local_port = local_port
        self._sock: socket.socket | None = None
        self._transport = None
        self._ssh = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._clients: list[threading.Thread] = []
        self.last_error: str | None = None

    # ── 生命周期 ──
    def start(self, timeout: float = 30.0) -> int:
        self._ssh = self.client.connect()
        self._transport = self._ssh.get_transport()
        if self._transport is None:
            raise ClusterError("SSH transport 建立失败")

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", self.local_port))
        self._sock.listen(16)
        self._sock.settimeout(1.0)
        self.local_port = self._sock.getsockname()[1]

        self._running = True
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

        # 立刻自测一次，确认隧道真的通（不要等第一次推理才发现）
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            try:
                ch = self._transport.open_channel(
                    "direct-tcpip", (self.node, self.remote_port), ("127.0.0.1", 0)
                )
                ch.close()
                return self.local_port
            except Exception as e:
                last = e
                time.sleep(1.5)
        self.stop()
        raise ClusterError(f"隧道自测失败（{self.node}:{self.remote_port}）：{last}")

    def stop(self) -> None:
        self._running = False
        for s in (self._sock,):
            try:
                if s:
                    s.close()
            except Exception:
                pass
        self._sock = None
        try:
            if self._ssh:
                self._ssh.close()
        except Exception:
            pass
        self._ssh = None
        self._transport = None

    @property
    def alive(self) -> bool:
        if not self._running or self._transport is None:
            return False
        try:
            return self._transport.is_active()
        except Exception:
            return False

    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.local_port}"

    # ── 内部 ──
    def _accept_loop(self) -> None:
        while self._running and self._sock is not None:
            try:
                conn, addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            t = threading.Thread(target=self._pump, args=(conn, addr), daemon=True)
            t.start()
            self._clients.append(t)
            self._clients = [x for x in self._clients if x.is_alive()]

    def _pump(self, conn: socket.socket, addr) -> None:
        chan = None
        try:
            if self._transport is None:
                return
            chan = self._transport.open_channel(
                "direct-tcpip", (self.node, self.remote_port), addr
            )
            # select retains the existing read-idle timeout; writes must wait
            # for a slow receiver instead of dropping a partial HTTP response.
            conn.settimeout(_POLL_TIMEOUT)
            while self._running:
                r, _, _ = select.select([conn, chan], [], [], _POLL_TIMEOUT)
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
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
        finally:
            for x in (chan, conn):
                try:
                    if x:
                        x.close()
                except Exception:
                    pass


# ─────────────────────────── 作业编排 ───────────────────────────


class ServiceSession:
    """一次"常驻推理服务"的完整生命周期：提交 → 等就绪 → 建隧道 → 用 → 收尾。

    这是 Web 后端对外的唯一入口对象。网页上的「启动服务 / 停止服务」
    按钮对应 start() / stop()。
    """

    class StartCancelled(ClusterError):
        """The user stopped this session while its startup was in progress."""

    def __init__(self, client: ClusterClient, model: str, port: int = 18080,
                 exclude: str = "gpu8002,gpu8003,gpu8012", extra_sbatch: str = ""):
        self.client = client
        self.model = model
        self.port = port
        self.exclude = exclude
        self.extra_sbatch = extra_sbatch
        self.jobid: str | None = None
        self.node: str | None = None
        self.tunnel: Tunnel | None = None
        self.state = "idle"        # idle|submitting|pending|running|ready|error|stopped
        self.message = ""
        self.started_at: float | None = None
        self.log_tail = ""
        self._stop_event = threading.Event()
        self._lifecycle_lock = threading.RLock()
        self._stop_lock = threading.Lock()
        self._release_pending = False
        self._health_result = None
        self._health_checked_at = 0.0
        self._health_worker = None

    @property
    def cancelled(self) -> bool:
        return self._stop_event.is_set()

    def _raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise self.StartCancelled("服务启动已取消")

    def _set_start_state(self, state: str) -> None:
        with self._lifecycle_lock:
            self._raise_if_cancelled()
            self.state = state

    # ── 启动 ──
    def start(self, wait_ready_sec: int = 900, progress=None) -> None:
        try:
            self._start(wait_ready_sec=wait_ready_sec, progress=progress)
        except Exception as exc:
            if not self.cancelled:
                raise
            # A submit/tunnel operation can return after stop() has completed.
            # Clean up its newly returned resources, without sweeping another
            # session's jobs, before reporting cancellation to the caller.
            self.stop(sweep_orphans=False)
            raise self.StartCancelled("服务启动已取消") from exc

    def _start(self, wait_ready_sec: int = 900, progress=None) -> None:
        def note(msg: str) -> None:
            with self._lifecycle_lock:
                self._raise_if_cancelled()
                self.message = msg
            if progress:
                progress(msg)
            print(f"[session] {msg}", flush=True)

        self._set_start_state("submitting")
        note("提交计算节点作业…")
        # Small live batches should reach one model together; replicas on the
        # same GPU split batch=2 into two batch=1 calls and compete for compute.
        # VLM_WORKERS remains an explicit override for larger offline workloads.
        workers = int(os.environ.get("VLM_WORKERS", "1") or 1)
        jobid = self.client.submit(
            "cluster/serve-vlm.sbatch",
            extra=f'--export=ALL,MODEL={self.model},PORT={self.port},'
                  f'VLM_WORKERS={workers} {self.extra_sbatch}',
        )
        with self._lifecycle_lock:
            # Retain the returned id even if stop happened during submission;
            # the cancellation handler must release this otherwise orphaned job.
            self.jobid = jobid
            self.started_at = time.time()
            self._raise_if_cancelled()
        note(f"作业 {self.jobid} 已提交")

        # 等 RUNNING 并拿到节点名
        t0 = time.time()
        self._set_start_state("pending")
        while time.time() - t0 < wait_ready_sec:
            self._raise_if_cancelled()
            st, node = self.client.job_state(self.jobid)
            self._raise_if_cancelled()
            if st in ("RUNNING", "COMPLETING") and node:
                self.node = node.split(",")[0]
                note(f"运行在 {self.node}（状态 {st}）")
                break
            if st in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"):
                self._set_start_state("error")
                self.log_tail = self.client.tail_log(self.jobid, "err")
                raise ClusterError(f"作业 {self.jobid} 失败（{st}）\n{self.log_tail[:2000]}")
            if st == "":
                self._set_start_state("error")
                self.log_tail = self.client.tail_log(self.jobid, "err")
                raise ClusterError(f"作业 {self.jobid} 已不在队列且未运行\n{self.log_tail[:2000]}")
            if progress and int(time.time() - t0) % 20 < 2:
                note(f"排队中…（{st}，{int(time.time() - t0)}s）")
            self._stop_event.wait(8)
        else:
            self._set_start_state("error")
            raise ClusterError(f"等待作业启动超时（{wait_ready_sec}s）")

        self._set_start_state("running")

        # 等模型加载完：轮询服务健康检查（先建隧道才能问）
        note("建立两级隧道…")
        with self._lifecycle_lock:
            self._raise_if_cancelled()
            tunnel = Tunnel(self.client, self.node, self.port)
            self.tunnel = tunnel
        local_port = tunnel.start()
        if self.cancelled:
            # stop() may already have detached the tunnel while start() waited.
            tunnel.stop()
            self._raise_if_cancelled()
        note(f"隧道就绪 127.0.0.1:{local_port} → {self.node}:{self.port}")

        note("等待模型加载（首次需从 /ssdfs 读 8.3GB 权重）…")
        t1 = time.time()
        import requests

        while time.time() - t1 < 900:
            self._raise_if_cancelled()
            try:
                r = requests.get(f"{tunnel.base_url()}/health", timeout=20)
                self._raise_if_cancelled()
                info = r.json()
                if info.get("ready"):
                    with self._lifecycle_lock:
                        self._health_result = info
                        self._health_checked_at = time.monotonic()
                    self._set_start_state("ready")
                    note(f"模型就绪：{info.get('model')} @ {info.get('device')}"
                         f"（加载 {info.get('load_sec')}s）")
                    return
                if info.get("error"):
                    self._set_start_state("error")
                    raise ClusterError(f"模型加载失败：{info['error']}")
                if progress:
                    note(f"加载中…（{info.get('loading') and '进行中' or '未开始'}）")
            except ClusterError:
                raise
            except Exception as e:
                # 服务还没起来时连接会被拒绝，这是正常的
                if progress:
                    note(f"等待服务端口响应…（{type(e).__name__}）")
            self._stop_event.wait(10)

        self._set_start_state("error")
        raise ClusterError("等待模型加载超时")

    # ── 使用 ──
    def base_url(self) -> str:
        if not self.tunnel:
            raise ClusterError("隧道未建立")
        return self.tunnel.base_url()

    def health(self, timeout: float = 20) -> dict:
        """问服务健康状态。**未就绪不是错误**，要如实返回 ready=false。

        ⚠️ 多 worker 场景下，网关在 worker 还没加载完时会返回 503
        （"暂无可用的 worker"）。这不是故障，是"正在启动" ——
        如果这里抛异常，本机就会把"正在加载模型"误报成"启动失败"
        （实测踩到：隧道自测/健康检查把加载中的状态当成了错误）。
        """
        import requests

        with self._lifecycle_lock:
            tunnel = self.tunnel
        try:
            r = requests.get(f"{self.base_url()}/health", timeout=timeout)
        except requests.exceptions.RequestException as e:
            info = {"ready": False, "state": "unreachable", "error": f"{type(e).__name__}: {e}"}
        except ClusterError:
            info = {"ready": False, "state": "unreachable", "error": "隧道未建立"}
        else:
            if r.status_code != 200:
                info = {"ready": False, "state": "starting", "error": f"HTTP {r.status_code}: {r.text[:200]}"}
            else:
                try:
                    info = r.json()
                    if not isinstance(info, dict):
                        raise ValueError("健康检查格式无效")
                except ValueError:
                    info = {"ready": False, "state": "bad-response", "error": "健康检查响应格式无效"}
        with self._lifecycle_lock:
            if self.tunnel is tunnel and self.state == "ready" and not self.cancelled:
                self._health_result = info
                self._health_checked_at = time.monotonic()
        return info

    def poll_health(self) -> None:
        """One short background probe per ten seconds, never block UI polling."""
        with self._lifecycle_lock:
            if self.state != "ready" or self.cancelled:
                return
            if self._health_worker and self._health_worker.is_alive():
                return
            if time.monotonic() - self._health_checked_at < 10:
                return
            self._health_worker = threading.Thread(target=self.health, kwargs={"timeout": 3}, daemon=True)
            self._health_worker.start()

    def infer(self, frames_b64: list[str], labels: list[str],
              max_new_tokens: int = 512, few_shot: bool = False,
              with_alert: bool = False, with_actions: bool = False,
              with_risks: bool = False, behaviors: dict | None = None) -> dict:
        import requests

        payload = {
            "frames": [{"b64": b} for b in frames_b64],
            "labels": labels,
            "max_new_tokens": max_new_tokens,
            "few_shot": few_shot,
            # 让模型自己判断该不该告警（aggregate.mode: ai）
            "with_alert": with_alert,
            # 让模型描述画面里每个人在做什么（摄像头监测页）
            "with_actions": with_actions,
            # 检查预设行为以外、有可见证据的安全风险。
            "with_risks": with_risks,
        }
        if behaviors is not None:
            payload["behaviors"] = behaviors
        r = None
        last_err: Exception | None = None
        request_started = time.perf_counter()
        # ⚠️ 允许一次重试：多 worker 场景下，网关可能刚好把请求转发给
        #    一个正在重启/刚挂掉的 worker（实测遇到 ConnectionError:
        #    RemoteDisconnected，整批窗口全部失败）。重试一次能挡住这类
        #    瞬时故障 —— 注意只重试**连接类**错误，不重试业务错误。
        for attempt in (1, 2):
            try:
                r = requests.post(f"{self.base_url()}/infer", json=payload,
                                  timeout=300)
                # Busy queues need time to drain. A timed-out request may
                # already be running on the GPU; resubmitting duplicates work.
                if r.status_code < 500 or r.status_code in (503, 504) or attempt == 2:
                    break
            except requests.exceptions.RequestException as e:
                last_err = e
                if attempt == 2:
                    raise ClusterError(
                        f"推理请求失败（已重试一次）：{type(e).__name__}: {e}") from e
                time.sleep(0.8)
        if r is None:                               # 理论到不了，防御
            raise ClusterError(f"推理请求失败：{last_err}")
        if r.status_code != 200:
            raise ClusterError(f"推理失败 HTTP {r.status_code}: {r.text[:500]}")
        result = r.json()
        result["roundtrip_sec"] = round(time.perf_counter() - request_started, 3)
        return result

    def ask_video(self, frames_b64: list[str], question: str, evidence: list[dict], context: str) -> dict:
        """Use the deployed VLM's dedicated visual QA protocol, without fake behavior definitions."""
        import requests

        health = self.health()
        if health.get("video_qa_protocol") != 1:
            raise ClusterError("当前模型服务尚未支持视频问答，请停止并重新启动推理服务以加载新版代码")
        if health.get("ready") is not True:
            raise ClusterError("模型尚未就绪，请稍后提问")
        started = time.perf_counter()
        try:
            response = requests.post(f"{self.base_url()}/video-qa", json={
                "frames": [{"b64": b} for b in frames_b64], "question": question,
                "evidence": evidence, "context": context, "max_new_tokens": 768,
            }, timeout=300)
            response.raise_for_status()
            result = response.json()
        except (requests.exceptions.RequestException, ValueError) as exc:
            raise ClusterError(f"视频问答请求失败：{type(exc).__name__}: {exc}") from exc
        if not isinstance(result, dict) or result.get("error") or not result.get("parsed"):
            raise ClusterError(str(result.get("error") if isinstance(result, dict) else "模型返回格式错误"))
        result["roundtrip_sec"] = round(time.perf_counter() - started, 3)
        return result

    def reload_behaviors(self, behaviors_json_path: str) -> None:
        import requests

        r = requests.post(
            f"{self.base_url()}/reload-behaviors",
            json={"path": behaviors_json_path}, timeout=30,
        )
        r.raise_for_status()

    # ── 收尾 ──
    def stop(self, sweep_orphans: bool = True) -> list[str]:
        """停止服务并释放计算节点。

        ⚠️ sweep_orphans 为什么重要（踩过）：
            服务作业号只存在 ServiceSession 内存里。网页重启、换端口、
            或上一次启动中途失败时，旧作业就变成"孤儿"——
            界面显示已停止，但集群上仍在烧机时（约 4 元/卡时）。
            实测中一次就留下两个孤儿作业在跑。
            所以停止时顺手扫一遍自己名下所有 civility-serve 作业，
            一并取消。只动自己启动的作业名，不碰别的。
        """
        self._stop_event.set()
        with self._stop_lock:
            return self._stop(sweep_orphans=sweep_orphans)

    def _stop(self, sweep_orphans: bool = True) -> list[str]:
        cancelled: list[str] = []
        with self._lifecycle_lock:
            if self.state == "stopped" and not self.jobid and self.tunnel is None and not self._release_pending:
                return []
            self.state = "stopping"
            self.message = "正在停止服务并释放计算节点…"
            self._release_pending = True
            tunnel = self.tunnel
        try:
            if tunnel:
                tunnel.stop()
                with self._lifecycle_lock:
                    if self.tunnel is tunnel:
                        self.tunnel = None
            with self._lifecycle_lock:
                jobid = self.jobid
            if jobid:
                self.client.cancel(jobid)
                cancelled.append(jobid)
                with self._lifecycle_lock:
                    if self.jobid == jobid:
                        self.jobid = None
            if sweep_orphans:
                for j in self.client.list_my_jobs():
                    if j.get("name") == "civility-serve" and j["jobid"] not in cancelled:
                        self.client.cancel(j["jobid"])
                        cancelled.append(j["jobid"])
        except Exception as exc:
            with self._lifecycle_lock:
                self.state = "error"
                self.message = f"停止或释放失败：{exc}。请重试停止并释放节点。"
            raise

        with self._lifecycle_lock:
            self.state = "stopped"
            self.node = None
            self.started_at = None
            self._release_pending = False
            self.message = ("已停止并释放计算节点"
                            + (f"（含 {len(cancelled) - 1} 个遗留作业）"
                               if len(cancelled) > 1 else ""))
        return cancelled

    def status(self) -> dict:
        with self._lifecycle_lock:
            worker = getattr(self, "_start_worker", None)
            state, message = self.state, self.message
            health = self._health_result
            if state == "ready" and (not self.tunnel or not self.tunnel.alive or (health is not None and health.get("ready") is not True)):
                state, message = "unreachable", "模型连接异常，正在自动检测恢复；计算节点仍需停止并释放。"
            return {
                "state": state, "message": message,
                "jobid": self.jobid, "node": self.node, "model": self.model,
                "local_url": self.base_url() if self.tunnel else None,
                "uptime_sec": round(time.time() - self.started_at, 1) if self.started_at else None,
                "tunnel_alive": self.tunnel.alive if self.tunnel else False,
                "release_pending": self._release_pending,
                "start_worker_alive": bool(worker and worker.is_alive()),
            }

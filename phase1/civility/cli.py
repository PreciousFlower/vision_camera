"""命令行入口 —— 第一阶段的全部操作都从这一个入口走。

    python -m civility.cli check                # 体检：配置、依赖、ffmpeg
    python -m civility.cli prompt              # 打印将要发给模型的完整 Prompt
    python -m civility.cli extract              # 视频 → 抽帧 + manifest
    python -m civility.cli run                  # 跑推理（mock/cloud/cluster）
    python -m civility.cli aggregate            # → 告警 alerts.jsonl
    python -m civility.cli score                # → 评测报告 report.md
    python -m civility.cli stability            # Prompt 一致性体检
    python -m civility.cli all                  # 一把梭：extract → run → aggregate → score
    python -m civility.cli cluster-prepare      # 打包给 A800 的作业包
    python -m civility.cli cluster-fetch        # 从 A800 拉结果
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import ROOT, load_behaviors, load_settings


def _force_utf8_console() -> None:
    """Windows 控制台默认 GBK，输出中文/emoji 会 UnicodeEncodeError 直接崩。

    在入口处把 stdout/stderr 切成 UTF-8，比在每个 print 里做兼容省事得多。
    （Windows Terminal / VS Code 终端都能正确显示 UTF-8。）
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def _cfg(args):
    cfg = load_settings(args.settings)
    if getattr(args, "backend", None):
        cfg["backend"] = args.backend
    bs = load_behaviors(cfg.get("paths", {}).get("behaviors"))
    return cfg, bs


def _request_options(cfg: dict) -> dict[str, bool]:
    """保持命令行、云端和集群的识别请求与当前告警设置一致。"""
    agg = cfg.get("aggregate") or {}
    return {"with_alert": str(agg.get("mode") or "").lower() == "ai",
            "with_actions": False,
            "with_risks": bool(agg.get("open_risks", True))}


def _run_dir(cfg: dict, name: str) -> Path:
    d = Path(cfg["paths"]["results_dir"]) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cluster_connect_kwargs() -> dict:
    """集群认证参数。

    优先免密（agent/私钥），这是最省事也最安全的做法。
    若尚未配置免密，把密码放进环境变量 VLM_SSH_PASSWORD —— 绝不写进代码或 YAML，
    也不要在 shell 里拼接（密码含 `}` 和 `&`，转义极易出错，
    详见 tju/11b-credentials.md）。
    """
    import os

    pw = os.environ.get("VLM_SSH_PASSWORD")
    if pw:
        return {"password": pw, "allow_agent": False, "look_for_keys": False}
    return {}


def _cluster_module(cfg: dict):
    """导入 aipc 仓库里的 cluster.py（paramiko 封装，自动绕 DNS 问题）。"""
    cl = cfg.get("cluster") or {}
    tools_dir = str(cl.get("tools_dir") or "")
    if tools_dir and tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    try:
        import cluster  # type: ignore

        return cluster
    except ImportError:
        print(f"❌ 无法从 {tools_dir} 导入 cluster.py")
        print("   检查 settings.yaml 的 cluster.tools_dir 是否指向 E:\\vibecoding\\aipc\\tools")
        return None


def _connect_or_hint(cfg: dict):
    """连接集群，失败时给出可执行的排查步骤而不是一句报错。"""
    cluster = _cluster_module(cfg)
    if cluster is None:
        return None, None
    kw = _cluster_connect_kwargs()
    try:
        c, sftp = cluster.sftp(timeout=20, **kw)
        return c, sftp
    except Exception as e:
        print(f"❌ 连接失败：{type(e).__name__}: {e}")
        print("   按顺序排查：")
        print("   1) 学校 VPN 连了吗？（不连则 100% 丢包，无公网入口）")
        print("      ping 192.168.212.236")
        print("   2) 端口是 10022，不是 22")
        print("   3) 换备用 IP：192.168.212.245")
        print("   4) 认证方式：推荐先配免密，用文档里那条 ssh-copy-id；")
        print("      或临时设置环境变量 VLM_SSH_PASSWORD（不要写进代码）")
        return None, None


def _manifest_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["data_dir"]) / "manifest.jsonl"


def _annotations_path(cfg: dict, args) -> Path:
    if getattr(args, "annotations", None):
        return Path(args.annotations)
    return Path(cfg["paths"]["data_dir"]) / "annotations.jsonl"


# ───────────────────────────── check ─────────────────────────────


def cmd_check(args) -> int:
    cfg, bs = _cfg(args)
    print(f"civility phase1 v{__version__}")
    print(f"项目根目录: {ROOT}")
    print(f"后端: {cfg.get('backend')}")

    ok = True

    # 1) 行为定义
    print(f"\n[1] 行为定义 ({cfg['paths']['behaviors']})")
    for b in bs.behaviors:
        mark = "启用" if b.enabled else "停用"
        usable = "" if b.is_usable or not b.enabled else "  ⚠️ 定义不完整，已忽略"
        print(f"    [{mark}] {b.id:20s} {b.name}{usable}")
    if not bs.enabled:
        print("    ❌ 没有任何启用的行为")
        ok = False

    # 2) ffmpeg
    print("\n[2] ffmpeg / ffprobe")
    try:
        from .extract import find_ffmpeg

        ff, fp = find_ffmpeg()
        print(f"    ✅ {ff}\n    ✅ {fp}")
    except Exception as e:
        print(f"    ❌ {e}")
        ok = False

    # 3) Python 依赖
    print("\n[3] Python 依赖")
    for mod in ("yaml", "requests", "paramiko"):
        try:
            __import__(mod)
            print(f"    ✅ {mod}")
        except ImportError:
            hint = "pip install pyyaml requests paramiko"
            print(f"    ❌ {mod} 缺失 —— {hint}")
            if mod in ("yaml", "requests"):
                ok = False

    # 4) 数据
    print(f"\n[4] 数据目录 {cfg['paths']['data_dir']}")
    vids = sorted(Path(cfg["paths"]["data_dir"]).glob("videos/*"))
    print(f"    视频: {len(vids)} 个")
    for v in vids[:10]:
        print(f"      - {v.name}")
    if not vids:
        print("    ⚠️ 没有视频。第一阶段不需要真摄像头，放几个 mp4 进去即可：")
        print(f"      {Path(cfg['paths']['data_dir']) / 'videos'}")
    ann = Path(cfg["paths"]["data_dir"]) / "annotations.jsonl"
    print(f"    标注: {'✅ ' + str(ann) if ann.exists() else '⚠️ 缺失 ' + str(ann)}")

    # 5) 后端可用性
    print("\n[5] 后端连通性")
    kind = (cfg.get("backend") or "mock").lower()
    if kind == "mock":
        print("    ✅ mock 后端，无需网络")
    elif kind == "cloud":
        from .config import get_api_key

        key = get_api_key(cfg)
        c = cfg.get("cloud") or {}
        print(f"    端点: {c.get('base_url')}  模型: {c.get('model')}")
        print(f"    API Key: {'✅ 已设置' if key else '❌ 未设置（环境变量 ' + str(c.get('api_key_env')) + '）'}")
        if not key:
            ok = False
    elif kind == "cluster":
        cl = cfg.get("cluster") or {}
        tools = Path(cl.get("tools_dir", ""))
        print(f"    集群结果文件: {cl.get('remote_workdir')}")
        print(f"    本地工具目录 cluster.py: {'✅' if (tools / 'cluster.py').exists() else '❌ ' + str(tools)}")
        print("    ⚠️ 集群需要先挂学校 VPN，否则完全不通")
    print()
    print("结论：" + ("✅ 可以开始" if ok else "❌ 有阻塞项，见上"))
    return 0 if ok else 1


# ───────────────────────────── prompt ─────────────────────────────


def cmd_prompt(args) -> int:
    cfg, bs = _cfg(args)
    from .prompts import build_system_prompt

    text = build_system_prompt(bs, **_request_options(cfg))
    out = Path(cfg["paths"]["results_dir"]) / "prompt-preview.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[已保存] {out}")
    print(
        "\n提示：这份 Prompt 就是喂给模型的全部内容。"
        "改识别效果先改它（或改 behaviors.yaml），不用动代码。"
    )
    return 0


# ───────────────────────────── extract ─────────────────────────────


def cmd_extract(args) -> int:
    cfg, bs = _cfg(args)
    from .extract import build_windows, extract_frames, probe_duration, write_manifest

    data_dir = Path(cfg["paths"]["data_dir"])
    vdir = Path(args.videos) if args.videos else data_dir / "videos"
    frames_root = Path(cfg["paths"]["frames_dir"])
    ex = cfg.get("extract") or {}
    dflt = bs.defaults or {}

    vids = sorted([p for p in vdir.glob("*") if p.suffix.lower() in (".mp4", ".avi", ".mkv", ".mov", ".flv", ".ts", ".m4v")])
    if not vids:
        print(f"❌ {vdir} 下没有视频文件")
        return 1

    all_windows = []
    for v in vids:
        dur = probe_duration(v)
        frames = extract_frames(
            v,
            frames_root / v.stem,
            fps=float(ex.get("fps", 1.0)),
            max_side=int(ex.get("max_side", 640)),
            jpeg_quality=int(ex.get("jpeg_quality", 82)),
            overwrite=args.overwrite,
        )
        ws = build_windows(
            v,
            frames,
            window_stride_sec=float(dflt.get("window_stride_sec", 4.0)),
            max_frames_per_window=int(ex.get("max_frames_per_window", 4)),
            window_duration_sec=float(ex.get("window_duration_sec", dflt.get("window_stride_sec", 4.0))),
        )
        all_windows += ws
        print(f"  {v.name}: {dur:.1f}s → {len(frames)} 帧 → {len(ws)} 个窗口")

    out = Path(args.out) if args.out else _manifest_path(cfg)
    write_manifest(all_windows, out)
    print(f"\n[manifest] {out}  共 {len(all_windows)} 个窗口")
    print("下一步：python -m civility.cli run")
    return 0


# ───────────────────────────── run ─────────────────────────────


def cmd_run(args) -> int:
    cfg, bs = _cfg(args)
    from .extract import read_manifest
    from .runner import behavior_fingerprint, run_windows, write_summary
    from .vlm import build_backend

    mp = Path(args.manifest) if args.manifest else _manifest_path(cfg)
    if not mp.exists():
        print(f"❌ manifest 不存在：{mp}\n先跑 python -m civility.cli extract")
        return 1
    windows = read_manifest(mp)

    options = _request_options(cfg)
    fp_ = behavior_fingerprint(bs, **options)
    rd = _run_dir(cfg, Path(mp).parent.name if Path(mp).parent.name != "data" else "default")
    out_path = Path(args.out) if args.out else rd / f"windows-{fp_}.jsonl"

    cluster_results = None
    if (cfg.get("backend") or "").lower() == "cluster":
        cluster_results = Path(args.cluster_results) if args.cluster_results else Path(cfg["paths"]["results_dir"]) / "cluster" / "results.jsonl"
        if not Path(cluster_results).exists():
            print(
                f"❌ 集群结果不存在：{cluster_results}\n"
                "顺序应为：cluster-prepare → 上传 → sbatch → cluster-fetch → run"
            )
            return 1

    backend = build_backend(cfg, cluster_results=cluster_results)
    conc = int(args.concurrency or (cfg.get("cloud") or {}).get("concurrency", 1))
    stats = run_windows(
        windows, backend, bs, out_path,
        concurrency=conc, retry_failed=args.retry_failed,
        **options,
    )
    stats["behavior_fingerprint"] = fp_
    write_summary(stats, rd / "run-stats.json")

    c = getattr(backend, "cost_cny", None)
    if callable(c):
        print(f"[成本] 累计 token 输入 {backend.tokens_in_total} / 输出 {backend.tokens_out_total}"
              f" ⇒ 约 {c():.4f} 元（单价以控制台账单为准）")

    print(f"\n[结果] {out_path}")
    print("下一步：python -m civility.cli aggregate && python -m civility.cli score")
    return 0


# ──────────────────────────── aggregate ────────────────────────────


def cmd_aggregate(args) -> int:
    cfg, bs = _cfg(args)
    from .aggregate import derive_events, write_alerts
    from .runner import behavior_fingerprint, load_results

    mp = Path(args.manifest) if args.manifest else _manifest_path(cfg)
    rd = _run_dir(cfg, Path(mp).parent.name if Path(mp).parent.name != "data" else "default")
    fp_ = behavior_fingerprint(bs, **_request_options(cfg))
    in_path = Path(args.input) if args.input else rd / f"windows-{fp_}.jsonl"

    results = load_results(in_path)
    if not results:
        print(f"❌ 没有结果可聚合：{in_path}")
        return 1

    alerts = derive_events(results, bs, cfg.get("aggregate") or {})
    out = Path(args.out) if args.out else rd / "alerts.jsonl"
    write_alerts(alerts, out)

    print(f"[聚合] {len(results)} 个窗口 → {len(alerts)} 条告警  →  {out}")
    for a in alerts[:20]:
        print(
            f"  {a.t_start:6.1f}-{a.t_end:6.1f}s  {a.behavior_name:8s} "
            f"conf={a.max_confidence:.2f}  {a.hits}/{a.span}  {Path(a.video).name}"
        )
    if len(alerts) > 20:
        print(f"  ... 另有 {len(alerts) - 20} 条")
    return 0


# ───────────────────────────── score ─────────────────────────────


def cmd_score(args) -> int:
    cfg, bs = _cfg(args)
    from .aggregate import derive_events
    from .runner import behavior_fingerprint, load_results
    from .score import (
        load_annotations,
        render_report,
        score_events,
        score_windows,
    )

    mp = Path(args.manifest) if args.manifest else _manifest_path(cfg)
    rd = _run_dir(cfg, Path(mp).parent.name if Path(mp).parent.name != "data" else "default")
    fp_ = behavior_fingerprint(bs, **_request_options(cfg))
    in_path = Path(args.input) if args.input else rd / f"windows-{fp_}.jsonl"

    results = load_results(in_path)
    if not results:
        print(f"❌ 没有结果可评分：{in_path}")
        return 1

    labels = load_annotations(_annotations_path(cfg, args))
    agg = cfg.get("aggregate") or {}
    min_conf = float(agg.get("min_confidence", 0.45))

    win = score_windows(results, labels, bs, min_confidence=min_conf)
    alerts = derive_events(results, bs, agg)
    evt = score_events(alerts, labels, bs)

    stats_file = rd / "run-stats.json"
    stats = json.loads(stats_file.read_text(encoding="utf-8")) if stats_file.exists() else None
    report = render_report(win, evt, bs, stats)

    out = Path(args.out) if args.out else rd / "report.md"
    out.write_text(report, encoding="utf-8")
    json_out = out.with_suffix(".json")
    json_out.write_text(json.dumps({"window": win, "event": evt, "stats": stats}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(report)
    print(f"\n[报告] {out}\n[机读] {json_out}")
    return 0


# ──────────────────────────── stability ────────────────────────────


def cmd_stability(args) -> int:
    cfg, bs = _cfg(args)
    from .aggregate import stability_report
    from .extract import read_manifest
    from .vlm import build_backend

    mp = Path(args.manifest) if args.manifest else _manifest_path(cfg)
    windows = read_manifest(mp)
    if not windows:
        print("❌ manifest 为空")
        return 1

    # 取 N 个窗口（默认跨不同视频均匀取），每个重复 R 次
    n = int(args.windows)
    step = max(1, len(windows) // n)
    sample = windows[::step][:n]

    backend = build_backend(cfg)
    repeated: dict[str, list[dict]] = {}
    for i, w in enumerate(sample, 1):
        objs = []
        for _ in range(int(args.repeat)):
            req = w.to_request()
            for key, value in _request_options(cfg).items():
                setattr(req, key, value)
            r = backend.infer(req, bs)
            if r.parsed:
                objs.append(r.parsed)
        repeated[w.window_id] = objs
        print(f"  [{i}/{len(sample)}] {w.window_id} → {len(objs)} 次有效输出")

    rep = stability_report(repeated, bs)
    out = _run_dir(cfg, "default") / "stability.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"\n[一致性] {out}")
    return 0


# ────────────────────────────── all ──────────────────────────────


def cmd_all(args) -> int:
    # `all` 复用其他子命令的处理函数，但那些函数依赖各自子解析器定义的属性。
    # 例如 cmd_extract 读 args.overwrite，而 all 的解析器上没有这个开关。
    # 这里统一补默认值，避免"换个入口就 AttributeError"。
    for attr, default in (
        ("overwrite", False),
        ("retry_failed", False),
        ("concurrency", None),
        ("cluster_results", None),
    ):
        if not hasattr(args, attr):
            setattr(args, attr, default)

    for fn in (cmd_extract, cmd_run, cmd_aggregate, cmd_score):
        rc = fn(args)
        if rc != 0:
            print(f"❌ {fn.__name__} 失败，中止")
            return rc
        print()
    return 0


# ───────────────────────── cluster 相关 ─────────────────────────


def cmd_cluster_prepare(args) -> int:
    """把 manifest + 帧 + Prompt + 集群脚本打包成一个可直接上传的目录。"""
    cfg, bs = _cfg(args)
    import shutil
    import tarfile

    from .extract import read_manifest
    from .prompts import build_system_prompt

    mp = Path(args.manifest) if args.manifest else _manifest_path(cfg)
    windows = read_manifest(mp)
    if not windows:
        print(f"❌ manifest 为空或不存在：{mp}")
        print("   manifest 由抽帧生成，请先跑：python -m civility.cli extract")
        return 1

    pkg = Path(cfg["paths"]["results_dir"]) / "cluster" / "pkg"
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir(parents=True)

    # 只带 manifest 里引用的帧，避免把整个 frames 目录传上去
    (pkg / "manifest.jsonl").write_text(
        "\n".join(json.dumps(w.__dict__, ensure_ascii=False) for w in windows) + "\n",
        encoding="utf-8",
    )
    options = _request_options(cfg)
    (pkg / "system_prompt.txt").write_text(
        build_system_prompt(bs, **options), encoding="utf-8")
    (pkg / "behaviors.json").write_text(
        json.dumps(
            {
                "defaults": bs.defaults,
                "request_options": options,
                "behaviors": [
                    {
                        "id": b.id,
                        "name": b.name,
                        "definition": b.definition,
                        "evidence_required": b.evidence_required,
                        "counter_examples": b.counter_examples,
                        "observable_cues": b.observable_cues,
                        "evidence_checks": b.evidence_checks,
                        "evidence_sequence": b.evidence_sequence,
                        "severity": b.severity,
                    }
                    for b in bs.enabled
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    img_dir = pkg / "images"
    img_dir.mkdir()
    n_img = 0
    for w in windows:
        for f in w.frames:
            src = Path(f)
            if src.exists():
                dst = img_dir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
                    n_img += 1

    # 把集群侧要跑的脚本 + 共享的 civility 包一起打包。
    # ⚠️ 必须带上 civility/ 包本身：集群脚本 import civility.prompts 来生成 Prompt，
    #    这样云端和集群用的是同一份 Prompt 代码，两边的准确率才有可比性。
    cluster_src = ROOT / "cluster"
    if cluster_src.exists():
        shutil.copytree(cluster_src, pkg / "cluster")
    pkg_src = ROOT / "civility"
    if pkg_src.exists():
        shutil.copytree(
            pkg_src,
            pkg / "civility",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

    total_mb = sum(p.stat().st_size for p in pkg.rglob("*") if p.is_file()) / 1e6
    print(f"[打包] {pkg}")
    print(f"  窗口 {len(windows)}  图片 {n_img}  体积 {total_mb:.1f} MB")
    if total_mb > 200:
        print("  ⚠️ 体积偏大。SFTP 上传约 1.25 MB/s，这样要几分钟。")
        print("     可减小 extract.max_side 或提高 window_stride_sec 来降体积。")

    tgz = pkg.with_suffix(".tar.gz")
    with tarfile.open(tgz, "w:gz") as tf:
        for p in pkg.rglob("*"):
            tf.add(p, arcname=p.relative_to(pkg.parent))
    print(f"[压缩] {tgz}  ({tgz.stat().st_size / 1e6:.1f} MB)")
    print("\n下一步：python -m civility.cli cluster-upload （或手工 scp）")
    return 0


def cmd_cluster_upload(args) -> int:
    cfg, bs = _cfg(args)
    cl = cfg.get("cluster") or {}
    tgz = Path(cfg["paths"]["results_dir"]) / "cluster" / "pkg.tar.gz"
    if not tgz.exists():
        print(f"❌ 找不到 {tgz}，先跑 cluster-prepare")
        return 1

    cluster = _cluster_module(cfg)
    if cluster is None:
        return 1
    kw = _cluster_connect_kwargs()

    print("⚠️ 需要先连学校 VPN。正在连接登录节点…")
    c, sftp = _connect_or_hint(cfg)
    if c is None:
        return 1
    print(f"✅ 已连接 {getattr(c, '_vlm_host', '?')}")

    wd = cl.get("remote_workdir", "/share/home/u00009/apps/civility-kit")
    remote_tgz = f"{wd}/pkg.tar.gz"
    try:
        cluster.run(f"mkdir -p {wd}", timeout=60, **kw)
        print(f"上传 → {remote_tgz} …")
        sftp.put(str(tgz), remote_tgz)
        rc, out, err = cluster.run(
            # ⚠️ 包内路径是 pkg/…（tar 的 arcname 是 pkg/xxx），
            #    所以 manifest.jsonl 在 pkg/manifest.jsonl。
            #    旧写法在 $WORKDIR 下直接 wc manifest.jsonl，必然失败 ——
            #    于是**解包其实成功了却报"解包失败"**，白折腾一轮（实测踩到）。
            f"cd {wd} && tar xzf pkg.tar.gz && ls -la && "
            f"echo '--- 窗口数 ---' && wc -l pkg/manifest.jsonl",
            timeout=180, **kw,
        )
        print(out or err)
        if rc != 0:
            print(f"❌ 解包失败 rc={rc}")
            return 1
    finally:
        try:
            sftp.close()
            c.close()
        except Exception:
            pass

    print("\n✅ 上传完成。下一步在集群上提交作业：")
    print("   ssh -p 10022 u00009@192.168.212.236")
    print(f"   cd {wd} && sbatch cluster/run-vlm.sbatch")
    return 0


def cmd_cluster_fetch(args) -> int:
    cfg, bs = _cfg(args)
    cl = cfg.get("cluster") or {}
    cluster = _cluster_module(cfg)
    if cluster is None:
        return 1
    kw = _cluster_connect_kwargs()

    wd = cl.get("remote_workdir", "/share/home/u00009/apps/civility-kit")
    local = Path(cfg["paths"]["results_dir"]) / "cluster"
    local.mkdir(parents=True, exist_ok=True)

    print("正在连接登录节点（需 VPN）…")
    try:
        rc, out, err = cluster.run(
            f"ls -la {wd}/results.jsonl 2>/dev/null; "
            f"squeue -u u00009 -o '%.10i %.20j %.9T %.11M %R' | head -20; "
            f"ls -1 {wd}/logs/ 2>/dev/null | tail -5",
            timeout=120, **kw,
        )
        print(out or err)
    except Exception as e:
        print(f"❌ 连接失败：{e}")
        print("   见上文排查顺序（VPN → 端口 → IP → 认证方式）")
        return 1

    c, sftp = _connect_or_hint(cfg)
    if c is None:
        return 1
    try:
        dst = local / "results.jsonl"
        try:
            sftp.get(f"{wd}/results.jsonl", str(dst))
            n = sum(1 for _ in open(dst, encoding="utf-8"))
            print(f"✅ 已拉回 {dst}（{n} 行）")
        except FileNotFoundError:
            print(f"⚠️ 远端还没有 {wd}/results.jsonl —— 作业可能仍在跑或失败了")
            print(f"   看日志：ssh 上去 tail -40 {wd}/logs/*.out")
            return 1
        try:
            sftp.get(f"{wd}/logs", str(local / "logs"))
        except Exception:
            pass
    except Exception as e:
        print(f"❌ 拉取失败：{e}")
        return 1
    finally:
        try:
            sftp.close()
            c.close()
        except Exception:
            pass
    return 0


# ───────────────────────────── main ─────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="civility",
        description="摄像头不文明行为识别 —— 第一阶段开发工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--settings", default=None, help="settings.yaml 路径")
    p.add_argument("--backend", default=None, choices=["mock", "cloud", "cluster"], help="覆盖后端")
    p.add_argument("--manifest", default=None, help="manifest.jsonl 路径")
    p.add_argument("--annotations", default=None, help="annotations.jsonl 路径")
    p.add_argument("--input", default=None, help="输入结果 jsonl 路径")
    p.add_argument("--out", default=None, help="输出路径")
    p.add_argument("--videos", default=None, help="视频目录（默认 data/videos）")

    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="体检：配置/依赖/ffmpeg/后端").set_defaults(func=cmd_check)
    sub.add_parser("prompt", help="打印完整 Prompt").set_defaults(func=cmd_prompt)

    e = sub.add_parser("extract", help="视频 → 抽帧 + manifest")
    e.add_argument("--overwrite", action="store_true", help="重新抽帧")
    e.set_defaults(func=cmd_extract)

    r = sub.add_parser("run", help="跑推理")
    r.add_argument("--concurrency", type=int, default=None)
    r.add_argument("--retry-failed", action="store_true", help="重跑上次失败的窗口")
    r.add_argument("--cluster-results", default=None, help="集群结果 jsonl")
    r.set_defaults(func=cmd_run)

    sub.add_parser("aggregate", help="窗口结果 → 告警").set_defaults(func=cmd_aggregate)
    sub.add_parser("score", help="算 P/R/F1 并出报告").set_defaults(func=cmd_score)

    s = sub.add_parser("stability", help="Prompt 一致性体检")
    s.add_argument("--windows", type=int, default=8, help="抽样窗口数")
    s.add_argument("--repeat", type=int, default=3, help="每窗口重复次数")
    s.set_defaults(func=cmd_stability)

    sub.add_parser("all", help="extract → run → aggregate → score").set_defaults(func=cmd_all)

    cp = sub.add_parser("cluster-prepare", help="打包给 A800 的作业包")
    cp.set_defaults(func=cmd_cluster_prepare)
    cu = sub.add_parser("cluster-upload", help="上传作业包到 A800")
    cu.set_defaults(func=cmd_cluster_upload)
    cf = sub.add_parser("cluster-fetch", help="从 A800 拉结果")
    cf.set_defaults(func=cmd_cluster_fetch)
    return p


def main(argv: list[str] | None = None) -> int:
    _force_utf8_console()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n[中断] 已完成的结果已落盘，重跑会自动续跑")
        return 130


if __name__ == "__main__":
    sys.exit(main())

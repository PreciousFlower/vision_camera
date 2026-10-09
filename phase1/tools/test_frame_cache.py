#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端复现/验证：页面上"AI 动作识别"的图**是不是真的跟着窗口变**。

用户反馈（原话）："ai动作识别中的图片始终是同一张，并不是人做对应动作的那一帧。"

这个缺陷有两半，各自单独测都不够 —— 必须**端到端**跑一遍：
    前端那一半：缩略图 URL 有没有缓存破坏参数 → tools/test_thumb_url.py 覆盖
    服务端那一半：/frames/<window>/<idx> 对不同窗口是否返回**不同的字节**
                → 本文件覆盖（这是唯一能证伪它的方式）

★ 为什么要起真服务、发真请求：
    只读代码能看出"URL 变了"，但证明不了"服务端真的会给出不同的图"。
    /frames 的帧号原来是拿 int(t_start) 推算的，看着对、其实只在特定步长下
    碰巧对（见 tools/test_frame_index.py 的实测）。这里用**真实 HTTP 响应字节**
    来判：不同窗口的图必须不同，同一窗口必须相同（缓存还得能用）。

⚠️ 隔离：用 CIVILITY_SETTINGS 指向一份临时 settings.yaml（frames_dir 改到临时
    目录），所以**不会**写进用户正在用的 data/frames/<job_id>。
⚠️ 用假模型（use_mock）：不连集群、不花卡时、不需要 VPN。
"""
from __future__ import annotations

import http.client
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(ROOT))

PORT = 8873        # 故意避开用户实例的 8848
TMP = ROOT / "runs" / "_tmp_frame_cache_test"


def log(m: str = "") -> None:
    print(m, flush=True)


def make_isolated_settings() -> Path:
    """复制真 settings.yaml，只把 frames_dir 改到临时目录。"""
    import yaml

    src = ROOT / "config" / "settings.yaml"
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    frames = TMP / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    cfg.setdefault("paths", {})["frames_dir"] = str(frames)
    # 行为定义仍然用真的（缩略图与行为无关，但服务启动时要读）
    out = TMP / "settings.yaml"
    out.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                   encoding="utf-8")
    return out


def wait_up(timeout: float = 40.0) -> dict:
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout:
        try:
            c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=3)
            c.request("GET", "/api/state")
            r = c.getresponse()
            data = json.loads(r.read().decode("utf-8"))
            c.close()
            return data
        except Exception as e:                              # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            time.sleep(0.4)
    raise RuntimeError(f"webui 没起来（最后错误 {last}）")


def req(method: str, path: str, body: dict | None = None):
    """返回 (status, headers, bytes)。"""
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=30)
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    hdrs = {"Content-Type": "application/json"} if payload else {}
    c.request(method, path, payload, hdrs)
    r = c.getresponse()
    data = r.read()
    hs = {k.lower(): v for k, v in r.getheaders()}
    st = r.status
    c.close()
    return st, hs, data


def main() -> int:
    if TMP.exists():
        shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)
    settings = make_isolated_settings()

    # ⚠️ 素材要用 **.mp4**，不要用 .h264（实测踩到）：
    #    裸 h264 没有容器，OpenCV 的 CAP_PROP_FRAME_COUNT 返回荒谬的负数
    #    （实测 -153722867280913），而"文件冒充实时源"这条路径是**按帧号跳转**
    #    取帧的 —— 帧数一错就一个窗口都取不到，测试表现为"窗口数不足"，
    #    看起来像被测功能坏了，其实是素材格式问题。
    #    （服务端已经为这种情况加了顺序读回退，见 capture.py 的 seq_mode；
    #      但测试用有容器的 mp4 更直接。）
    demo = ROOT / "runs" / "demo_cam.mp4"
    if not demo.exists():
        alt = ROOT / "runs" / "demo_cam.h264"
        if alt.exists():
            log(f"⚠️ 缺 {demo.name}，用 ffmpeg 转一份…")
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                            "-r", "20", "-i", str(alt), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", str(demo)], check=False)
    if not demo.exists():
        log(f"⚠️ 缺少素材 {demo}，跳过（用 tools/make_demo_stream.py 生成）")
        return 0

    env = dict(os.environ)
    env["CIVILITY_SETTINGS"] = str(settings)
    env["CIVILITY_MOCK_LATENCY"] = "0.05"
    log(f"起隔离实例 127.0.0.1:{PORT}（settings={settings.relative_to(ROOT)}）")
    proc = subprocess.Popen(
        [sys.executable, "-m", "webui.server", "--port", str(PORT),
         "--no-auto-port"],
        cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    failed: list[str] = []

    def check(cond: bool, label: str, extra: str = "") -> None:
        log(f"  {'✅' if cond else '❌'} {label}" + (f"　{extra}" if extra else ""))
        if not cond:
            failed.append(label)

    try:
        wait_up()
        log("实例已就绪\n")

        # 用"视频文件当实时源"跑持续监测（免摄像头、免集群）
        st, _, body = req("POST", "/api/live/config",
                          {"window_stride_sec": 2.0,
                           "continuous_batch_windows": 2})
        log(f"[config] HTTP {st} {body.decode('utf-8')[:160]}")
        st, _, body = req("POST", "/api/live/start",
                          {"source": str(demo), "continuous": True,
                           "use_mock": True, "mock_scene": "littering",
                           "with_actions": True})
        if st != 200:
            log(f"❌ 启动监测失败：HTTP {st} {body.decode('utf-8')[:300]}")
            return 1
        jid = json.loads(body.decode("utf-8"))["job_id"]
        log(f"[start] job={jid}")

        # 等够几个窗口
        wins: list[dict] = []
        t0 = time.time()
        while time.time() - t0 < 60:
            _, _, b = req("GET", f"/api/job/{jid}")
            j = json.loads(b.decode("utf-8"))
            wins = j.get("window_results") or []
            if len(wins) >= 3:
                break
            time.sleep(1.0)
        log(f"[wait] 拿到 {len(wins)} 个窗口结果")
        if len(wins) < 3:
            log("❌ 窗口数不足，无法验证"); return 1

        req("POST", f"/api/stop/{jid}")

        # ★ 判据：用页面**真实的** frameThumbUrl 生成 URL，再走真 HTTP 取字节。
        #   这样前后端两半是被同一条判据连起来的。
        import re
        html = (ROOT / "webui" / "static" / "index.html").read_text(encoding="utf-8")
        m = re.search(r"^function\s+frameThumbUrl\s*\(", html, re.M)
        if not m:
            log("❌ 页面里找不到 frameThumbUrl"); return 1
        i = html.index("{", m.end() - 1)
        depth = 0
        for j2 in range(i, len(html)):
            if html[j2] == "{":
                depth += 1
            elif html[j2] == "}":
                depth -= 1
                if depth == 0:
                    fn = html[m.start(): j2 + 1]
                    break
        log("\n[A] 服务端：不同窗口的缩略图必须返回**不同字节**")
        log("     （用页面真实的 frameThumbUrl 生成 URL）")
        urls = []
        for w in wins[:3]:
            js = (fn + "\nconsole.log(frameThumbUrl(%s, %s));"
                  % (json.dumps(jid), json.dumps(w, ensure_ascii=False)))
            tmpjs = TMP / "u.js"
            tmpjs.write_text(js, encoding="utf-8")
            r = subprocess.run(["node", str(tmpjs)], capture_output=True,
                               text=True, encoding="utf-8")
            if r.returncode != 0:
                log(f"❌ node 执行失败：{(r.stderr or '')[:400]}"); return 1
            urls.append(r.stdout.strip().splitlines()[-1])

        blobs = []
        for u, w in zip(urls, wins[:3]):
            st, hs, d = req("GET", u)
            log(f"   {w['window_id']} frame0={w.get('frame0')} → {u}")
            log(f"      HTTP {st}  {len(d)} 字节  "
                f"cache-control={hs.get('cache-control', '(未设置)')}")
            blobs.append(d)

        uniq = len({hash(b) for b in blobs})
        check(uniq == len(blobs),
              f"{len(blobs)} 个窗口的图互不相同（实际不同 {uniq} 张）",
              "全部相同 = 用户报的那个 bug")
        check(all(len(b) > 1000 for b in blobs), "每张图都是有效 JPEG（>1KB）")

        log("\n[B] 同一窗口重复请求必须返回**相同字节**（缓存还得能用）")
        u0 = urls[0]
        st, hs, d1 = req("GET", u0)
        st2, hs2, d2 = req("GET", u0)
        check(d1 == d2, "同一 URL 两次请求字节一致")

        log("\n[C] 实时画面那两路必须带 no-store（它们不是缩略图，不能缓存）")
        for path, name in ((f"/api/job/{jid}/preview", "/preview"),
                           (f"/api/job/{jid}/latest", "/latest")):
            st, hs, _ = req("GET", path)
            cc = (hs.get("cache-control") or "").lower()
            check("no-store" in cc, f"{name} 返回 Cache-Control: {cc or '(空)'}")

        log("\n[D] 帧号确实来自采集记录（不是拿时间戳推算）")
        bad = [w for w in wins if int(w.get("frame0", -1)) < 0]
        check(not bad, f"每个窗口记录都带真实 frame0（异常 {len(bad)} 个）")
        # frame0 必须严格递增，且步长等于"每窗帧数"
        f0 = [int(w["frame0"]) for w in wins]
        nf = [int(w.get("frames") or 0) for w in wins]
        check(all(b > a for a, b in zip(f0, f0[1:])),
              f"frame0 严格递增：{f0[:8]}")
        check(all(b - a == n for a, b, n in zip(f0, f0[1:], nf)),
              f"相邻窗口的帧号差 == 每窗帧数（{f0[:5]} / frames={nf[:5]}）")

        log("\n" + "=" * 62)
        if failed:
            log(f"❌ {len(failed)} 项失败：")
            for f in failed:
                log(f"   · {f}")
            return 1
        log("✅ 全部通过 —— 每张卡片显示的确实是**它自己那个窗口**的画面")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        shutil.rmtree(TMP, ignore_errors=True)
        log("（隔离实例已停，临时目录已清理）")


if __name__ == "__main__":
    raise SystemExit(main())

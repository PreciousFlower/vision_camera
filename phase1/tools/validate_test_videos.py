"""Run the user's clips through the deployed model, in an isolated short-lived job.

Outputs raw model answers, frame timestamps, accepted decisions and a Markdown report.
Cancels only the allocation created by this script, including when validation fails.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import time

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_validation_cluster import configure_auth
from civility.config import behavior_payload, load_behaviors, load_settings
from civility.extract import build_windows, extract_frames, write_manifest
from civility.prompts import sanitize
from civility.recognition import generation_budget, infer_with_review
from webui.cluster_client import ClusterClient, Tunnel

ROOT = Path(__file__).resolve().parents[1]


def prepare(output, remote):
    cfg, bs = load_settings(), load_behaviors()
    package = output / "package"
    package.mkdir(parents=True, exist_ok=True)
    for directory in ("civility", "cluster"):
        shutil.copytree(ROOT / directory, package / directory, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (package / "behaviors.json").write_text(json.dumps({"defaults": bs.defaults,
        "behaviors": [asdict(b) for b in bs.enabled]}, ensure_ascii=False, indent=2), encoding="utf-8")
    cases = []
    for index, video in enumerate(sorted((ROOT.parent / "test").glob("*.mp4"))):
        frames = extract_frames(video, package / "images" / f"clip-{index}",
                                fps=cfg["extract"]["fps"], max_side=cfg["extract"]["max_side"])
        windows = build_windows(video, frames, window_stride_sec=bs.defaults["window_stride_sec"],
                                window_duration_sec=cfg["extract"]["window_duration_sec"],
                                max_frames_per_window=cfg["extract"]["max_frames_per_window"])
        write_manifest(windows, output / f"clip-{index}-manifest.jsonl")
        expected = "smoking" if video.stem == "吸烟" else "no_safety_helmet" if video.stem == "安全帽" else "littering"
        for window in windows:
            cases.append({"video": video.name, "expected": expected, "window_id": window.window_id,
                          "t_start": window.t_start, "t_end": window.t_end, "labels": window.labels,
                          "frames": [{"path": remote + "/" + Path(p).relative_to(package).as_posix()}
                                     for p in window.frames]})
        # Earlier parts of the same clip are hard negatives for the action that comes later.
        if video.stem in ("吸烟", "乱丢垃圾"):
            cut = 1.5 if expected == "smoking" else 3.0
            subset = [(t, p) for t, p in frames if t < cut]
            cases.append({"video": video.name, "expected": "negative", "window_id": video.stem + "__negative",
                          "t_start": 0, "t_end": cut, "labels": [f"t={t:.1f}s" for t, p in subset],
                          "frames": [{"path": remote + "/" + Path(p).relative_to(package).as_posix()}
                                     for t, p in subset]})
    (output / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    return cfg, bs, package, cases


def write_report(output, rows, bs, elapsed, job_id):
    summary = {}
    for row in rows:
        clean = sanitize(row.get("parsed") or {}, bs, frame_count=len(row["frames"]))
        hits = [bid for bid, v in clean["violations"].items() if v["violation"] and v["alert"] is not False]
        row["accepted_hits"] = hits
        record = summary.setdefault(row["video"], {"expected": row["expected"], "positive_hits": 0,
                                                   "negative_false_hits": [], "errors": 0, "missing_process": 0})
        if row["expected"] == "negative":
            record["negative_false_hits"].extend(hits)
        elif row["expected"] in hits:
            record["positive_hits"] += 1
            if not any(p.get("changed") for p in clean.get("people", [])):
                record["missing_process"] += 1
        if row.get("error"):
            record["errors"] += 1
    (output / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 2026-10-10 用户视频真模型验证", "", "模型：Qwen/Qwen3-VL-4B-Instruct；同济 A800。",
             f"作业：{job_id}；本次占用及等待约 {elapsed:.1f} 秒，脚本结束释放。", "",
             "|视频|目标|命中窗口|前段负例误报|缺少过程描述|错误|", "|---|---|---:|---|---:|---:|"]
    for video, item in summary.items():
        lines.append(f"|{video}|{item['expected']}|{item['positive_hits']}|{', '.join(item['negative_false_hits']) or '无'}|{item['missing_process']}|{item['errors']}|")
    lines += ["", "这是用户提供的少量样例测试，不能作为总体准确率。每窗原始回答、证据与时间戳见 results.json。", ""]
    for row in rows:
        lines += [f"## {row['video']} · {row['t_start']}-{row['t_end']} 秒", "",
                  f"命中：{', '.join(row['accepted_hits']) or '无'}；错误：{row.get('error') or '无'}", "",
                  "```json", row.get("raw") or "{}", "```", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return all(item["positive_hits"] > 0 and not item["negative_false_hits"] and not item["errors"]
               and not item["missing_process"]
               for item in summary.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--include-legacy-cup", action="store_true", help="Also replay the screenshot's 4-frame input")
    parser.add_argument("--output", default=str(ROOT / "runs/test-video-validation/model"))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    remote = "/share/home/u00009/apps/civility-validation-20261010"
    cfg, bs, package, cases = prepare(output, remote)
    if args.include_legacy_cup:
        cup = next(c for c in cases if c["video"].startswith("420c8"))
        cases.append({**cup, "window_id": "cup__legacy_4_frames", "t_start": 0, "t_end": 4,
                      "labels": ["t=0.0s", "t=1.0s", "t=2.0s", "t=3.0s"],
                      "frames": [cup["frames"][i] for i in (0, 3, 6, 9)], "sampling": "legacy_4_frames"})
    if args.prepare_only:
        print(f"Prepared {len(cases)} windows: {output}")
        return 0
    configure_auth()
    cc = cfg["cluster"]
    client = ClusterClient(remote, cc["remote_python"], cc["remote_model_root"])
    ok, info = client.check_vpn()
    if not ok:
        raise RuntimeError(f"Cluster unavailable: {info}")
    _, active, _ = client.run('squeue -u u00009 -h -o "%i %j %T"', timeout=30)
    if active.strip():
        raise RuntimeError("Another allocation is active; validation does not cancel or interfere with it")
    print(f"Uploading {len(cases)} test windows and current recognition code", flush=True)
    errors = client.upload_tree(package, remote)
    if errors:
        raise RuntimeError(f"Upload failed: {errors[:3]}")
    port = 18180
    job_script = f'''#!/bin/bash
#SBATCH --job-name=civility-validation
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48g
#SBATCH --time=00:15:00
#SBATCH --exclude={cc['exclude_nodes']}
#SBATCH --output={remote}/validation-%j.log
export PYTHONPATH={remote}
export HF_HOME={cc['remote_model_root']}/hf
export HF_HUB_CACHE={cc['remote_model_root']}/hf/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd {remote}
exec {cc['remote_python']} cluster/vlm_service.py --pkg {remote} --behaviors {remote}/behaviors.json --model {cc['remote_model']} --bind 0.0.0.0 --port {port}
'''
    (package / "validation.sbatch").write_text(job_script, encoding="utf-8", newline="\n")
    conn, sftp = client.sftp()
    try:
        sftp.put(str(package / "validation.sbatch"), remote + "/validation.sbatch")
    finally:
        sftp.close()
        conn.close()
    job_id, tunnel = None, None
    rows = []
    started = time.time()
    try:
        job_id = client.submit("validation.sbatch")
        (output / "job.json").write_text(json.dumps({"job_id": job_id, "remote": remote}), encoding="utf-8")
        print(f"Validation job {job_id}; maximum allocation 15 minutes", flush=True)
        for _ in range(60):
            status, node = client.job_state(job_id)
            if status == "RUNNING" and node:
                break
            if status in ("FAILED", "CANCELLED", "TIMEOUT"):
                raise RuntimeError(f"Job failed: {status}")
            time.sleep(5)
        else:
            raise TimeoutError("Allocation did not start")
        for attempt in range(30):
            tunnel = Tunnel(client, node.split(",")[0], port)
            try:
                tunnel.start()
                break
            except Exception:
                tunnel.stop()
                tunnel = None
                time.sleep(3)
        if tunnel is None:
            raise RuntimeError("Inference tunnel did not open")
        for attempt in range(60):
            health = requests.get(tunnel.base_url() + "/health", timeout=20).json()
            if health.get("ready"):
                print(f"Deployed model ready: {health.get('model')} on {health.get('device')}", flush=True)
                break
            if health.get("error"):
                raise RuntimeError(health["error"])
            time.sleep(5)
        else:
            raise TimeoutError("Model did not become ready")
        for i, case in enumerate(cases, 1):
            def send(frames, labels, **options):
                response = requests.post(tunnel.base_url() + "/infer", json={
                    "frames": frames, "labels": labels, **options}, timeout=180)
                response.raise_for_status()
                return response.json()
            result = infer_with_review(send, case["frames"], case["labels"], bs,
                       max_new_tokens=generation_budget(bs, with_alert=True, with_actions=True, with_risks=True),
                       with_alert=True, with_actions=True, with_risks=True)
            row = {**case, **result}
            rows.append(row)
            (output / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            hits = [k for k, v in (row.get("parsed") or {}).get("violations", {}).items() if v.get("violation")]
            print(f"[{i}/{len(cases)}] {case['video']} {case['t_start']}-{case['t_end']}s: {hits}, error={row.get('error')}", flush=True)
        passed = write_report(output, rows, bs, time.time() - started, job_id)
        return 0 if passed else 1
    finally:
        if tunnel:
            tunnel.stop()
        if job_id:
            client.run(f"scancel {job_id}", timeout=30)
            code, allocation, _ = client.run(f"squeue -j {job_id} -h -o '%T'", timeout=30)
            print(f"Released own validation job {job_id}; remaining state: {allocation.strip() or 'absent'}", flush=True)
        if rows and not (output / "report.md").exists():
            write_report(output, rows, bs, time.time() - started, job_id)


if __name__ == "__main__":
    raise SystemExit(main())

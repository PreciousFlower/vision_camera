"""Replay user clips inside a slow-inference gap, optionally using real Qwen.

The static padding emulates time spent waiting for a previous GPU response.
All action pictures are decoded from the user's clips; no detection result is
fabricated. Validation outputs stay in an ignored directory. Allocate at most
one 15-minute job and always release only that job.
"""
from __future__ import annotations

import argparse
import base64
import copy
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from civility.config import behavior_payload, load_behaviors, load_settings
from civility.recognition import generation_budget, infer_with_review
from webui.motion_highlights import MotionHighlights
from webui import server as S

OUT = ROOT / "runs/_live_priority_20261011"
CASES = [("吸烟.mp4", "smoking"), ("乱丢垃圾.mp4", "littering"),
         ("安全帽.mp4", "no_safety_helmet")]


def make_history(video: Path, config: dict):
    cap = cv2.VideoCapture(str(video))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        images = []
        for index in range(int(duration * 8)):
            cap.set(cv2.CAP_PROP_POS_MSEC, index * 125)
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            side = config["extract"]["max_side"]
            if max(h, w) > side:
                scale = side / max(h, w)
                frame = cv2.resize(frame, (round(w * scale), round(h * scale)))
            images.append(frame)
    finally:
        cap.release()
    if not images:
        raise RuntimeError(f"Cannot decode {video.name}")
    tracker, history, costs = MotionHighlights(), [], []
    # Previously examined time ends at 108; latest view is 124 after 16s wait.
    for index in range(193):
        timestamp = 100 + index / 8
        clip_index = min(len(images) - 1, max(0, index - 64))
        frame = images[clip_index]
        started = time.perf_counter()
        tracker.add(timestamp, frame)
        costs.append((time.perf_counter() - started) * 1000)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        history.append((timestamp, encoded.tobytes()))
    return history, list(tracker.scores), costs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-model", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(load_settings())
    cfg["paths"]["frames_dir"] = str(OUT / "frames")
    rows, batches = [], []
    report = {"status": "preparing", "model_requested": args.real_model,
              "baseline_commit": "f6b2d6d36d288f46be1f3f896d8d73aa0d2a4221", "cases": rows}

    def persist():
        (OUT / "real-validation.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    persist()
    for case_index, (filename, behavior) in enumerate(CASES):
        history, peaks, costs = make_history(ROOT.parent / "test" / filename, cfg)
        windows_by_mode = []
        for mode, highlights in (("before", []), ("after", peaks)):
            reader = SimpleNamespace(dead=False, snapshot_history=lambda: history,
                                     snapshot_highlights=lambda: highlights)
            job = S.JobRecord(f"replay-{case_index}-{mode}", filename, is_live=True,
                continuous=True, source_kind="local", capture_started_at=100,
                last_recent_wall=108, last_batch_sec=16)
            with patch.object(S.STATE, "cfg", cfg), patch.object(S.time, "time", return_value=124.1), \
                    patch.object(S.STATE, "log_line"):
                windows = S.capture_recent_windows(job, reader)
            windows_by_mode.append(windows)
        before, after = windows_by_mode
        assert len(after) == len(before) == 2
        assert [w.t_end for w in after] == [22, 24]
        assert all(len(a.frames) == len(b.frames) for a, b in zip(after, before))
        clip_times = [round(t - 8, 3) for t in after[0].sampling["frame_times"] if t < 16]
        assert clip_times, "No real action from the skipped interval was retained"
        if behavior == "littering":
            assert any(6 <= t <= 7 for t in clip_times), "Release sequence was crowded out by walking"
            assert any(t < 5 for t in clip_times), "No preceding held-object frame retained"
        row = {"filename": filename, "target_behavior": behavior,
            "frames_before": [len(w.frames) for w in before],
            "frames_after": [len(w.frames) for w in after],
            "retained_clip_times_sec": clip_times, "sampling": after[0].sampling,
            "motion_hint_mean_ms": round(statistics.mean(costs), 3),
            "motion_hint_p95_ms": round(sorted(costs)[int(len(costs) * .95)], 3),
            "baseline_labels": before[0].labels, "new_labels": after[0].labels}
        rows.append(row)
        batches.append(after)
        persist()
        print(json.dumps(row, ensure_ascii=False), flush=True)
    report["status"] = "cpu_passed"
    persist()
    if not args.real_model:
        return 0

    from check_validation_cluster import configure_auth
    from webui.cluster_client import ClusterClient, ServiceSession, deploy_service_package
    configure_auth()
    cc, bs = cfg["cluster"], load_behaviors()
    client = ClusterClient(cc["remote_workdir"], cc["remote_python"], cc["remote_model_root"])
    if client.list_my_jobs():
        raise RuntimeError("An existing allocation is present; no additional GPU job started")
    deploy_service_package(client, ROOT, behavior_payload(bs))
    session = ServiceSession(client, cc["remote_model"], port=18191,
        exclude=cc["exclude_nodes"], extra_sbatch="--time=00:15:00")
    try:
        session.start(wait_ready_sec=240, progress=lambda message: print(message, flush=True))
        report["job_id"], report["node"] = session.jobid, session.node
        report["health"] = session.health()
        persist()
        for row, windows in zip(rows, batches):
            window = windows[0]
            encoded = [base64.b64encode(Path(p).read_bytes()).decode("ascii") for p in window.frames]
            result = infer_with_review(session.infer, encoded, window.labels, bs,
                max_new_tokens=generation_budget(bs, with_alert=True, with_actions=True, with_risks=True),
                with_alert=True, with_actions=True, with_risks=True)
            row["real_model"] = result
            persist()
            if result.get("error") or not result.get("parsed"):
                raise RuntimeError(f"Real model request failed for {row['filename']}")
            target = result["parsed"]["violations"][row["target_behavior"]]
            row["confirmed_target"] = bool(target.get("violation") and target.get("alert")
                and not target.get("evidence_issues"))
            if not row["confirmed_target"]:
                raise RuntimeError(f"Real model did not confirm target in {row['filename']}")
        report["status"] = "passed"
    except Exception as exc:
        report["status"], report["error"] = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        job_id = session.jobid
        report["released_job_ids"] = session.stop(sweep_orphans=False)
        report["allocation_after_release"] = client.job_state(job_id)[0] if job_id else "not allocated"
        persist()
        print("Released validation allocation:", report["released_job_ids"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""One short, isolated real-model validation; release only the allocation we own."""
from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from check_validation_cluster import configure_auth
from civility.config import behavior_payload, load_behaviors, load_settings
from webui.cluster_client import ClusterClient, ServiceSession, deploy_service_package
from webui import server as S
from fastapi.testclient import TestClient

OUTPUT = ROOT / "runs/features-20261011"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    configure_auth()
    cfg = copy.deepcopy(load_settings())
    cc = cfg["cluster"]
    client = ClusterClient(cc["remote_workdir"], cc["remote_python"], cc["remote_model_root"])
    if client.list_my_jobs():
        raise RuntimeError("An existing allocation is present; do not create a second GPU job")
    print("Uploading current visual QA and detection service source…", flush=True)
    deploy_service_package(client, ROOT, behavior_payload(load_behaviors()))
    session = ServiceSession(client, cc["remote_model"], port=18190, exclude=cc["exclude_nodes"], extra_sbatch="--time=00:15:00")
    results = []
    started = time.time()
    def persist(**extra):
        (OUTPUT / "real-model.json").write_text(json.dumps({"job_id": session.jobid, "node": session.node,
            "results": results, "elapsed_sec": round(time.time()-started, 1), **extra}, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        session.start(wait_ready_sec=240, progress=lambda message: persist(status=message))
        health = session.health()
        if health.get("video_qa_protocol") != 1:
            raise RuntimeError("Deployed worker did not load the video QA protocol")
        # Deployment can be slow. Re-import local endpoints after model startup
        # so edits made while uploading are not mistaken for a stale web process.
        import webui.video_library as library_module
        importlib.reload(library_module)
        importlib.reload(S)
        cfg["paths"]["data_dir"] = str(OUTPUT / "data")
        cfg["paths"]["frames_dir"] = str(OUTPUT / "data/frames")
        cfg["paths"]["results_dir"] = str(OUTPUT)
        S.STATE.cfg, S.STATE.session, S.STATE.jobs, S.STATE.uploads = cfg, session, {}, {}
        with TestClient(S.create_app()) as api:
            cases = [("吸烟.mp4", "哪一帧有人吸烟？请给出时间和可见证据。"),
                     ("乱丢垃圾.mp4", "丢垃圾的动作是怎样发生的？请引用前后画面及时间。"),
                     ("安全帽.mp4", "画面中有没有人未戴安全帽？在哪一帧可以看出？")]
            for index, (filename, question) in enumerate(cases):
                video = ROOT.parent / "test" / filename
                previous = next((m for m in S.video_library().list() if m["filename"] == filename and m["state"] == "done" and not m["use_mock"]), None)
                if previous:
                    job_id = previous["video_id"]
                    windows, alerts = previous["windows_seen"], previous["alerts"]
                    print(f"Reusing real analysis archive after process restart: {filename}", flush=True)
                else:
                    with video.open("rb") as source:
                        uploaded = api.post("/api/upload", files={"file": (filename, source, "video/mp4")})
                    uploaded.raise_for_status()
                    job_id = uploaded.json()["job_id"]
                    job = S.STATE.jobs[job_id]
                    job.with_actions = True
                    print(f"Real analysis: {filename}", flush=True)
                    S.run_job(job_id)
                    if job.state != "done":
                        raise RuntimeError(f"Analysis failed: {job.error}")
                    windows, alerts = job.windows_seen, job.alerts
                response = api.post(f"/api/videos/{job_id}/ask", json={"question": question})
                row = {"filename": filename, "job_id": job_id, "windows": windows,
                       "alerts": alerts, "status": response.status_code, "qa": response.json()}
                results.append(row)
                persist(status="validating")
                if response.status_code != 200:
                    raise RuntimeError(f"Real QA failed: {response.text}")
                for ref in row["qa"]["citations"]:
                    frame = api.get(ref["image_url"])
                    if frame.status_code != 200 or not frame.content.startswith(b"\xff\xd8"):
                        raise RuntimeError("A real-model citation cannot be opened as a JPEG frame")
                print(json.dumps(row["qa"], ensure_ascii=False), flush=True)
            # Archive works independently of current in-memory jobs.
            S.STATE.jobs.clear()
            if len(api.get("/api/videos").json()["videos"]) != 3:
                raise RuntimeError("Persistent video history failed")
        persist(status="passed", health={k: health.get(k) for k in ("model", "device", "video_qa_protocol")})
    except Exception as exc:
        persist(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        job_id = session.jobid
        released = session.stop(sweep_orphans=False)
        previous = json.loads((OUTPUT / "real-model.json").read_text(encoding="utf-8"))
        previous["released_job_ids"] = released
        previous["allocation_after_release"] = client.job_state(job_id)[0] if job_id else "not allocated"
        (OUTPUT / "real-model.json").write_text(json.dumps(previous, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Validation allocation released:", released, flush=True)


if __name__ == "__main__":
    main()

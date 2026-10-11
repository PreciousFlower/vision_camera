"""Back up and deploy service code without allocating a GPU or restarting local apps."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_validation_cluster import configure_auth
from civility.config import behavior_payload, load_behaviors, load_settings
from webui.cluster_client import ClusterClient, deploy_service_package

ROOT = Path(__file__).resolve().parents[1]


def main():
    configure_auth()
    cfg = load_settings()["cluster"]
    client = ClusterClient(cfg["remote_workdir"], cfg["remote_python"], cfg["remote_model_root"])
    ok, info = client.check_vpn()
    if not ok:
        raise RuntimeError(f"Cluster unavailable: {info}")
    workdir = shlex.quote(client.workdir)
    backup = f"recognition-before-{datetime.now():%Y%m%d-%H%M%S}.tar.gz"
    code, out, err = client.run(
        f"cd {workdir} && tar -czf {shlex.quote(backup)} "
        "pkg/civility pkg/cluster pkg/behaviors.json cluster/serve-vlm.sbatch", timeout=90)
    if code:
        raise RuntimeError(f"Backup failed: {err or out}")
    print(f"Saved remote source backup: {client.workdir}/{backup}", flush=True)
    deploy_service_package(client, ROOT, behavior_payload(load_behaviors()))
    expected = {f"pkg/{name}": hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in ("civility/config.py", "civility/prompts.py", "civility/recognition.py",
                             "cluster/vlm_service.py", "cluster/vision_cache.py")}
    expected["cluster/serve-vlm.sbatch"] = hashlib.sha256((ROOT / "cluster/serve-vlm.sbatch").read_bytes()).hexdigest()
    paths = " ".join(shlex.quote(name) for name in expected)
    code, out, err = client.run(f"cd {workdir} && sha256sum {paths}", timeout=60)
    if code:
        raise RuntimeError(f"Remote verification failed: {err}")
    actual = {line.split(maxsplit=1)[1].strip(): line.split()[0] for line in out.splitlines()}
    if actual != expected:
        raise RuntimeError("Uploaded service files differ from local files")
    code, out, err = client.run(
        f"cd {workdir} && {shlex.quote(client.python)} -m compileall -q pkg/civility pkg/cluster", timeout=60)
    if code:
        raise RuntimeError(f"Remote code compilation failed: {err or out}")
    result = {"workdir": client.workdir, "backup": backup, "verified_sha256": actual,
              "gpu_job_started": False, "local_server_restarted": False}
    target = ROOT / "runs/test-video-validation/deployment.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    speed_target = ROOT / "runs/latency-20261010/deployment.json"
    speed_target.parent.mkdir(parents=True, exist_ok=True)
    speed_target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Service source and behavior definitions deployed; hashes and compilation passed.", flush=True)


if __name__ == "__main__":
    main()

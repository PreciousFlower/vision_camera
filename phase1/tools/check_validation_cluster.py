"""Read-only connection and allocation check. Never prints credentials."""
from pathlib import Path
import os
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from civility.config import load_settings
from webui.cluster_client import ClusterClient


def configure_auth():
    if not os.environ.get("VLM_SSH_PASSWORD"):
        source = Path(__file__).resolve().parents[2] / "tju/11b-credentials.md"
        if source.exists():
            match = re.search(r'cluster\.connect\(password="([^"]+)"', source.read_text(encoding="utf-8"))
            if match:
                os.environ["VLM_SSH_PASSWORD"] = match.group(1)


def main():
    configure_auth()
    cfg = load_settings()["cluster"]
    client = ClusterClient(cfg["remote_workdir"], cfg["remote_python"], cfg["remote_model_root"])
    ok, info = client.check_vpn()
    print("Cluster connection:", ok, info)
    if not ok:
        return 2
    code, jobs, _ = client.run('squeue -u u00009 -h -o "%i %j %T %N"', timeout=30)
    print("Current allocations:", jobs.strip() or "none")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

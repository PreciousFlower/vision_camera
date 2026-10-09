#!/bin/bash
#SBATCH --job-name=tunnel-probe
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=16g
#SBATCH --time=0-00:20:00
#SBATCH --exclude=gpu8002,gpu8003,gpu8012
#SBATCH --output=/share/home/u00009/apps/civility-kit/logs/tunnel-%j.out
#SBATCH --error=/share/home/u00009/apps/civility-kit/logs/tunnel-%j.err

# 在计算节点上常驻一个最小 HTTP 服务，供本机经 SSH 隧道验证连通性。
# 服务用纯标准库实现 —— 集群没装 web 框架，且计算节点装包受网络/权限限制
# （实测 pip --dry-run 显示"Would install"，但正式环境不应依赖这一点）。
#
# 为什么常驻而不是跑完就退：就是要让本机有足够时间建隧道打进来。

PORT=${PORT:-18080}
BIND=${BIND:-0.0.0.0}
echo "NODE=$(hostname)"
echo "PORT=$PORT"
echo "BIND=$BIND"
echo "READY_AT=$(date +%s)"

cat > "$HOME/apps/civility-kit/cluster/_mini_http.py" <<'PYEOF'
import http.server, json, os, socket, sys, time

PORT = int(sys.argv[1])
START = time.time()
HITS = []


class H(http.server.BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        HITS.append((time.time(), self.client_address[0], self.path))
        self._send(200, {
            "ok": True,
            "hostname": socket.gethostname(),
            "path": self.path,
            "client": self.client_address[0],
            "uptime_sec": round(time.time() - START, 1),
            "hits": len(HITS),
        })

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(n) if n else b""
        HITS.append((time.time(), self.client_address[0], self.path))
        self._send(200, {
            "ok": True,
            "hostname": socket.gethostname(),
            "echo_bytes": len(raw),
            "hits": len(HITS),
        })

    def log_message(self, *a):
        pass


srv = http.server.ThreadingHTTPServer((sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1", PORT), H)
print(f"serving on {srv.server_address} host={socket.gethostname()}", flush=True)
srv.serve_forever()
PYEOF

# 20 分钟时限，服务最多活 19 分钟
timeout 1140 /share/home/u00009/apps/conda/envs/vlm/bin/python \
  "$HOME/apps/civility-kit/cluster/_mini_http.py" "$PORT" "$BIND"
echo "TUNNEL_PROBE_EXITED"

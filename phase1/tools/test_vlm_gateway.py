#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证"同卡多 worker 网关"：轮询分配、坏后端自动跳过、转发要能双向传大数据。

为什么值得单独测：
    这个网关是"用一张卡跑多份模型、把结论速度拉近翻倍"的关键部件，
    但它跑在计算节点上、出问题只会表现为"网页偶尔超时"——
    在集群上调这种东西成本很高（要占卡、要等排队）。
    所以逻辑必须在本地用假后端验透：两条打印端口号的 HTTP 服务就够了。

用法：
    python tools/test_vlm_gateway.py
"""
from __future__ import annotations

import json
import socket
import subprocess  # noqa: F401  （保留给将来起真后端用）
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    results.append((ok, name, detail))
    print(f"  {'✅' if ok else '❌'} {name}" + (f" — {detail}" if detail else ""),
          flush=True)
    return ok


def make_backend(port: int, tag: str, *, big: bool = False, ready: bool = True):
    """假后端：告诉客户端"是哪个 worker 答的"；big=True 回大包（验分片转发）。

    ready=False 模拟"HTTP 端口起来了但模型还在加载"的 worker ——
    网关**必须**跳过它（实测这个场景害得第一批请求全部失败）。
    """
    class H(BaseHTTPRequestHandler):
        def _send(self, payload: dict):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):                       # noqa: N802
            if self.path.startswith("/health"):
                # 真实服务返回的就是这个形状：{"ready": true, ...}
                return self._send({"ready": ready, "worker": tag,
                                   "state": "ready" if ready else "loading"})
            self._send({"worker": tag, "path": self.path,
                        "pad": "x" * (200000 if big else 0)})

        def do_POST(self):                      # noqa: N802
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n)
            self._send({"worker": tag, "got": len(body)})

        def log_message(self, *a):              # 静音
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def get(port: int, path: str = "/health", timeout: int = 30) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}",
                                timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    from cluster.vlm_gateway import Gateway

    print("=" * 74)
    print("同卡多 worker 网关验证（本地假后端，不占卡、不花钱）")
    print("=" * 74)

    a = make_backend(18081, "A")
    b = make_backend(18082, "B")
    gw = Gateway(18080, [("127.0.0.1", 18081), ("127.0.0.1", 18082)])
    threading.Thread(target=gw.serve, daemon=True).start()
    time.sleep(0.8)

    print("\n[1] 轮询分配：连发 6 个请求，应该 A/B 交替")
    seen = []
    for _ in range(6):
        seen.append(get(18080)["worker"])
    check(seen.count("A") == 3 and seen.count("B") == 3,
          "请求被轮流分到两个 worker", f"序列 {' '.join(seen)}")
    check(seen[0] != seen[1], "相邻两次落在不同 worker（真的轮询）")

    print("\n[2] 大响应要能完整穿过网关（分片转发正确性）")
    c = make_backend(18083, "C", big=True)
    gw2 = Gateway(18084, [("127.0.0.1", 18083)])
    threading.Thread(target=gw2.serve, daemon=True).start()
    time.sleep(0.5)
    # ⚠️ 注意取 /big 而不是默认的 /health —— /health 现在返回的是
    #    {"ready":...} 这种小 JSON（网关要靠它判断就绪），没有 pad 字段
    d = get(18084, "/big")
    check(d.get("worker") == "C" and len(d.get("pad", "")) == 200000,
          "200KB 响应完整转发（没有截断）",
          f"收到 {len(d.get('pad',''))} 字节")

    print("\n[3] POST 请求体也要能穿过（真正的推理请求是 POST + 大 body）")
    req = urllib.request.Request(
        "http://127.0.0.1:18080/infer", data=b"z" * 100000,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        body = json.loads(r.read().decode("utf-8"))
    check(body.get("got") == 100000, "100KB 请求体完整转发",
          f"后端收到 {body.get('got')} 字节，由 {body.get('worker')} 处理")

    print("\n[4] 坏后端自动跳过（不能让一个坏 worker 拖垮整体）")
    gw3 = Gateway(18085, [("127.0.0.1", 18999), ("127.0.0.1", 18081)])
    threading.Thread(target=gw3.serve, daemon=True).start()
    time.sleep(0.5)
    ok_seq = [get(18085)["worker"] for _ in range(3)]
    check(all(w == "A" for w in ok_seq),
          "第一个后端挂了，请求自动落到活着的那个",
          f"序列 {' '.join(ok_seq)}（18999 是故意不存在的端口）")

    print("\n[5] 全部后端都挂 → 返回 503 而不是挂住")
    gw4 = Gateway(18086, [("127.0.0.1", 18998)])
    threading.Thread(target=gw4.serve, daemon=True).start()
    time.sleep(0.5)
    code = None
    try:
        get(18086)
    except urllib.error.HTTPError as e:
        code = e.code
    except Exception as e:        code = f"{type(e).__name__}"
    check(code == 503, "没有可用后端时快速返回 503", f"实际 {code}")

    print("\n[6] 模型没加载完的 worker 必须被跳过（实测踩到的启动竞态）")
    # ⚠️ 用一个**没被占用的新端口**（18088）来模拟"正在加载"的 worker。
    #    第一版复用了 18082，而那个端口上早先已经有一个 ready=True 的假后端，
    #    新后端绑不上去 → 旧后端继续应答 → 测试误判成"网关没跳过"。
    make_backend(18088, "B", ready=False)
    gw5 = Gateway(18087, [("127.0.0.1", 18088), ("127.0.0.1", 18081)])
    threading.Thread(target=gw5.serve, daemon=True).start()
    time.sleep(0.5)
    seq = [get(18087)["worker"] for _ in range(4)]
    check(all(w == "A" for w in seq),
          "未就绪的 worker 被跳过，请求都落到就绪的那个",
          f"序列 {' '.join(seq)}（18088 报告 ready=false）")

    # 缓存到期后必须重新探测（不能把 worker 永久拉黑）
    gw5.health.clear()
    seq2 = [get(18087)["worker"] for _ in range(2)]
    check(all(w == "A" for w in seq2),
          "重新探测后仍未就绪 → 继续只用就绪的那台",
          f"序列 {' '.join(seq2)}（说明是每次都判定，而不是一次判死）")

    for s in (a, b, c):
        s.shutdown()

    print("\n" + "=" * 74)
    bad = [x for x in results if not x[0]]
    if bad:
        print(f"❌ {len(bad)} 项未通过")
        for _, n, d in bad:
            print(f"   · {n} — {d}")
        return 1
    print("✅ 全部通过 —— 网关可以上集群（轮询正确、能扛大包、坏后端自动跳过）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

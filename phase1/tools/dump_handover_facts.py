"""一次性清单导出：把手交文档需要的"当前事实"打成可核对的文本。

为什么单独写一个工具而不是直接念代码：
    交接文档里凡是写数字的地方都必须能被复核。这里把行为定义、关键开关、
    各条测试的存在性一次性打出来，写文档时照抄、对方接手后也能随时重跑确认。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from civility.config import load_behaviors, load_settings  # noqa: E402

cfg = load_settings()
bs = load_behaviors(cfg["paths"]["behaviors"])

print("=" * 74)
print("后端与开关")
print("=" * 74)
print(f"  backend                : {cfg.get('backend')}")
print(f"  aggregate.mode         : {(cfg.get('aggregate') or {}).get('mode')}")
print(f"  min_confidence         : {(cfg.get('aggregate') or {}).get('min_confidence')}")
print(f"  dedup_sec              : {(cfg.get('aggregate') or {}).get('dedup_sec')}")
ex = cfg.get("extract") or {}
print(f"  extract.max_side       : {ex.get('max_side')}")
print(f"  extract.fps            : {ex.get('fps')}")
print(f"  extract.max_frames/win : {ex.get('max_frames_per_window')}")
lv = cfg.get("live") or {}
for k in ("window_stride_sec", "continuous_batch_windows", "infer_workers",
          "pipeline", "motion_gate", "motion_threshold", "max_windows"):
    print(f"  live.{k:<20}: {lv.get(k)}")
sv = cfg.get("service") or {}
for k in ("idle_release_min", "max_run_hours", "gpu_price_per_hour",
          "check_orphans_on_start"):
    print(f"  service.{k:<18}: {sv.get(k)}")
cl = cfg.get("cluster") or {}
for k in ("remote_workdir", "remote_python", "remote_model_root",
          "remote_model", "partition", "service_port", "exclude_nodes"):
    print(f"  cluster.{k:<18}: {cl.get(k)}")

print()
print("=" * 74)
print(f"行为定义（共 {len(bs.behaviors)} 条）")
print("=" * 74)
print(f"  defaults: {bs.defaults}")
for b in bs.behaviors:
    flag = "启用" if b.enabled else "停用"
    print(f"  [{flag}] {b.id:<16} {b.name}   severity={b.severity}")

print()
print("=" * 74)
print("测试与工具清单（区分 tools/ 与 tests/）")
print("=" * 74)
for d in ("tools", "tests"):
    p = ROOT / d
    if not p.exists():
        continue
    names = sorted(x.name for x in p.glob("*.py")
                   if x.name[:5] in ("test_", "check") or x.name[:5] == "smoke"
                   or x.name[:5] == "diag_")
    print(f"  {d}/  （{len(names)} 个测试/诊断工具）")
    for n in names:
        print(f"      {n}")

"""编排器 —— 跑 manifest，产出逐窗口结果。

两个工程属性决定了它好不好用：

1. **断点续跑**：每条结果立刻追加写 JSONL，重跑时跳过已完成的 window_id。
   第一阶段会反复调 Prompt，一次跑几百个窗口，中途限流/断网是常态；
   没有续跑就得从头再来，成本和时间都受不了。
   （同一套思路在集群侧脚本里也用了 —— 见 tju/10-cluster-handover.md §5.4）

2. **结果文件与 Prompt 版本绑定**：换 Prompt 必须换输出文件，
   否则旧结果会被当成新结果复用。所以文件名里带 behaviors.yaml 的哈希。
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from .config import BehaviorSet
from .extract import Window
from .vlm import Backend, WindowResult


def behavior_fingerprint(bs: BehaviorSet) -> str:
    """行为定义 + 采样参数的指纹，用来给结果文件命名。

    ⚠️ 这是防"张冠李戴"的关键：改了 Prompt 但复用旧结果文件，
    会得到一份看起来合理、实际对不上的评测结论 —— 这种错误极难发现。
    """
    parts = []
    for b in bs.enabled:
        parts.append(
            "|".join(
                [
                    b.id,
                    b.name,
                    b.definition,
                    b.evidence_required,
                    ";".join(b.counter_examples),
                    ";".join(b.observable_cues),
                ]
            )
        )
    parts.append(json.dumps(bs.defaults, sort_keys=True, ensure_ascii=False))
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:10]


def load_done(out_path: Path) -> dict[str, dict]:
    """读回已完成的结果，用于断点续跑。"""
    done: dict[str, dict] = {}
    if not out_path.exists():
        return done
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # 容忍被中断写坏的最后一行
            if rec.get("window_id"):
                done[rec["window_id"]] = rec
    return done


def run_windows(
    windows: list[Window],
    backend: Backend,
    bs: BehaviorSet,
    out_path: str | Path,
    *,
    concurrency: int = 1,
    retry_failed: bool = False,
    verbose: bool = True,
) -> dict:
    """跑完全部窗口，返回统计信息。

    并发只用于 IO 密集的云端后端；集群后端是读本地结果，并发无意义。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = load_done(out_path)
    todo = [
        w
        for w in windows
        if w.window_id not in done
        or (retry_failed and done[w.window_id].get("error"))
    ]

    if verbose:
        print(f"[run] 总窗口 {len(windows)}，已完成 {len(windows) - len(todo)}，本轮待跑 {len(todo)}")

    lock = threading.Lock()
    t0 = time.time()
    finished = 0
    errors = 0

    def work(w: Window) -> WindowResult:
        return backend.infer(w.to_request(), bs)

    concurrency = max(1, int(concurrency))
    with open(out_path, "a", encoding="utf-8") as f:
        if concurrency == 1 or len(todo) <= 1:
            it = (work(w) for w in todo)
        else:
            ex = ThreadPoolExecutor(max_workers=concurrency)
            futs = [ex.submit(work, w) for w in todo]
            it = (fu.result() for fu in as_completed(futs))

        for res in it:
            rec = asdict(res)
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()  # 立即落盘 —— 续跑的前提
                finished += 1
                if res.error:
                    errors += 1
                if verbose and finished % 10 == 0:
                    el = time.time() - t0
                    rate = finished / el if el > 0 else 0
                    print(
                        f"  {finished}/{len(todo)}  {rate:.2f} win/s  "
                        f"错误 {errors}  ETA {(len(todo) - finished) / rate:.0f}s"
                        if rate > 0
                        else f"  {finished}/{len(todo)}"
                    )

    elapsed = time.time() - t0
    stats = {
        "total_windows": len(windows),
        "ran_now": finished,
        "skipped_done": len(windows) - len(todo),
        "errors": errors,
        "elapsed_sec": round(elapsed, 2),
        "windows_per_sec": round(finished / elapsed, 3) if elapsed > 0 and finished else None,
        "out_path": str(out_path),
        "backend": backend.name,
    }
    if verbose:
        print(f"[run] 完成：{stats}")
    return stats


def write_summary(stats: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return path


def load_results(path: str | Path) -> list[WindowResult]:
    """把 JSONL 读回成 WindowResult 列表（供聚合/评分使用）。"""
    path = Path(path)
    out: list[WindowResult] = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.append(WindowResult(**d))
    return out

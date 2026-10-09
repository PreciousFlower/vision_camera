#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""回归：帧序号映射（页面上"AI 看到的到底是哪一帧"）。

为什么值得专门测：
    用户反馈"ai动作识别中的图片始终是同一张，并不是人做对应动作的那一帧"。
    查下来是**三个缺陷叠加**，其中两个在服务端：
      ① 缩略图 URL 没带缓存破坏参数，而 /frames 也不返回缓存头
         → 浏览器启发式缓存，永远显示第一张；
      ② 缩略图固定取窗口第 0 帧（= 窗口**开头**那一帧），动作可能在后面几帧；
      ③ 帧号是拿 `int(t_start) + idx` **推算**的 —— 只在
         "窗口步长 == 帧间隔、且一帧没丢"时才成立。步长 2 秒、帧间隔 2 秒时
         它碰巧对；一旦改步长就指到别的帧上，而页面看起来完全正常。

    ①②没法在纯 Python 里测（要靠 HTTP 头），③可以 —— 而且它最危险，
    因为错了不会报错，只会静默显示错误的画面。

测什么：
    A. frame_index_of 能正确解析 f_00023.jpg / f_00023_ab12cd.jpg，非法名返回 -1
    B. build_windows 给每个窗口填的 idx0 与真实帧文件一致
       （含"改步长"这个最容易踩的场景）
    C. Window 能经 manifest 序列化/反序列化后保留 idx0
       （否则集群往返一趟回来就丢了，页面又退回旧算法）
    D. 旧 manifest（没有 idx0 字段）仍能读入，idx0 落到 -1（向后兼容）
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

FAILED: list[str] = []


def check(cond: bool, label: str, extra: str = "") -> None:
    print(f"  {'✅' if cond else '❌'} {label}" + (f"　{extra}" if extra else ""))
    if not cond:
        FAILED.append(label)


def main() -> int:
    from civility.extract import (Window, build_windows, frame_index_of,
                                  read_manifest, write_manifest)

    print("\n[A] frame_index_of：文件名 → 全局帧序号")
    cases = [
        ("f_00023.jpg", 23),
        ("f_00000.jpg", 0),
        (r"C:\x\job\f_00109.jpg", 109),
        ("/tmp/f_12345.jpg", 12345),
        # 非法/无关的名字必须返回 -1，而不是猜。
        # ⚠️ 这里曾经写过一个"带指纹的变体 f_00023_ab12cd.jpg"并期望解析出 23 ——
        #    但那种命名在本项目里**并不存在**（当初考虑过给帧名加指纹以便启用
        #    浏览器缓存，最后没做）。预期一个不存在的格式，等于给自己埋假测试。
        #    现在按"格式收紧"的立场：认不出就返回 -1，让调用方回退。
        ("f_00023_ab12cd.jpg", -1),
        ("preview_0001234567890.jpg", -1),
        ("f_abc.jpg", -1),
        ("f_00023.png", -1),
        ("00023.jpg", -1),
        ("f_00023_grid.jpg", -1),
    ]
    for name, want in cases:
        got = frame_index_of(name)
        check(got == want, f"{name} → {got}", f"(期望 {want})")

    print("\n[B] build_windows 填的 idx0 必须与真实帧文件对得上")
    # 造 12 张帧。★ 间隔要说明白：这里 frames 的间隔代表**固定 fps 抽帧**的结果
    #（视频文件路径），所以窗口步长必须 >= (帧数-1)×间隔 才能凑够帧。
    # 实时流那条路径不受这个约束：capture.py 把取样间隔算成 stride/(帧数-1)，
    # 末帧正好落在窗口末尾，天生对齐。
    frames = [(i * 2.0, f"data/frames/job1/f_{i:05d}.jpg") for i in range(12)]
    for stride, n_frames in ((4.0, 4), (6.0, 4), (6.0, 3)):
        ws = build_windows("job1", frames, window_stride_sec=stride,
                           max_frames_per_window=n_frames)
        ok = bool(ws)
        detail = []
        for w in ws:
            # 关键判据：idx0 必须等于该窗口第一帧文件名的数字
            want = frame_index_of(w.frames[0])
            if w.idx0 != want:
                ok = False
                detail.append(f"{w.window_id}: idx0={w.idx0} 实际={want}")
        check(ok, f"步长 {stride}s / {n_frames} 帧：{len(ws)} 个窗口的 idx0 全对",
              "；".join(detail))
        check(bool(ws) and all(w.idx0 >= 0 for w in ws),
              f"步长 {stride}s：idx0 都已填（不是 -1）")

    print("\n[B1] 帧不够时的行为（步长太小 → 一个窗口都组不出，且必须是 0 而非错值）")
    ws_small = build_windows("job1", frames, window_stride_sec=2.0,
                             max_frames_per_window=4)
    check(len(ws_small) == 0,
          f"步长 2s / 帧间隔 2s：产出 {len(ws_small)} 个窗口（期望 0）",
          "每桶只有 1 帧，<2 帧的窗口按设计被跳过")

    print("\n[B2] 老算法会算错的那个场景（改步长后 int(t_start) 指到别的帧）")
    # 步长 6 秒、帧间隔 2 秒：老算法 i = int(t_start) 会指向 t_start/2 张之后，
    # 也就是**超出本窗口 3 倍的偏移** —— 这正是"图片与动作对不上"的成因。
    ws6 = build_windows("job1", frames, window_stride_sec=6.0,
                        max_frames_per_window=3)
    if len(ws6) >= 2:
        w = ws6[1]
        old_algo = int(w.t_start) + 0
        real = w.idx0
        check(old_algo != real,
              f"老算法确实会指错：int(t_start)={old_algo} vs 真实={real}",
              f"（窗口 {w.window_id}，差 {abs(old_algo - real)} 帧）")
    else:
        check(False, "步长 6s 下窗口数不足，无法验证老算法的错误")

    print("\n[C] manifest 往返：idx0 不能丢")
    # ⚠️ 临时目录必须放在 **workspace 内**：写到 %TEMP% 会被沙箱
    #    （workspace-write）拒绝，报 PermissionError —— 那是策略拦截，不是 bug。
    tmp = PROJECT / "runs" / "_tmp_frame_index_test"
    tmp.mkdir(parents=True, exist_ok=True)
    p = tmp / "manifest.jsonl"
    write_manifest(ws6, p)
    back = read_manifest(p)
    check(len(back) == len(ws6), f"往返窗口数一致（{len(back)}）")
    check(all(a.idx0 == b.idx0 for a, b in zip(ws6, back)),
          "往返后 idx0 完全一致")

    print("\n[D] 向后兼容：旧 manifest 没有 idx0 字段")
    old = tmp / "old.jsonl"
    old.write_text(json.dumps({
        "window_id": "v__w0000", "video": "v", "t_start": 0.0, "t_end": 4.0,
        "frames": ["data/frames/job1/f_00000.jpg"],
        "labels": ["t=0.0s"],
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        ob = read_manifest(old)
        check(len(ob) == 1 and ob[0].idx0 == -1,
              "旧记录能读入且 idx0 = -1（让调用方回退旧算法，不崩）")
    except Exception as e:                                  # noqa: BLE001
        check(False, f"旧 manifest 读取失败：{type(e).__name__}: {e}")

    print("\n[E] Window 直接构造（不传 idx0）也不能崩")
    try:
        w = Window(window_id="x", video="v", t_start=0.0, t_end=2.0,
                   frames=["a.jpg"], labels=["t=0"])
        check(w.idx0 == -1, "默认 idx0 = -1")
    except Exception as e:                                  # noqa: BLE001
        check(False, f"构造失败：{type(e).__name__}: {e}")

    # 收尾：清掉本次用的临时目录。
    # 踩过的坑：第一版没清，于是在 runs/ 下留了一个 _tmp_frame_index_test/。
    # 后果不只是"多一个目录" —— runs/ 要进版本库，残留的测试垃圾会被一起提交，
    # 而且以后每次跑测试都会再产生一份。
    shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 62)
    if FAILED:
        print(f"❌ {len(FAILED)} 项失败：")
        for f in FAILED:
            print(f"   · {f}")
        return 1
    print("✅ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

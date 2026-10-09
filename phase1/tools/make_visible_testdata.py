#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成"有可见内容"的验证视频 —— 用 PIL 逐帧绘制，可控且可自检。

为什么换掉 ffmpeg 方案（两次踩坑）：
    第一次：背景 0x333333，模型答"画面全黑"，其实答对了 —— 视频本身没内容。
    第二次：用 drawbox 的 `x='80+45*t'` 表达运动，但 ffmpeg 的表达式
            没有按预期逐帧求值，橙色人形根本没出现在画面里，
            生成的还是静态图 —— 又一次"以为在测模型，其实在测我的脚本"。
    PIL 逐帧绘制没有表达式解析的不确定性，而且能在生成后**断言帧确实不同**，
    从根上避免"测试数据本身是坏的"这类最浪费时间的错误。

四类测试，每类只验证一件事：
    1. content_check  —— 中文大字。能不能"看见"（同时是 OCR 测试）。
    2. frame_order    —— 圆点随时间横向移动。能不能区分帧的先后。
                        **这是时序判断的前提，也是本系统成败的关键。**
    3. litter_visual  —— 人形走向垃圾桶旁的途中把物体丢到地上。
                        看行为识别通路是否成立。
    4. clean_visual   —— 干净场景。测误报。

用法：
    python tools/make_visible_testdata.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
OUT = PROJECT / "data" / "videos"
FONTDIR = PROJECT / "assets" / "fonts"

W, H, FPS, DUR = 640, 360, 10, 14
SRC_FONTS = [
    Path("C:/Windows/Fonts/simhei.ttf"),
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simsun.ttc"),
]


def font(size: int) -> ImageFont.FreeTypeFont:
    FONTDIR.mkdir(parents=True, exist_ok=True)
    dst = FONTDIR / "cjk.ttf"
    if not dst.exists():
        for src in SRC_FONTS:
            if src.exists():
                shutil.copy2(src, dst)
                break
        else:
            raise RuntimeError("找不到中文字体")
    return ImageFont.truetype(str(dst), size)


def center_text(d: ImageDraw.ImageDraw, y: int, text: str, f, fill) -> None:
    w = d.textlength(text, font=f)
    d.text(((W - w) / 2, y), text, font=f, fill=fill)


# ─────────────────────────── 四类帧生成器 ───────────────────────────


def frames_content_check(t: float) -> Image.Image:
    """蓝底 + 中文大字 + 英文。测"看得见"与 OCR。"""
    im = Image.new("RGB", (W, H), (47, 111, 191))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 250, W, H], fill=(185, 185, 185))
    center_text(d, 60, "停车场 3 号门", font(56), (255, 255, 255))
    center_text(d, 150, "SAFE ZONE", font(36), (0, 229, 255))
    center_text(d, 285, "禁止吸烟", font(40), (255, 221, 51))
    return im


def frames_motion_visual(t: float) -> Image.Image:
    """人形走过垃圾桶，**什么都没丢**。测两件事：
        ① 能否看出人在画面中移动（时序理解的实用形式）
        ② 人靠近垃圾桶但没丢东西时，会不会误报 littering

    为什么要专门做这个用例：
        最初设计的是"绿色圆点在移动"，但模型答"无人员活动"是**忠实于指令**的
        —— scene 字段被定义为"谁在哪做什么（关于人）"，画面里没人时它当然不描述圆点。
        那是我把测试问错了，不是模型看不见。
        换成"人在动"，既测了时序理解，又测了最容易发生的误报场景
        （人出现在垃圾桶旁边），一举两得。
    """
    im = Image.new("RGB", (W, H), (58, 58, 58))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 290, W, H], fill=(90, 90, 90))
    d.rectangle([470, 200, 530, 292], fill=(60, 90, 60))          # 垃圾桶
    d.text((472, 268), "桶", font=font(20), fill=(200, 230, 200))

    px = 90 + (t / DUR) * 400
    d.ellipse([px - 14, 130, px + 14, 158], fill=(240, 160, 48))
    d.rectangle([px - 17, 158, px + 17, 250], fill=(240, 160, 48))
    d.rectangle([px - 17, 250, px - 4, 292], fill=(200, 130, 30))
    d.rectangle([px + 4, 250, px + 17, 292], fill=(200, 130, 30))

    center_text(d, 20, f"t={t:.0f}s", font(26), (170, 170, 170))
    return im


def frames_litter_visual(t: float) -> Image.Image:
    """人形从右往左走，第 5 秒把物体丢到地上，然后继续走开。

    刻意做成"人走开、物体留在地上"——这正是 littering 定义里
    要求的动作链条（丢弃 + 离开 + 不再拾取）。
    """
    im = Image.new("RGB", (W, H), (58, 58, 58))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 290, W, H], fill=(90, 90, 90))       # 地面
    d.rectangle([430, 200, 490, 292], fill=(60, 90, 60))  # 垃圾桶
    d.text((432, 268), "桶", font=font(20), fill=(200, 230, 200))

    # 人形：从 x=120 走到 x=520，线性
    px = 120 + (t / DUR) * 400
    d.ellipse([px - 14, 130, px + 14, 158], fill=(240, 160, 48))      # 头
    d.rectangle([px - 17, 158, px + 17, 250], fill=(240, 160, 48))    # 身
    d.rectangle([px - 17, 250, px - 4, 292], fill=(200, 130, 30))     # 腿
    d.rectangle([px + 4, 250, px + 17, 292], fill=(200, 130, 30))

    # 手中的物体：第 5 秒前跟着手，之后落在地上固定位置
    if t < 5.0:
        d.rectangle([px + 20, 196, px + 36, 212], fill=(255, 255, 255))
    else:
        d.rectangle([300, 276, 316, 292], fill=(255, 255, 255))

    center_text(d, 20, f"t={t:.0f}s", font(26), (170, 170, 170))
    return im


def frames_clean_visual(t: float) -> Image.Image:
    """干净场景：只有地面、墙、和一条静止的标线。无人物、无动作。"""
    im = Image.new("RGB", (W, H), (48, 48, 48))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 285, W, H], fill=(85, 85, 85))
    d.rectangle([0, 0, W, 4], fill=(70, 70, 70))
    d.line([100, 320, 540, 320], fill=(120, 120, 120), width=4)
    center_text(d, 150, "无人区域", font(52), (153, 153, 153))
    return im


# (文件名, 帧生成器, 说明, 是否应当有画面变化)
# 第 4 个字段很关键：content_check 与 clean_visual **本来就该是静态的**
# （固定文字牌 / 空场景）。没有这个区分，"全静态"告警会一直误报，
# 人就会开始忽略它 —— 那这条自检就白加了。
SPECS = [
    ("content_check_vis.mp4", frames_content_check,
     "蓝底 + 中文大字「停车场 3 号门」「禁止吸烟」+ 英文（测可见性与 OCR）",
     False),
    ("motion_visual_vis.mp4", frames_motion_visual,
     "人形走过垃圾桶但不丢东西（测人物位移感知 + 误报抵抗）",
     True),
    ("litter_visual_vis.mp4", frames_litter_visual,
     "人形走向垃圾桶，第 5 秒把白色物体丢到地上后继续走开",
     True),
    ("clean_visual_vis.mp4", frames_clean_visual,
     "干净场景，无人物无动作（测误报）",
     False),
]


def render(name: str, gen, tmpdir: Path) -> Path:
    """逐帧画图 → ffmpeg 合成 mp4，并断言帧确实在变化。"""
    tmpdir.mkdir(parents=True, exist_ok=True)
    for f in tmpdir.glob("*.png"):
        f.unlink()

    n = int(DUR * FPS)
    sigs = []
    for i in range(n):
        t = i / FPS
        im = gen(t)
        im.save(tmpdir / f"{i:04d}.png")
        # 用缩略图灰度做"这一帧是否与上一帧不同"的签名
        sigs.append(list(im.convert("L").resize((32, 18)).getdata()))

    changed = sum(1 for i in range(1, n) if sigs[i] != sigs[i - 1])
    dst = OUT / name
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-framerate", str(FPS), "-i", str(tmpdir / "%04d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
         str(dst)],
        check=True, capture_output=True, encoding="utf-8", errors="replace",
    )
    return dst, changed, n


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = PROJECT / "runs" / "_frames_tmp"

    print("=" * 74)
    print("生成验证视频（PIL 逐帧绘制，生成后自检帧是否真的在变化）")
    print("=" * 74)

    ok = 0
    for name, gen, desc, should_move in SPECS:
        try:
            dst, changed, n = render(name, gen, tmp)
        except subprocess.CalledProcessError as e:
            print(f"  ❌ {name}: ffmpeg 失败 {(e.stderr or '')[:200]}")
            continue
        except Exception as e:
            print(f"  ❌ {name}: {type(e).__name__}: {e}")
            continue

        static = changed == 0
        # 断言：该动的必须动，该静的必须静
        good = (not should_move and static) or (should_move and not static)
        detail = ("全静态（预期）" if static and not should_move
                  else f"{changed}/{n - 1} 帧有变化")
        if not good:
            detail += "  ⚠️ 与预期不符"
        size = dst.stat().st_size / 1024
        print(f"  {'✅' if good else '⚠️'} {name:26s} {size:6.1f} KB  {detail}")
        print(f"      {desc}")
        if good:
            ok += 1

    print()
    print(f"生成 {len(SPECS)} 个，其中 {ok} 个通过帧变化自检 → {OUT}")
    print()
    print("对照期望：")
    print("  content_check → 应能说出「停车场 3 号门」「禁止吸烟」")
    print("  frame_order   → 应能说出绿点在移动/位置在变（时序能力）")
    print("  litter_visual → 应命中 littering，依据提到丢弃/物体落地")
    print("  clean_visual  → 应为全 false（任何命中都是误报）")
    print()
    print("跑验证：python -m civility.cli extract && python tools/verify_visible.py")
    return 0 if ok == len(SPECS) else 1


if __name__ == "__main__":
    sys.exit(main())

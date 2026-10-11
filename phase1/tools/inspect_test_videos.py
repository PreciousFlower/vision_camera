"""Create timestamped contact sheets and a sampling report for the user's clips."""
from pathlib import Path
import json
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from civility.extract import extract_frames, probe_duration, build_windows


def main():
    project = Path(__file__).resolve().parents[2]
    output = project / "phase1/runs/test-video-validation"
    output.mkdir(parents=True, exist_ok=True)
    report = []
    for index, video in enumerate(sorted((project / "test").glob("*.mp4"))):
        folder = output / f"clip-{index}"
        frames = extract_frames(video, folder, fps=2, max_side=960)
        thumbs = []
        for t, path in frames:
            picture = Image.open(path).convert("RGB")
            picture.thumbnail((320, 180))
            tile = Image.new("RGB", (320, 208), "#111111")
            tile.paste(picture, ((320 - picture.width) // 2, 0))
            ImageDraw.Draw(tile).text((8, 187), f"{t:.1f}s", fill="white")
            thumbs.append(tile)
        sheet = Image.new("RGB", (320 * 4, 208 * ((len(thumbs) + 3) // 4)), "#111111")
        for i, tile in enumerate(thumbs):
            sheet.paste(tile, ((i % 4) * 320, (i // 4) * 208))
        sheet.save(folder / "contact.jpg")
        old_frames = frames[::2]
        old = build_windows(video, old_frames, window_stride_sec=4, max_frames_per_window=4)
        item = {"name": video.name, "duration": probe_duration(video),
                "folder": str(folder), "old_windows": [w.labels for w in old]}
        report.append(item)
        print(json.dumps(item, ensure_ascii=False))
    (output / "sampling-before.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Persistent file-video analyses and bounded retrieval of verifiable frames."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

from civility.extract import find_ffmpeg, sample_evenly

_ID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class RevisionConflict(ValueError):
    pass


class VideoLibrary:
    def __init__(self, root: Path, frames_root: Path, uploads_root: Path):
        self.root, self.frames_root, self.uploads_root = root.resolve(), frames_root.resolve(), uploads_root.resolve()
        self.lock = threading.RLock()

    def folder(self, video_id: str) -> Path:
        if not _ID.fullmatch(video_id):
            raise ValueError("视频编号不合法")
        return self.root / video_id

    def begin(self, job) -> None:
        with self.lock:
            job._analysis_revision = uuid.uuid4().hex
            folder = self.folder(job.job_id)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "windows.jsonl").write_text("", encoding="utf-8")
            self.save(job)

    def record(self, job, rec: dict) -> None:
        with self.lock:
            with (self.folder(job.job_id) / "windows.jsonl").open("a", encoding="utf-8") as out:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def save(self, job) -> None:
        with self.lock:
            folder = self.folder(job.job_id)
            folder.mkdir(parents=True, exist_ok=True)
            source_spec = getattr(job, "_archive_source", None) or job.source_spec
            source = Path(source_spec).resolve() if source_spec else None
            # Only uploaded file videos are exposed through this library.
            if source is not None and not source.is_relative_to(self.uploads_root):
                source = None
            metadata = {**job.summary(), "video_id": job.job_id, "updated_at": time.time(),
                        "source_path": str(source) if source else None,
                        "analysis_revision": getattr(job, "_analysis_revision", ""),
                        "archived_windows": getattr(job, "_archived_windows", job.windows_seen),
                        "archive_error": getattr(job, "_archive_error", "")}
            temp = folder / f"metadata-{uuid.uuid4().hex}.tmp"
            try:
                temp.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
                os.replace(temp, folder / "metadata.json")
            finally:
                temp.unlink(missing_ok=True)

    def load(self, video_id: str) -> dict:
        with self.lock:
            return json.loads((self.folder(video_id) / "metadata.json").read_text(encoding="utf-8"))

    def windows(self, video_id: str) -> list[dict]:
        with self.lock:
            rows = []
            with (self.folder(video_id) / "windows.jsonl").open(encoding="utf-8") as source:
                for line in source:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue  # A partial final write must not hide completed windows.
                    if isinstance(row, dict):
                        rows.append(row)
            return rows

    def snapshot(self, video_id: str, revision: str = "") -> tuple[dict, list[dict]]:
        with self.lock:
            metadata = self.load(video_id)
            rows = self.windows(video_id)
            current = self.load(video_id).get("analysis_revision")
            if current != metadata.get("analysis_revision") or (revision and current != revision):
                raise RevisionConflict("视频已重新分析，请刷新证据")
            return metadata, rows

    def list(self) -> list[dict]:
        items = []
        for path in self.root.glob("*/metadata.json"):
            try:
                item = self.public(self.load(path.parent.name))
            except (OSError, ValueError, TypeError):
                continue
            if item.get("state") in ("done", "stopped", "error") and item.get("windows_seen", 0):
                items.append(item)
        return sorted(items, key=lambda item: -item.get("updated_at", 0))

    def source(self, metadata: dict) -> Path | None:
        raw = metadata.get("source_path")
        path = Path(raw).resolve() if raw else None
        return path if path and path.is_relative_to(self.uploads_root) and path.is_file() else None

    def public(self, metadata: dict) -> dict:
        return {k: v for k, v in metadata.items() if k not in ("source_path", "source_spec")} | {
            "source_available": self.source(metadata) is not None}

    def delete(self, video_id: str) -> dict:
        """Remove one completed archive and its owned local files, under one lock."""
        with self.lock:
            metadata = self.load(video_id)
            if metadata.get("state") not in ("done", "stopped", "error"):
                raise ValueError("视频仍在分析，不能删除")
            folder = self.folder(video_id)
            frames = self.frames_root / video_id
            # Preflight all targets before removing anything, including junctions.
            for target, root in ((folder, self.root), (frames, self.frames_root)):
                resolved = target.resolve()
                if resolved == root or not resolved.is_relative_to(root):
                    raise ValueError("视频文件路径不在对应的保存目录中，拒绝删除")
                if target.is_symlink() or (hasattr(target, "is_junction") and target.is_junction()):
                    raise ValueError("视频目录为链接，拒绝删除")
            raw = metadata.get("source_path")
            source = Path(raw) if raw else None
            owned_source = bool(source and source.stem == video_id
                                and source.resolve().is_relative_to(self.uploads_root)
                                and not source.is_symlink())
            # A migrated archive may share a source. It never owns that file.
            if owned_source:
                for path in self.root.glob("*/metadata.json"):
                    if path.parent == folder:
                        continue
                    try:
                        other = self.load(path.parent.name).get("source_path")
                        if other and Path(other).resolve() == source.resolve():
                            owned_source = False
                            break
                    except (OSError, ValueError, TypeError):
                        continue
            # Keep metadata until all media removal succeeds, so a failed delete
            # remains visible and can be retried (e.g. a Windows file is in use).
            if frames.exists():
                shutil.rmtree(frames)
            source_removed = False
            if owned_source and source.exists():
                source.unlink()
                source_removed = True
            shutil.rmtree(folder)
            return {"video_id": video_id, "source_removed": source_removed,
                    "source_retained": bool(source and source.exists())}

    def frame(self, video_id: str, frame_id: str) -> tuple[Path, dict]:
        match = re.fullmatch(r"f_(\d{1,10})", frame_id)
        if not match:
            raise ValueError("帧编号不合法")
        index = int(match[1])
        with self.lock:
            metadata, rows = self.snapshot(video_id)
            found = next((r for r in rows if index in (r.get("frame_indices") or [])), None)
            if found is None:
                raise FileNotFoundError("该帧没有保存在分析记录中")
            return self._frame_from_row(video_id, found, index, metadata)

    def frame_bytes(self, video_id: str, frame_id: str, revision: str = "") -> bytes:
        # Read before releasing the deletion/reanalysis lock. FileResponse reads
        # later, after the route returns, and can otherwise serve another run.
        with self.lock:
            metadata = self.load(video_id)
            if revision and revision != metadata.get("analysis_revision"):
                raise RevisionConflict("视频已重新分析，请刷新证据")
            path, _ = self.frame(video_id, frame_id)
            data = path.read_bytes()
            if self.load(video_id).get("analysis_revision") != metadata.get("analysis_revision"):
                raise RevisionConflict("视频已重新分析，请刷新证据")
            return data

    def _frame_from_row(self, video_id: str, row: dict, index: int, metadata: dict | None = None) -> tuple[Path, dict]:
        with self.lock:
            # A caller may hold an old row while deletion or reanalysis wins.
            # Check membership before reading a retained shared source or mkdir.
            if metadata is None:
                metadata, rows = self.snapshot(video_id)
                if row not in rows:
                    raise RevisionConflict("证据窗口已变更，请刷新证据")
            elif self.load(video_id).get("analysis_revision") != metadata.get("analysis_revision"):
                raise RevisionConflict("证据窗口已变更，请刷新证据")
            return self._restore_frame(video_id, row, index, metadata)

    def _restore_frame(self, video_id: str, row: dict, index: int, metadata: dict) -> tuple[Path, dict]:
        info = self.frame_info(row, index)
        candidate = self.frames_root / video_id / f"f_{index:05d}.jpg"
        if candidate.is_file() and candidate.resolve().is_relative_to(self.frames_root):
            return candidate, info
        restored = self.frames_root / video_id / "qa_recovered" / f"f_{index:05d}.jpg"
        if restored.is_file() and restored.resolve().is_relative_to(self.frames_root):
            return restored, info | {"recovered": True}
        source = self.source(metadata)
        if source is None:
            raise FileNotFoundError("旧帧与原视频已清理，无法恢复这张证据画面")
        with self.lock:
            # Recheck after source lookup as well; callbacks / another console
            # may have removed the archive. Never recreate a deleted cache.
            current = self.load(video_id)
            if (current.get("analysis_revision") != metadata.get("analysis_revision")
                    or current.get("source_path") != metadata.get("source_path")):
                raise RevisionConflict("证据窗口已变更，请刷新证据")
            if not restored.exists():
                restored.parent.mkdir(parents=True, exist_ok=True)
                temp = restored.with_name(f"{restored.stem}-{uuid.uuid4().hex}.jpg")
                ffmpeg, _ = find_ffmpeg()
                try:
                    # Recover only this archived timestamp, never re-run detection.
                    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", str(info["time_sec"]),
                                    "-i", str(source), "-frames:v", "1", "-vf", "scale='min(960,iw)':-2",
                                    "-q:v", "3", "-y", str(temp)], check=True, capture_output=True, timeout=30)
                    if not temp.is_file():
                        raise FileNotFoundError("原视频无法恢复这个时间点")
                    os.replace(temp, restored)
                finally:
                    temp.unlink(missing_ok=True)
            return restored, info | {"recovered": True}

    @staticmethod
    def frame_info(row: dict, index: int) -> dict:
        indices = row.get("frame_indices") or []
        position = indices.index(index)
        labels = row.get("frame_labels") or []
        label = labels[position] if position < len(labels) else ""
        match = re.fullmatch(r"t=(\d+(?:\.\d+)?)s", label)
        if not match:
            raise ValueError("该分析缺少可靠的帧时间，不能猜测证据位置")
        times = row.get("frame_times") or []
        timestamp = times[position] if position < len(times) else float(match[1])
        if not isinstance(timestamp, (int, float)) or not 0 <= timestamp < 1e8:
            raise ValueError("帧时间无效")
        return {"frame_id": f"f_{index}", "sample_index": index, "sample_number": index + 1,
                "time_sec": timestamp, "window_id": row["window_id"], "recovered": False}

    def retrieve(self, video_id: str, question: str, limit: int = 18) -> tuple[list[dict], str, list[str]]:
        metadata, rows = self.snapshot(video_id)
        if not rows:
            raise ValueError("视频没有可用于问答的分析窗口")
        query = question.lower()
        aliases = {"smoking": ["吸烟", "抽烟", "烟", "smoking"],
                   "littering": ["垃圾", "丢弃", "丢东西", "扔东西", "litter"],
                   "no_helmet": ["安全帽", "头盔", "helmet"]}
        terms = set(re.findall(r"[a-z_]+|[\u4e00-\u9fff]{2,}", query))
        for behavior, words in aliases.items():
            if any(word in query for word in words):
                terms.update([behavior, *words])
        time_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:秒|s\b)", query)
        requested_time = float(time_match[1]) if time_match else None
        frame_match = re.search(r"第\s*(\d+)\s*帧", query)
        requested_index = int(frame_match[1]) - 1 if frame_match else None

        def score(row):
            text = json.dumps({k: row.get(k) for k in ("scene", "people", "violations", "risks")}, ensure_ascii=False).lower()
            value = sum(1 for term in terms if term in text)
            for bid, violation in (row.get("violations") or {}).items():
                if bid in terms and (violation.get("violation") or violation.get("hit")):
                    value += 10
            if requested_time is not None:
                value += 100 if row["t_start"] <= requested_time <= row["t_end"] else 0
            if requested_index is not None and requested_index in (row.get("frame_indices") or []):
                value += 100
            return value

        ranked = sorted(enumerate(rows), key=lambda pair: (-score(pair[1]), pair[0]))
        positive = [row for _, row in ranked if score(row) > 0][:3]
        selected = positive or sample_evenly(rows, min(6, len(rows)))
        seen, evidence, missing, essential = set(), [], [], set()
        per_window = max(2, limit // len(selected))
        for row in selected:
            indices = row.get("frame_indices") or []
            candidates = list(sample_evenly(indices, per_window))
            if indices:
                essential.update([indices[0], indices[-1]])
            if requested_index is not None and requested_index in indices:
                candidates.append(requested_index)
                essential.add(requested_index)
            # Keep explicitly cited action frames as well as temporal endpoints.
            for v in (row.get("violations") or {}).values():
                refs = v.get("evidence") or {}
                for n in (refs.values() if isinstance(refs, dict) else []):
                    if isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= len(indices):
                        candidates.append(indices[n - 1])
                        essential.add(indices[n - 1])
            for index in candidates:
                if index in seen:
                    continue
                seen.add(index)
                try:
                    path, info = self._frame_from_row(video_id, row, index, metadata)
                    evidence.append(info | {"path": str(path)})
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    missing.append(str(exc))
        evidence.sort(key=lambda frame: frame["time_sec"])
        if len(evidence) > limit:
            priority = [e for e in evidence if e["sample_index"] in essential]
            others = [e for e in evidence if e["sample_index"] not in essential]
            if len(priority) >= limit:
                evidence = sample_evenly(priority, limit)
            else:
                budget = limit - len(priority)
                evidence = priority + (others[:1] if budget == 1 else sample_evenly(others, budget))
            evidence.sort(key=lambda frame: frame["time_sec"])
        if not evidence:
            raise FileNotFoundError("所有候选证据帧均不可用；请重新上传并分析视频")
        context_rows = [{k: r.get(k) for k in ("window_id", "t_start", "t_end", "scene", "violations", "risks", "error")}
                        for r in selected]
        context = json.dumps({"total_windows": len(rows), "selected_windows": context_rows,
                              "selection": "按问题检索；未匹配时抽取全视频概览，未覆盖所有帧"}, ensure_ascii=False)
        return evidence, context[:24000], sorted(set(missing))

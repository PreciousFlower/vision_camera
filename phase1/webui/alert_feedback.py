"""Human review records, separate from model output and aggregation policy."""
import hashlib
import json
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from civility.extract import sample_evenly
from .video_library import VideoLibrary


class FeedbackStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.lock = threading.RLock()

    def path(self, job_id: str) -> Path:
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', job_id):
            raise ValueError('任务编号无效')
        return self.root / (job_id + '.json')

    def list(self, job_id: str) -> list[dict]:
        with self.lock:
            path = self.path(job_id)
            return json.loads(path.read_text(encoding='utf-8')) if path.exists() else []

    def save(self, job_id: str, revision: str, row: dict, body: dict, behaviors: dict, *, use_mock=False, is_live=False) -> dict:
        label, note, behavior = body.get('label'), body.get('note', ''), body.get('behavior_id')
        if label not in ('correct', 'false_positive', 'missed') or not isinstance(note, str) or len(note) > 1000:
            raise ValueError('请选择正确、误报或漏报；备注最多 1000 字')
        if not isinstance(behavior, str) or behavior not in behaviors:
            raise ValueError('反馈行为不存在')
        violation = (row.get('violations') or {}).get(behavior)
        risks = [r for r in row.get('risks') or [] if 'ai_risk:' + r.get('risk_id', '') == behavior]
        model_result = violation if violation is not None else (risks[0] if risks else None)
        if label in ('correct', 'false_positive') and model_result is None:
            raise ValueError('该窗口没有对应的模型判断；可提交漏报反馈')
        frame_id = body.get('frame_id')
        frame = None
        if frame_id:
            if not isinstance(frame_id, str) or not re.fullmatch(r'f_\d{1,10}', frame_id):
                raise ValueError('反馈帧编号无效')
            index = int(frame_id[2:])
            if index not in (row.get('frame_indices') or []):
                raise ValueError('反馈帧不属于该窗口')
            frame = VideoLibrary.frame_info(row, index)
        key = hashlib.sha256(json.dumps([revision, row['window_id'], behavior]).encode()).hexdigest()[:24]
        with self.lock:
            entries = self.list(job_id)
            previous = next((item for item in entries if item['feedback_id'] == key), None)
            entry = {'feedback_id': key, 'job_id': job_id, 'analysis_revision': revision,
                     'window_id': row['window_id'], 'behavior_id': behavior, 'behavior_name': behaviors[behavior],
                     'label': label, 'note': note.strip(), 'frame': frame,
                     't_start': row.get('t_start'), 't_end': row.get('t_end'),
                     'model_result': model_result, 'scene': row.get('scene'),
                     'use_mock': use_mock, 'is_live': is_live,
                     'created_at': previous['created_at'] if previous else time.time(), 'updated_at': time.time()}
            entries = [item for item in entries if item['feedback_id'] != key]
            if len(entries) >= 10000:
                raise ValueError('该任务反馈记录已达上限，请先导出')
            entries.append(entry)
            path = self.path(job_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_symlink() or not path.resolve().is_relative_to(self.root):
                raise ValueError('反馈文件路径无效')
            temp = path.with_name(f'{job_id}-{uuid.uuid4().hex}.tmp')
            try:
                temp.write_text(json.dumps(entries, ensure_ascii=False, allow_nan=False), encoding='utf-8')
                os.replace(temp, path)
            finally:
                temp.unlink(missing_ok=True)
            return entry

    def delete(self, job_id: str):
        with self.lock:
            self.path(job_id).unlink(missing_ok=True)

    @contextmanager
    def deleting(self, job_id: str):
        """Hide feedback atomically; restore it if media deletion fails."""
        with self.lock:
            path = self.path(job_id)
            staged = path.with_suffix('.delete-' + uuid.uuid4().hex + '.tmp')
            if path.is_symlink() or not path.resolve().is_relative_to(self.root):
                raise ValueError('反馈文件路径无效')
            moved = path.exists()
            if moved:
                os.replace(path, staged)
            try:
                yield
            except BaseException:
                if moved:
                    os.replace(staged, path)
                raise
            else:
                # A locked temporary file is no longer a public review record.
                try: staged.unlink(missing_ok=True)
                except OSError: pass


def evidence_rows(rows: list[dict], window_id: str, behavior_id: str, alerts: list[dict]) -> tuple[dict, list[dict]]:
    anchor = next((row for row in rows if row.get('window_id') == window_id), None)
    if anchor is None:
        raise FileNotFoundError('该窗口已清理或不存在')
    ids = {window_id}
    for alert in alerts:
        if alert.get('behavior_id') == behavior_id and window_id in (alert.get('window_ids') or []):
            ids.update(alert['window_ids'])
    positions = [i for i, row in enumerate(rows) if row.get('window_id') in ids]
    selected = [rows[i] for i in sorted({p for i in positions for p in (i-1, i, i+1) if 0 <= p < len(rows)})]
    return anchor, selected


def evidence_frames(rows: list[dict], limit=32) -> tuple[list[dict], list[str]]:
    frames, warnings, seen = [], [], set()
    for row in rows:
        for position, index in enumerate(row.get('frame_indices') or []):
            try:
                frame = VideoLibrary.frame_info(row, index)
            except (ValueError, IndexError, TypeError):
                warnings.append('部分旧窗口缺少可靠的帧时间，未推算证据位置')
                continue
            key = (row['window_id'], index)
            if key not in seen:
                seen.add(key)
                frames.append(frame | {'position': position})
    frames.sort(key=lambda item: item['time_sec'])
    if len(frames) > limit:
        # Keep action evidence and first/last frames when thinning the context.
        essential = set()
        for row in rows:
            indices = row.get('frame_indices') or []
            if indices:
                essential.update((row['window_id'], i) for i in (indices[0], indices[-1]))
            for violation in (row.get('violations') or {}).values():
                refs = violation.get('evidence') or {}
                if isinstance(refs, dict):
                    for n in refs.values():
                        if type(n) is int and 1 <= n <= len(indices):
                            essential.add((row['window_id'], indices[n-1]))
        priority = [f for f in frames if (f['window_id'], f['sample_index']) in essential]
        others = [f for f in frames if (f['window_id'], f['sample_index']) not in essential]
        budget = min(limit-len(priority), len(others))
        frames = sample_evenly(priority, limit) if len(priority) >= limit else priority + (others[:budget] if budget < 2 else sample_evenly(others, budget))
        frames.sort(key=lambda item: item['time_sec'])
        warnings.append('证据较多，保留动作引用和前后关键帧；可在对应窗口查看全部抽样帧')
    return frames, sorted(set(warnings))

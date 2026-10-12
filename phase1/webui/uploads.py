"""Bounded uploads and a durable reference for a file awaiting analysis."""
import asyncio
import json
import os
import re
import threading
import uuid
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import JSONResponse

CHUNK_BYTES = 1024 * 1024
DEFAULT_MAX_BYTES = 1024 * 1024 * 1024
VIDEO_SUFFIXES = {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.m4v', '.mpeg', '.mpg'}


class UploadGuard:
    """Apply limits before multipart parsing/spooling, including chunked bodies."""
    def __init__(self, app, limit, concurrency=2):
        self.app, self.limit = app, limit
        self.slots = threading.BoundedSemaphore(concurrency)

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['path'] != '/api/upload':
            return await self.app(scope, receive, send)
        budget = self.limit() + CHUNK_BYTES  # bounded multipart headers/boundaries
        length = dict(scope.get('headers', [])).get(b'content-length', b'0')
        try:
            oversized = int(length) > budget
        except ValueError:
            return await JSONResponse({'detail': '上传长度无效'}, 400)(scope, receive, send)
        if oversized:
            return await JSONResponse({'detail': '视频超过上传大小限制'}, 413)(scope, receive, send)
        if not self.slots.acquire(False):
            return await JSONResponse({'detail': '上传处理中，请稍后重试'}, 429)(scope, receive, send)
        received = 0
        async def bounded_receive():
            nonlocal received
            message = await receive()
            received += len(message.get('body', b''))
            if received > budget:
                raise HTTPException(413, '视频超过上传大小限制')
            return message
        try:
            await self.app(scope, bounded_receive, send)
        finally:
            self.slots.release()


def manifest(root: Path, job_id: str) -> Path:
    if not re.fullmatch(r'[a-f0-9]{12}', job_id):
        raise ValueError('上传编号无效')
    return root / '_pending' / (job_id + '.json')


async def save_upload(file, root: Path, max_bytes: int) -> tuple[str, Path, int]:
    suffix = Path(file.filename or '').suffix.lower()
    temp = dst = reference = None
    try:
        if suffix not in VIDEO_SUFFIXES:
            raise HTTPException(400, '请选择支持的视频文件（MP4 / AVI / MKV / MOV / WebM 等）')
        if file.size is not None and file.size > max_bytes:
            raise HTTPException(413, '视频超过上传大小限制')
        root.mkdir(parents=True, exist_ok=True)
        job_id = uuid.uuid4().hex[:12]
        dst = root / (job_id + suffix)
        temp = root / (job_id + '.part')
        size = 0
        with temp.open('xb') as output:
            while chunk := await file.read(CHUNK_BYTES):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(413, '视频超过上传大小限制')
                await asyncio.to_thread(output.write, chunk)
        if size == 0:
            raise HTTPException(400, '空文件')
        os.replace(temp, dst)
        reference = manifest(root, job_id)
        reference.parent.mkdir(exist_ok=True)
        reference.write_text(json.dumps({'filename': file.filename, 'source_name': dst.name}, ensure_ascii=False), encoding='utf-8')
        return job_id, dst, size
    except BaseException:
        for path in (temp, dst, reference):
            if path is not None:
                path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


def load_upload(root: Path, job_id: str) -> tuple[Path, str]:
    record = json.loads(manifest(root, job_id).read_text(encoding='utf-8'))
    path = root / record['source_name']
    if path.stem != job_id or path.suffix not in VIDEO_SUFFIXES or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('上传文件不存在或路径无效')
    return path, record['filename']

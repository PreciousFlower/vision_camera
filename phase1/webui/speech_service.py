"""Bounded local neural speech, independent of the detection/GPU queue."""
from __future__ import annotations

import asyncio
from collections import OrderedDict
import math
import io
import os
from pathlib import Path
import threading
import time

VOICES = {"xiaoxiao": "zf_xiaoxiao", "yunxi": "zm_yunxi"}


def validate_speech(body: dict) -> tuple[str, str, int]:
    text, voice, rate = body.get("text"), body.get("voice", "xiaoxiao"), body.get("rate", 1)
    if not isinstance(text, str) or not text.strip() or len(text) > 240:
        raise ValueError("播报文字需为 1 至 240 字")
    if not isinstance(voice, str) or voice not in VOICES:
        raise ValueError("请选择自然女声或自然男声")
    if isinstance(rate, bool) or not isinstance(rate, (float, int)) or not math.isfinite(rate) or not .85 <= rate <= 1.15:
        raise ValueError("播报语速需为 0.85 至 1.15")
    return text.strip(), voice, round((rate - 1) * 100)


class SpeechService:
    def __init__(self):
        self.slots = threading.BoundedSemaphore(2)
        self.lock = threading.Lock()
        self.render_lock = threading.Lock()
        self.cache = OrderedDict()
        self.max_audio_bytes = 2 * 1024 * 1024
        self.max_cache_bytes = 8 * 1024 * 1024
        self.model = None
        self.g2p = None
        self.models_dir = Path(os.environ.get("CIVILITY_SPEECH_MODELS") or
                               Path(__file__).resolve().parents[1] / "data/speech_models")

    async def synthesize(self, text: str, voice: str, rate: int) -> bytes:
        key = (text, voice, rate)
        with self.lock:
            now = time.monotonic()
            for old_key, (created, _) in list(self.cache.items()):
                if now - created > 900:
                    del self.cache[old_key]
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key][1]
        if not self.slots.acquire(blocking=False):
            raise BlockingIOError("自然语音正在生成，请使用系统语音")
        try:
            audio = await asyncio.wait_for(self._render(text, voice, rate), timeout=10)
            with self.lock:
                self.cache[key] = (time.monotonic(), audio)
                self.cache.move_to_end(key)
                while len(self.cache) > 32 or sum(len(data) for _, data in self.cache.values()) > self.max_cache_bytes:
                    self.cache.popitem(last=False)
            return audio
        finally:
            self.slots.release()

    async def _render(self, text: str, voice: str, rate: int) -> bytes:
        if not self.render_lock.acquire(blocking=False):
            raise BlockingIOError("自然语音正在生成，请使用系统语音")
        # The worker owns the lock until inference actually exits, including
        # after a client timeout. Timed-out requests cannot pile up CPU work.
        return await asyncio.to_thread(self._render_local, text, voice, rate)

    def _render_local(self, text: str, voice: str, rate: int) -> bytes:
        try:
            import onnxruntime as ort
            from kokoro_onnx import Kokoro
            from misaki.zh import ZHG2P
            import soundfile as sf

            if self.model is None:
                options = ort.SessionOptions()
                options.intra_op_num_threads = 2
                options.inter_op_num_threads = 1
                session = ort.InferenceSession(str(self.models_dir / "kokoro-v1.0.int8.onnx"),
                                               sess_options=options, providers=["CPUExecutionProvider"])
                self.model = Kokoro.from_session(session, str(self.models_dir / "voices-v1.0.bin"))
                self.g2p = ZHG2P()
            phonemes, _ = self.g2p(text)
            samples, sample_rate = self.model.create(phonemes, voice=VOICES[voice],
                                                      speed=1 + rate / 100, is_phonemes=True)
            output = io.BytesIO()
            sf.write(output, samples, sample_rate, format="WAV", subtype="PCM_16")
            audio = output.getvalue()
            if len(audio) <= 44 or len(audio) > self.max_audio_bytes:
                raise ValueError("自然语音音频为空或超过限制")
            return audio
        finally:
            self.render_lock.release()

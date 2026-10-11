"""Small, bounded motion hints for sampling; only the VLM decides behavior.

The reader already encodes at 8fps. At that same cadence we compare tiny gray
tiles, retaining timestamps and scores only. Image bytes stay in its existing
ring buffer. Highlights may recover a brief action between two model batches;
they cannot guarantee that every action in an overloaded stream is examined.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import deque


class MotionHighlights:
    def __init__(self, limit: int = 256) -> None:
        self.scores = deque(maxlen=limit)
        self.previous = None

    def add(self, timestamp: float, frame) -> None:
        import cv2
        import numpy as np

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        small = cv2.resize(gray, (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32)
        if self.previous is not None:
            change = small - self.previous
            # Uniform exposure changes should not crowd out a small hand/object.
            change = np.abs(change - np.median(change))
            tiles = change.reshape(6, 8, 8, 8).mean(axis=(1, 3))
            self.scores.append((timestamp, float(tiles.max())))
        self.previous = small

    def clear(self) -> None:
        self.scores.clear()
        self.previous = None


def action_frames(history: list, highlights: list, *, start: float, end: float,
                  max_frames: int = 6, threshold: float = 2.5) -> list:
    """Pick at most two separated peaks with real before/peak/after frames.

    Scores are sampling hints, never evidence of a specific violation. Frames
    remain ordered, unique, bounded, and taken only from the supplied history.
    """
    if max_frames < 3 or len(history) < 3 or end <= start:
        return []
    times = [t for t, _ in history]
    candidates = [(t, score) for t, score in highlights
                  if start < t < end and score >= threshold]
    if not candidates:
        return []
    # Cover separate phases of actual movement. Two stronger walking peaks
    # must not displace the slightly weaker later release/landing action.
    if max_frames >= 6:
        middle = (min(t for t, _ in candidates) + max(t for t, _ in candidates)) / 2
        phases = [[pair for pair in candidates if pair[0] <= middle],
                  [pair for pair in candidates if pair[0] > middle]]
    else:
        phases = [candidates]
    chosen = []
    peaks = []
    for phase in phases:
        for timestamp, _ in sorted(phase, key=lambda value: (-value[1], -value[0])):
            if any(abs(timestamp - previous) < 0.75 for previous in peaks):
                continue
            index = bisect_left(times, timestamp)
            if index >= len(times) or abs(times[index] - timestamp) > 0.001:
                continue
            # ±0.375s gives context beyond the encoded frame immediately next door.
            before = max(0, bisect_left(times, timestamp - 0.375))
            after = min(len(times) - 1, bisect_left(times, timestamp + 0.375))
            if before >= index or after <= index:
                continue
            group = [history[before], history[index], history[after]]
            if group[0][0] < start or group[-1][0] > end:
                continue
            extra = [pair for pair in group if pair[0] not in {t for t, _ in chosen}]
            if len(chosen) + len(extra) > max_frames:
                continue
            chosen.extend(extra)
            peaks.append(timestamp)
            break
    return sorted(chosen, key=lambda pair: pair[0])


def merge_sampling(prior: list, fresh: list, highlights: list, *, budget: int) -> list:
    """Keep fresh samples and contextual endpoints; swap background for peaks."""
    from civility.extract import sample_evenly

    # The normal image count is also the maximum: no extra GPU image work.
    budget = min(budget, len(prior) + len(fresh))
    required = {t: data for t, data in fresh}
    if prior and len(required) < budget:
        required[prior[0][0]] = prior[0][1]
    if prior and len(required) < budget:
        required[prior[-1][0]] = prior[-1][1]
    for timestamp, data in highlights:
        if timestamp in required or len(required) < budget:
            required[timestamp] = data
    remaining = [(t, data) for t, data in prior if t not in required]
    slots = max(0, budget - len(required))
    background = (sample_evenly(remaining, slots) if slots >= 2
                  else [remaining[len(remaining) // 2]] if slots == 1 and remaining else [])
    for timestamp, data in background:
        required[timestamp] = data
    return sorted(required.items())[:budget]

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time


@dataclass
class DelayedFrame:
    ready_at: float
    rgb: tuple[int, int, int]
    beat: bool = False
    downbeat: bool = False
    bpm: float = 0.0
    flux: float = 0.0
    bass: float = 0.0
    energy: float = 0.0
    state: str = ""
    palette: str = ""
    tension: float = 0.0
    pre_drop: bool = False
    drop_event: bool = False


class OutputDelay:
    """Hold light updates so they line up with delayed Bluetooth speakers."""

    def __init__(self, delay_ms: float = 200.0) -> None:
        self.delay_s = max(0.0, float(delay_ms) / 1000.0)
        self._queue: deque[DelayedFrame] = deque()

    @property
    def delay_ms(self) -> float:
        return self.delay_s * 1000.0

    def push(self, rgb: tuple[int, int, int], **meta: object) -> None:
        self._queue.append(
            DelayedFrame(
                ready_at=time.perf_counter() + self.delay_s,
                rgb=rgb,
                beat=bool(meta.get("beat", False)),
                downbeat=bool(meta.get("downbeat", False)),
                bpm=float(meta.get("bpm", 0.0) or 0.0),
                flux=float(meta.get("flux", 0.0) or 0.0),
                bass=float(meta.get("bass", 0.0) or 0.0),
                energy=float(meta.get("energy", 0.0) or 0.0),
                state=str(meta.get("state", "") or ""),
                palette=str(meta.get("palette", "") or ""),
                tension=float(meta.get("tension", 0.0) or 0.0),
                pre_drop=bool(meta.get("pre_drop", False)),
                drop_event=bool(meta.get("drop_event", False)),
            )
        )
        cutoff = time.perf_counter() - 0.05
        while len(self._queue) > 1 and self._queue[0].ready_at < cutoff - 2.0:
            self._queue.popleft()

    def pop_ready(self) -> DelayedFrame | None:
        if not self._queue:
            return None
        if self.delay_s <= 0.0 or self._queue[0].ready_at <= time.perf_counter():
            return self._queue.popleft()
        return None

    def pop_all_ready(self) -> list[DelayedFrame]:
        ready: list[DelayedFrame] = []
        while True:
            frame = self.pop_ready()
            if frame is None:
                break
            ready.append(frame)
        return ready

    def clear(self) -> None:
        self._queue.clear()

    def __len__(self) -> int:
        return len(self._queue)

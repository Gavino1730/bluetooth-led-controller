from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MonitorInfo:
    index: int
    left: int
    top: int
    width: int
    height: int

    @property
    def label(self) -> str:
        return f"#{self.index}  {self.width}x{self.height}  @({self.left},{self.top})"


def list_monitors() -> list[MonitorInfo]:
    """Return individual monitors (mss index 1+)."""
    import mss

    with mss.mss() as sct:
        out: list[MonitorInfo] = []
        for i, mon in enumerate(sct.monitors):
            if i == 0:
                continue  # virtual "all monitors" desktop
            out.append(
                MonitorInfo(
                    index=i,
                    left=int(mon["left"]),
                    top=int(mon["top"]),
                    width=int(mon["width"]),
                    height=int(mon["height"]),
                )
            )
        return out


class ScreenCapture:
    """Fast full-monitor grab via mss, downscaled for color analysis."""

    def __init__(self, monitor: int = 1, max_width: int = 160) -> None:
        import mss

        self.monitor = max(1, int(monitor))
        self.max_width = max(48, int(max_width))
        self._sct = mss.mss()
        if self.monitor >= len(self._sct.monitors):
            raise ValueError(
                f"Monitor {self.monitor} not found "
                f"(available: 1..{len(self._sct.monitors) - 1})"
            )
        mon = self._sct.monitors[self.monitor]
        self.width = int(mon["width"])
        self.height = int(mon["height"])
        self.label = (
            f"monitor {self.monitor} ({self.width}x{self.height})"
        )

    def grab(self) -> np.ndarray:
        """Return RGB uint8 array, downscaled so width ≈ max_width."""
        shot = self._sct.grab(self._sct.monitors[self.monitor])
        # mss BGRA → RGB
        frame = np.frombuffer(shot.bgra, dtype=np.uint8).reshape(shot.height, shot.width, 4)
        rgb = frame[:, :, 2::-1].copy()  # B,G,R → R,G,B contiguous
        return self._downscale(rgb)

    def _downscale(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        if w <= self.max_width:
            return rgb
        scale = self.max_width / float(w)
        new_w = self.max_width
        new_h = max(1, int(round(h * scale)))
        # Integer box downsample — fast enough for ~15 Hz ambient sync
        y_idx = (np.linspace(0, h - 1, new_h)).astype(np.int32)
        x_idx = (np.linspace(0, w - 1, new_w)).astype(np.int32)
        return rgb[y_idx][:, x_idx]

    def close(self) -> None:
        try:
            self._sct.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self) -> ScreenCapture:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

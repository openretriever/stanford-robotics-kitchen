"""Streams a run to an mp4: wide view, caption bar, and Astra's own 320px frame.

Frames are written as they are produced, never accumulated. A six-decision run
is roughly 4,000 frames; held in memory at 1280x720x3 that would be about 11 GB.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

BANNER_H = 76
PIP = 300


@dataclass
class Recorder:
    path: str
    width: int = 1280
    height: int = 720
    fps: int = 50
    stride: int = 4                 # capture every Nth control step
    pip_every: int = 5              # re-render Astra's inset only this often
    title: str = "SRC kitchen  -  Astra drawer search"
    _writer: object = None
    _n: int = 0
    _frames: int = 0
    _pip: object = None
    _pip_age: int = 0
    _pip_failed: bool = False
    banner: dict = field(default_factory=dict)

    def __post_init__(self):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # h264 + yuv420p is what QuickTime and browsers will actually play;
        # odd dimensions break yuv420p, so both are kept even.
        self._writer = imageio.get_writer(
            self.path, fps=self.fps, codec="libx264", pixelformat="yuv420p",
            macro_block_size=1, ffmpeg_params=["-crf", "20"])

    def set(self, **fields):
        self.banner.update({k: v for k, v in fields.items() if v is not None})

    def capture(self, world, *, force=False):
        self._n += 1
        if not force and self._n % self.stride:
            return
        wide = world.render_large(width=self.width, height=self.height - BANNER_H)
        frame = Image.new("RGB", (self.width, self.height), (18, 18, 20))
        frame.paste(wide, (0, BANNER_H))
        self._draw_banner(frame)
        try:                                    # Astra's actual input, inset
            if self._pip_failed:
                raise RuntimeError("inset previously failed")
            # Re-rendering the model's 320px view every frame doubles the cost of
            # the whole recording for an inset that barely changes between them.
            if self._pip is None or self._pip_age >= self.pip_every:
                self._pip = world.render().resize((PIP, PIP))
                self._pip_age = 0
            self._pip_age += 1
            frame.paste(self._pip, (self.width - PIP - 16, self.height - PIP - 16))
            d = ImageDraw.Draw(frame)
            d.rectangle([self.width - PIP - 16, self.height - PIP - 34,
                         self.width - 16, self.height - PIP - 16], fill=(0, 0, 0))
            d.text((self.width - PIP - 10, self.height - PIP - 30),
                   "drawer_view 320x320 (what Astra sees)", fill=(235, 235, 235))
        except Exception as error:
            # The inset is a nicety; losing it must not abandon the recording.
            # But swallowing the reason silently would hide a broken renderer for
            # a whole run, so say it once and then stop trying.
            if not self._pip_failed:
                self._pip_failed = True
                print(f"  [recorder] inset disabled: {type(error).__name__}: {error}")
        self._writer.append_data(np.asarray(frame))
        self._frames += 1

    def hold(self, world, seconds=1.2):
        for _ in range(int(seconds * self.fps)):
            self.capture(world, force=True)

    def _draw_banner(self, frame):
        d = ImageDraw.Draw(frame)
        d.rectangle([0, 0, self.width, BANNER_H], fill=(24, 24, 28))
        d.text((16, 10), self.title, fill=(245, 245, 245))
        left = "   ".join(
            str(self.banner[k]) for k in ("decision", "action", "drawer") if self.banner.get(k))
        d.text((16, 34), left[:110], fill=(150, 200, 255))
        note = self.banner.get("note", "")
        if note:
            d.text((16, 54), note[:150], fill=(200, 200, 160))
        right = self.banner.get("metric", "")
        if right:
            d.text((self.width - 330, 34), str(right)[:46], fill=(220, 220, 220))

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        return self._frames

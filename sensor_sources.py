"""Data sources: a real phone running phyphox, or a simulator for offline testing.

phyphox remote access (https://phyphox.org/wiki/index.php/Remote-interface_communication):
    GET /get?accX=<t>|acc_time&accY=<t>|acc_time&accZ=<t>|acc_time&acc_time=<t>
        -> new samples recorded after time t (incremental polling)
    GET /control?cmd=start | stop | clear
    GET /config  -> experiment description, including the buffer names

Both sources return the same structure, so the rest of the pipeline does not
care where the data comes from.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import requests


@dataclass
class Samples:
    """A batch of new samples: time [s] plus one array per channel."""

    t: np.ndarray
    channels: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.t)


class PhyphoxSource:
    """Polls a phone running phyphox with 'Allow remote access' enabled."""

    def __init__(
        self,
        url: str,
        channels: tuple[str, ...] = ("accX", "accY", "accZ"),
        time_buffer: str = "acc_time",
        timeout: float = 2.0,
    ) -> None:
        self.url = url.rstrip("/")
        self.channels = channels
        self.time_buffer = time_buffer
        self.timeout = timeout
        self._last_t = 0.0

    def _get(self, path: str, params: str = "") -> dict:
        r = requests.get(f"{self.url}/{path}{params}", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def buffer_names(self) -> list[str]:
        """Buffer names of the experiment currently open in phyphox."""
        cfg = self._get("config")
        return [b["name"] for b in cfg.get("buffers", [])]

    def start(self) -> None:
        self._get("control", "?cmd=start")

    def stop(self) -> None:
        self._get("control", "?cmd=stop")

    def clear(self) -> None:
        self._get("control", "?cmd=clear")
        self._last_t = 0.0

    def read_new(self) -> Samples:
        """Fetch only the samples recorded since the previous call."""
        tb = self.time_buffer
        parts = [f"{tb}={self._last_t}"]
        parts += [f"{ch}={self._last_t}|{tb}" for ch in self.channels]
        data = self._get("get", "?" + "&".join(parts))["buffer"]

        t = np.asarray(data[tb]["buffer"], dtype=float)
        chans = {ch: np.asarray(data[ch]["buffer"], dtype=float) for ch in self.channels}

        # phyphox can return None/NaN at buffer edges: keep only complete rows
        n = min([len(t)] + [len(v) for v in chans.values()])
        t = t[:n]
        chans = {k: v[:n] for k, v in chans.items()}
        ok = np.isfinite(t)
        for v in chans.values():
            ok &= np.isfinite(v)
        t = t[ok]
        chans = {k: v[ok] for k, v in chans.items()}

        if len(t):
            self._last_t = float(t[-1])
        return Samples(t, chans)


class SimulatedSource:
    """Generates phone-like accelerometer data so the app runs without a phone.

    Signal model per axis: sum of sinusoids (e.g. a vibrating machine at a
    fundamental plus harmonics) + white noise. A fault can be switched on,
    which adds a new frequency component, e.g. a bearing defect.
    """

    def __init__(
        self,
        fs: float = 200.0,
        channels: tuple[str, ...] = ("accX", "accY", "accZ"),
        base_freq: float = 25.0,
        noise: float = 0.05,
        seed: int | None = 0,
    ) -> None:
        self.fs = fs
        self.channels = channels
        self.base_freq = base_freq
        self.noise = noise
        self.fault = False
        self.fault_freq = 67.0
        self._rng = np.random.default_rng(seed)
        self._t0 = time.monotonic()
        self._n = 0  # samples emitted so far

    def start(self) -> None:
        self._t0 = time.monotonic()
        self._n = 0

    stop = clear = start

    def _signal(self, t: np.ndarray, axis: int) -> np.ndarray:
        f0 = self.base_freq
        amp = (1.0, 0.6, 0.3)[axis % 3]
        x = amp * np.sin(2 * np.pi * f0 * t + axis)
        x += 0.3 * amp * np.sin(2 * np.pi * 2 * f0 * t)
        if self.fault:
            x += 0.5 * np.sin(2 * np.pi * self.fault_freq * t)
        return x + self.noise * self._rng.standard_normal(len(t))

    def generate(self, n: int) -> Samples:
        """Return the next n samples (deterministic, no wall clock)."""
        t = (self._n + np.arange(n)) / self.fs
        self._n += n
        return Samples(t, {ch: self._signal(t, i) for i, ch in enumerate(self.channels)})

    def read_new(self) -> Samples:
        """Return all samples that 'arrived' since the last call, in real time."""
        due = int((time.monotonic() - self._t0) * self.fs) - self._n
        return self.generate(max(due, 0))

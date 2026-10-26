"""Signal processing for accelerometer data: resampling, FFT, RMS, peaks.

Phone sensors do not sample at a perfectly constant rate, so the signal is
first resampled onto a uniform grid. Without that step the FFT is not valid.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def estimate_fs(t: np.ndarray) -> float:
    """Sampling rate from timestamps (median interval is robust to jitter)."""
    if len(t) < 2:
        return 0.0
    return 1.0 / float(np.median(np.diff(t)))


def resample_uniform(t: np.ndarray, x: np.ndarray, fs: float | None = None):
    """Linear interpolation of an irregularly sampled signal onto a uniform grid."""
    fs = fs or estimate_fs(t)
    if fs <= 0 or len(t) < 2:
        return t, x, fs
    tu = np.arange(t[0], t[-1], 1.0 / fs)
    return tu, np.interp(tu, t, x), fs


def rms(x: np.ndarray) -> float:
    """Root mean square after removing the mean (DC / gravity offset)."""
    if len(x) == 0:
        return 0.0
    x = x - np.mean(x)
    return float(np.sqrt(np.mean(x**2)))


def amplitude_spectrum(x: np.ndarray, fs: float):
    """One-sided amplitude spectrum with a Hann window.

    Scaling: a sinusoid of amplitude A shows up as a peak of height ~A.
    """
    n = len(x)
    if n < 4:
        return np.array([]), np.array([])
    x = x - np.mean(x)
    w = np.hanning(n)
    X = np.fft.rfft(x * w)
    amp = 2.0 * np.abs(X) / np.sum(w)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    return f, amp


def dominant_peaks(f: np.ndarray, amp: np.ndarray, n: int = 3, min_rel: float = 0.1):
    """The n largest spectral peaks above min_rel * max amplitude, as (freq, amp)."""
    if len(amp) == 0:
        return []
    idx, _ = signal.find_peaks(amp, height=min_rel * amp.max())
    idx = idx[np.argsort(amp[idx])[::-1][:n]]
    return [(float(f[i]), float(amp[i])) for i in sorted(idx, key=lambda i: f[i])]


def spectrogram(x: np.ndarray, fs: float, window_s: float = 1.0):
    """Short-time Fourier transform: how the spectrum changes over time."""
    nper = max(16, int(window_s * fs))
    if len(x) < nper:
        return None
    f, t, S = signal.spectrogram(x - np.mean(x), fs=fs, nperseg=nper, noverlap=nper // 2)
    return f, t, 10 * np.log10(S + 1e-12)


def magnitude(channels: dict[str, np.ndarray]) -> np.ndarray:
    """Vector magnitude sqrt(x^2 + y^2 + z^2): orientation-independent."""
    return np.sqrt(sum(v**2 for v in channels.values()))

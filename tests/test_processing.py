import numpy as np

import processing as P
from sensor_sources import SimulatedSource


def test_fft_finds_known_frequency_and_amplitude():
    fs, f0, A = 200.0, 25.0, 1.5
    t = np.arange(0, 10, 1 / fs)
    f, amp = P.amplitude_spectrum(A * np.sin(2 * np.pi * f0 * t), fs)
    peaks = P.dominant_peaks(f, amp, n=1)
    assert abs(peaks[0][0] - f0) < 0.2
    assert abs(peaks[0][1] - A) < 0.1


def test_rms_of_sine():
    t = np.arange(0, 1, 1 / 1000)
    assert abs(P.rms(2 * np.sin(2 * np.pi * 10 * t) + 9.81) - 2 / np.sqrt(2)) < 1e-2


def test_resample_handles_jittery_timestamps():
    rng = np.random.default_rng(1)
    t = np.cumsum(0.005 + 0.001 * rng.standard_normal(2000).clip(-3, 3))
    x = np.sin(2 * np.pi * 20 * t)
    tu, xu, fs = P.resample_uniform(t, x)
    assert np.allclose(np.diff(tu), 1 / fs)
    f, amp = P.amplitude_spectrum(xu, fs)
    assert abs(P.dominant_peaks(f, amp, n=1)[0][0] - 20) < 0.5


def test_simulator_fault_adds_frequency():
    src = SimulatedSource(fs=200, base_freq=25)
    src.fault = True
    s = src.generate(2000)
    f, amp = P.amplitude_spectrum(s.channels["accX"], 200)
    freqs = [pf for pf, _ in P.dominant_peaks(f, amp, n=4)]
    assert any(abs(pf - 67) < 0.5 for pf in freqs)
    assert any(abs(pf - 25) < 0.5 for pf in freqs)

"""Live dashboard: stream phone accelerometer data, show time signal, FFT and spectrogram.

Run:  streamlit run app.py
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import processing as P
from sensor_sources import PhyphoxSource, SimulatedSource

st.set_page_config(page_title="Phone Sensor Pipeline", layout="wide")
st.title("📱 Phone-as-Sensor Vibration Pipeline")
st.caption("phyphox → Python → signal processing → live dashboard")

# ---------------- sidebar: source and settings ----------------
with st.sidebar:
    st.header("Source")
    mode = st.radio("Data source", ["Simulated", "Phone (phyphox)"])
    url = st.text_input("phyphox URL", "http://192.168.0.10:8080",
                        disabled=mode != "Phone (phyphox)")
    window_s = st.slider("Display window [s]", 2, 30, 10)
    refresh = st.slider("Refresh interval [s]", 0.2, 2.0, 0.5)
    channel = st.selectbox("Analyse", ["accX", "accY", "accZ", "magnitude"],
                           help="Magnitude is orientation-independent but nonlinear: "
                                "without gravity it creates harmonics at 2×f.")

    if mode == "Simulated":
        st.header("Simulator")
        base = st.slider("Machine frequency [Hz]", 5.0, 80.0, 25.0)
        fault = st.toggle("Inject fault component (67 Hz)")

    c1, c2 = st.columns(2)
    run = c1.toggle("Run", value=False)
    if c2.button("Clear"):
        st.session_state.pop("buf", None)
        if "src" in st.session_state:
            try:
                st.session_state.src.clear()
            except Exception as e:  # noqa: BLE001
                st.warning(f"Clear failed: {e}")

# ---------------- source in session state ----------------
key = (mode, url)
if st.session_state.get("src_key") != key:
    st.session_state.src = SimulatedSource() if mode == "Simulated" else PhyphoxSource(url)
    st.session_state.src_key = key
    st.session_state.pop("buf", None)
src = st.session_state.src
if isinstance(src, SimulatedSource):
    src.base_freq, src.fault = base, fault

if "buf" not in st.session_state:
    st.session_state.buf = pd.DataFrame(columns=["t", "accX", "accY", "accZ"])


def pull() -> str | None:
    """Append new samples to the rolling buffer. Returns an error message or None."""
    try:
        s = src.read_new()
    except Exception as e:  # noqa: BLE001
        return f"Could not reach phyphox at {url}: {e}"
    if len(s):
        new = pd.DataFrame({"t": s.t, **s.channels})
        buf = pd.concat([st.session_state.buf, new], ignore_index=True)
        tmax = buf["t"].iloc[-1]
        st.session_state.buf = buf[buf["t"] >= tmax - window_s].reset_index(drop=True)
    return None


@st.fragment(run_every=refresh if run else None)
def live_view() -> None:
    if run:
        err = pull()
        if err:
            st.error(err)
            st.info("Check: phone and laptop on the same Wi-Fi, phyphox experiment "
                    "running, and 'Allow remote access' enabled (⋮ menu).")
            return

    buf = st.session_state.buf
    if len(buf) < 32:
        st.info("Press **Run** in the sidebar to start streaming.")
        return

    t = buf["t"].to_numpy(float)
    chans = {c: buf[c].to_numpy(float) for c in ("accX", "accY", "accZ")}
    x = P.magnitude(chans) if channel == "magnitude" else chans[channel]
    tu, xu, fs = P.resample_uniform(t, x)
    f, amp = P.amplitude_spectrum(xu, fs)
    peaks = P.dominant_peaks(f, amp)

    # metrics
    m = st.columns(4)
    m[0].metric("Sampling rate", f"{fs:.0f} Hz")
    m[1].metric("RMS", f"{P.rms(xu):.3f} m/s²")
    m[2].metric("Dominant frequency", f"{peaks[np.argmax([a for _, a in peaks])][0]:.1f} Hz"
                if peaks else "–")
    m[3].metric("Samples in window", f"{len(t)}")

    left, right = st.columns(2)

    fig = go.Figure()
    for c, v in chans.items():
        fig.add_trace(go.Scatter(x=t, y=v, name=c, mode="lines", line=dict(width=1)))
    fig.update_layout(title="Time signal", xaxis_title="t [s]",
                      yaxis_title="a [m/s²]", height=340, margin=dict(t=40, b=40))
    left.plotly_chart(fig, width="stretch")

    fig = go.Figure(go.Scatter(x=f, y=amp, mode="lines", name="amplitude"))
    for pf, pa in peaks:
        fig.add_annotation(x=pf, y=pa, text=f"{pf:.1f} Hz", showarrow=True, arrowhead=2)
    fig.update_layout(title=f"Amplitude spectrum ({channel})", xaxis_title="f [Hz]",
                      yaxis_title="amplitude [m/s²]", height=340, margin=dict(t=40, b=40))
    right.plotly_chart(fig, width="stretch")

    spec = P.spectrogram(xu, fs)
    if spec is not None:
        sf, st_, S = spec
        fig = go.Figure(go.Heatmap(x=st_ + tu[0], y=sf, z=S, colorscale="Viridis",
                                   colorbar=dict(title="dB")))
        fig.update_layout(title="Spectrogram", xaxis_title="t [s]", yaxis_title="f [Hz]",
                          height=320, margin=dict(t=40, b=40))
        st.plotly_chart(fig, width="stretch")

    csv = io.StringIO()
    buf.to_csv(csv, index=False)
    st.download_button("Download window as CSV", csv.getvalue(), "recording.csv", "text/csv")


live_view()

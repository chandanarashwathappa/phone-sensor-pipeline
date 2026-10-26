"""Command-line recorder: stream from the phone (or simulator) into a CSV file.

Examples:
    python record.py --sim --seconds 10
    python record.py --url http://192.168.0.10:8080 --seconds 30 --out vibration.csv
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

import processing as P
from sensor_sources import PhyphoxSource, SimulatedSource


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--url", help="phyphox remote access URL, e.g. http://192.168.0.10:8080")
    ap.add_argument("--sim", action="store_true", help="use the simulator instead of a phone")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out", default="recording.csv")
    args = ap.parse_args()

    if not args.sim and not args.url:
        ap.error("give --url or --sim")
    src = SimulatedSource() if args.sim else PhyphoxSource(args.url)

    if not args.sim:
        print("Buffers in current phyphox experiment:", src.buffer_names())
        src.clear()
        src.start()
    else:
        src.start()

    frames, t_end = [], time.monotonic() + args.seconds
    while time.monotonic() < t_end:
        s = src.read_new()
        if len(s):
            frames.append(pd.DataFrame({"t": s.t, **s.channels}))
        time.sleep(0.2)

    if not args.sim:
        src.stop()

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(args.out, index=False)

    t = df["t"].to_numpy()
    x = P.magnitude({c: df[c].to_numpy() for c in ("accX", "accY", "accZ")})
    tu, xu, fs = P.resample_uniform(t, x)
    f, amp = P.amplitude_spectrum(xu, fs)
    print(f"Saved {len(df)} samples to {args.out}  (fs ≈ {fs:.0f} Hz, RMS {P.rms(xu):.3f} m/s²)")
    print("Dominant peaks:", ", ".join(f"{pf:.1f} Hz" for pf, _ in P.dominant_peaks(f, amp)))


if __name__ == "__main__":
    main()

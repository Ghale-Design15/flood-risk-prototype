"""
Pre-demo warm-up for the Render free instance.

Run this a few minutes before a live demo. It hits /health until the service
answers quickly, so the first real prediction during the demo doesn't pay the
cold-start delay (the instance sleeps after ~15 min idle; the first call reloads
the container and, for the LSTM, TensorFlow).

Usage (cross-platform, no extra dependencies):
    python backend/warmup.py                             # uses API_URL or the default
    python backend/warmup.py https://flood-risk-api.onrender.com
    python backend/warmup.py --lstm                      # also preload the LSTM (TensorFlow)
    API_URL=https://... python backend/warmup.py

It (1) wakes the container via /health, then (2) fires one real prediction so the
model cache is loaded before the demo. Pass --lstm to also warm the LSTM path,
which lazy-loads TensorFlow and is otherwise slow on its first call.

Exit code 0 once the service is warm and a prediction succeeds, 1 otherwise.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

DEFAULT_URL = "https://flood-risk-api.onrender.com"
WARM_SECONDS = 3.0      # a /health under this is considered warm
MAX_WAIT_SECONDS = 150  # give up after this (a cold start is usually 30-60s)
GAP_SECONDS = 5

SAMPLE_FEATURES = {"level_lag1": 0.73, "level_lag2": 0.69,
                   "level_roll7": 0.68, "level_change3": 0.06}
SAMPLE_SERIES = [0.60, 0.62, 0.63, 0.64, 0.66, 0.67, 0.68,
                 0.69, 0.70, 0.71, 0.72, 0.72, 0.73, 0.73]  # 14 levels for the LSTM window


def get(url: str) -> tuple[int, float]:
    """Return (status_code, elapsed_seconds) for one GET; (0, elapsed) on failure."""
    start = time.time()
    try:
        with urllib.request.urlopen(url, timeout=95) as resp:
            resp.read(1)
            return resp.status, time.time() - start
    except Exception:
        return 0, time.time() - start


def post(url: str, body: dict) -> tuple[int, float]:
    start = time.time()
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=95) as resp:
            resp.read()
            return resp.status, time.time() - start
    except Exception:
        return 0, time.time() - start


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    warm_lstm = "--lstm" in sys.argv[1:]
    base = (args[0] if args else os.getenv("API_URL") or DEFAULT_URL).rstrip("/")

    print(f"Warming {base}/health")
    print("(first call may take 30-60s if the instance was asleep)\n")

    deadline = time.time() + MAX_WAIT_SECONDS
    attempt = 0
    warm = False
    while time.time() < deadline:
        attempt += 1
        code, elapsed = get(base + "/health")
        state = "OK" if code == 200 else f"HTTP {code}"
        print(f"  attempt {attempt}: /health {state} in {elapsed:0.1f}s")
        if code == 200 and elapsed <= WARM_SECONDS:
            warm = True
            break
        if code == 200:
            print("  responded but slow (still warming) — pinging again...")
        time.sleep(GAP_SECONDS)

    if not warm:
        print("\nGave up: the service did not warm up in time. Check the Render URL/logs.")
        return 1

    # Warm the model cache with one real prediction (the demo's default path).
    code, elapsed = post(base + "/predict", SAMPLE_FEATURES)
    print(f"  /predict (default/ensemble): HTTP {code} in {elapsed:0.1f}s")
    ok = code == 200

    if warm_lstm:
        code, elapsed = post(base + "/predict_series", {"levels": SAMPLE_SERIES, "model": "lstm"})
        print(f"  /predict_series (lstm, loads TensorFlow): HTTP {code} in {elapsed:0.1f}s")
        ok = ok and code == 200

    print("\nWarm and prediction path loaded. Demo-ready." if ok
          else "\nWarm, but a prediction call did not return 200 — check the logs.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

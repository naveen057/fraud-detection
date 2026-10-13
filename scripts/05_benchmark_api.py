"""Load-test a RUNNING API (http://localhost:8000) and report end-to-end latency.

    python scripts/05_benchmark_api.py --n 500 --url http://localhost:8000
"""
import argparse

import _common  # noqa: F401
import numpy as np
import requests

from fraud.config import RAW_FEATURES, TARGET
from fraud.data import load_processed, time_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000"); ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    test = time_split(load_processed())[2].sample(a.n, random_state=0)
    recs = test[RAW_FEATURES].to_dict("records")
    s = requests.Session()
    for r in recs[:20]: s.post(f"{a.url}/predict", json=r)  # warm-up
    lat = []
    for r in recs:
        resp = s.post(f"{a.url}/predict", json=r); resp.raise_for_status()
        lat.append(resp.elapsed.total_seconds() * 1000)
    lat = np.array(lat)
    print(f"{a.n} sequential requests | mean {lat.mean():.1f} ms | p50 {np.percentile(lat,50):.1f} | "
          f"p95 {np.percentile(lat,95):.1f} | p99 {np.percentile(lat,99):.1f} ms")
    print("server-side /metrics:", s.get(f"{a.url}/metrics").json())


if __name__ == "__main__":
    main()

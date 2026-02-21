#!/usr/bin/env python3
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import requests


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (p / 100.0)
    floor = int(k)
    ceil = min(floor + 1, len(ordered) - 1)
    if floor == ceil:
        return ordered[floor]
    return ordered[floor] + (ordered[ceil] - ordered[floor]) * (k - floor)


def one_request(url: str, host_header: str, timeout: int, event_id: str) -> Tuple[float, int]:
    payload = {
        "raw": {
            "event_id": event_id,
            "timestamp": 1735691403,
            "src_port": 12345,
            "dst_port": 80,
            "protocol": "TCP",
            "packet_count": 10,
            "conn_state": "SF",
            "bytes_transferred": 1024.0,
        }
    }
    headers = {"Content-Type": "application/json"}
    if host_header:
        headers["Host"] = host_header

    started = time.perf_counter()
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return elapsed_ms, response.status_code
    except Exception:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return elapsed_ms, 0


def run_benchmark(url: str, host_header: str, requests_count: int, concurrency: int, timeout: int, warmup: int) -> Dict[str, float]:
    for idx in range(warmup):
        one_request(url, host_header, timeout, f"warmup-{idx}")

    latencies: List[float] = []
    status_codes: List[int] = []

    begin = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(one_request, url, host_header, timeout, f"bench-{idx}")
            for idx in range(requests_count)
        ]
        for future in as_completed(futures):
            elapsed_ms, code = future.result()
            latencies.append(elapsed_ms)
            status_codes.append(code)
    total_s = time.perf_counter() - begin

    ok_rate = sum(1 for code in status_codes if code == 200) / max(len(status_codes), 1)

    return {
        "requests": requests_count,
        "concurrency": concurrency,
        "ok_rate": round(ok_rate, 4),
        "rps": round(requests_count / max(total_s, 1e-9), 2),
        "p50_ms": round(percentile(latencies, 50), 2),
        "p95_ms": round(percentile(latencies, 95), 2),
        "p99_ms": round(percentile(latencies, 99), 2),
        "max_ms": round(max(latencies) if latencies else 0.0, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Ray Serve HTTP latency benchmark")
    parser.add_argument("--url", default="http://127.0.0.1/infer")
    parser.add_argument("--host-header", default="serving.localhost")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--concurrency", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--target-p99-ms", type=float, default=80.0)
    parser.add_argument("--baseline-json", type=str, default="")
    parser.add_argument("--require-improvement-pct", type=float, default=0.0)
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    results = run_benchmark(
        url=args.url,
        host_header=args.host_header,
        requests_count=args.requests,
        concurrency=args.concurrency,
        timeout=args.timeout,
        warmup=args.warmup,
    )

    print(json.dumps(results, indent=2))

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(results, indent=2), encoding="utf-8")

    if results["ok_rate"] < 1.0:
        print("FAIL: Some requests failed (status != 200)")
        return 1

    if args.baseline_json:
        baseline = json.loads(Path(args.baseline_json).read_text(encoding="utf-8"))
        baseline_p99 = float(baseline["p99_ms"])
        current_p99 = float(results["p99_ms"])
        improvement_pct = ((baseline_p99 - current_p99) / baseline_p99) * 100.0 if baseline_p99 > 0 else 0.0
        print(f"Improvement vs baseline p99: {improvement_pct:.2f}%")
        if improvement_pct < args.require_improvement_pct:
            print(
                f"FAIL: improvement {improvement_pct:.2f}% is below required {args.require_improvement_pct:.2f}%"
            )
            return 1

    if results["p99_ms"] > args.target_p99_ms:
        print(f"FAIL: p99 {results['p99_ms']:.2f} ms is above target {args.target_p99_ms:.2f} ms")
        return 1

    print("PASS: Latency target satisfied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

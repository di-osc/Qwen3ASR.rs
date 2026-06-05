#!/usr/bin/env python3
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def file_url(path: Path) -> str:
    return Path(path).resolve().as_uri()


def wait_health(base_url: str, timeout_sec: float) -> None:
    deadline = time.time() + timeout_sec
    health_url = f"{base_url.rstrip('/')}/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.5)
    raise TimeoutError(f"service not healthy after {timeout_sec:.1f}s: {health_url}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark vASR /transcribe HTTP throughput")
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--wait-health-sec", type=float, default=300.0)
    args = parser.parse_args()

    files = sorted(args.audio_dir.glob("*.wav"))[: args.limit]
    if not files:
        print(f"no wav files in {args.audio_dir}", file=sys.stderr)
        return 1

    wait_health(args.base_url, args.wait_health_sec)

    payload = {
        "inputs": [{"url": file_url(path)} for path in files],
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/transcribe",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=3600) as response:
        raw = response.read()
    client_wall = time.perf_counter() - start

    result = json.loads(raw)
    perf = result.get("inference_performance", {})
    data = result.get("data", [])
    bad = [item for item in data if item.get("is_bad")]

    print(f"request_files={len(files)}")
    print(f"response_items={len(data)} bad={len(bad)}")
    print(
        "server_performance "
        f"batch_wall_seconds={perf.get('batch_wall_seconds')} "
        f"num_items={perf.get('num_items')} "
        f"throughput_items_per_second={perf.get('throughput_items_per_second')} "
        f"total_audio_duration_seconds={perf.get('total_audio_duration_seconds')} "
        f"speedup={perf.get('speedup')} "
        f"rtf={perf.get('rtf')}"
    )
    print(f"client_wall_seconds={client_wall:.3f}")
    if bad:
        for item in bad[:5]:
            print(
                "bad_item "
                f"reason={item.get('bad_reason')} "
                f"component={item.get('bad_componet') or item.get('bad_component')}",
                file=sys.stderr,
            )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

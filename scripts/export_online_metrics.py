#!/usr/bin/env python3
"""Export online-training log lines to CSV for open-source evidence plots."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


_UPDATE_RE = re.compile(
    r"update=(?P<update>\d+)/\d+\s+"
    r"steps=(?P<steps>\d+)\s+"
    r"episodes=(?P<episodes>\d+)\s+"
    r"collect=(?P<collect>[0-9.]+)s\s+"
    r"infer=(?P<infer>[0-9.]+)s\s+"
    r"(?:stack=(?P<stack>[0-9.]+)s\s+)?"
    r"ppo=(?P<ppo>[0-9.]+)s\s+"
    r"total=(?P<total>[0-9.]+)s\s+"
    r"step/s=(?P<steps_per_sec>[0-9.]+)\s+"
    r".*?loss=(?P<loss>[0-9.]+)\s+"
    r"kl=(?P<kl>[+-]?[0-9.]+)\s+"
    r"entropy=(?P<entropy>[0-9.]+)\s+"
    r"ef=(?P<ef>[+-]?[0-9.]+)"
)

_FIELDS = (
    "update",
    "steps",
    "episodes",
    "collect",
    "infer",
    "stack",
    "ppo",
    "total",
    "steps_per_sec",
    "loss",
    "kl",
    "entropy",
    "ef",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="keep every N-th matched update row (default 1 = all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stride <= 0:
        raise ValueError(f"stride must be > 0, got {args.stride}")
    if not args.log.is_file():
        raise FileNotFoundError(args.log)
    rows: list[dict[str, str]] = []
    with args.log.open() as handle:
        for line in handle:
            match = _UPDATE_RE.search(line)
            if match is None:
                continue
            update = int(match.group("update"))
            if (update - 1) % args.stride != 0:
                continue
            row = {key: (match.group(key) or "") for key in _FIELDS}
            rows.append(row)
    if not rows:
        raise RuntimeError(f"no update rows parsed from {args.log}")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()

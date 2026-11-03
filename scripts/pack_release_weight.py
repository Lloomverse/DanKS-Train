#!/usr/bin/env python3
"""Pack a checkpoint for release under a version and date archive name."""

from __future__ import annotations

import argparse
import hashlib
import re
import zipfile
from pathlib import Path

VERSION = re.compile(r"v\d+\.\d+\.\d+")
DATE = re.compile(r"\d{8}")
BLOCKED = (
    "yu" + "zhenbo",
    "duole" + ".com",
    "cursor" + "agent",
    "top10_selector_online_",
    "/root/",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite a checkpoint archive name to <version>_<YYYYMMDD>.pt."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def pack(source: Path, version: str, date: str, output: Path) -> str:
    if VERSION.fullmatch(version) is None:
        raise RuntimeError(f"version must look like v0.1.1, got {version}")
    if DATE.fullmatch(date) is None:
        raise RuntimeError(f"date must be YYYYMMDD, got {date}")
    if not source.is_file():
        raise RuntimeError(f"source is not a file: {source}")
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    if not output.parent.is_dir():
        raise RuntimeError(f"output directory does not exist: {output.parent}")

    root = f"{version}_{date}.pt"
    with zipfile.ZipFile(source) as zin:
        names = zin.namelist()
        if not names:
            raise RuntimeError("checkpoint archive is empty")
        prefixes = {name.split("/", 1)[0] for name in names}
        if len(prefixes) != 1:
            raise RuntimeError(f"checkpoint archive has multiple roots: {sorted(prefixes)}")
        old = prefixes.pop()
        if old == root:
            raise RuntimeError(f"archive root is already {root}")
        members = [(name, zin.getinfo(name), zin.read(name)) for name in names]

    with zipfile.ZipFile(output, "w") as zout:
        for name, info, data in members:
            if not name.startswith(old + "/"):
                raise RuntimeError(f"unexpected member {name}")
            new_info = zipfile.ZipInfo(
                filename=root + name[len(old):],
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            new_info.compress_type = info.compress_type
            new_info.external_attr = info.external_attr
            new_info.create_system = info.create_system
            new_info.flag_bits = info.flag_bits
            zout.writestr(new_info, data, compress_type=info.compress_type)

    with zipfile.ZipFile(source) as zin, zipfile.ZipFile(output) as zout:
        old_names = zin.namelist()
        new_names = zout.namelist()
        if len(old_names) != len(new_names):
            raise RuntimeError("member count changed")
        for old_name, new_name in zip(old_names, new_names):
            if not new_name.startswith(root + "/"):
                raise RuntimeError(f"member was not renamed: {new_name}")
            if zin.read(old_name) != zout.read(new_name):
                raise RuntimeError(f"member bytes changed: {old_name}")

    raw = output.read_bytes()
    for item in BLOCKED:
        if item.encode() in raw:
            output.unlink()
            raise RuntimeError(f"packed file still contains {item}")
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    args = parse_args()
    digest = pack(args.source.resolve(), args.version, args.date, args.output.resolve())
    print(f"{digest}  {args.output.resolve()}")


if __name__ == "__main__":
    main()

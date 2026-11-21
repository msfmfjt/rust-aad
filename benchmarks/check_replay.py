#!/usr/bin/env python3
"""Validate every external path output and seeded input adjoint in a replay."""
import argparse
import json
import math
from pathlib import Path
import struct


def check(bundle, actual):
    info = json.loads((bundle / "manifest.json").read_text())
    if info["schema"] != 1:
        raise ValueError("unknown replay schema")
    width = info["outputs"] + info["inputs"]
    row = struct.Struct(f"<{width}d")
    tolerance = 2e-13 if info["workload"] == "affine32-v1" else 2e-12
    with (bundle / "expected.f64le").open("rb") as expected, actual.open("rb") as observed:
        for path in range(info["paths"]):
            left, right = expected.read(row.size), observed.read(row.size)
            if min(len(left), len(right)) != row.size:
                raise ValueError(f"short row at path {path}")
            for j, (want, got) in enumerate(zip(row.unpack(left), row.unpack(right))):
                if not (math.isfinite(want) and math.isfinite(got) and abs(want-got) <= tolerance*(1+abs(want))):
                    raise ValueError(f"path {path}, column {j}: {got} expected {want}")
        if expected.read(1) or observed.read(1):
            raise ValueError("unexpected trailing data")
    return info["paths"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("actual", type=Path)
    args = parser.parse_args()
    print(f"Verified {check(args.bundle, args.actual)} paths")

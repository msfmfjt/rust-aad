#!/usr/bin/env python3
"""Compare measured medians after checking workload identity and environment."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import sys

from run import PHASES, matrix

ENVIRONMENT = ("system", "os_version", "cpu", "arch", "cpu_features", "threads",
               "rustc", "profile", "build_env", "matrix")
IDENTITY = ("schema", "paths", "workload", "inputs", "nodes", "seeds", "input_fnv1a64",
            "program_fnv1a64", "verified_paths")


def load(path, backend):
    records = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            r = json.loads(line)
            if r["backend"] != backend:
                continue
            key = r["workload"], r["paths"]
            if key in records:
                raise ValueError(f"duplicate {key}")
            records[key] = r
    if not records:
        raise ValueError(f"{path}: no records for {backend}")
    profile = next(iter(records.values()))["environment"]["matrix"]
    if set(matrix(backend, profile)) != records.keys():
        raise ValueError(f"{path}: incomplete {profile} matrix for {backend}")
    return records


def compare(left, right):
    for key in left.keys() & right.keys():
        a,b = left[key],right[key]
        for k in IDENTITY:
            if a[k] != b[k]:
                raise ValueError(f"{key}: different {k}")
        for k in ENVIRONMENT:
            if a["environment"][k] != b["environment"][k]:
                raise ValueError(f"{key}: different {k}")
        for k in ("output_means", "input_vjp_means"):
            if len(a[k])!=len(b[k]) or any(not math.isfinite(x) or not math.isfinite(y) or
                    abs(x-y)>2e-12*(1+abs(x)) for x,y in zip(a[k],b[k])):
                raise ValueError(f"{key}: differing {k}")
        if set(a["phases"]) != PHASES or set(b["phases"]) != PHASES:
            raise ValueError(f"{key}: missing phases")
    keys = [k for k in left if k in right]
    if not keys:
        raise ValueError("no shared workloads")
    for key in keys:
        a,b = left[key],right[key]
        for phase in sorted(PHASES):
            ta,tb = a["phases"][phase],b["phases"][phase]
            if ta["units"] != tb["units"]:
                # Cold startup processes one native batch, whose width can differ.
                continue
            if any(len(t["samples_ns"]) != 11 or t["median_ns"] <= 0 for t in (ta,tb)):
                raise ValueError(f"{key}: invalid timing samples")
            yield (*key,phase,ta["units"],ta["median_ns"],tb["median_ns"],ta["median_ns"]/tb["median_ns"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left_file",type=Path)
    parser.add_argument("left_backend")
    parser.add_argument("right_file",type=Path)
    parser.add_argument("right_backend")
    args = parser.parse_args()
    rows = list(compare(load(args.left_file,args.left_backend),load(args.right_file,args.right_backend)))
    writer = csv.writer(sys.stdout)
    writer.writerow(("workload","paths","phase","units","left_median_ns","right_median_ns","left_over_right"))
    writer.writerows(rows)

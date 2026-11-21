#!/usr/bin/env python3
"""Run the registered workloads sequentially and save validated JSONL records."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AFFINE = "affine32-v1"
AVERAGES = [f"average-{n}-v1" for n in (1, 12, 120)]
BRANCHES = [f"branch-{n}-v1" for n in (1, 50, 99)]
WIDTHS = [f"width-{n}-v1" for n in (1, 10, 100, 1000)]
CASES = [AFFINE, *AVERAGES, *BRANCHES, *WIDTHS]
PHASES = {"record_compile", "prepare_workspace", "cold_record_vjp", "forward", "forward_vjp"}


def output(*cmd, cwd=ROOT):
    return subprocess.check_output(cmd, cwd=cwd, text=True).strip()


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def host():
    if sys.platform == "darwin":
        cpu = output("sysctl", "-n", "machdep.cpu.brand_string")
        features = output("sysctl", "-n", "machdep.cpu.features")
    elif sys.platform.startswith("linux"):
        info = dict(line.split(":", 1) for line in Path("/proc/cpuinfo").read_text().splitlines() if ":" in line)
        info = {k.strip(): v.strip() for k, v in info.items()}
        cpu = info.get("model name", info.get("Processor", platform.machine()))
        features = info.get("flags", info.get("Features", "unknown"))
    else:
        cpu = platform.processor()
        features = os.environ.get("PROCESSOR_IDENTIFIER", "unknown")
    return {"system": platform.system(), "os_version": platform.version(), "arch": platform.machine(),
            "cpu": cpu, "cpu_features": features, "threads": 1}


def windows_peak(process):
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("page_faults", wintypes.DWORD)] + [
            (k, ctypes.c_size_t) for k in ("peak", "working", "peak_paged", "paged", "peak_nonpaged",
                                           "nonpaged", "pagefile", "peak_pagefile")]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    query = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    query.restype = wintypes.BOOL
    if not query(int(process._handle), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.peak


def observe(cmd):
    # One child per case. A temporary stdout file avoids a pipe-capacity dependency.
    with tempfile.TemporaryFile() as stream, subprocess.Popen(cmd, stdout=stream) as child:
        if os.name == "nt":
            code = child.wait()
            peak = windows_peak(child)
        else:
            _, status, usage = os.wait4(child.pid, 0)
            code = child.returncode = os.waitstatus_to_exitcode(status)
            peak = usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        if code:
            raise subprocess.CalledProcessError(code, cmd)
        stream.seek(0)
        return json.load(stream), peak


def build(manifest, binary, toolchain, target):
    executables = []
    for mode in ("timing", "allocations"):
        directory = target / mode
        cmd = ["cargo", "+" + toolchain, "build", "--release", "--locked", "--manifest-path",
               str(manifest), "--bin", binary, "--target-dir", str(directory)]
        if mode == "allocations":
            cmd += ["--features", "count-allocations"]
        subprocess.run(cmd, cwd=manifest.parent, check=True)
        executables.append(directory / "release" / (binary + (".exe" if os.name == "nt" else "")))
    return executables


def matrix(backend, profile):
    for case in CASES:
        if backend == "rust-simd2" and case not in [AFFINE, *WIDTHS]:
            continue
        sizes = (100,) if profile == "smoke" else (
            (10_000, 100_000, 1_000_000) if case in AVERAGES else (10_000,))
        for size in sizes:
            yield case, size


def validate(timing, allocations, case, paths, backend, fixtures):
    for record in (timing, allocations):
        if (record["schema"], record["backend"], record["workload"], record["paths"],
            record["verified_paths"]) != (1, backend, case, paths, paths):
            raise ValueError(f"wrong workload or incomplete verification for {case}")
        if set(record["phases"]) != PHASES:
            raise ValueError("missing phases")
        frozen = fixtures[case]
        if (record["program_fnv1a64"] != frozen["program_fnv1a64"] or
                record["input_fnv1a64"] != frozen["inputs"][str(paths)]):
            raise ValueError(f"fixture digest mismatch for {case}/{paths}")
    if {k:v for k,v in timing.items() if k != "phases"} != {k:v for k,v in allocations.items() if k != "phases"}:
        raise ValueError("instrumentation changed the validated results")
    for phase in PHASES:
        times, alloc = timing["phases"][phase], allocations["phases"][phase]
        if len(times["samples_ns"]) != 11 or times["units"] != alloc["units"]:
            raise ValueError(f"inconsistent samples for {phase}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--backend", action="append", help="default: rust-scalar and rust-simd2")
    parser.add_argument("--manifest", type=Path, default=HERE / "Cargo.toml")
    parser.add_argument("--binary", default="rust-aad-bench")
    parser.add_argument("--toolchain", default="1.98.1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    binaries = build(manifest, args.binary, args.toolchain, manifest.parent / "target" / "matrix")
    metadata = host()
    metadata.update({"revision": output("git", "rev-parse", "HEAD", cwd=manifest.parent),
                     "dirty": bool(output("git", "status", "--porcelain", cwd=manifest.parent)),
                     "rustc": output("rustc", "+" + args.toolchain, "-Vv"),
                     "profile": "release", "matrix": args.profile,
                     "build_env": {k:v for k,v in os.environ.items() if k in ("RUSTFLAGS","CARGO_ENCODED_RUSTFLAGS")
                                   or k.startswith(("CARGO_PROFILE_RELEASE_", "CARGO_TARGET_"))},
                     "lock_sha256": sha256(manifest.parent / "Cargo.lock"),
                     "binary_sha256": {name:sha256(path) for name,path in zip(("timing","allocations"), binaries)}})
    fixtures = json.loads((HERE / "fixtures.json").read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a complete matrix. Preserve any previous result after failure.
    with tempfile.TemporaryDirectory(dir=args.output.parent) as staging:
        staged = Path(staging) / "matrix.jsonl"
        with staged.open("w", encoding="utf-8") as stream:
            for backend in args.backend or ["rust-scalar", "rust-simd2"]:
                for case, paths in matrix(backend, args.profile):
                    print(f"{backend} {case} {paths:,}", file=sys.stderr, flush=True)
                    (timing, peak), (allocations, _) = [observe([binpath, backend, case, str(paths)]) for binpath in binaries]
                    validate(timing, allocations, case, paths, backend, fixtures)
                    timing.update({"allocation_phases": allocations["phases"], "peak_rss_bytes": peak, "environment": metadata})
                    print(json.dumps(timing, allow_nan=False, separators=(",", ":")), file=stream, flush=True)
        os.replace(staged, args.output)


if __name__ == "__main__":
    main()

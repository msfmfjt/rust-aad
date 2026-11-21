# Reproducible benchmarks

Baseline on `msfmfjt/rust-pricing-library` commit `f82a8b2c3fb2231959ae4104ec30778cdba683a1`. Measure *both* original and independent versions with identical inputs and workloads.

| Workload | Cases |
|---|---|
| Vanilla/Asian payoff | 10k, 100k, 1M paths; 1, 12, 120 observations |
| Branch-heavy barrier/autocall | rare, 50/50 and frequent branch selection; invalid inactive branches |
| Bergomi path | 1F/2F, 120 steps, price+all first-order risks |
| LSV calibration | finite particle calibration VJP, external to generic tape |
| Hull-White hybrid | path+calibration+market mapping, application-only |

Collect graph compilation time, initial workspace allocation, warm forward-only, warm forward + all-gradient VJP, per-call allocations, peak RSS, confidence bands and CPU features, with at least ten warmed iterations. Separate model-specific costs from generic graph cost. Require exact replay agreement where the original contracts require it; otherwise document numeric tolerances. Vendor AADC/XAD numbers are third-party data until both run on the same host with the same workload and equivalent differentiation semantics.

## P0 gate

Use full application fixtures for application-level performance comparisons. The numeric kernel below provides a portable first workload.

## Initial executable benchmark

`cargo run --release --example baseline -- 100000` runs a 32-observation arithmetic-average payoff graph with fixed deterministic inputs. It reports warmed forward-only and forward+VJP median nanoseconds for seven repetitions as a JSON line. CI includes a manual `workflow_dispatch` baseline job that stores the JSON line and host/toolchain metadata as artifacts.

This is an **interpreter-only smoke benchmark**, not a valid AADC/XAD or pre-extraction Pricing comparison. GitHub-hosted runners have variable performance; compare release builds on the same pinned host, and benchmark allocations/peak memory separately. The warm loop includes input construction, graph evaluation and result reads, not graph compilation.

### Workspace allocation measurement

Run `cargo test --release --locked --test allocations`. Its standalone process
counts system-allocator requests for workspace construction and for reuse.
The 1,030-node fixture alternates a 1,024-add chain and a short lazy branch.
On aarch64 macOS (rustc 1.92.0, release), the original stack at `2d3d149`
requested 99,942 bytes across seven workspace allocations. Storing graph-local
16-bit slots and reserving two pending tasks per node reduced this to 34,022
bytes across seven allocations on the same fixture. Both versions recorded
zero allocations and reallocations in 256 forward calls and separately in 256
VJP calls. These numbers measure requested heap bytes, excluding graph storage,
allocator overhead and process RSS.

### Fixed numeric workload for scalar, SIMD and external replay

Run `cargo run --locked --release --features simd --example fixed_batch`.
The workload ID is `affine32-v1`. On one thread, generate 10,000 paths indexed
from zero. Each row contains two binary64 numbers, in this order:

```text
x = ((47*i mod 1021) - 510) / 256
q = 1 + ((31*i mod 127) - 63) / 4096
```

Concatenate little-endian f64 bytes in row order. The FNV-1a 64-bit input
digest is `a7bf6211c9ab67d9`. Record two input nodes, a single constant 1/64,
and 32 separate multiplications and additions for `s = s*q + 1/64`, starting
from `s = x`. The two outputs are `[s, s*x]` (68 nodes). The VJP seeds are
`[1, -0.5]`. Keep f64, separate multiply/add, and all paths, outputs and input
adjoints in an external implementation. Before timing, compare every value and
gradient with the independent forward derivative recurrence in the example.
The tolerance is `2e-13 * (1 + abs(reference))`.

Prepack inputs in each backend's native layout, compile once, and allocate one
workspace. Time forward and forward-plus-VJP separately after one warm-up for
each case. JSONL reports minimum, median and maximum wall time in nanoseconds
over eleven samples of the entire 10,000-path workload. Timings exclude graph
construction, data conversion, validation against the oracle and workspace
creation. Library range/shape checks remain enabled. The simd2 executor also
prepares a static tape, so backend timing differences include dispatch and
scheduling as well as packed arithmetic.

PR runs retain the four records and host/toolchain metadata per OS as
`affine32-<os>` artifacts. For an AADC or XAD comparison, record the actual
library version and compiler/optimization flags, repeat on the same host and
report recording/JIT costs separately. The repository currently measures Rust
backends only.

## Local benchmark matrix

From the repository root, use Python 3.9+ and Rust 1.98.1 (stable).
The runner builds a separate, locked benchmark package. The production crate
keeps its existing dependencies and minimum Rust version.

```sh
python3 benchmarks/run.py --profile smoke --output benchmarks/results/smoke.jsonl
python3 benchmarks/run.py --profile full --output benchmarks/results/full.jsonl
```

| Fixture ID | Full run | Computation and derivative |
|---|---:|---|
| `affine32-v1` | 10,000 paths | Two outputs and the seeded derivative specified above |
| `average-{1,12,120}-v1` | 10,000, 100,000 and 1,000,000 paths for each observation count | Positive part of the average minus 100, left derivative operand on a tie |
| `branch-{1,50,99}-v1` | 10,000 paths each | Twelve nested lazy selections, exactly 1%, 50% or 99% of paths hit an early exit. Inactive branches contain invalid logarithms |
| `width-{1,10,100,1000}-v1` | 10,000 paths each | Sum and sum of squares, output seeds 0.5 and −0.25, all input sensitivities |

Smoke uses 100 paths for each shape. The scalar backend covers all cases. The
two-lane backend covers affine and width cases, which use supported arithmetic.
The full matrix has 17 scalar and five SIMD records. The largest input matrix
occupies 960 MB. Run cases sequentially on an idle host.

Each JSON line describes one backend, graph and path count. Input and graph
digests are checked against `benchmarks/fixtures.json`. Each output and seeded
input adjoint on every path must agree with a separate formula before timing.
The absolute tolerance is scaled by `1 + abs(reference)` with coefficient
2e−13 for affine and 2e−12 elsewhere. Per-coordinate means are for inspection.

The five phases are recording/compilation from the portable program, workspace
preparation, fresh graph + workspace + first VJP on one native batch, warmed
forward over all paths, and warmed forward including the full seeded reverse
sweep. Data generation and validation are outside the timers. Each phase has
one warm-up followed by 11 samples. The median interval uses the second and
tenth order statistics (98.83% distribution-free coverage for an independent,
continuous sample). Samples from one process can be correlated. The binary
reports the original samples as well as the interval. The native batch has
one scalar or two SIMD paths. RSS covers the entire timing process, including
inputs and correctness checks. Requested heap allocation counts and bytes come
from a separate instrumented release executable. They exclude allocator overhead.
`environment` records the OS, CPU, compiler, release build overrides, source
revision, dirty flag, lockfile and executable digests. The runner also accepts
`--manifest`, `--binary` and `--backend` for Rust reference adapters implementing
the same measurement and validation contract.

### External replay

Export a bundle into a new directory, for example:

```sh
cargo +1.98.1 run --locked --manifest-path benchmarks/Cargo.toml --release -- --export branch-50-v1 10000 branch-replay
python3 benchmarks/check_replay.py branch-replay external-results.f64le
```

`program.json` lists nodes in dependency order, output indices and VJP seeds.
Each node has one opcode and its zero-based operands or immediate literal.
`greater` returns a zero or one and has zero adjoint. `select` evaluates only
the chosen arm. `max` sends the adjoint to the left operand on a tie.
`inputs.f64le` holds contiguous path-major rows of little-endian binary64.
Emit one row per path containing all output values in program order followed
by all derivatives of the seeded output sum in input order. The checker
validates this stream against `expected.f64le` and rejects nonfinite, missing
and trailing data. `manifest.json` records dimensions and the byte digests.
Apply the same precision, graph, seeds and correctness gate before timing any
external engine. Record its version, compilation settings, threading, SIMD
width, timing boundary and any retaping in its report. Keep application-level
calibration and model benchmarks in the separate Pricing Library.

Compare two complete local reports with the same host, compiler, build flags,
verified fixtures and sample counts. For instance:

```sh
python3 benchmarks/compare.py benchmarks/results/full.jsonl rust-scalar benchmarks/results/full.jsonl rust-simd2
```

CSV ratios are left median divided by right median. Only common workloads
with equal phase units appear. The one-batch cold measurements for different
lane widths remain available separately in the original JSONL.

### Public XAD tape replay

The optional C++ driver runs the two-output affine fixture on the public XAD
2.2.0 tape. XAD is built from a separate checkout. With CMake, a C++20
compiler, and the benchmark binary built as above:

```sh
git clone --depth 1 --branch v2.2.0 https://github.com/auto-differentiation/xad.git /tmp/xad-2.2.0
cmake -S benchmarks/xad -B /tmp/xad-affine -DXAD_SOURCE_DIR=/tmp/xad-2.2.0 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/xad-affine --target xad-affine --config Release
cargo +1.98.1 run --locked --release --manifest-path benchmarks/Cargo.toml -- --export affine32-v1 10000 /tmp/affine32-replay
/tmp/xad-affine/xad-affine /tmp/affine32-replay /tmp/xad-output.f64le
python3 benchmarks/check_replay.py /tmp/affine32-replay /tmp/xad-output.f64le
```

The driver validates both outputs and both input adjoints on every path. It
reuses a tape object and records each path afresh. Its eleven timed sweeps
include recording, forward evaluation, reverse and storing every path result.
This differs from the Rust prepared-graph phases. The single-threaded snapshot
below reports these phase boundaries separately.

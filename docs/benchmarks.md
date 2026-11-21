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
report recording/JIT costs separately. The local matrix and XAD tape replay below extend this original fixture.

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

## Measured local snapshot, 2026-11-21

Apple M3 Max (arm64), single worker, Rust 1.98.1 release, with default Rust
target flags. The generic Rust suite used b7d15ce and the Pricing reference
used 16799c5, whose production crates match f82a8b2. Each row below gives the
median of 11 warmed samples. On the 10,000-path affine32-v1 graph:

| Backend | Forward over all paths | Forward + seeded VJP | Hot VJP allocations |
|---|---:|---:|---:|
| rust-aad scalar | 4.675 ms | 6.953 ms | 0 |
| rust-aad SIMD, two paths per batch | 1.447 ms | 3.227 ms | 0 |
| Pricing Library native | 3.637 ms | 10.721 ms | 60,000 |

At 120 observations and one million paths, scalar rust-aad took 1.260 s
forward and 1.948 s forward + VJP. Pricing took 0.648 s and 2.045 s.
The timing-process peak RSS values were 962,428,928 and 963,018,752 bytes,
including 960 MB of prepacked inputs. Across the full 22 rust-aad records,
each warmed forward and forward + VJP reported zero allocations and
reallocations. These measurements apply to the registered shapes on this host.

The public XAD 2.2.0 tape driver validated all 40,000 affine output and adjoint
values and measured 2.592 ms for its record/forward/reverse/store pass. The
XAD driver uses Apple Clang 21 with the flags in its result file. The
standalone Rust timings use reusable compiled graphs. AADC and XAD JIT
measurements remain open.

Native application medians were 0.548 ms and 0.539 ms for complete 32-path
forward + path pullback on the 120-step one- and two-factor Bergomi fixtures.
On 256 particles, finite LSV calibration took 0.065 ms and the calibration
VJP took 0.029 ms. Hull–White with calibrated LSV took 0.422 ms for price and
all native risks over 64 independent antithetic units. These are measurements
of the existing Pricing Library implementations.

Full records include individual samples, intervals, memory and numerical
outputs. The Rust matrices are gzipped JSON Lines and can be passed directly to
`benchmarks/compare.py`. The application fixture and
native comparison sources are on the separate Pricing integration PR.

- [Generic Rust measurements](../benchmarks/evidence/m3-max-2026-11-21/local-rust.jsonl.gz)
- [Native Pricing measurements](../benchmarks/evidence/m3-max-2026-11-21/local-pricing.jsonl.gz)
- [Native application measurements](../benchmarks/evidence/m3-max-2026-11-21/local-models.jsonl.gz)
- [XAD tape measurements and build flags](../benchmarks/evidence/m3-max-2026-11-21/xad-tape.json)
- [Pricing reference and application fixtures](https://github.com/msfmfjt/rust-pricing-library/tree/16799c5bfdac3be1e3a06fa996fe8e35836221ad/verification/standalone-aad)

## Same-host comparison with an external compiled AD engine

Start with 10,000 paths on each registered graph. The affine and width cases
exercise multiple seeded outputs. The average and branch cases cover the
payoff kink and lazy, domain-invalid arms. Restrict each paired timing to
semantics and arithmetic both engines support. Compare scalar against scalar
and report the best supported single-worker SIMD configuration separately.
Run the two engines on the same physical host, without concurrent benchmark
jobs. Record the CPU, OS, compiler, optimisation flags, SDK and engine
versions, actual execution backend, lane count, worker count and binary or
package digest. Record a fallback or unsupported operation explicitly.

Use these two batch boundaries for a public batch API that materialises
results. Inputs and seeds are prepacked, the workspace and destination arrays
exist before the timer, and the timer covers all 10,000 paths.

| Phase | Work completed before stopping the timer |
|---|---|
| `forward_store_outputs` | Compute and store all primal outputs, path-major |
| `forward_vjp_store_gradients` | Evaluate primal and seeded reverse once, then store every input adjoint, path-major |

One VJP seed per output defines the weighted sum. A full Jacobian requires
different work. If a binding also returns primal outputs in the second phase,
allocates returned arrays, or requires per-path language calls, record those
costs. A timing with different output or allocation boundaries belongs in a
separate API-level result. Keep initial recording, JIT compilation, workspace
preparation, input packing and file I/O outside these two timers. Report
setup and first use independently. Confirm actual compiled execution and
separate any interpreter fallback.

For example, run the following from the repository root with fresh output
paths. The binary's optional final argument enables the two stored-result
phases. It writes the last observed output and gradient for every path.

```sh
cargo +1.98.1 build --locked --release --manifest-path benchmarks/Cargo.toml
BIN=benchmarks/target/release/rust-aad-bench
"$BIN" --export affine32-v1 10000 affine-inputs
"$BIN" rust-simd2 affine32-v1 10000 rust-results.f64le > rust-timing.json
python3 benchmarks/check_replay.py affine-inputs rust-results.f64le
```

Pass the same `program.json`, `inputs.f64le` and seeds to the external
driver. Check its path-major output using the last command with its result
file. Match the graph and input digests against `benchmarks/fixtures.json`.
For performance comparisons, retain one warm-up and 11 timed samples per
phase per engine. Repeat in alternating engine order to assess run-order
effects. Report both medians, raw samples and the ratio of medians with its
direction specified. If setup is material, compare total time for a specified
number of replays as measured setup plus that number times the warm batch
time. Figures from other hardware or different workloads cannot establish
performance parity on this matrix.

## Scalar traversal tuning on Apple M3 Max

On the same Apple M3 Max (macOS 27.0.1), compare `bb27c46` and `6c07966`
with Rust 1.98.1 release builds, default target flags and one worker. Both use
the same locked benchmark package and prepacked binary64 inputs. Fixed scalar
graphs now compile their reachable execution order once. Dynamic graphs
retain a compact stack of suspended parents and a 16-bit active-node trace.

| Scalar workload | Paths | Forward before → after | Forward + VJP before → after |
|---|---:|---:|---:|
| Two-output affine, 32 steps | 10,000 | 4.37 → 1.85 ms | 6.87 → 4.29 ms |
| Average, 120 observations | 10,000 | 12.41 → 4.68 ms | 19.26 → 8.27 ms |
| Two outputs, 1,000 inputs | 10,000 | 207.52 → 74.13 ms | 352.98 → 175.76 ms |
| Lazy branch, 50% early exit | 10,000 | 4.20 → 4.43 ms | 6.57 → 6.14 ms |
| Average, 120 observations | 1,000,000 | 1.275 → 0.483 s | 1.936 → 0.858 s |

The 10,000-path matrix covers 11 scalar and five SIMD cases. Each binary ran
one warm-up and 11 samples per phase. Four paired runs alternated the order
of binaries and workloads. The table reports the median of the four per-run
medians. The million-path case ran one pair, baseline first. For reproduction,
build each revision with `cargo +1.98.1 build --release --locked --manifest-path
benchmarks/Cargo.toml` and run its `rust-aad-bench BACKEND WORKLOAD PATHS`
executable for each case. Recording, workspace setup and validation are outside
the two displayed timers.

Recording/compilation for the 120-observation graph increased from 1.23 to
2.38 µs. Workspace construction for that graph fell from 9,086 to 4,904
requested bytes and from seven to four allocations. The separate 1,030-node
lazy allocation regression fell from 34,022 to 21,662 requested bytes. Across
all 16 cases, both binaries recorded zero allocations and reallocations in
warm forward and VJP, including when writing results into preallocated arrays.
Every output and seeded input derivative agreed with its oracle. The complete
10,000-path result streams from the two revisions were byte-identical for
each backend and workload.

Gains varied with the graph. Forward alone took about 4–5% longer on the 50%
and 99% early-exit cases. Both saw about 7% higher throughput for forward +
VJP. The two-lane SIMD arithmetic kernels used their existing execution
strategy. Their warm VJP medians ranged from nearly unchanged on affine to
7% longer on the 100-input case.

[Raw timings, setup and allocation counts, result digests and build metadata](../benchmarks/evidence/m3-max-2026-11-21/scalar-traversal.json.gz)

## Ordered scalar sums and dot products on Apple M3 Max

The 120-observation graph at one million paths took 830.31 ms for forward
plus VJP at `68208da` and 172.58 ms at `d04989d`, a 4.81× throughput gain.
This measures repeated evaluation of an already-compiled graph with one
workspace. The compiler now recognizes runs of additions and dot products
and emits loops that retain the original order of floating-point operations.

| Scalar workload | Paths | Forward before → after (ms) | Forward + VJP before → after (ms) | VJP including forward, speedup |
|---|---:|---:|---:|---:|
| Average, 120 observations | 1,000,000 | 455.13 → 123.57 | 830.31 → 172.58 | 4.81× |
| Average, 12 observations | 10,000 | 0.666 → 0.208 | 1.093 → 0.430 | 2.54× |
| Sum and sum of squares, 100 inputs | 10,000 | 7.30 → 1.16 | 17.07 → 2.61 | 6.54× |
| Sum and sum of squares, 1,000 inputs | 10,000 | 74.41 → 16.30 | 177.70 → 29.21 | 6.08× |
| Lazy branch, 50% early exit | 10,000 | 4.26 → 4.23 | 6.12 → 5.56 | 1.10× |

At 10,000 paths, the geometric mean of the per-case forward + VJP speedups
was 2.06× across 11 scalar cases, 1.01× across five two-lane SIMD cases and
1.65× across all 16. The one-input scalar average and width cases took
2–3% longer. The scalar affine case stayed within 1%. Recording and compiling
the 120-observation graph took 2.42 → 2.92 µs in the million-path runs.
Preparing its scalar workspace requested 4,904 bytes in each revision.

Both builds used Rust 1.98.1, the locked benchmark dependencies, the default
release profile, and one worker on macOS 27.0.1. Each phase ran one warm-up
and 11 timed samples per process. Each binary ran four times per 10,000-path
case and twice on the million-path case. The starting binary alternated
between rounds. Values above are medians of process medians. The replay
timers exclude setup, input generation, validation and copying all path
results into long-lived arrays.

Every path's outputs and seeded input derivatives passed the independent
numerical checks. Separate allocation-instrumented runs of all 16 cases at
10,000 paths produced byte-identical result streams across revisions. Both
versions recorded zero allocations and reallocations during repeated forward
and VJP, including the phases that copied results into preallocated arrays.

To repeat the million-path measurement, build and run the following at each
revision on the same host. The JSON includes individual samples and setup times.

```sh
cargo +1.98.1 build --locked --release --manifest-path benchmarks/Cargo.toml
benchmarks/target/release/rust-aad-bench rust-scalar average-120-v1 1000000
```

[All timing samples, allocation counts, result digests and build metadata](../benchmarks/evidence/m3-max-2026-11-21/scalar-reductions.json.gz)

## Scalar sum chains and two-lane reductions on Apple M3 Max

At one million paths, the 120-observation graph took **173.22 → 52.25 ms**
for a forward evaluation and seeded VJP, or **3.32×** the throughput of
`a4e4d28`. Forward alone took 121.78 → 38.82 ms. The new core revision
was `53c4b21`. Both release builds used Rust 1.98.1, one worker, default
target flags and macOS 27.0.1.

| Workload | Paths | Measurement | Before (ms) | After (ms) | Speedup |
|---|---:|---|---:|---:|---:|
| Scalar average, 120 inputs | 1,000,000 | Forward + VJP | 173.22 | 52.25 | 3.32× |
| Scalar average, 120 inputs | 10,000 | Forward + VJP and copy every gradient | 1.864 | 0.902 | 2.07× |
| Scalar average, 120 inputs | 10,000 | Forward + VJP with a different seed per path | 1.852 | 0.756 | 2.45× |
| SIMD sum and sum of squares, 100 inputs | 10,000 | Forward + VJP | 11.454 | 2.715 | 4.22× |
| SIMD sum and sum of squares, 1,000 inputs | 10,000 | Forward + VJP | 109.70 | 26.75 | 4.10× |
| SIMD sum and sum of squares, one input | 10,000 | Forward + VJP | 0.181 | 0.201 | 0.90× |

The registered average uses a unit seed and the ordinary VJP returns a
workspace slice. Its derivatives are uniform across the inputs. The workspace
rewrites the vector when its value changes. The copy run additionally wrote
every gradient into a preallocated path-major array. The variable-seed
example used the same graph and rows with seed `1 + path_index / 10000`,
checking each gradient against the independently seeded formula before timing.

Across the 16 registered 10,000-path cases, the sum of the forward + VJP
medians fell from 182.02 to 84.46 ms (2.15×). Wide SIMD graphs contribute
most of that time. Giving every case equal weight, the geometric mean speedup
was 1.55× overall, 1.30× for 11 scalar cases and 2.27× for five SIMD cases.
The three lazy scalar cases took roughly 1–3% longer. Preparing the 1,000-input
SIMD executor requested 308,264 bytes across nine allocations, up from 304,120
bytes across seven. All 16 workloads recorded zero warm allocations and
reallocations in both builds, including the stored-result phases.

Each process ran one warm-up and 11 samples per phase. The table uses the
median of the process medians. The binaries ran in alternating order across
four pairs per registered 10,000-path case, two pairs for the million-path
case, two pairs for stored results, and eight pairs for variable seeds.
Recording, workspace preparation, numerical checks and disk I/O were outside
the replay timers. Every timed workload passed its pathwise numerical checks.
At 10,000 paths, the complete value and VJP streams were byte-identical on
each backend across revisions.

Run the million-path case using the release command above. Supply a new
output file as the last argument for the stored-result measurements. To run
the variable-seed control, use the same example source at both revisions.

```sh
benchmarks/target/release/rust-aad-bench rust-scalar average-120-v1 10000 values-and-gradients.f64le
cargo +1.98.1 run --locked --release --manifest-path benchmarks/Cargo.toml --example variable_seeds
```

[Raw timings, setup costs, allocation counts, output digests and build identifiers](../benchmarks/evidence/m3-max-2026-11-21/checked-replay.json.gz)

## Exact sums, repeated arithmetic and lazy branches on Apple M3 Max

Against `081579f`, revision `34d57af` reduced the 120-observation,
one-million-path forward + VJP from **52.39 to 23.90 ms (2.19×)**.
Forward alone took 38.79 to 19.20 ms. Both builds ran the same scalar API on
one worker with Rust 1.98.1, default release flags and macOS 27.0.1.

| Forward + VJP workload | Paths | Before (ms) | After (ms) | Speedup |
|---|---:|---:|---:|---:|
| Scalar average, 120 observations | 1,000,000 | 52.395 | 23.898 | 2.19× |
| Scalar average, 120 observations, copy every input gradient | 10,000 | 0.887 | 0.405 | 2.19× |
| Scalar affine recurrence, two outputs | 10,000 | 4.257 | 1.116 | 3.81× |
| Scalar lazy branch, 50% early exit | 10,000 | 5.661 | 3.905 | 1.45× |
| Scalar sum and sum of squares, 1,000 inputs | 10,000 | 26.051 | 8.978 | 2.90× |
| SIMD sum and sum of squares, 1,000 inputs | 10,000 | 26.743 | 12.091 | 2.21× |

The original average fixture uses binary fractions on a common grid. Its
inputs satisfy the runtime check for exact integer summation. At 10,000 paths
with a different VJP seed per row, it took 0.754 to 0.334 ms (2.26×). A
control filled the low 30 mantissa bits with deterministic pseudorandom data
to exercise the floating-point fallback. With a unit seed, the control took
0.519 to 0.540 ms. With varying seeds it took 0.754 to 0.802 ms. These
represent 4% and 6% longer elapsed times. All control outputs and input
adjoints passed the pathwise formula checks.

Across the 16 registered 10,000-path cases, the sum of forward + VJP medians
fell from 84.50 to 38.92 ms (2.17×). The geometric mean of the 16 speedups
was 1.68×, with 1.60× across 11 scalar cases and 1.88× across five SIMD
cases. The 12-observation scalar average took 7% longer for forward + VJP
and 29% longer for forward alone. Compiling the 120-observation graph took
3.13 to 4.35 µs in the million-path runs. At 50% early exit, branch graph
compilation took 0.79 to 2.54 µs and workspace preparation requested 2,422
bytes, down from 2,666.

Each process ran a warm-up and 11 timed samples. Both the 10,000-path matrix
and the million-path case used four pairs of processes with alternating
binary order. Stored-gradient measurements used two pairs. The precision and
seed controls used eight pairs each. Timings above are medians of process
medians. The replay timers exclude setup and verification. Stored-result
timers include copies into preallocated arrays and exclude file I/O.

In separate allocation runs, both revisions produced byte-identical streams
of values and VJPs at 10,000 paths for all 16 cases. All warm replay phases,
including result storage, recorded zero allocations and reallocations.
Use the existing release benchmark commands above to repeat the registered
workloads. For the precision controls, use the example from `34d57af` in
both builds and run these commands in each checkout.

```sh
cargo +1.98.1 run --locked --release --manifest-path benchmarks/Cargo.toml --example variable_seeds -- --full-precision --unit-seed
cargo +1.98.1 run --locked --release --manifest-path benchmarks/Cargo.toml --example variable_seeds -- --full-precision
```

[Paired samples, numerical checks, result hashes, setup and allocation measurements](../benchmarks/evidence/m3-max-2026-11-21/ordered-fold.json.gz)

## Losslessly prepared input paths on Apple M3 Max

On macOS 27.0.1, one worker on an Apple M3 Max evaluated the registered
`average-120-v1` workload with Rust 1.98.1 release binaries and default target
flags. Both revisions read the same path-major inputs, preserve ordered f64
semantics and return a borrowed output or gradient. We measured the merge
commit `84fb9a5` and implementation `c8a628a`. Each cell below is the median
of four process medians, each based on 11 timed evaluations after one warmup.
The three binaries ran in varying order. Graph construction, input generation
and oracle checks precede these timings.

| Operation | Previous f64 | Current f64 | Prepared inputs | Previous / prepared |
|---|---:|---:|---:|---:|
| Forward, 1M paths | 19.259 ms | 16.420 ms | 7.207 ms | 2.67× |
| Forward + VJP, 1M paths | 24.028 ms | 20.910 ms | 11.034 ms | 2.18× |
| Forward + VJP, 10k paths | 0.239 ms | 0.207 ms | 0.109 ms | 2.20× |
| Forward + VJP, 10k paths, write all gradients | 0.405 ms | 0.367 ms | 0.293 ms | 1.38× |

The per-round warm VJP ratios for one million paths ranged from 2.14×
to 2.19×. Every value in this fixture fits exactly on a common binary
grid. Converting the existing `Vec<f64>` took 88.5 ms (median of four).
Its 960 MB of inputs occupy 240 MB after conversion. Both buffers coexist
during packing. The representation stores individual input values and can be
shared by workers or different graphs. Every call reads its row and computes
its result. An eligible scalar sum uses an integer reduction because every
ordered prefix fits exactly. A different graph expands the row into worker
scratch. Workspace construction allocates an additional zero-gradient array
of eight bytes per input on sum-chain graphs.

### Including the initial conversion

These are separate wall-clock measurements starting from the original f64
matrix. They include workspace construction, input conversion for the
prepared path and the indicated number of full VJP sweeps. Timers stop before
teardown. The output seed changes each sweep, cycling through 17 values from
+1 to −1. We alternate process order and report the median of four runs.

| Sweeps of 1M paths | Previous f64 | Prepared inputs | Previous / prepared |
|---:|---:|---:|---:|
| 1 | 26.3 ms | 101.4 ms | 0.26× |
| 8 | 194.0 ms | 180.0 ms | 1.08× |
| 16 | 386.0 ms | 267.7 ms | 1.44× |
| 256 | 6327.4 ms | 3011.7 ms | 2.10× |

### Coverage and controls

The 16 registered scalar and SIMD cases at 10k paths produced byte-identical
output and gradient files across revisions. Their warm allocation counts
were zero. With the ordinary slice API, the sum of their individual VJP
medians was 38.96 ms before and 38.76 ms after (1.01×).
The prepared executor checked all rows in the 11 scalar fixtures against the
slice API with alternating seeds. Its warm forwards, VJPs and retained VJPs
also allocated zero bytes. Benefits depend on the computation. For example,
`width-100-v1` took 0.974 ms on the previous slice API
and 1.110 ms using prepared inputs in a single 11-sample process.

The full-precision control perturbs low significand bits in each input. It
kept all 960 MB in f64 storage and took 54.08 ms per million-path VJP
in one 11-sample process. The evidence also includes eight pairs of 10k-path
slice-API runs with the existing variable-seed and full-precision controls.
[Raw samples, allocation counts, binary hashes and per-case digests](../benchmarks/evidence/m3-max-2026-11-21/prepared-inputs.json.gz).

Run from the repository root. Use `--features count-allocations` for a separate
allocation run. To reproduce the setup-inclusive baseline, copy
`benchmarks/examples/f64_sweeps.rs` and its `support/sweeps.rs` module into a
checkout of `84fb9a5`, then run the last command there.

```sh
cargo +1.98.1 run --locked --release --manifest-path benchmarks/Cargo.toml -- rust-scalar average-120-v1 1000000
cargo +1.98.1 run --locked --release --manifest-path benchmarks/Cargo.toml --example prepared -- average-120-v1 1000000
cargo +1.98.1 run --locked --release --manifest-path benchmarks/Cargo.toml --example prepared -- average-120-v1 1000000 full
cargo +1.98.1 run --locked --release --manifest-path benchmarks/Cargo.toml --example prepared -- --sweeps 1000000 256
cargo +1.98.1 run --locked --release --manifest-path benchmarks/Cargo.toml --example f64_sweeps -- 1000000 256
```

## Three path VJP experiments

One worker on an Apple M3 Max (macOS 27.0.1) ran Rust 1.98.1 release
builds with the default target flags at `1a0f536`. The reference in each
pair uses the existing scalar VJP in the same binary. Each time is the median
of four process medians in milliseconds. Each process warmed up once and timed
11 sweeps. We reversed trial and method order on alternating rounds. All
inputs are ordinary path-major f64 values. Input generation, graph and
workspace construction and correctness checks precede the clock.

### Two paths per SIMD instruction

Each sweep evaluates one million paths with 120 inputs and a unit output
seed. The direct method reads two rows as it sums, preserving addition order
within each lane. The interleaved method copies each pair into reusable
scratch inside the timer, then invokes the packed-pair API. Forward outputs
and every input adjoint in both lanes and both layouts matched scalar f64 bits
for every row. The fixture has an exact binary grid. The full-precision
control perturbs the low 16 significand bits and exercises the ordered f64
fallback in the scalar evaluator.

| Inputs | Scalar ms | Direct SIMD ms | Copy + SIMD ms | Scalar / direct |
|---|---:|---:|---:|---:|
| fixture | 21.31 | 37.20 | 52.68 | 0.57× |
| full | 44.46 | 37.20 | 52.72 | 1.20× |

Use direct row SIMD for the full-precision average on this host. The scalar
exact-sum path wins on the grid fixture. The measurements include the cost
of reading path-major input in all three cases.

### Reusing a forward evaluation

Each sweep uses 10,000 paths and two outputs. The first two seeds select
individual outputs, followed by distinct weighted combinations. Timings
include every forward and reverse sweep. Both methods consume each borrowed
gradient. A separate check verifies forward values and every derivative for
each path and seed against separate scalar calls. Only one path's primal
values and active trace are live at a time.

| Graph | Seed vectors | Repeated VJP ms | Shared forward ms | Repeated / shared |
|---|---:|---:|---:|---:|
| `affine32-v1` | 1 | 1.085 | 1.111 | 0.98× |
| `affine32-v1` | 2 | 2.194 | 1.725 | 1.27× |
| `affine32-v1` | 4 | 4.396 | 2.952 | 1.49× |
| `affine32-v1` | 8 | 8.741 | 5.416 | 1.61× |
| `width-100-v1` | 1 | 0.960 | 0.950 | 1.01× |
| `width-100-v1` | 2 | 1.927 | 1.338 | 1.44× |
| `width-100-v1` | 4 | 3.891 | 2.129 | 1.83× |
| `width-100-v1` | 8 | 7.683 | 3.625 | 2.12× |

### Adding gradients directly to totals

Each sweep evaluates the same one million paths and produces one sum per
input. Both methods use the same checked sequential f64 addition in original
path order. The stored method writes every gradient to a preallocated matrix
before reading it back to sum. The streamed method consumes the borrowed
gradient immediately. Both reproduce the bit pattern from a separate pathwise
sum of scalar VJPs. Timings include the additions, copies and clearing the
totals. Destination allocation is outside the timer.

| Inputs | Store then sum ms | Stream ms | Stored / streamed | Peak process RSS, stored / streamed (MB) |
|---|---:|---:|---:|---:|
| fixture | 69.12 | 51.57 | 1.34× | 1924 / 964 |
| full | 126.57 | 112.08 | 1.13× | 1924 / 964 |

The result buffers use 960,000,960 bytes for the matrix and totals or 960 bytes
for just the totals. Both processes also hold the same 960 MB of inputs and
a per-worker workspace. RSS includes the executable, reference checks and
other live memory. All 26 timed configurations used zero warm allocations in
separate instrumented runs. Output/gradient files in the existing 16-case
10k-path matrix matched revision `5aec0ed` byte for byte.

Run the example in the separate benchmark package. Switch the SIMD method
among `scalar`, `rows` and `interleave`, the multi-seed method between
`repeat` and `shared`, and the aggregate method between `store` and `stream`.
Use either `fixture` or `full` for average inputs. For multiple seeds choose
`affine32-v1` or `width-100-v1` with 1, 2, 4 or 8 directions.

```sh
cargo +1.98.1 run --locked --release --manifest-path benchmarks/Cargo.toml --example path_vjp -- simd rows average-120-v1 1000000 full 1
cargo +1.98.1 run --locked --release --manifest-path benchmarks/Cargo.toml --example path_vjp -- multi shared width-100-v1 10000 fixture 8
cargo +1.98.1 run --locked --release --manifest-path benchmarks/Cargo.toml --example path_vjp -- aggregate stream average-120-v1 1000000 fixture 1
```

[All samples, measured allocations, peak RSS and existing matrix result digests](../benchmarks/evidence/m3-max-2026-11-21/path-vjp.json.gz).

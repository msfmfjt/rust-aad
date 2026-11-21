# rust-aad

An **experimental** Rust reverse-mode AD library for reusable numerical computation graphs. The long-term research target is financial Monte Carlo throughput comparable to MatLogica AADC; **no such performance claim is currently substantiated**. XAD is an architecture reference, not a dependency.

## Status

P1 prototype, not a released library. This bootstrap includes a small, default-feature zero-dependency `f64` graph interpreter with multiple outputs, arbitrary VJP seeds, lazy conditionals, and reusable per-worker workspace. It is **not yet** a full implementation of the parent Pricing Library's payoff graph and does not contain JIT, custom VJP, or checkpointing.

## Example

```rust
use rust_aad_core::GraphBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut b = GraphBuilder::new();
    let spot = b.input();
    let strike = b.input();
    let intrinsic = b.sub(spot, strike)?;
    let zero = b.constant(0.0)?;
    let payoff = b.max(intrinsic, zero)?;
    let graph = b.compile(&[payoff])?;
    let mut workspace = graph.workspace();
    let v = graph.evaluate(&[110.0, 100.0], &mut workspace)?;
    assert_eq!(v, &[10.0]);
    let greeks = graph.vjp(&[110.0, 100.0], &[1.0], &mut workspace)?;
    assert_eq!(greeks, &[1.0, -1.0]);
    Ok(())
}
```

## Develop

```bash
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo doc --no-deps
```

The original Rust Pricing Library is at https://github.com/msfmfjt/rust-pricing-library. Independence from financial types is a design requirement; do not copy product/model types into this crate.

See [architecture](docs/architecture.md) and [benchmark plan](docs/benchmarks.md).

## Important numerical limitations

- Branches are lazy: inactive domain-invalid branches are not evaluated. For `select`, zero means false and any nonzero value means true; there is no adjoint to the selection condition.
- `min`/`max` break ties to the left input. No special pathwise derivative is implied for digital discontinuities.
- Evaluation uses an iterative task stack instead of Rust recursion. The regression suite includes a 50,000-step dependency chain.
- Division reverse uses exponent scaling for extreme finite operands.
- Input validation rejects nonfinite values. The gradient seed and intermediate errors are checked but extreme cancellation/overflow paths require more adversarial testing.
- The core forbids unsafe Rust and has no C++/Python dependencies.

## License

MIT OR Apache-2.0.

## CI and baseline

GitHub Actions tests on Linux, macOS, and Windows. Open **Actions → Rust CI → Run workflow** to run a manual Linux interpreter-only benchmark and retrieve `aad-baseline-linux`. GitHub-hosted runner results are smoke measurements, not stable performance comparison evidence. The benchmark is run locally with `cargo run --release --example baseline -- 100000`.

Graph-local `NodeId`s reject foreign handles; VJP checks nonfinite adjoints. Evaluation is iterative, and the active execution trace is reused by the reverse sweep. No validated AADC parity is claimed.

## Development status (iterative graph evaluator)

Graph evaluation now uses an explicit per-workspace stack rather than recursive Rust calls, preserving lazy selection of the active branch. Workspace identity is checked against the compiled graph, and regression tests cover deep graphs. The project is still experimental and has **not** demonstrated AADC-comparable performance.

## Reverse workspace and active-path sweep (experimental)

Branch-free scalar graphs share a compiled sequence of reachable nodes across workers. Graphs with a reachable `select` use a per-workspace stack and record the nodes executed on each path. Both modes reverse their forward order, preserving floating-point accumulation for shared nodes and repeated outputs. `Workspace::executed_nodes()` reports the forward count, including the completed prefix after a numerical error. The allocator regression measures workspace construction and reuse.

The numeric semantics (tie-breaking for `min/max`, pathwise derivative through the chosen select branch, and error handling) remain experimental.

## Stable numerical adjoints (experimental)

The division VJP computes the denominator derivative with exponent scaling when
intermediate binary64 products or denominator squares overflow or underflow.
The implementation aims to retain mathematically finite derivatives instead of
silently emitting infinities or zeros. Truly nonfinite adjoints remain errors.
Regression tests cover both exponent extremes and workspace recovery after an
error. Correct rounding, cross-platform bitwise identity, and release quality
are **not** claimed; run Rust CI before use.


## Experimental two-path SIMD

Enable `--features simd` and call `CompiledGraph::batch2()` once per worker.
The result owns a prepared two-lane tape. Pass one `[f64; 2]` per input to
`evaluate`, or pass inputs and one seed pair per output to `vjp`. Lane 0 and
lane 1 are independent paths. Dispatch an odd final path to the scalar workspace.
The kernel supports reachable constants, inputs, add, subtract, multiply and
negate. A graph with any other reachable operation returns `None`; use the
scalar evaluator for lazy branches, scaled division and transcendental functions.

The [local benchmark matrix](docs/benchmarks.md#local-benchmark-matrix) checks
every path's values and adjoints, measures setup, forward, VJP, allocations and
peak memory, and exports binary fixtures for replay with other AD libraries.

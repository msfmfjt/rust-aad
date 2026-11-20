# rust-aad

An **experimental** Rust reverse-mode AD library for reusable numerical computation graphs. The long-term research target is financial Monte Carlo throughput comparable to MatLogica AADC; **no such performance claim is currently substantiated**. XAD is an architecture reference, not a dependency.

## Status

P1 prototype, not a released library. This bootstrap includes a small, zero-dependency `f64` graph interpreter with multiple outputs, arbitrary VJP seeds, lazy conditionals, and reusable per-worker workspace. It is **not yet** a full implementation of the parent Pricing Library's payoff graph and does not contain JIT, SIMD batch kernels, custom VJP, or checkpointing. No benchmark results have been produced.

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
- Evaluation uses an iterative task stack instead of Rust recursion. Deep-graph tests exist, but successful compilation and platform behavior still require CI validation.
- Division reverse uses the straightforward formula and does not yet include the Pricing Library's scaled arithmetic for extreme exponents.
- Input validation rejects nonfinite values. The gradient seed and intermediate errors are checked but extreme cancellation/overflow paths require more adversarial testing.
- The core forbids unsafe Rust and has no C++/Python dependencies.

## License

MIT OR Apache-2.0.

## CI and baseline

GitHub Actions tests on Linux, macOS, and Windows. Once this source is pushed to GitHub, open **Actions → Rust CI → Run workflow** to run a manual Linux interpreter-only benchmark and retrieve `aad-baseline-linux`. GitHub-hosted runner results are smoke measurements, not stable performance comparison evidence. The benchmark is run locally with `cargo run --release --example baseline -- 100000`.

Graph-local `NodeId`s reject foreign handles; VJP checks nonfinite adjoints. Evaluation is iterative, and the active execution trace is reused by the reverse sweep. No validated AADC parity is claimed.

## Development status (iterative graph evaluator)

Graph evaluation now uses an explicit per-workspace stack rather than recursive Rust calls, preserving lazy selection of the active branch. Workspace identity is checked against the compiled graph, and regression tests cover deep graphs. The project is still experimental and has **not** demonstrated AADC-comparable performance. A Rust toolchain and CI run are required to validate compilation and behavior.

## Reverse workspace and active-path sweep (experimental)

`CompiledGraph::workspace()` preallocates the primal, adjoint, input-gradient, iterative evaluation stack and active-node buffers. Each forward call clears the active trace and records executed nodes in dependency-first order. VJP traverses only this active trace in reverse, rather than scanning every compiled node. A large branch not selected at runtime is not visited in reverse. `Workspace::executed_nodes()` and `Workspace::capacities()` expose simple diagnostics for regression tests; capacities alone do not prove zero allocations by the system allocator.

The numeric semantics (tie-breaking for `min/max`, pathwise derivative through the chosen select branch, and error handling) remain experimental. The project has not yet been built or benchmarked in this conversation; CI execution is still required.

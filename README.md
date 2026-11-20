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
- This prototype currently uses recursive interpretation (maximum permitted graph nodes is not a stack-depth guarantee). Deep graphs require an explicit iterative scheduler before production use.
- Division reverse uses the straightforward formula and does not yet include the Pricing Library's scaled arithmetic for extreme exponents.
- Input validation rejects nonfinite values. The gradient seed and intermediate errors are checked but extreme cancellation/overflow paths require more adversarial testing.
- The core forbids unsafe Rust and has no C++/Python dependencies.

## License

MIT OR Apache-2.0.

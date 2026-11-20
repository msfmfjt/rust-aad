# Architecture and extraction roadmap

This independent repository owns only generic numerical graph and AD infrastructure. No dependency on pricing, market data, dates, underlying IDs, financial payoff semantics or model calibration.

## Existing baseline

Source: `msfmfjt/rust-pricing-library` at commit `f82a8b2c3fb2231959ae4104ec30778cdba683a1`.

- `engine/aad/workspace.rs`: tile, checkpoint policy, aligned SoA primal/adjoint workspace.
- `engine/payoff/{compile,program,tape}.rs`: source-to-compiled tape, conditional lazy execution, scalar payoff reverse, graph fingerprints.
- `engine/processes/capabilities.rs`: financial model-specific path pullback.
- `engine/calibration/capabilities.rs`: calibration pullback.
- `engine/risk/hull_white/aad.rs`: composes path, payoff, dividend, curve and calibration adjoints.

The bootstrap graph is a deliberately smaller independent prototype; it is **not yet an extraction** of all these capabilities.

## Proposed interfaces

- `GraphBuilder` builds numeric operations with graph-local `NodeId` handles.
- `CompiledGraph` is immutable and should be shareable between threads.
- `Workspace` owns mutable scratch storage; one per worker.
- `evaluate` reuses primal workspace; `vjp` accepts arbitrary output seeds and returns input adjoints.
- Later: iterative execution scheduler, compact register allocation, generalized custom VJP, checkpoint/recompute policies, batched SIMD, optional isolated JIT backend.

## Integration

The financial project maps `(underlying, date, pre/post-dividend)` observations to numeric inputs and maps returned input adjoints back to financial observations. Model-specific VJPs remain in the financial project. First integration must compare prices, Greeks, branch behavior, and replay metadata against the unchanged pricing baseline.

## Safety / compatibility

Keep `#![forbid(unsafe_code)]` in the core. Do not promise bitwise identity across hardware or different backends. Do not publish until performance and numerical contracts, CI, licenses and crates.io naming have been reviewed.

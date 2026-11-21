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
- Iterative evaluation now uses a per-workspace explicit task stack; inactive conditional branches are not evaluated, and the reverse sweep follows visited nodes.
- Workspaces carry a compiled-graph identity, not just shape, and cannot be used with another graph.
- Still pending: compact register allocation, generalized custom VJP, checkpoint/recompute policies, broader SIMD opcode coverage, optional isolated JIT backend.

## Integration

The financial project maps `(underlying, date, pre/post-dividend)` observations to numeric inputs and maps returned input adjoints back to financial observations. Model-specific VJPs remain in the financial project. First integration must compare prices, Greeks, branch behavior, and replay metadata against the unchanged pricing baseline.

## Safety / compatibility

Keep `#![forbid(unsafe_code)]` in the core. Do not promise bitwise identity across hardware or different backends. Do not publish until performance and numerical contracts, CI, licenses and crates.io naming have been reviewed.

## Iterative evaluator constraints (experimental)

The compiler traverses reachable nodes from the declared outputs in left-first dependency order. When this traversal encounters a select, the scalar interpreter evaluates each output on demand. It suspends a parent while computing an unvisited dependency and reads only the chosen arm of a select. A suspended frame holds one 16-bit node index. On graphs with a fixed traversal order, the compiler retains that order and workspaces replay it directly. Both traversals use explicit stacks to handle deep graphs.

Tests include a 50,000-step dependency chain, an invalid inactive branch with 20,000 nested nodes, graph identity checks, and repeated VJP calls.

## Active execution trace for reverse (P1 iteration)

Branch-free graphs store a single forward order on the compiled graph. The SIMD arithmetic executor uses it when preparing its tape. Lazy evaluation records the active order in the workspace for each path. Reversing either order visits every consumer before its dependencies. A node's adjoint is therefore final when it is visited, and the reverse interpreter checks it for finiteness at that point. This also covers repeated output seeds. The primal and adjoint arrays and the input-gradient buffer are reused across calls.

The scalar compiler groups consecutive steps when they form an ordered sum
of contiguous input slots or an ordered dot product over contiguous primal
slots. Each group needs at least four terms. A graph output or another use of
an intermediate ends the group at that intermediate. Shared inputs retain
their primal values for other consumers. The dot product supports overlapping
operand ranges, including sums of squares.

The loops preserve the primitive multiply, add and adjoint accumulation order.
They check the final sum for finiteness. Once an addition produces infinity or
NaN, further additions cannot produce a finite value. A failed check replays
the original schedule to locate the first failure and count the completed nodes.
Reverse checks shared input adjoints in their original visit order.

Static schedules record whether they visit every declared input. Each input
then passes a finiteness check when it is read individually or as part of an
ordered sum. A failed evaluation checks the inputs before reporting the
original failing operation. Graphs with unreachable inputs or lazy branches
continue to validate all inputs up front.

For a single ordered sum followed by a single-output chain of add, subtract,
multiply, minimum or maximum with right-hand constants, the scalar compiler
records the constants and the multipliers in reverse order. A clamp selecting
its constant cuts the derivative to the sum. Ties select the left operand.
Every input has the same derivative, so a workspace can retain the last
uniform gradient vector across calls. Each call reads its inputs and seed.

The compiler bounds reverse growth by a power of two per multiplier and
reserves enough exponent bits to add all contributions to a shared constant.
The fast path checks the seed and each successive primal against the resulting
limit. Outside that interval the primitive reverse rules also check adjoints
at constant nodes and report the original error, if any.

The two-lane arithmetic tape maps the ordered sum and dot-product blocks to
its compact slots during preparation. It preserves multiplication and addition
order independently in each lane, including for overlapping dot operands.

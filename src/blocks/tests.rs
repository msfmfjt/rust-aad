//! Compare optimized schedules against the same graph's original primitive steps.
use super::Block;
use crate::{AadError, CompiledGraph, GraphBuilder, Workspace};

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn check(
    graph: &mut CompiledGraph,
    optimized: &mut Workspace,
    reference: &mut Workspace,
    inputs: &[f64],
    seeds: &[f64],
) -> Result<(), AadError> {
    let value = graph.evaluate(inputs, optimized).map(bits);
    let count = optimized.executed_nodes();
    let vjp = graph.vjp(inputs, seeds, optimized).map(bits);

    let schedule = graph.straight_line.as_mut().unwrap();
    let blocks = std::mem::take(&mut schedule.blocks);
    let chain = schedule.sum_chain.take();
    let covered = std::mem::replace(&mut schedule.covers_all_inputs, false);
    assert_eq!(value, graph.evaluate(inputs, reference).map(bits));
    assert_eq!(count, reference.executed_nodes());
    assert_eq!(vjp, graph.vjp(inputs, seeds, reference).map(bits));
    let schedule = graph.straight_line.as_mut().unwrap();
    schedule.blocks = blocks;
    schedule.sum_chain = chain;
    schedule.covers_all_inputs = covered;
    vjp.map(|_| ())
}

#[test]
fn sums_preserve_intermediate_outputs_errors_and_workspace_reuse() {
    let mut builder = GraphBuilder::new();
    let x: Vec<_> = (0..14).map(|_| builder.input()).collect();
    let mut sum = builder.constant(-0.0).unwrap();
    let mut sums = Vec::new();
    for &input in &x {
        sum = builder.add(sum, input).unwrap();
        sums.push(sum);
    }
    let mut graph = builder.compile(&[sum, sums[6], x[9], sum]).unwrap();
    let blocks = &graph.straight_line.as_ref().unwrap().blocks;
    assert!(
        blocks
            .iter()
            .any(|b| matches!(b, Block::InputSum { shared: true, .. }))
    );
    assert!(
        blocks
            .iter()
            .any(|b| matches!(b, Block::InputSum { shared: false, .. }))
    );
    let (mut ws, mut reference) = (graph.workspace(), graph.workspace());

    let mut inputs = [0.0; 14];
    inputs[7..10].copy_from_slice(&[1e16, -1e16, 1.0]);
    check(
        &mut graph,
        &mut ws,
        &mut reference,
        &inputs,
        &[1.0, 0.5, -1.0, -0.0],
    )
    .unwrap();
    check(&mut graph, &mut ws, &mut reference, &[-0.0; 14], &[-0.0; 4]).unwrap();
    assert_eq!(
        check(
            &mut graph,
            &mut ws,
            &mut reference,
            &inputs,
            &[f64::MAX, 0.0, f64::MAX, 0.0]
        ),
        Err(AadError::NonFiniteResult { node: x[9].0 })
    );
    inputs[3..5].fill(f64::MAX);
    assert_eq!(
        check(&mut graph, &mut ws, &mut reference, &inputs, &[0.0; 4]),
        Err(AadError::NonFiniteResult { node: sums[4].0 })
    );
    check(&mut graph, &mut ws, &mut reference, &[1.0; 14], &[1.0; 4]).unwrap();
}

#[test]
fn dot_preserves_overlapping_operands_shared_products_and_first_failing_node() {
    let mut builder = GraphBuilder::new();
    let x: Vec<_> = (0..16).map(|_| builder.input()).collect();
    // Observing the inputs first makes their primal slots available to the dot.
    let mut outputs = x.clone();
    let mut sum = x[0];
    let mut products = Vec::new();
    for pair in x.windows(2) {
        let term = builder.mul(pair[0], pair[1]).unwrap();
        sum = builder.add(sum, term).unwrap();
        products.push(term);
    }
    outputs.extend([sum, products[6], sum]);
    let mut graph = builder.compile(&outputs).unwrap();
    let blocks = &graph.straight_line.as_ref().unwrap().blocks;
    assert_eq!(
        blocks
            .iter()
            .filter(|b| matches!(b, Block::Dot { .. }))
            .count(),
        2
    );
    let (mut ws, mut reference) = (graph.workspace(), graph.workspace());
    let mut inputs = [1.0; 16];
    inputs[1] = -1e16;
    let mut seeds = [0.0; 19];
    seeds[0] = 1e16;
    seeds[16] = 1.0;
    seeds[17] = -2.0;
    check(&mut graph, &mut ws, &mut reference, &inputs, &seeds).unwrap();
    inputs[10..12].fill(1e200);
    assert_eq!(
        check(&mut graph, &mut ws, &mut reference, &inputs, &[0.0; 19]),
        Err(AadError::NonFiniteResult {
            node: products[10].0
        })
    );
    seeds[16] = f64::MAX;
    seeds[18] = f64::MAX;
    assert_eq!(
        check(&mut graph, &mut ws, &mut reference, &[0.0; 16], &seeds),
        Err(AadError::NonFiniteResult { node: sum.0 })
    );
    check(
        &mut graph,
        &mut ws,
        &mut reference,
        &[-0.0; 16],
        &[-0.0; 19],
    )
    .unwrap();
}

#[test]
fn scalar_chain_matches_primitive_rounding_ties_seeds_and_cold_errors() {
    let mut b = GraphBuilder::new();
    let x: Vec<_> = (0..17).map(|_| b.input()).collect();
    let bias = b.constant(-0.5).unwrap();
    let q = b.constant(-1.0 / 3.0).unwrap();
    let hi = b.constant(0.75).unwrap();
    let mut s = bias;
    for i in x {
        s = b.add(s, i).unwrap();
    }
    s = b.sub(s, bias).unwrap();
    s = b.min(s, hi).unwrap();
    s = b.max(s, bias).unwrap();
    s = b.mul(s, q).unwrap();
    s = b.add(s, bias).unwrap();
    s = b.mul(s, q).unwrap();
    let mut g = b.compile(&[s]).unwrap();
    assert!(g.straight_line.as_ref().unwrap().sum_chain.is_some());
    let (mut w, mut reference) = (g.workspace(), g.workspace());
    let mut state = 43_u64;
    for _ in 0..1024 {
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            f64::from_bits((1023u64 << 52) | (state >> 12)) - 1.5
        };
        let row: [f64; 17] = std::array::from_fn(|_| next());
        check(&mut g, &mut w, &mut reference, &row, &[next()]).unwrap();
    }
    for v in [0.75, -0.5, 0.0, -0.0, 1e190] {
        let mut row = [0.0; 17];
        row[0] = v;
        for seed in [-0.0, 1.0, -1.0, 1e180, 2.0, 2.0, f64::from_bits(1)] {
            check(&mut g, &mut w, &mut reference, &row, &[seed]).unwrap();
        }
    }
    let mut row = [f64::MAX; 17];
    assert!(matches!(
        check(&mut g, &mut w, &mut reference, &row, &[0.0]),
        Err(AadError::NonFiniteResult { .. })
    ));
    row[16] = f64::NAN;
    assert_eq!(
        check(&mut g, &mut w, &mut reference, &row, &[0.0]),
        Err(AadError::NonFiniteInput)
    );
    check(&mut g, &mut w, &mut reference, &[0.0; 17], &[1.0]).unwrap();
}

#[test]
fn cancelling_primals_still_check_shared_constant_adjoints() {
    let mut b = GraphBuilder::new();
    let x: Vec<_> = (0..8).map(|_| b.input()).collect();
    let c = b.constant(0.0).unwrap();
    let sum = x.into_iter().fold(c, |s, i| b.add(s, i).unwrap());
    let out = b.add(sum, c).unwrap();
    let mut g = b.compile(&[out]).unwrap();
    assert!(g.straight_line.as_ref().unwrap().sum_chain.is_some());
    let (mut w, mut reference) = (g.workspace(), g.workspace());
    check(&mut g, &mut w, &mut reference, &[0.0; 8], &[1.0]).unwrap();
    assert_eq!(
        check(&mut g, &mut w, &mut reference, &[0.0; 8], &[f64::MAX]),
        Err(AadError::NonFiniteResult { node: c.0 })
    );
    check(&mut g, &mut w, &mut reference, &[0.0; 8], &[1.0]).unwrap();
}

#[test]
fn paired_moments_replay_overflow_and_refresh_partial_gradients() {
    let mut b = GraphBuilder::new();
    let x: Vec<_> = (0..24).map(|_| b.input()).collect();
    let zero = b.constant(0.0).unwrap();
    let (mut sum, mut squares) = (zero, zero);
    let mut first_square = zero;
    for (i, &leaf) in x.iter().enumerate() {
        sum = b.add(sum, leaf).unwrap();
        let square = b.mul(leaf, leaf).unwrap();
        if i == 0 {
            first_square = square;
        }
        squares = b.add(squares, square).unwrap();
    }
    let mut g = b.compile(&[sum, squares, sum]).unwrap();
    let schedule = g.straight_line.as_ref().unwrap();
    assert!(
        schedule
            .blocks
            .iter()
            .any(|r| matches!(r, Block::Moments(_)))
    );
    assert!(!schedule.reset_ranges.is_empty());
    let (mut w, mut reference) = (g.workspace(), g.workspace());
    let mut inputs: [f64; 24] = std::array::from_fn(|i| (i as f64).sin());
    for seeds in [
        [0.125, -2.25, 0.0],
        [1.0, 0.0, -1.0],
        [0.0; 3],
        [1e85, -1e84, -1e85],
    ] {
        check(&mut g, &mut w, &mut reference, &inputs, &seeds).unwrap();
    }
    inputs.fill(1e152);
    assert_eq!(
        check(&mut g, &mut w, &mut reference, &inputs, &[0.0, 1e200, 0.0]),
        Err(AadError::NonFiniteResult { node: x[23].0 })
    );
    inputs[0] = 1e200;
    assert_eq!(
        check(&mut g, &mut w, &mut reference, &inputs, &[0.0; 3]),
        Err(AadError::NonFiniteResult {
            node: first_square.0
        })
    );
    check(&mut g, &mut w, &mut reference, &[-0.0; 24], &[-0.0; 3]).unwrap();
}

#[test]
fn recurrence_splits_at_observed_states_and_checks_intermediate_adjoints() {
    let mut b = GraphBuilder::new();
    let (x, q, bias) = (b.input(), b.input(), b.input());
    let mut state = x;
    let mut states = Vec::new();
    for _ in 0..20 {
        let product = b.mul(state, q).unwrap();
        state = b.add(product, bias).unwrap();
        states.push(state);
    }
    let mut g = b.compile(&[state, states[10], state]).unwrap();
    assert_eq!(
        g.straight_line
            .as_ref()
            .unwrap()
            .blocks
            .iter()
            .filter(|r| matches!(r, Block::Recurrence(_)))
            .count(),
        2
    );
    let (mut w, mut reference) = (g.workspace(), g.workspace());
    for scale in [-0.0, 0.125, -0.75, 1.0, 1.625] {
        check(
            &mut g,
            &mut w,
            &mut reference,
            &[1.0, scale, 0.125],
            &[1.75, -2.5, 0.3],
        )
        .unwrap();
    }
    assert_eq!(
        check(
            &mut g,
            &mut w,
            &mut reference,
            &[0.0, 2.0, 0.0],
            &[f64::MAX, 0.0, 0.0]
        ),
        Err(AadError::NonFiniteResult { node: states[18].0 })
    );
    check(
        &mut g,
        &mut w,
        &mut reference,
        &[-0.0, -1.0, -0.0],
        &[-0.0; 3],
    )
    .unwrap();
}

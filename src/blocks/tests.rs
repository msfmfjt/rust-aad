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

    let blocks = std::mem::take(&mut graph.straight_line.as_mut().unwrap().blocks);
    assert_eq!(value, graph.evaluate(inputs, reference).map(bits));
    assert_eq!(count, reference.executed_nodes());
    assert_eq!(vjp, graph.vjp(inputs, seeds, reference).map(bits));
    graph.straight_line.as_mut().unwrap().blocks = blocks;
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

use rust_aad_core::{AadError, GraphBuilder};

#[test]
fn weighted_multi_output_vjp_and_workspace_reuse() {
    let mut b = GraphBuilder::new();
    let x = b.input();let y = b.input();
    let xy = b.mul(x,y).unwrap();
    let sum = b.add(x,y).unwrap();
    let g = b.compile(&[xy,sum]).unwrap();let mut w=g.workspace();
    assert_eq!(g.evaluate(&[2.0,3.0],&mut w).unwrap(), &[6.0,5.0]);
    assert_eq!(g.vjp(&[2.0,3.0],&[1.0,2.0],&mut w).unwrap(), &[5.0,4.0]);
    assert_eq!(g.vjp(&[5.0,7.0],&[1.0,0.0],&mut w).unwrap(), &[7.0,5.0]);
}
#[test]
fn select_is_lazy_for_invalid_inactive_branch_and_reverse_tracks_branch() {
    let mut b=GraphBuilder::new();let x=b.input();let z=b.constant(0.0).unwrap();
    let cond=b.greater(x,z).unwrap();
    let positive=b.ln(x).unwrap();let negative=b.neg(x).unwrap();
    let out=b.select(cond,positive,negative).unwrap();
    let g=b.compile(&[out]).unwrap();let mut w=g.workspace();
    assert_eq!(g.evaluate(&[-2.0],&mut w).unwrap(), &[2.0]);
    assert_eq!(g.vjp(&[-2.0],&[1.0],&mut w).unwrap(), &[-1.0]);
    assert_eq!(g.vjp(&[2.0],&[1.0],&mut w).unwrap(), &[0.5]);
}
#[test]
fn min_tie_goes_left() {
    let mut b=GraphBuilder::new();let x=b.input();let y=b.input();
    let o=b.min(x,y).unwrap();let g=b.compile(&[o]).unwrap();let mut w=g.workspace();
    assert_eq!(g.vjp(&[2.0,2.0],&[1.0],&mut w).unwrap(), &[1.0,0.0]);
}
#[test]
fn invalid_shapes_are_rejected() {
    let mut b=GraphBuilder::new();let x=b.input();let g=b.compile(&[x]).unwrap();let mut w=g.workspace();
    assert_eq!(g.evaluate(&[],&mut w).unwrap_err(), AadError::InputShape);
    assert_eq!(g.vjp(&[2.0],&[],&mut w).unwrap_err(), AadError::SeedShape);
}

#[test]
fn foreign_node_handles_are_rejected_even_if_indices_match() {
    let mut left = GraphBuilder::new();
    let foreign = left.input();
    let mut right = GraphBuilder::new();
    let local = right.input();
    assert_eq!(right.add(local, foreign).unwrap_err(), AadError::InvalidNode);
    assert_eq!(right.compile(&[foreign]).err(), Some(AadError::InvalidNode));
}

#[test]
fn nonfinite_intermediate_adjoint_is_reported() {
    let mut b = GraphBuilder::new();
    let x = b.input();
    let y = b.input();
    let product = b.mul(x, y).unwrap();
    let graph = b.compile(&[product]).unwrap();
    let mut workspace = graph.workspace();
    // The primal is finite, but d(x*y)/dy * seed overflows binary64.
    assert!(matches!(
        graph.vjp(&[1e308, 1e-308], &[1e308], &mut workspace),
        Err(AadError::NonFiniteResult { .. })
    ));
}

#[test]
fn repeated_workspace_calls_do_not_leak_old_branches_or_gradients() {
    let mut b = GraphBuilder::new();
    let x = b.input();
    let zero = b.constant(0.0).unwrap();
    let condition = b.greater(x, zero).unwrap();
    let square = b.mul(x, x).unwrap();
    let negative = b.neg(x).unwrap();
    let output = b.select(condition, square, negative).unwrap();
    let graph = b.compile(&[output]).unwrap();
    let mut w = graph.workspace();
    for (input, expected) in [(2.0, 4.0), (-3.0, -1.0), (5.0, 10.0), (-2.0, -1.0)] {
        assert_eq!(graph.vjp(&[input], &[1.0], &mut w).unwrap(), &[expected]);
    }
}

#[test]
fn very_deep_graph_uses_explicit_stack_not_native_recursion() {
    let mut b = GraphBuilder::new();
    let x = b.input();
    let one = b.constant(1.0).unwrap();
    let mut y = x;
    for _ in 0..50_000 { y = b.add(y, one).unwrap(); }
    let g = b.compile(&[y]).unwrap();
    let mut w = g.workspace();
    assert_eq!(g.evaluate(&[2.0], &mut w).unwrap(), &[50_002.0]);
    assert_eq!(g.vjp(&[2.0], &[1.0], &mut w).unwrap(), &[1.0]);
}

#[test]
fn branch_with_deep_invalid_inactive_arm_is_not_evaluated() {
    let mut b = GraphBuilder::new();
    let x = b.input();
    let zero = b.constant(0.0).unwrap();
    let condition = b.greater(x, zero).unwrap();
    let mut invalid = b.ln(x).unwrap();
    for _ in 0..20_000 { let temp = b.ln(invalid).unwrap(); invalid = b.exp(temp).unwrap(); }
    let safe = b.neg(x).unwrap();
    let out = b.select(condition, invalid, safe).unwrap();
    let g = b.compile(&[out]).unwrap();
    let mut w = g.workspace();
    assert_eq!(g.vjp(&[-2.0], &[1.0], &mut w).unwrap(), &[-1.0]);
}

#[test]
fn workspace_must_belong_to_compiled_graph_even_when_shapes_match() {
    let mut b1=GraphBuilder::new();
    let x=b1.input();
    let g1=b1.compile(&[x]).unwrap();
    let mut b2=GraphBuilder::new();
    let y=b2.input();
    let g2=b2.compile(&[y]).unwrap();
    let mut w=g1.workspace();
    assert_eq!(g2.evaluate(&[1.0],&mut w).unwrap_err(), AadError::InvalidWorkspace);
}

#[test]
fn reverse_uses_only_executed_nodes_and_reuses_scratch() {
    let mut b = GraphBuilder::new();
    let x = b.input();
    let zero = b.constant(0.0).unwrap();
    let positive = b.greater(x, zero).unwrap();
    // This costly branch is invalid for x < 0 and must not run or be reversed.
    let mut expensive = b.ln(x).unwrap();
    for _ in 0..10_000 {
        expensive = b.exp(expensive).unwrap();
        expensive = b.ln(expensive).unwrap();
    }
    let safe = b.neg(x).unwrap();
    let result = b.select(positive, expensive, safe).unwrap();
    let g = b.compile(&[result]).unwrap();
    let mut w = g.workspace();
    let original = w.capacities();
    for _ in 0..100 {
        assert_eq!(g.vjp(&[-2.0], &[1.0], &mut w).unwrap(), &[-1.0]);
        assert!(w.executed_nodes() < 10);
        assert_eq!(w.capacities(), original);
    }
    assert_eq!(g.vjp(&[1.0], &[1.0], &mut w).unwrap(), &[1.0]);
    assert!(w.executed_nodes() > 20_000);
    assert_eq!(w.capacities(), original);
    assert_eq!(g.vjp(&[-2.0], &[1.0], &mut w).unwrap(), &[-1.0]);
    assert!(w.executed_nodes() < 10);
}

#[test]
fn shared_graph_multi_output_seeds_accumulate_and_reset() {
    let mut b = GraphBuilder::new();
    let x = b.input();
    let y = b.input();
    let shared = b.mul(x, y).unwrap();
    let twice = b.add(shared, shared).unwrap();
    let g = b.compile(&[shared, twice, shared]).unwrap();
    let mut w = g.workspace();
    assert_eq!(g.vjp(&[2.0, 3.0], &[1.0, 2.0, 3.0], &mut w).unwrap(), &[24.0, 16.0]);
    assert_eq!(g.vjp(&[2.0, 3.0], &[0.0, 1.0, 0.0], &mut w).unwrap(), &[6.0, 4.0]);
    assert_eq!(w.executed_nodes(), g.node_count());
}

#[test]
fn failed_branch_does_not_poison_next_workspace_call() {
    let mut b = GraphBuilder::new();
    let x = b.input();
    let one = b.constant(1.0).unwrap();
    let z = b.ln(x).unwrap();
    let result = b.add(z, one).unwrap();
    let g = b.compile(&[result]).unwrap();
    let mut w = g.workspace();
    assert!(matches!(g.vjp(&[-1.0], &[1.0], &mut w), Err(AadError::NonFiniteResult { .. })));
    assert_eq!(g.vjp(&[1.0], &[1.0], &mut w).unwrap(), &[1.0]);
}

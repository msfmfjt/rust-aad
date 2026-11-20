#![cfg(feature = "simd")]
use rust_aad_core::{AadError, GraphBuilder};

#[test]
fn lanes_match_scalar_for_shared_nodes_duplicate_outputs_and_distinct_seeds() {
    let mut b = GraphBuilder::new();
    let x = b.input();
    let y = b.input();
    let _unused_input = b.input();
    let _dead_log = b.ln(y).unwrap();
    let c = b.constant(0.75).unwrap();
    let shared = b.mul(x, y).unwrap();
    let scaled = b.mul(shared, c).unwrap();
    let square = b.mul(shared, shared).unwrap();
    let sum = b.add(square, scaled).unwrap();
    let difference = b.sub(sum, y).unwrap();
    let result = b.neg(difference).unwrap();
    let g = b.compile(&[shared, result, shared]).unwrap();
    let mut w = g.workspace();
    let mut batch = g.batch2().unwrap();
    let seeds = [[0.5, 0.0], [2.0, -3.0], [-0.25, 0.0]];
    for x in -8..8 {
        let inputs = [[x as f64, 7.0 - x as f64], [-2.0, 1.5], [4.0, -3.0]];
        let values: [[f64; 2]; 3] = batch.evaluate(&inputs).unwrap().try_into().unwrap();
        let grads = batch.vjp(&inputs, &seeds).unwrap();
        for lane in 0..2 {
            let v = inputs.map(|pair| pair[lane]);
            let s = seeds.map(|pair| pair[lane]);
            assert_eq!(g.evaluate(&v, &mut w).unwrap(), values.map(|p| p[lane]));
            for (&want, pair) in g.vjp(&v, &s, &mut w).unwrap().iter().zip(grads) {
                assert_eq!(want, pair[lane]);
            }
        }
    }
}

#[test]
fn signed_zero_and_error_recovery() {
    let mut b = GraphBuilder::new();
    let x = b.input();
    let neg = b.neg(x).unwrap();
    let square = b.mul(x, x).unwrap();
    let g = b.compile(&[neg, square]).unwrap();
    let mut batch = g.batch2().unwrap();
    assert_eq!(
        batch.evaluate(&[[0.0, -0.0]]).unwrap()[0].map(f64::to_bits),
        [-0.0_f64, 0.0].map(f64::to_bits)
    );
    assert_eq!(batch.evaluate(&[]), Err(AadError::InputShape));
    assert_eq!(
        batch.evaluate(&[[1.0, f64::NAN]]),
        Err(AadError::NonFiniteInput)
    );
    assert_eq!(
        batch.evaluate(&[[1.0, f64::MAX]]),
        Err(AadError::NonFiniteResult { node: 2 })
    );
    assert_eq!(batch.vjp(&[[1.0, 2.0]], &[]), Err(AadError::SeedShape));
    assert_eq!(
        batch.vjp(&[[1.0, 2.0]], &[[0.0; 2], [0.0, f64::INFINITY]]),
        Err(AadError::SeedShape)
    );
    assert_eq!(
        batch.vjp(&[[1.0, 2.0]], &[[0.0; 2], [0.0, f64::MAX]]),
        Err(AadError::NonFiniteResult { node: 0 })
    );
    assert_eq!(
        batch.vjp(&[[1.0, 2.0]], &[[0.0; 2], [0.0, 1.0]]).unwrap(),
        &[[0.0, 4.0]]
    );
}

#[test]
fn reachable_unsupported_ops_use_scalar_lazy_semantics() {
    let mut b = GraphBuilder::new();
    let x = b.input();
    let zero = b.constant(0.0).unwrap();
    let positive = b.greater(x, zero).unwrap();
    let log = b.ln(x).unwrap();
    let out = b.select(positive, log, zero).unwrap();
    let g = b.compile(&[out]).unwrap();
    assert!(g.batch2().is_none());
    assert_eq!(g.evaluate(&[-1.0], &mut g.workspace()).unwrap(), &[0.0]);
}

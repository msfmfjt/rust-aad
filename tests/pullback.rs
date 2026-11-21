use rust_aad_core::{AadError, GraphBuilder};

#[test]
fn shared_forward_preserves_lazy_trace_multiple_outputs_and_seed_error_recovery() {
    let mut b = GraphBuilder::new();
    let x = b.input();
    let gate = b.input();
    let log = b.ln(x).unwrap();
    let square = b.mul(x, x).unwrap();
    let choice = b.select(gate, log, square).unwrap();
    let g = b.compile(&[choice, x, choice]).unwrap();
    let (mut w, mut reference) = (g.workspace(), g.workspace());
    for row in [[-2., 0.], [4., 1.], [0., 0.]] {
        let mut saved = g.linearize(&row, &mut w).unwrap();
        assert_eq!(saved.outputs(), g.evaluate(&row, &mut reference).unwrap());
        for seed in [
            [0., 2., -1.],
            [1., 0., -1.],
            [f64::MAX, 0., f64::MAX],
            [-0.0, 0.25, 0.5],
        ] {
            assert_eq!(
                saved.vjp(&seed).map(bits),
                g.vjp(&row, &seed, &mut reference).map(bits)
            );
        }
        assert_eq!(saved.vjp(&[]), Err(AadError::SeedShape));
        assert_eq!(saved.vjp(&[0., f64::NAN, 0.]), Err(AadError::SeedShape));
        assert_eq!(saved.vjp(&[0., 1., 0.]).unwrap(), &[1., 0.]);
    }
    assert!(matches!(
        g.linearize(&[-1., 1.], &mut w),
        Err(AadError::NonFiniteResult { .. })
    ));
    assert!(g.linearize(&[], &mut w).is_err());
    let mut b = GraphBuilder::new();
    let x = b.input();
    let other = b.compile(&[x]).unwrap();
    assert!(matches!(
        g.linearize(&[1., 1.], &mut other.workspace()),
        Err(AadError::InvalidWorkspace)
    ));
}
fn bits(x: &[f64]) -> Vec<u64> {
    x.iter().map(|v| v.to_bits()).collect()
}

use rust_aad_core::{AadError, GraphBuilder, PreparedInputs};

#[test]
fn packing_is_all_or_nothing_and_preserves_unusual_values() {
    let mut b = GraphBuilder::new();
    let x = b.input();
    let g = b.compile(&[x]).unwrap();
    for (data, bytes) in [
        (vec![0.0, -1.0, 2.5, 1.25], 8),
        (vec![0.0; 129], 258),
        (vec![-0.0, 1.0], 16),
        (vec![1.0, f64::from_bits(1.0f64.to_bits() + 1)], 16),
        (vec![f64::MIN_POSITIVE, f64::from_bits(1)], 16),
        (vec![f64::MAX, -f64::MAX], 16),
        (vec![f64::NAN, 2.0, f64::INFINITY], 24),
    ] {
        let inputs = PreparedInputs::new(data.len(), 1, data.clone()).unwrap();
        assert_eq!(inputs.storage_bytes(), bytes);
        let mut worker = inputs.replay(&g).unwrap();
        for (path, x) in data.into_iter().enumerate() {
            if x.is_finite() {
                assert_eq!(worker.evaluate(path).unwrap()[0].to_bits(), x.to_bits());
                assert_eq!(worker.vjp(path, &[0.25]), Ok(&[0.25][..]));
            } else {
                assert_eq!(worker.evaluate(path), Err(AadError::NonFiniteInput));
                assert_eq!(worker.vjp(path, &[f64::NAN]), Err(AadError::SeedShape));
            }
        }
    }
}

#[test]
fn prepared_sums_match_an_ordered_fold_and_seeded_gradients() {
    let width = 120;
    for initial in [0.0, -0.0, 0.375] {
        for scale in [-2.0, -0.0, 1.0, 4.0] {
            let mut b = GraphBuilder::new();
            let mut s = b.constant(initial).unwrap();
            for _ in 0..width {
                let x = b.input();
                s = b.add(s, x).unwrap();
            }
            let c = b.constant(scale).unwrap();
            let s = b.mul(s, c).unwrap();
            let zero = b.constant(0.0).unwrap();
            let s = b.max(s, zero).unwrap();
            let g = b.compile(&[s]).unwrap();
            let rows: Vec<_> = (0..19 * width)
                .map(|i| match i / width % 4 {
                    0 => 0.0,
                    1 => -2.5,
                    2 => 1.25,
                    _ => (i as i32 % 9 - 4) as f64 / 8.0,
                })
                .collect();
            let data = PreparedInputs::new(19, width, rows.clone()).unwrap();
            assert_eq!(data.storage_bytes(), rows.len() * 2);
            let mut worker = data.replay(&g).unwrap();
            for (i, row) in rows.chunks_exact(width).enumerate().rev() {
                let primal = row.iter().fold(initial, |s, x| s + x) * scale;
                assert_eq!(
                    worker.evaluate(i).unwrap()[0].to_bits(),
                    primal.max(0.0).to_bits()
                );
                for seed in [0.0, -0.0, 0.375, -3.0, 1.0] {
                    let derivative = if primal >= 0.0 {
                        0.0 + seed * scale
                    } else {
                        0.0
                    };
                    let grad = worker.vjp(i, &[seed]).unwrap();
                    assert_eq!(grad.len(), width);
                    assert!(grad.iter().all(|v| v.to_bits() == derivative.to_bits()));
                    assert_eq!(worker.outputs()[0].to_bits(), primal.max(0.0).to_bits());
                }
            }
            assert_eq!(worker.vjp(0, &[]), Err(AadError::SeedShape));
            assert_eq!(worker.vjp(0, &[f64::INFINITY]), Err(AadError::SeedShape));
        }
    }
}

#[test]
fn lazy_errors_extreme_seeds_and_later_rows_keep_their_semantics() {
    let mut b = GraphBuilder::new();
    let x = b.input();
    let gate = b.input();
    let log = b.ln(x).unwrap();
    let s = b.select(gate, log, x).unwrap();
    let g = b.compile(&[s, x]).unwrap();
    let data = PreparedInputs::new(3, 2, vec![-1., 1., -1., 0., 2., 1.]).unwrap();
    let mut worker = data.replay(&g).unwrap();
    assert_eq!(
        worker.evaluate(0),
        Err(AadError::NonFiniteResult { node: 2 })
    );
    assert_eq!(worker.evaluate(1), Ok(&[-1., -1.][..]));
    assert_eq!(
        worker.vjp(1, &[f64::MAX; 2]),
        Err(AadError::NonFiniteResult { node: 0 })
    );
    assert_eq!(worker.vjp(2, &[2., 3.]), Ok(&[4., 0.][..]));
}

#[test]
fn constant_graphs_empty_matrices_and_shape_errors() {
    assert!(matches!(
        PreparedInputs::new(usize::MAX, 2, Vec::new()),
        Err(AadError::InputShape)
    ));
    assert!(matches!(
        PreparedInputs::new(2, 3, vec![0.; 5]),
        Err(AadError::InputShape)
    ));
    let mut b = GraphBuilder::new();
    let c = b.constant(-0.0).unwrap();
    let g = b.compile(&[c]).unwrap();
    let inputs = PreparedInputs::new(2, 0, vec![]).unwrap();
    let mut worker = inputs.replay(&g).unwrap();
    assert_eq!(
        worker.evaluate(1).unwrap()[0].to_bits(),
        (-0.0f64).to_bits()
    );
    assert!(worker.vjp(1, &[1.]).unwrap().is_empty());
    assert_eq!(worker.evaluate(2), Err(AadError::InputShape));
    assert_eq!(worker.vjp(usize::MAX, &[1.]), Err(AadError::InputShape));
    let empty = PreparedInputs::new(0, 2, vec![]).unwrap();
    assert!(matches!(empty.replay(&g), Err(AadError::InputShape)));
}

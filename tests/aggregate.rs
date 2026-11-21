use rust_aad_core::{AccumulationError, accumulate_gradient};

#[test]
fn streamed_add_matches_retained_rows_and_rejects_an_entire_invalid_row() {
    let mut s = [-0.0, 0., 0.];
    let rows = [[0., 1e16, 1.25], [-0., 1., -2.5], [0., -1e16, 0.5]];
    let mut expected = s;
    for row in rows {
        accumulate_gradient(&mut s, &row).unwrap();
    }
    for k in 0..s.len() {
        for row in rows {
            expected[k] += row[k];
        }
    }
    assert_eq!(s.map(f64::to_bits), expected.map(f64::to_bits));
    assert_eq!(
        accumulate_gradient(&mut s, &[]),
        Err(AccumulationError::Shape)
    );
    assert_eq!(
        accumulate_gradient(&mut s, &[1., f64::NAN, 0.]),
        Err(AccumulationError::NonFinite { input: 1 })
    );
    let mut max = [1., f64::MAX];
    assert_eq!(
        accumulate_gradient(&mut max, &[2., f64::MAX]),
        Err(AccumulationError::NonFinite { input: 1 })
    );
    assert_eq!(max, [1., f64::MAX]);
    assert_eq!(s.map(f64::to_bits), expected.map(f64::to_bits));
    accumulate_gradient(&mut [], &[]).unwrap();
}

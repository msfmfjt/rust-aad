use rust_aad_bench::fixture::Case;

#[test]
fn branch_frequencies_and_inactive_gradients_match_the_contract() {
    for percent in [1, 50, 99] {
        let case = Case::Branch(percent);
        let mut hits = 0;
        for path in 0..10_000 {
            let x: Vec<_> = (0..12).map(|j| case.input(path, j)).collect();
            let positives: Vec<_> = (0..12).filter(|&j| x[j] > 0.).collect();
            let mut grad = [0.; 12];
            case.oracle(&x, &mut [0.], &mut grad);
            if let [hit] = positives[..] {
                hits += 1;
                assert!(grad[hit + 1..].iter().all(|v| *v == 0.));
            } else {
                assert!(positives.is_empty());
            }
        }
        assert_eq!(hits, percent * 100);
    }
}

#[test]
fn averaged_payoff_sends_the_tie_adjoint_left() {
    let mut grad = [0.; 12];
    let mut output = [0.];
    Case::Average(12).oracle(&[100.; 12], &mut output, &mut grad);
    assert_eq!(output, [0.]);
    assert_eq!(grad, [1. / 12.; 12]);
}

#[test]
fn retained_results_export_every_lane_and_coordinate_in_path_order() {
    use rust_aad_bench::{Scalar, Simd, run_with_output};
    let rows = 4;
    let mut scalar = Vec::new();
    let mut simd = Vec::new();
    let a = run_with_output::<Scalar>("width-10-v1", rows, Some(&mut scalar)).unwrap();
    let b = run_with_output::<Simd>("width-10-v1", rows, Some(&mut simd)).unwrap();
    assert_eq!(a["input_fnv1a64"], b["input_fnv1a64"]);
    assert_eq!(scalar, simd);
    let case = Case::Width(10);
    let mut expected = Vec::new();
    for path in 0..rows {
        let xs: Vec<_> = (0..10).map(|j| case.input(path, j)).collect();
        let (mut ys, mut gs) = ([0.; 2], [0.; 10]);
        case.oracle(&xs, &mut ys, &mut gs);
        for v in ys.into_iter().chain(gs) {
            expected.extend_from_slice(&v.to_le_bytes());
        }
    }
    assert_eq!(scalar, expected);
    for record in [a, b] {
        for phase in ["forward_store_outputs", "forward_vjp_store_gradients"] {
            assert_eq!(record["retained_phases"][phase]["units"], rows);
        }
    }
}

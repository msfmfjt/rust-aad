//! Replay the fixed 120-input graph while changing its output seed on every path.
use rust_aad_bench::{Backend, Scalar, fixture::Case, measure};
use std::hint::black_box;

fn main() {
    let case = Case::Average(120);
    let paths = 10_000;
    let n = case.inputs();
    let graph = Scalar::compile(&case.program());
    let mut workspace = graph.workspace();
    let mut inputs = Vec::with_capacity(paths * n);
    let mut seeds = Vec::with_capacity(paths);
    let mut full_precision = false;
    let mut unit_seed = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--full-precision" => full_precision = true,
            "--unit-seed" => unit_seed = true,
            _ => panic!("unknown option {arg}"),
        }
    }
    let mut state = 43_u64;
    for path in 0..paths {
        inputs.extend((0..n).map(|i| {
            let value = case.input(path, i);
            if full_precision {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                f64::from_bits(value.to_bits() | (state >> 34))
            } else {
                value
            }
        }));
        seeds.push([if unit_seed {
            1.0
        } else {
            1.0 + path as f64 / paths as f64
        }]);
    }
    let mut value = [0.0];
    let mut gradient = vec![0.0; n];
    for (row, seed) in inputs.chunks_exact(n).zip(&seeds) {
        case.oracle(row, &mut value, &mut gradient);
        assert_eq!(graph.evaluate(row, &mut workspace).unwrap(), value);
        let actual = graph.vjp(row, seed, &mut workspace).unwrap();
        for (&got, &want) in actual.iter().zip(&gradient) {
            assert_eq!(got.to_bits(), (want * seed[0]).to_bits());
        }
    }
    let timing = measure(paths, || {
        for (row, seed) in inputs.chunks_exact(n).zip(&seeds) {
            black_box(
                graph
                    .vjp(black_box(row), black_box(seed), &mut workspace)
                    .unwrap(),
            );
        }
    });
    println!("{timing}");
}

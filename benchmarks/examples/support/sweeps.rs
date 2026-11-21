//! Shared single-worker, setup-inclusive benchmark for repeated path VJPs.
use rust_aad_bench::{Backend, Scalar, fixture::Case};
use rust_aad_core::CompiledGraph;
use serde_json::json;
use std::{hint::black_box, time::Duration};

pub fn run(
    paths: usize,
    passes: usize,
    replay: impl FnOnce(&CompiledGraph, Vec<f64>, usize, usize) -> Duration,
) {
    assert!((1..=1_000_000).contains(&paths) && passes > 0);
    let case = Case::Average(120);
    let graph = Scalar::compile(&case.program());
    let width = case.inputs();
    let inputs = (0..paths)
        .flat_map(|i| (0..width).map(move |j| case.input(i, j)))
        .collect();
    let elapsed = replay(&graph, inputs, paths, passes);
    println!(
        "{}",
        json!({"paths":paths,"passes":passes,"elapsed_ns":elapsed.as_nanos() as u64})
    );
}

pub fn with_seeds(passes: usize, mut sweep: impl FnMut(&[f64])) {
    for pass in 0..passes {
        let seed = [1.0 - (pass % 17) as f64 / 8.0];
        sweep(black_box(&seed));
    }
}

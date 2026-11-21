//! Benchmark the input conversion and warm replay as separate operations.
#[path = "support/sweeps.rs"]
mod sweeps;
use rust_aad_bench::{Backend, Scalar, fixture::Case, measure};
use rust_aad_core::PreparedInputs;
use serde_json::json;
use std::{env, hint::black_box, time::Instant};

fn main() {
    let args: Vec<_> = env::args().collect();
    if let [_, flag, paths, passes] = args.as_slice()
        && flag == "--sweeps"
    {
        sweeps::run(
            paths.parse().unwrap(),
            passes.parse().unwrap(),
            |graph, values, paths, passes| {
                let start = Instant::now();
                let data = PreparedInputs::new(paths, graph.input_count(), values).unwrap();
                let mut worker = data.replay(graph).unwrap();
                sweeps::with_seeds(passes, |seed| {
                    for i in 0..paths {
                        black_box(worker.vjp(black_box(i), black_box(seed)).unwrap());
                    }
                });
                start.elapsed()
            },
        );
        return;
    }
    let [_, workload, paths, tail @ ..] = args.as_slice() else {
        panic!("expected WORKLOAD PATHS [fixture|full], or --sweeps PATHS PASSES");
    };
    assert!(tail.len() <= 1);
    let case = Case::parse(workload).expect("registered workload");
    let paths = paths.parse::<usize>().unwrap();
    let precision = args.get(3).map(String::as_str).unwrap_or("fixture");
    assert!((1..=1_000_000).contains(&paths));
    let width = case.inputs();
    let p = case.program();
    let graph = Scalar::compile(&p);
    // The raw reference rows are independently regenerated for verification.
    let input = |path, j| {
        let x = case.input(path, j);
        match precision {
            "fixture" => x,
            "full" => f64::from_bits(x.to_bits() ^ ((path as u64 * 41 + j as u64 + 1) & 0xffff)),
            _ => panic!("choose fixture or full"),
        }
    };
    let values = (0..paths)
        .flat_map(|i| (0..width).map(move |j| input(i, j)))
        .collect();
    let start = Instant::now();
    let inputs = PreparedInputs::new(paths, width, values).unwrap();
    let prepare_ns = start.elapsed().as_nanos() as u64;
    let mut replay = inputs.replay(&graph).unwrap();
    let mut work = graph.workspace();
    let mut row = vec![0.0; width];
    let alternate: Vec<_> = p.seeds.iter().map(|s| -0.375 * s).collect();
    for i in 0..paths {
        for (j, x) in row.iter_mut().enumerate() {
            *x = input(i, j);
        }
        let expected = graph.evaluate(&row, &mut work).unwrap();
        assert!(
            expected
                .iter()
                .zip(replay.evaluate(i).unwrap())
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        // Verify all input derivatives with changing seeds too.
        let seeds = if i % 2 == 0 {
            &p.seeds[..]
        } else {
            &alternate[..]
        };
        let expected = graph.vjp(&row, seeds, &mut work).unwrap();
        assert!(
            expected
                .iter()
                .zip(replay.vjp(i, seeds).unwrap())
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }
    let forward = measure(paths, || {
        for i in 0..paths {
            black_box(replay.evaluate(black_box(i)).unwrap());
        }
    });
    let reverse = measure(paths, || {
        for i in 0..paths {
            black_box(replay.vjp(black_box(i), black_box(&p.seeds)).unwrap());
        }
    });
    let mut retained = serde_json::Value::Null;
    if paths <= 10000 {
        let mut grads = vec![0.0; paths * width];
        retained = measure(paths, || {
            for (i, dest) in grads.chunks_exact_mut(width).enumerate() {
                dest.copy_from_slice(replay.vjp(black_box(i), black_box(&p.seeds)).unwrap());
            }
            black_box(&grads);
        });
    }
    println!(
        "{}",
        json!({"workload":args[1],"paths":paths,"precision":precision,"bytes":inputs.storage_bytes(),
        "verified_paths":paths,"prepare_ns":prepare_ns,"forward":forward,"forward_vjp":reverse,"retained_vjp":retained})
    );
}

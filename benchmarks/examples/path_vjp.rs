//! Compare three path VJP techniques on identical, uncompressed path-major f64 rows.
use rust_aad_bench::{Backend, Scalar, fixture::Case, measure};
use rust_aad_core::{CompiledGraph, accumulate_gradient};
use serde_json::{Value, json};
use std::{env, hint::black_box, mem::size_of_val};

fn same(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    assert!(a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()));
}

fn simd(method: &str, graph: &CompiledGraph, rows: &[f64]) -> Value {
    let width = graph.input_count();
    let paths = rows.len() / width;
    let mut ws = graph.workspace();
    let mut batch = graph.batch2().unwrap();
    let mut scratch = vec![[0.0; 2]; width];
    let seed = [1.0];
    let pair = [[1.0; 2]];
    let chunks = rows.chunks_exact(2 * width);
    for block in chunks.clone() {
        let (a, b) = block.split_at(width);
        for ((dest, &x), &y) in scratch.iter_mut().zip(a).zip(b) {
            *dest = [x, y];
        }
        let values = batch.evaluate_rows(a, b).unwrap()[0];
        assert_eq!(
            values.map(f64::to_bits),
            batch.evaluate(&scratch).unwrap()[0].map(f64::to_bits)
        );
        for (k, row) in [a, b].into_iter().enumerate() {
            let v = graph.evaluate(row, &mut ws).unwrap()[0];
            assert_eq!(v.to_bits(), values[k].to_bits());
            let expected = graph.vjp(row, &seed, &mut ws).unwrap();
            for direct in [true, false] {
                let grad = if direct {
                    batch.vjp_rows(a, b, &pair)
                } else {
                    batch.vjp(&scratch, &pair)
                }
                .unwrap();
                assert_eq!(grad.len(), width);
                for (g, c) in grad.iter().zip(expected) {
                    assert_eq!(g[k].to_bits(), c.to_bits());
                }
                assert_eq!(batch.outputs()[0][k].to_bits(), v.to_bits());
            }
        }
    }
    match method {
        "scalar" => measure(paths, || {
            for row in rows.chunks_exact(width) {
                black_box(
                    graph
                        .vjp(black_box(row), black_box(&seed), &mut ws)
                        .unwrap(),
                );
            }
        }),
        "rows" | "interleave" => measure(paths, || {
            for block in chunks.clone() {
                let (a, b) = black_box(block).split_at(width);
                if method == "rows" {
                    black_box(batch.vjp_rows(a, b, black_box(&pair)).unwrap());
                } else {
                    for ((dest, &x), &y) in scratch.iter_mut().zip(a).zip(b) {
                        *dest = [x, y];
                    }
                    black_box(batch.vjp(black_box(&scratch), black_box(&pair)).unwrap());
                }
            }
            if !chunks.remainder().is_empty() {
                black_box(graph.vjp(chunks.remainder(), &seed, &mut ws).unwrap());
            }
        }),
        _ => panic!("unknown SIMD method"),
    }
}

fn multi(method: &str, graph: &CompiledGraph, rows: &[f64], directions: usize) -> Value {
    let seeds: Vec<Vec<f64>> = (0..directions)
        .map(|k| {
            (0..graph.output_count())
                .map(|j| {
                    if k < graph.output_count() {
                        f64::from(k == j)
                    } else {
                        ((k + 1) as f64 * (j + 1) as f64 * 0.125) - 1.375
                    }
                })
                .collect()
        })
        .collect();
    let width = graph.input_count();
    let mut ws = graph.workspace();
    let mut reference = graph.workspace();
    for row in rows.chunks_exact(width) {
        let mut pb = graph.linearize(row, &mut ws).unwrap();
        same(pb.outputs(), graph.evaluate(row, &mut reference).unwrap());
        for s in &seeds {
            same(
                pb.vjp(s).unwrap(),
                graph.vjp(row, s, &mut reference).unwrap(),
            );
        }
    }
    measure(rows.len() / width, || {
        for row in rows.chunks_exact(width) {
            let row = black_box(row);
            match method {
                "repeat" => {
                    for s in &seeds {
                        black_box(graph.vjp(row, black_box(s), &mut ws).unwrap());
                    }
                }
                "shared" => {
                    let mut pb = graph.linearize(row, &mut ws).unwrap();
                    for s in &seeds {
                        black_box(pb.vjp(black_box(s)).unwrap());
                    }
                }
                _ => panic!("unknown multiple-VJP method"),
            }
        }
    })
}

fn aggregate(method: &str, graph: &CompiledGraph, rows: &[f64], seed: &[f64]) -> Value {
    let n = graph.input_count();
    let paths = rows.len() / n;
    let mut ws = graph.workspace();
    let mut sums = vec![0.; n];
    let mut expected = vec![0.; n];
    for row in rows.chunks_exact(n) {
        for (t, &g) in expected
            .iter_mut()
            .zip(graph.vjp(row, seed, &mut ws).unwrap())
        {
            *t += g;
        }
    }
    let mut retained = if method == "store" {
        vec![0.; rows.len()]
    } else {
        Vec::new()
    };
    let mut run = || {
        sums.fill(0.0);
        match method {
            "store" => {
                for (row, dest) in rows.chunks_exact(n).zip(retained.chunks_exact_mut(n)) {
                    dest.copy_from_slice(
                        graph.vjp(black_box(row), black_box(seed), &mut ws).unwrap(),
                    );
                }
                for grad in black_box(&retained).chunks_exact(n) {
                    accumulate_gradient(&mut sums, grad).unwrap();
                }
            }
            "stream" => {
                for row in rows.chunks_exact(n) {
                    let grad = graph.vjp(black_box(row), black_box(seed), &mut ws).unwrap();
                    accumulate_gradient(&mut sums, grad).unwrap();
                }
            }
            _ => panic!("unknown aggregate method"),
        }
        black_box(&sums);
    };
    let mut result = measure(paths, &mut run);
    same(&sums, &expected);
    result["gradient_and_total_bytes"] =
        json!(size_of_val(retained.as_slice()) + size_of_val(sums.as_slice()));
    result
}

fn main() {
    let args: Vec<_> = env::args().skip(1).collect();
    let [trial, method, id, paths, precision, directions] = args.as_slice() else {
        panic!("TRIAL METHOD CASE PATHS fixture|full DIRECTIONS")
    };
    let paths = paths.parse::<usize>().unwrap();
    let directions = directions.parse::<usize>().unwrap();
    assert!((1..=1_000_000).contains(&paths) && (1..=8).contains(&directions));
    let case = Case::parse(id).unwrap();
    let p = case.program();
    let graph = Scalar::compile(&p);
    let width = case.inputs();
    let rows: Vec<_> = (0..paths)
        .flat_map(|i| {
            (0..width).map(move |j| {
                let x = case.input(i, j);
                match precision.as_str() {
                    "fixture" => x,
                    "full" => {
                        f64::from_bits(x.to_bits() ^ ((i as u64 * 41 + j as u64 + 1) & 0xffff))
                    }
                    _ => panic!("unknown precision"),
                }
            })
        })
        .collect();
    let result = match trial.as_str() {
        "simd" => {
            assert!(matches!(case, Case::Average(_)));
            simd(method, &graph, &rows)
        }
        "multi" => multi(method, &graph, &rows, directions),
        "aggregate" => aggregate(method, &graph, &rows, &p.seeds),
        _ => panic!("unknown trial"),
    };
    println!(
        "{}",
        json!({"trial":trial,"method":method,"workload":id,"paths":paths,"precision":precision,"directions":directions,"result":result})
    );
}

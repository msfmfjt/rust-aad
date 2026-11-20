//! Fixed two-input, two-output, 32-step arithmetic workload for external replay.
use rust_aad_core::{CompiledGraph, GraphBuilder};
use std::{hint::black_box, time::Instant};

const PATHS: usize = 10_000;
const SAMPLES: usize = 11;
const SEEDS: [f64; 2] = [1.0, -0.5];

fn build() -> CompiledGraph {
    let mut b = GraphBuilder::new();
    let x = b.input();
    let q = b.input();
    let bias = b.constant(1.0 / 64.0).unwrap();
    let mut s = x;
    for _ in 0..32 {
        let product = b.mul(s, q).unwrap();
        s = b.add(product, bias).unwrap();
    }
    let z = b.mul(s, x).unwrap();
    b.compile(&[s, z]).unwrap()
}

fn input(i: usize) -> [f64; 2] {
    [
        ((i * 47 % 1021) as f64 - 510.0) / 256.0,
        1.0 + ((i * 31 % 127) as f64 - 63.0) / 4096.0,
    ]
}

// Independent forward derivatives for the weighted output s - 0.5*s*x.
fn reference([x, q]: [f64; 2]) -> ([f64; 2], [f64; 2]) {
    let (mut s, mut dx, mut dq) = (x, 1.0, 0.0);
    for _ in 0..32 {
        dq = dq * q + s;
        dx *= q;
        s = s * q + 1.0 / 64.0;
    }
    let weight = 1.0 - 0.5 * x;
    ([s, s * x], [dx * weight - 0.5 * s, dq * weight])
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 2e-13 * (1.0 + expected.abs()),
        "{actual} vs {expected}"
    );
}

fn measure(label: &str, phase: &str, nodes: usize, hash: u64, mut run: impl FnMut()) {
    run();
    let mut ns = [0; SAMPLES];
    for elapsed in &mut ns {
        let start = Instant::now();
        run();
        *elapsed = start.elapsed().as_nanos();
    }
    ns.sort_unstable();
    println!(
        "{{\"workload\":\"affine32-v1\",\"backend\":\"{label}\",\"phase\":\"{phase}\",\"paths\":{PATHS},\"nodes\":{nodes},\"input_fnv1a64\":\"{hash:016x}\",\"samples\":{SAMPLES},\"min_ns\":{},\"median_ns\":{},\"max_ns\":{}}}",
        ns[0],
        ns[SAMPLES / 2],
        ns[SAMPLES - 1]
    );
}

fn main() {
    let g = build();
    let mut scalar = g.workspace();
    let mut batch = g.batch2().unwrap();
    let paths: Vec<_> = (0..PATHS).map(input).collect();
    let pairs: Vec<_> = paths
        .chunks_exact(2)
        .map(|p| [[p[0][0], p[1][0]], [p[0][1], p[1][1]]])
        .collect();
    let pair_seeds = SEEDS.map(|s| [s; 2]);
    let mut hash = 0xcbf29ce484222325_u64;
    for pair in &paths {
        for byte in pair.iter().flat_map(|v| v.to_le_bytes()) {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
    }
    assert_eq!(
        hash, 0xa7bf6211c9ab67d9,
        "affine32-v1 input fixture changed"
    );
    assert_eq!(g.node_count(), 68);
    for (index, p) in pairs.iter().enumerate() {
        let values: [[f64; 2]; 2] = batch.evaluate(p).unwrap().try_into().unwrap();
        let gradients = batch.vjp(p, &pair_seeds).unwrap();
        for lane in 0..2 {
            let row = &paths[index * 2 + lane];
            let (expected, expected_grad) = reference(*row);
            for (k, &v) in g.evaluate(row, &mut scalar).unwrap().iter().enumerate() {
                assert_close(v, expected[k]);
                assert_close(values[k][lane], expected[k]);
            }
            for (k, &v) in g.vjp(row, &SEEDS, &mut scalar).unwrap().iter().enumerate() {
                assert_close(v, expected_grad[k]);
                assert_close(gradients[k][lane], expected_grad[k]);
            }
        }
    }
    let nodes = g.node_count();
    measure("scalar", "forward", nodes, hash, || {
        for row in &paths {
            black_box(g.evaluate(black_box(row), &mut scalar).unwrap());
        }
    });
    measure("simd2", "forward", nodes, hash, || {
        for p in &pairs {
            black_box(batch.evaluate(black_box(p)).unwrap());
        }
    });
    measure("scalar", "forward_vjp", nodes, hash, || {
        for row in &paths {
            black_box(
                g.vjp(black_box(row), black_box(&SEEDS), &mut scalar)
                    .unwrap(),
            );
        }
    });
    measure("simd2", "forward_vjp", nodes, hash, || {
        for p in &pairs {
            black_box(batch.vjp(black_box(p), black_box(&pair_seeds)).unwrap());
        }
    });
}

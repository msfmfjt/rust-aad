//! Standalone process so test-runner allocations cannot affect the counter.
use rust_aad_core::GraphBuilder;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, Stats, StatsAlloc};
use std::{alloc::System, hint::black_box};

#[global_allocator]
static ALLOCATOR: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

fn check_reuse(label: &str, stats: Stats) {
    println!("{label}: {stats:?}");
    assert_eq!(stats, Stats::default(), "{label} used the allocator");
}

fn main() {
    let mut b = GraphBuilder::new();
    let x = b.input();
    let gate = b.input();
    let one = b.constant(1.0).unwrap();
    let mut chain = b.ln(x).unwrap();
    for _ in 0..1024 {
        chain = b.add(chain, one).unwrap();
    }
    let negative = b.neg(x).unwrap();
    let chosen = b.select(gate, chain, negative).unwrap();
    let graph = b.compile(&[chosen, chosen]).unwrap();

    let mut region = Region::new(ALLOCATOR);
    let mut ws = graph.workspace();
    let setup = region.change();
    println!("workspace ({} nodes): {setup:?}", graph.node_count());
    assert!(setup.allocations > 0);

    let paths = [[2.0, 1.0], [-2.0, 0.0]];
    region.reset();
    for i in 0..256 {
        black_box(graph.evaluate(&paths[i % 2], &mut ws).unwrap());
    }
    check_reuse("256 forward calls", region.change());

    region.reset();
    for i in 0..256 {
        black_box(graph.vjp(&paths[i % 2], &[1.0, 0.5], &mut ws).unwrap());
    }
    check_reuse("256 forward+VJP calls", region.change());

    region.reset();
    assert!(graph.evaluate(&[-2.0, 1.0], &mut ws).is_err());
    assert_eq!(
        graph.vjp(&paths[1], &[1.0, 0.0], &mut ws).unwrap(),
        &[-1.0, 0.0]
    );
    check_reuse("error and recovery", region.change());
    check_straight_line();
    #[cfg(feature = "simd")]
    check_batch();
}

#[cfg(feature = "simd")]
fn check_batch() {
    let mut b = GraphBuilder::new();
    let x = b.input();
    let y = b.mul(x, x).unwrap();
    let graph = b.compile(&[x, y]).unwrap();
    let mut batch = graph.batch2().unwrap();
    let region = Region::new(ALLOCATOR);
    for i in 0..256 {
        let x = [[i as f64, -(i as f64)]];
        black_box(batch.evaluate(&x).unwrap());
        black_box(batch.vjp(&x, &[[0.5, -0.5], [1.0, 2.0]]).unwrap());
    }
    check_reuse("256 SIMD pairs per method", region.change());
}

fn check_straight_line() {
    let mut b = GraphBuilder::new();
    let x = b.input();
    let square = b.mul(x, x).unwrap();
    let graph = b.compile(&[x, square]).unwrap();
    let mut ws = graph.workspace();
    let region = Region::new(ALLOCATOR);
    for i in 0..256 {
        let inputs = [f64::from(i)];
        black_box(graph.evaluate(&inputs, &mut ws).unwrap());
        black_box(graph.vjp(&inputs, &[0.5, 2.0], &mut ws).unwrap());
    }
    check_reuse("256 straight-line calls per method", region.change());
}

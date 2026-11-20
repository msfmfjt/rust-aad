//! Reproducible *local* interpreter baseline, not a vendor comparison.
//! Run: cargo run --release --example baseline -- 100000
use rust_aad_core::GraphBuilder;
use std::{env, hint::black_box, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let paths: usize = env::args().nth(1).as_deref().unwrap_or("100000").parse()?;
    assert!((1..=5_000_000).contains(&paths), "paths out of range");
    let mut builder = GraphBuilder::new();
    let input = builder.input();
    let strike = builder.constant(100.0)?;
    let zero = builder.constant(0.0)?;
    let mut state = input;
    let mut average = zero;
    for _ in 0..32 {
        let multiplier = builder.constant(1.001)?;
        state = builder.mul(state, multiplier)?;
        average = builder.add(average, state)?;
    }
    let n = builder.constant(32.0)?;
    let average = builder.div(average, n)?;
    let intrinsic = builder.sub(average, strike)?;
    let payoff = builder.max(intrinsic, zero)?;
    let graph = builder.compile(&[payoff])?;
    let mut ws = graph.workspace();
    let mut forward = Vec::with_capacity(7);
    let mut reverse = Vec::with_capacity(7);
    let mut sum = 0.0;
    // Warm up and then report 7 independent repetitions; fixed input sequence.
    for repetition in 0..8 {
        let start = Instant::now();
        for i in 0..paths {
            let s = 80.0 + ((i * 7919 % 10007) as f64) / 100.0;
            sum += black_box(graph.evaluate(black_box(&[s]), &mut ws)?[0]);
        }
        let forward_ns = start.elapsed().as_nanos() as u64;
        let start = Instant::now();
        for i in 0..paths {
            let s = 80.0 + ((i * 7919 % 10007) as f64) / 100.0;
            sum += black_box(graph.vjp(black_box(&[s]), &[1.0], &mut ws)?[0]);
        }
        let reverse_ns = start.elapsed().as_nanos() as u64;
        if repetition > 0 {
            forward.push(forward_ns);
            reverse.push(reverse_ns);
        }
    }
    forward.sort_unstable();
    reverse.sort_unstable();
    println!(
        "{{\"benchmark\":\"asian32_interpreter\",\"paths\":{paths},\"nodes\":{},\"forward_median_ns\":{},\"forward_vjp_median_ns\":{},\"checksum\":{sum:.12}}}",
        graph.node_count(),
        forward[3],
        reverse[3]
    );
    Ok(())
}

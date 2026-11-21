//! Include workspace preparation and repeated VJPs over the original input slice.
#[path = "support/sweeps.rs"]
mod sweeps;
use std::{env, hint::black_box, time::Instant};

fn main() {
    let mut args = env::args().skip(1);
    let paths = args.next().expect("path count").parse().unwrap();
    let passes = args.next().expect("sweep count").parse().unwrap();
    sweeps::run(paths, passes, |graph, inputs, _, passes| {
        let start = Instant::now();
        let mut work = graph.workspace();
        sweeps::with_seeds(passes, |seed| {
            for row in inputs.chunks_exact(graph.input_count()) {
                black_box(
                    graph
                        .vjp(black_box(row), black_box(seed), &mut work)
                        .unwrap(),
                );
            }
        });
        start.elapsed()
    });
}

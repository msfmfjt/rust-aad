//! Shared workload, correctness and measurement contract for benchmark backends.
pub mod fixture;
use fixture::{Case, Op, Program};
use rust_aad_core::{Batch2, CompiledGraph, GraphBuilder, Workspace};
use serde_json::{Value, json};
use std::hint::black_box;
use std::io::{self, Write};
#[cfg(not(feature = "count-allocations"))]
use std::time::Instant;

pub const SAMPLES: usize = 11;

#[cfg(feature = "count-allocations")]
#[global_allocator]
static ALLOCATOR: &stats_alloc::StatsAlloc<std::alloc::System> = &stats_alloc::INSTRUMENTED_SYSTEM;

pub trait Backend {
    const NAME: &'static str;
    const LANES: usize = 1;
    type Graph;
    type Work;
    fn compile(p: &Program) -> Self::Graph;
    fn prepare(g: &Self::Graph) -> Self::Work;
    fn forward<'a>(g: &Self::Graph, w: &'a mut Self::Work, x: &[f64]) -> &'a [f64];
    fn vjp<'a>(g: &Self::Graph, w: &'a mut Self::Work, x: &[f64], seeds: &[f64]) -> &'a [f64];
}

fn lower(p: &Program) -> CompiledGraph {
    let mut b = GraphBuilder::new();
    let mut slots = Vec::with_capacity(p.nodes.len());
    for op in &p.nodes {
        let id = match *op {
            Op::Input(_) => b.input(),
            Op::Constant(v) => b.constant(v).unwrap(),
            Op::Add(i, j) => b.add(slots[i], slots[j]).unwrap(),
            Op::Sub(i, j) => b.sub(slots[i], slots[j]).unwrap(),
            Op::Mul(i, j) => b.mul(slots[i], slots[j]).unwrap(),
            Op::Max(i, j) => b.max(slots[i], slots[j]).unwrap(),
            Op::Greater(i, j) => b.greater(slots[i], slots[j]).unwrap(),
            Op::Neg(i) => b.neg(slots[i]).unwrap(),
            Op::Log(i) => b.ln(slots[i]).unwrap(),
            Op::Select(i, j, k) => b.select(slots[i], slots[j], slots[k]).unwrap(),
        };
        slots.push(id);
    }
    b.compile(&p.outputs.iter().map(|&i| slots[i]).collect::<Vec<_>>())
        .unwrap()
}
pub struct Scalar;
impl Backend for Scalar {
    const NAME: &'static str = "rust-scalar";
    type Graph = CompiledGraph;
    type Work = Workspace;
    fn compile(p: &Program) -> Self::Graph {
        lower(p)
    }
    fn prepare(g: &Self::Graph) -> Self::Work {
        g.workspace()
    }
    fn forward<'a>(g: &Self::Graph, w: &'a mut Self::Work, x: &[f64]) -> &'a [f64] {
        g.evaluate(x, w).unwrap()
    }
    fn vjp<'a>(g: &Self::Graph, w: &'a mut Self::Work, x: &[f64], seeds: &[f64]) -> &'a [f64] {
        g.vjp(x, seeds, w).unwrap()
    }
}
pub struct Simd;
impl Backend for Simd {
    const NAME: &'static str = "rust-simd2";
    const LANES: usize = 2;
    type Graph = CompiledGraph;
    type Work = Batch2;
    fn compile(p: &Program) -> Self::Graph {
        lower(p)
    }
    fn prepare(g: &Self::Graph) -> Self::Work {
        g.batch2()
            .expect("this workload needs a non-SIMD operation")
    }
    fn forward<'a>(_: &Self::Graph, w: &'a mut Self::Work, x: &[f64]) -> &'a [f64] {
        w.evaluate(x.as_chunks::<2>().0).unwrap().as_flattened()
    }
    fn vjp<'a>(_: &Self::Graph, w: &'a mut Self::Work, x: &[f64], seeds: &[f64]) -> &'a [f64] {
        w.vjp(x.as_chunks::<2>().0, seeds.as_chunks::<2>().0)
            .unwrap()
            .as_flattened()
    }
}

pub fn fnv(hash: u64, bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .fold(hash, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100000001b3))
}
pub const FNV_OFFSET: u64 = 0xcbf29ce484222325;

fn check_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        actual.is_finite()
            && expected.is_finite()
            && (actual - expected).abs() <= tolerance * (1. + expected.abs()),
        "observed {actual} expected {expected}"
    );
}

// The allocation build makes a single untimed observation. Release timing uses
// the normal allocator. Stop both measurements before destroying returned objects.
pub fn measure<T>(units: usize, mut operation: impl FnMut() -> T) -> Value {
    drop(black_box(operation()));
    #[cfg(feature = "count-allocations")]
    {
        let region = stats_alloc::Region::new(ALLOCATOR);
        let result = black_box(operation());
        let stats = region.change();
        drop(result);
        json!({"units":units,"allocations":stats.allocations,"reallocations":stats.reallocations,
        "bytes_allocated":stats.bytes_allocated,"bytes_per_unit":stats.bytes_allocated as f64/units as f64})
    }
    #[cfg(not(feature = "count-allocations"))]
    {
        let mut ns = [0_u64; SAMPLES];
        for sample in &mut ns {
            let start = Instant::now();
            let result = black_box(operation());
            *sample = start.elapsed().as_nanos() as u64;
            drop(result);
        }
        let mut sorted = ns;
        sorted.sort_unstable();
        let median = sorted[SAMPLES / 2];
        json!({"units":units,"samples_ns":ns,"median_ns":median,"min_ns":sorted[0],"max_ns":sorted[SAMPLES-1],
            "median_ci_ns":[sorted[1],sorted[SAMPLES-2]],"units_per_second":units as f64*1e9/median.max(1) as f64})
    }
}

/// Retain all path results in contiguous path-major order. For SIMD, the
/// borrowed result is coordinate-major within each batch of lanes.
fn store_batch(src: &[f64], dest: &mut [f64], width: usize, lanes: usize) {
    if lanes == 1 {
        dest.copy_from_slice(src);
        return;
    }
    for (lane, row) in dest.chunks_exact_mut(width).enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            *value = src[j * lanes + lane];
        }
    }
}

pub fn run<B: Backend>(id: &str, paths: usize) -> Value {
    run_with_output::<B>(id, paths, None).expect("no file I/O in the ordinary benchmark")
}

pub fn run_with_output<B: Backend>(
    id: &str,
    paths: usize,
    retained: Option<&mut dyn Write>,
) -> io::Result<Value> {
    let case = Case::parse(id).expect("unknown workload");
    let tolerance = if matches!(case, Case::Affine) {
        2e-13
    } else {
        2e-12
    };
    let n = case.inputs();
    let p = case.program();
    let lanes = B::LANES;
    assert!(paths > 0 && paths <= 1_000_000 && paths.is_multiple_of(lanes));
    let stride = n * lanes;
    let mut inputs = vec![0.; paths * n];
    let mut hash = FNV_OFFSET;
    for (k, block) in inputs.chunks_exact_mut(stride).enumerate() {
        for lane in 0..lanes {
            for j in 0..n {
                let x = case.input(k * lanes + lane, j);
                block[j * lanes + lane] = x;
                hash = fnv(hash, &x.to_le_bytes());
            }
        }
    }
    let graph = B::compile(&p);
    let mut w = B::prepare(&graph);
    let seeds: Vec<_> = p
        .seeds
        .iter()
        .flat_map(|&s| std::iter::repeat_n(s, lanes))
        .collect();
    let (mut y, mut grad) = (vec![0.; p.outputs.len()], vec![0.; n]);
    let (mut sum_y, mut sum_grad) = (y.clone(), grad.clone());
    let mut row = vec![0.; n];
    let mut values = vec![0.; y.len() * lanes];
    for block in inputs.chunks_exact(stride) {
        values.copy_from_slice(B::forward(&graph, &mut w, block));
        let grads = B::vjp(&graph, &mut w, block, &seeds);
        assert_eq!(grads.len(), n * lanes);
        for lane in 0..lanes {
            for (j, x) in row.iter_mut().enumerate() {
                *x = block[j * lanes + lane];
            }
            case.oracle(&row, &mut y, &mut grad);
            for (j, &want) in y.iter().enumerate() {
                let actual = values[j * lanes + lane];
                check_close(actual, want, tolerance);
                sum_y[j] += actual;
            }
            for (j, &want) in grad.iter().enumerate() {
                let actual = grads[j * lanes + lane];
                check_close(actual, want, tolerance);
                sum_grad[j] += actual;
            }
        }
    }
    for sum in sum_y.iter_mut().chain(&mut sum_grad) {
        *sum /= paths as f64;
    }
    let compile = measure(1, || B::compile(black_box(&p)));
    let prepare = measure(1, || B::prepare(black_box(&graph)));
    let cold = measure(lanes, || {
        let g = B::compile(black_box(&p));
        let mut work = B::prepare(&g);
        black_box(B::vjp(
            &g,
            &mut work,
            black_box(&inputs[..stride]),
            black_box(&seeds),
        ));
        (g, work)
    });
    let forward = measure(paths, || {
        for x in inputs.chunks_exact(stride) {
            black_box(B::forward(&graph, &mut w, black_box(x)));
        }
    });
    let reverse = measure(paths, || {
        for x in inputs.chunks_exact(stride) {
            black_box(B::vjp(&graph, &mut w, black_box(x), black_box(&seeds)));
        }
    });
    let mut report = json!({"schema":1,"workload":id,"backend":B::NAME,"lanes":lanes,"paths":paths,"inputs":n,
        "nodes":p.nodes.len(),"seeds":p.seeds,"input_fnv1a64":format!("{hash:016x}"),
        "program_fnv1a64":format!("{:016x}",fnv(FNV_OFFSET,&serde_json::to_vec(&p).unwrap())),
        "verified_paths":paths,"output_means":sum_y,"input_vjp_means":sum_grad,
        "phases":{"record_compile":compile,"prepare_workspace":prepare,"cold_record_vjp":cold,"forward":forward,"forward_vjp":reverse}});
    if let Some(writer) = retained {
        let outputs = p.outputs.len();
        let mut all_y = vec![0.; paths * outputs];
        let mut all_g = vec![0.; paths * n];
        let forward = measure(paths, || {
            for (x, dest) in inputs
                .chunks_exact(stride)
                .zip(all_y.chunks_exact_mut(outputs * lanes))
            {
                store_batch(
                    B::forward(&graph, &mut w, black_box(x)),
                    dest,
                    outputs,
                    lanes,
                );
            }
            black_box(&all_y);
        });
        let vjp = measure(paths, || {
            for (x, dest) in inputs
                .chunks_exact(stride)
                .zip(all_g.chunks_exact_mut(stride))
            {
                store_batch(
                    B::vjp(&graph, &mut w, black_box(x), black_box(&seeds)),
                    dest,
                    n,
                    lanes,
                );
            }
            black_box(&all_g);
        });
        for (outputs, gradient) in all_y.chunks_exact(outputs).zip(all_g.chunks_exact(n)) {
            for v in outputs.iter().chain(gradient) {
                writer.write_all(&v.to_le_bytes())?;
            }
        }
        writer.flush()?;
        report["retained_phases"] =
            json!({"forward_store_outputs":forward,"forward_vjp_store_gradients":vjp});
    }
    Ok(report)
}

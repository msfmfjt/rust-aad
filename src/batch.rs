//! Two independent f64 evaluations per instruction, using safe SIMD operations.
use super::{AadError, CompiledGraph, Op};
use wide::f64x2;
mod reductions;
use reductions::Region;

#[derive(Clone, Copy)]
enum Arithmetic {
    Input(usize),
    Constant(f64),
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Neg(usize),
}

struct Step {
    source_node: usize,
    op: Arithmetic,
}

/// Experimental two-path executor. Each input, output and VJP seed is one
/// pair of independent path values. Prepare one executor per worker.
pub struct Batch2 {
    steps: Vec<Step>,
    regions: Box<[Region]>,
    reset_ranges: Box<[std::ops::Range<usize>]>,
    covers_all_inputs: bool,
    outputs: Vec<usize>,
    primal: Vec<f64x2>,
    adjoint: Vec<f64x2>,
    values: Vec<[f64; 2]>,
    gradients: Vec<[f64; 2]>,
}

impl CompiledGraph {
    /// Prepare a two-lane forward/reverse tape for reachable input, constant,
    /// add, subtract, multiply and negate nodes. Returns `None` for graphs
    /// requiring other operations. Those graphs can use the scalar workspace.
    pub fn batch2(&self) -> Option<Batch2> {
        let schedule = self.straight_line.as_ref()?;
        let order = &schedule.order;
        let mut slots = vec![0; self.nodes.len()];
        let mut steps = Vec::with_capacity(order.len());
        for &source in order {
            let i = usize::from(source);
            let op = match self.nodes[i] {
                Op::Input(k) => Arithmetic::Input(k),
                Op::Constant(x) => Arithmetic::Constant(x),
                Op::Add(a, b) => Arithmetic::Add(slots[a], slots[b]),
                Op::Sub(a, b) => Arithmetic::Sub(slots[a], slots[b]),
                Op::Mul(a, b) => Arithmetic::Mul(slots[a], slots[b]),
                Op::Neg(a) => Arithmetic::Neg(slots[a]),
                _ => return None,
            };
            slots[i] = steps.len();
            steps.push(Step { source_node: i, op });
        }
        Some(Batch2 {
            reset_ranges: crate::Block::reset_ranges(&schedule.blocks, order, steps.len(), |id| {
                slots[id]
            }),
            regions: schedule
                .blocks
                .iter()
                .map(|block| Region::prepare(block, &slots))
                .collect(),
            covers_all_inputs: schedule.covers_all_inputs,
            primal: vec![f64x2::ZERO; steps.len()],
            adjoint: vec![f64x2::ZERO; steps.len()],
            steps,
            outputs: self.outputs.iter().map(|o| slots[o.0]).collect(),
            values: vec![[0.0; 2]; self.output_count()],
            gradients: vec![[0.0; 2]; self.input_count()],
        })
    }
}

impl Batch2 {
    /// Evaluate two paths. A nonfinite value in either lane fails the call.
    pub fn evaluate(&mut self, inputs: &[[f64; 2]]) -> Result<&[[f64; 2]], AadError> {
        if inputs.len() != self.gradients.len() {
            return Err(AadError::InputShape);
        }
        if !self.covers_all_inputs && !inputs.iter().flatten().all(|v| v.is_finite()) {
            return Err(AadError::NonFiniteInput);
        }
        if self.regions.is_empty() || self.forward_regions(inputs).is_err() {
            if let Err(error) = Self::forward_steps(&self.steps, 0, inputs, &mut self.primal) {
                if self.covers_all_inputs && !inputs.iter().flatten().all(|v| v.is_finite()) {
                    return Err(AadError::NonFiniteInput);
                }
                return Err(error);
            }
        }
        for (value, &slot) in self.values.iter_mut().zip(&self.outputs) {
            *value = self.primal[slot].to_array();
        }
        Ok(&self.values)
    }

    fn forward_steps(
        steps: &[Step],
        start: usize,
        inputs: &[[f64; 2]],
        p: &mut [f64x2],
    ) -> Result<(), AadError> {
        for (i, step) in steps.iter().enumerate() {
            let value = match step.op {
                Arithmetic::Input(k) => f64x2::new(inputs[k]),
                Arithmetic::Constant(v) => f64x2::splat(v),
                Arithmetic::Add(a, b) => p[a] + p[b],
                Arithmetic::Sub(a, b) => p[a] - p[b],
                Arithmetic::Mul(a, b) => p[a] * p[b],
                // Flip the sign bit, including for both signed zeros.
                Arithmetic::Neg(a) => p[a] ^ f64x2::splat(-0.0),
            };
            if !value.is_finite().all() {
                return Err(AadError::NonFiniteResult {
                    node: step.source_node,
                });
            }
            p[start + i] = value;
        }
        Ok(())
    }
    /// Run forward and return one input-gradient vector per lane, using that
    /// lane's output seeds. Repeated output handles accumulate their seeds.
    pub fn vjp(
        &mut self,
        inputs: &[[f64; 2]],
        seeds: &[[f64; 2]],
    ) -> Result<&[[f64; 2]], AadError> {
        if seeds.len() != self.outputs.len() || !seeds.iter().flatten().all(|v| v.is_finite()) {
            return Err(AadError::SeedShape);
        }
        self.evaluate(inputs)?;
        let adj = &mut self.adjoint;
        let ranges = if self.regions.is_empty() {
            &[][..]
        } else {
            &self.reset_ranges
        };
        crate::reset_adjoints(adj, ranges, f64x2::ZERO);
        for (&o, &pair) in self.outputs.iter().zip(seeds) {
            adj[o] += f64x2::new(pair);
        }
        if self.regions.is_empty() {
            Self::reverse_steps(&self.steps, 0, &self.primal, adj, &mut self.gradients)?;
        } else {
            self.reverse_regions(inputs)?;
        }
        Ok(&self.gradients)
    }
    fn reverse_steps(
        steps: &[Step],
        start: usize,
        p: &[f64x2],
        adj: &mut [f64x2],
        gradients: &mut [[f64; 2]],
    ) -> Result<(), AadError> {
        for (i, step) in steps.iter().enumerate().rev() {
            let seed = adj[start + i];
            if !seed.is_finite().all() {
                return Err(AadError::NonFiniteResult {
                    node: step.source_node,
                });
            }
            match step.op {
                Arithmetic::Input(k) => gradients[k] = seed.to_array(),
                Arithmetic::Constant(_) => {}
                Arithmetic::Add(a, b) => {
                    adj[a] += seed;
                    adj[b] += seed;
                }
                Arithmetic::Sub(a, b) => {
                    adj[a] += seed;
                    adj[b] -= seed;
                }
                Arithmetic::Mul(a, b) => {
                    adj[a] += seed * p[b];
                    adj[b] += seed * p[a];
                }
                Arithmetic::Neg(a) => adj[a] -= seed,
            }
        }
        Ok(())
    }
}

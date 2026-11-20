//! Two independent f64 evaluations per instruction, using safe SIMD operations.
use super::{AadError, CompiledGraph, Op};
use wide::f64x2;

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
        // Follow the scalar evaluator's left-first, output-by-output order.
        // Preserve the order of floating-point accumulation for shared nodes.
        let mut slots = vec![None; self.nodes.len()];
        let mut steps = Vec::new();
        let mut stack = Vec::new();
        for root in &self.outputs {
            stack.push((root.0, false));
            while let Some((i, ready)) = stack.pop() {
                if slots[i].is_some() {
                    continue;
                }
                let op = self.nodes[i];
                if !ready {
                    stack.push((i, true));
                    match op {
                        Op::Input(_) | Op::Constant(_) => {}
                        Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) => {
                            stack.push((b.0, false));
                            stack.push((a.0, false));
                        }
                        Op::Neg(a) => stack.push((a.0, false)),
                        _ => return None,
                    }
                    continue;
                }
                let op = match op {
                    Op::Input(k) => Arithmetic::Input(k),
                    Op::Constant(x) => Arithmetic::Constant(x),
                    Op::Add(a, b) => Arithmetic::Add(slots[a.0]?, slots[b.0]?),
                    Op::Sub(a, b) => Arithmetic::Sub(slots[a.0]?, slots[b.0]?),
                    Op::Mul(a, b) => Arithmetic::Mul(slots[a.0]?, slots[b.0]?),
                    Op::Neg(a) => Arithmetic::Neg(slots[a.0]?),
                    _ => return None,
                };
                slots[i] = Some(steps.len());
                steps.push(Step { source_node: i, op });
            }
        }
        Some(Batch2 {
            primal: vec![f64x2::ZERO; steps.len()],
            adjoint: vec![f64x2::ZERO; steps.len()],
            steps,
            outputs: self
                .outputs
                .iter()
                .map(|o| slots[o.0])
                .collect::<Option<_>>()?,
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
        if !inputs.iter().flatten().all(|v| v.is_finite()) {
            return Err(AadError::NonFiniteInput);
        }
        let p = &mut self.primal;
        for (i, step) in self.steps.iter().enumerate() {
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
            p[i] = value;
        }
        for (value, &slot) in self.values.iter_mut().zip(&self.outputs) {
            *value = p[slot].to_array();
        }
        Ok(&self.values)
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
        adj.fill(f64x2::ZERO);
        for (&o, &pair) in self.outputs.iter().zip(seeds) {
            adj[o] += f64x2::new(pair);
        }
        for (i, step) in self.steps.iter().enumerate().rev() {
            let seed = adj[i];
            if !seed.is_finite().all() {
                return Err(AadError::NonFiniteResult {
                    node: step.source_node,
                });
            }
            match step.op {
                Arithmetic::Input(k) => self.gradients[k] = seed.to_array(),
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
                    adj[a] += seed * self.primal[b];
                    adj[b] += seed * self.primal[a];
                }
                Arithmetic::Neg(a) => adj[a] -= seed,
            }
        }
        Ok(&self.gradients)
    }
}

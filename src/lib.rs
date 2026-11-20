//! Experimental f64 graph with lazy branches and reusable primal/adjoint storage.
//! Not yet a performance replacement for AADC/XAD.
#![forbid(unsafe_code)]
use std::{
    error::Error,
    fmt,
    sync::atomic::{AtomicU64, Ordering},
};

const MAX_NODES: usize = 65_536;
static NEXT_GRAPH_ID: AtomicU64 = AtomicU64::new(1);

/// Graph-local handle; cannot be constructed outside the library.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(usize, u64);

#[derive(Clone, Copy, Debug)]
enum Op {
    Input(usize),
    Constant(f64),
    Add(NodeId, NodeId),
    Sub(NodeId, NodeId),
    Mul(NodeId, NodeId),
    Div(NodeId, NodeId),
    Neg(NodeId),
    Exp(NodeId),
    Ln(NodeId),
    Greater(NodeId, NodeId),
    Min(NodeId, NodeId),
    Max(NodeId, NodeId),
    Select {
        condition: NodeId,
        positive: NodeId,
        zero: NodeId,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AadError {
    InvalidNode,
    EmptyOutputs,
    GraphTooLarge,
    InputShape,
    SeedShape,
    InvalidConstant,
    NonFiniteInput,
    NonFiniteResult { node: usize },
    InvalidWorkspace,
}
impl fmt::Display for AadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AAD error: {self:?}")
    }
}
impl Error for AadError {}

/// Construct an immutable graph. Nodes are topologically ordered by construction.
pub struct GraphBuilder {
    nodes: Vec<Op>,
    input_count: usize,
    graph_id: u64,
}
impl Default for GraphBuilder {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            input_count: 0,
            graph_id: NEXT_GRAPH_ID.fetch_add(1, Ordering::Relaxed),
        }
    }
}
impl GraphBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    fn push(&mut self, op: Op) -> NodeId {
        let id = NodeId(self.nodes.len(), self.graph_id);
        self.nodes.push(op);
        id
    }
    pub fn input(&mut self) -> NodeId {
        let i = self.input_count;
        self.input_count += 1;
        self.push(Op::Input(i))
    }
    pub fn constant(&mut self, v: f64) -> Result<NodeId, AadError> {
        if !v.is_finite() {
            return Err(AadError::InvalidConstant);
        }
        Ok(self.push(Op::Constant(v)))
    }
    fn unary(&mut self, a: NodeId, f: impl FnOnce(NodeId) -> Op) -> Result<NodeId, AadError> {
        self.check(a)?;
        Ok(self.push(f(a)))
    }
    fn binary(
        &mut self,
        a: NodeId,
        b: NodeId,
        f: impl FnOnce(NodeId, NodeId) -> Op,
    ) -> Result<NodeId, AadError> {
        self.check(a)?;
        self.check(b)?;
        Ok(self.push(f(a, b)))
    }
    fn check(&self, node: NodeId) -> Result<(), AadError> {
        if node.1 != self.graph_id || node.0 >= self.nodes.len() {
            Err(AadError::InvalidNode)
        } else {
            Ok(())
        }
    }
    pub fn add(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, AadError> {
        self.binary(a, b, Op::Add)
    }
    pub fn sub(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, AadError> {
        self.binary(a, b, Op::Sub)
    }
    pub fn mul(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, AadError> {
        self.binary(a, b, Op::Mul)
    }
    pub fn div(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, AadError> {
        self.binary(a, b, Op::Div)
    }
    pub fn min(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, AadError> {
        self.binary(a, b, Op::Min)
    }
    pub fn max(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, AadError> {
        self.binary(a, b, Op::Max)
    }
    pub fn greater(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, AadError> {
        self.binary(a, b, Op::Greater)
    }
    pub fn neg(&mut self, a: NodeId) -> Result<NodeId, AadError> {
        self.unary(a, Op::Neg)
    }
    pub fn exp(&mut self, a: NodeId) -> Result<NodeId, AadError> {
        self.unary(a, Op::Exp)
    }
    pub fn ln(&mut self, a: NodeId) -> Result<NodeId, AadError> {
        self.unary(a, Op::Ln)
    }
    /// Lazy select; does not evaluate the inactive branch. A nonzero condition selects `positive`.
    pub fn select(
        &mut self,
        condition: NodeId,
        positive: NodeId,
        zero: NodeId,
    ) -> Result<NodeId, AadError> {
        self.check(condition)?;
        self.check(positive)?;
        self.check(zero)?;
        Ok(self.push(Op::Select {
            condition,
            positive,
            zero,
        }))
    }
    pub fn compile(self, outputs: &[NodeId]) -> Result<CompiledGraph, AadError> {
        if outputs.is_empty() {
            return Err(AadError::EmptyOutputs);
        }
        if self.nodes.len() > MAX_NODES {
            return Err(AadError::GraphTooLarge);
        }
        for &o in outputs {
            self.check(o)?;
        }
        Ok(CompiledGraph {
            nodes: self.nodes.into_boxed_slice(),
            outputs: outputs.into(),
            input_count: self.input_count,
            graph_id: self.graph_id,
        })
    }
}

/// Immutable graph, shareable across workers. Each worker needs its own Workspace.
pub struct CompiledGraph {
    nodes: Box<[Op]>,
    outputs: Box<[NodeId]>,
    input_count: usize,
    graph_id: u64,
}
impl CompiledGraph {
    pub fn input_count(&self) -> usize {
        self.input_count
    }
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    pub fn workspace(&self) -> Workspace {
        Workspace {
            primal: vec![0.0; self.nodes.len()],
            adjoint: vec![0.0; self.nodes.len()],
            visited: vec![false; self.nodes.len()],
            output: vec![0.0; self.outputs.len()],
            input_adjoints: vec![0.0; self.input_count],
            stack: Vec::with_capacity(self.nodes.len().saturating_mul(3)),
            active: Vec::with_capacity(self.nodes.len()),
            graph_id: self.graph_id,
        }
    }
    fn preflight(&self, inputs: &[f64], w: &Workspace) -> Result<(), AadError> {
        if inputs.len() != self.input_count {
            return Err(AadError::InputShape);
        }
        if inputs.iter().any(|v| !v.is_finite()) {
            return Err(AadError::NonFiniteInput);
        }
        if w.graph_id != self.graph_id
            || w.primal.len() != self.nodes.len()
            || w.adjoint.len() != self.nodes.len()
            || w.visited.len() != self.nodes.len()
            || w.active.capacity() < self.nodes.len()
            || w.output.len() != self.outputs.len()
            || w.input_adjoints.len() != self.input_count
        {
            return Err(AadError::InvalidWorkspace);
        }
        Ok(())
    }
    // An explicit evaluation stack avoids recursion depth proportional to graph depth.
    // Phase 0 = schedule prerequisites, 1 = evaluate, 2 = resolve a lazy select.
    fn eval(&self, id: NodeId, inputs: &[f64], w: &mut Workspace) -> Result<f64, AadError> {
        w.stack.push((id, 0));
        while let Some((id, phase)) = w.stack.pop() {
            let i = id.0;
            if w.visited[i] {
                continue;
            }
            let op = self.nodes[i];
            if phase == 0 {
                match op {
                    Op::Input(_) | Op::Constant(_) => w.stack.push((id, 1)),
                    Op::Select { condition, .. } => {
                        w.stack.push((id, 2));
                        if !w.visited[condition.0] {
                            w.stack.push((condition, 0));
                        }
                    }
                    Op::Neg(a) | Op::Exp(a) | Op::Ln(a) => {
                        w.stack.push((id, 1));
                        if !w.visited[a.0] {
                            w.stack.push((a, 0));
                        }
                    }
                    Op::Add(a, b)
                    | Op::Sub(a, b)
                    | Op::Mul(a, b)
                    | Op::Div(a, b)
                    | Op::Greater(a, b)
                    | Op::Min(a, b)
                    | Op::Max(a, b) => {
                        w.stack.push((id, 1));
                        if !w.visited[b.0] {
                            w.stack.push((b, 0));
                        }
                        if !w.visited[a.0] {
                            w.stack.push((a, 0));
                        }
                    }
                }
                continue;
            }
            if phase == 2 {
                if let Op::Select {
                    condition,
                    positive,
                    zero,
                } = op
                {
                    let selected = if w.primal[condition.0] != 0.0 {
                        positive
                    } else {
                        zero
                    };
                    w.stack.push((id, 1));
                    if !w.visited[selected.0] {
                        w.stack.push((selected, 0));
                    }
                }
                continue;
            }
            let p = &w.primal;
            let value = match op {
                Op::Input(k) => inputs[k],
                Op::Constant(v) => v,
                Op::Neg(a) => -p[a.0],
                Op::Exp(a) => p[a.0].exp(),
                Op::Ln(a) => p[a.0].ln(),
                Op::Select {
                    condition,
                    positive,
                    zero,
                } => {
                    p[if p[condition.0] != 0.0 {
                        positive.0
                    } else {
                        zero.0
                    }]
                }
                Op::Add(a, b) => p[a.0] + p[b.0],
                Op::Sub(a, b) => p[a.0] - p[b.0],
                Op::Mul(a, b) => p[a.0] * p[b.0],
                Op::Div(a, b) => p[a.0] / p[b.0],
                Op::Greater(a, b) => {
                    if p[a.0] > p[b.0] {
                        1.0
                    } else {
                        0.0
                    }
                }
                Op::Min(a, b) => p[a.0].min(p[b.0]),
                Op::Max(a, b) => p[a.0].max(p[b.0]),
            };
            if !value.is_finite() {
                w.stack.clear();
                return Err(AadError::NonFiniteResult { node: i });
            }
            w.primal[i] = value;
            w.visited[i] = true;
            // Forward executes dependencies before consumers. This execution-order
            // trace is reverse-topological, even for lazy branches and shared DAGs.
            w.active.push(i);
        }
        Ok(w.primal[id.0])
    }
    /// Evaluate all outputs. The workspace owns returned values; each call overwrites them.
    pub fn evaluate<'a>(
        &self,
        inputs: &[f64],
        w: &'a mut Workspace,
    ) -> Result<&'a [f64], AadError> {
        self.preflight(inputs, w)?;
        w.visited.fill(false);
        w.stack.clear();
        w.active.clear();
        for (k, &id) in self.outputs.iter().enumerate() {
            w.output[k] = self.eval(id, inputs, w)?;
        }
        Ok(&w.output)
    }
    /// Vector-Jacobian product: J(inputs)^T * output_seeds. Multiple outputs supported.
    pub fn vjp<'a>(
        &self,
        inputs: &[f64],
        output_seeds: &[f64],
        w: &'a mut Workspace,
    ) -> Result<&'a [f64], AadError> {
        if output_seeds.len() != self.outputs.len() || output_seeds.iter().any(|v| !v.is_finite()) {
            return Err(AadError::SeedShape);
        }
        self.evaluate(inputs, w)?;
        w.adjoint.fill(0.0);
        w.input_adjoints.fill(0.0);
        for (&o, &seed) in self.outputs.iter().zip(output_seeds) {
            w.adjoint[o.0] += seed;
        }
        // Only reverse nodes executed on this invocation. A large inactive
        // branch has no sweep cost; active storage is reused across calls.
        for &i in w.active.iter().rev() {
            let seed = w.adjoint[i];
            if seed == 0.0 {
                continue;
            }
            let p = &w.primal;
            match self.nodes[i] {
                Op::Input(k) => w.input_adjoints[k] += seed,
                Op::Constant(_) | Op::Greater(..) => {}
                Op::Add(a, b) => {
                    w.adjoint[a.0] += seed;
                    w.adjoint[b.0] += seed;
                }
                Op::Sub(a, b) => {
                    w.adjoint[a.0] += seed;
                    w.adjoint[b.0] -= seed;
                }
                Op::Mul(a, b) => {
                    w.adjoint[a.0] += seed * p[b.0];
                    w.adjoint[b.0] += seed * p[a.0];
                }
                Op::Div(a, b) => {
                    w.adjoint[a.0] += seed / p[b.0];
                    w.adjoint[b.0] += division_denominator_adjoint(seed, p[a.0], p[b.0]);
                }
                Op::Neg(a) => w.adjoint[a.0] -= seed,
                Op::Exp(a) => w.adjoint[a.0] += seed * p[i],
                Op::Ln(a) => w.adjoint[a.0] += seed / p[a.0],
                Op::Min(a, b) => {
                    if p[a.0] <= p[b.0] {
                        w.adjoint[a.0] += seed
                    } else {
                        w.adjoint[b.0] += seed
                    }
                }
                Op::Max(a, b) => {
                    if p[a.0] >= p[b.0] {
                        w.adjoint[a.0] += seed
                    } else {
                        w.adjoint[b.0] += seed
                    }
                }
                Op::Select {
                    condition,
                    positive,
                    zero,
                } => {
                    let selected = if p[condition.0] != 0.0 {
                        positive
                    } else {
                        zero
                    };
                    w.adjoint[selected.0] += seed;
                }
            }
            if !w.adjoint[i].is_finite() {
                return Err(AadError::NonFiniteResult { node: i });
            }
        }
        if let Some((i, _)) = w.adjoint.iter().enumerate().find(|(_, v)| !v.is_finite()) {
            return Err(AadError::NonFiniteResult { node: i });
        }
        if w.input_adjoints.iter().any(|v| !v.is_finite()) {
            return Err(AadError::NonFiniteResult { node: 0 });
        }
        Ok(&w.input_adjoints)
    }
}
/// Mutable per-worker cache. Do not share between concurrent evaluations.
pub struct Workspace {
    primal: Vec<f64>,
    adjoint: Vec<f64>,
    visited: Vec<bool>,
    output: Vec<f64>,
    input_adjoints: Vec<f64>,
    stack: Vec<(NodeId, u8)>,
    active: Vec<usize>,
    graph_id: u64,
}

/// Observable allocation capacities for regression tests; these are not byte-accurate
/// allocator measurements and do not guarantee allocation-free execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspaceCapacities {
    pub primal: usize,
    pub adjoint: usize,
    pub visited: usize,
    pub output: usize,
    pub input_adjoints: usize,
    pub stack: usize,
    pub active: usize,
}

impl Workspace {
    #[must_use]
    pub fn capacities(&self) -> WorkspaceCapacities {
        WorkspaceCapacities {
            primal: self.primal.capacity(),
            adjoint: self.adjoint.capacity(),
            visited: self.visited.capacity(),
            output: self.output.capacity(),
            input_adjoints: self.input_adjoints.capacity(),
            stack: self.stack.capacity(),
            active: self.active.capacity(),
        }
    }

    /// Number of nodes executed by the most recent forward or VJP call.
    #[must_use]
    pub fn executed_nodes(&self) -> usize {
        self.active.len()
    }
}

/// Compute -seed*numerator/denominator^2 without discarding a representable
/// derivative when an intermediate product or square overflows/underflows.
/// Forward division has already checked its result for finiteness.
fn division_denominator_adjoint(seed: f64, numerator: f64, denominator: f64) -> f64 {
    let product = -seed * numerator;
    let square = denominator * denominator;
    if square.is_normal() && product.is_normal() {
        return product / square;
    }
    if seed == 0.0 || numerator == 0.0 {
        return product;
    }
    // Decompose nonzero finite binary64 values into [1,2) significands
    // and base-two exponents. Inputs here are finite from the primal check.
    let (sf, se) = binary_parts(seed);
    let (nf, ne) = binary_parts(numerator);
    let (df, de) = binary_parts(denominator);
    let fraction = -sf * nf / (df * df);
    scale_binary(fraction, se + ne - 2 * de)
}

fn binary_parts(value: f64) -> (f64, i32) {
    let (value, correction) = if value.is_subnormal() {
        (value * 18_014_398_509_481_984.0, -54)
    } else {
        (value, 0)
    };
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32 - 1023 + correction;
    let significand = f64::from_bits((bits & 0x800f_ffff_ffff_ffff) | (1023_u64 << 52));
    (significand, exponent)
}

fn scale_binary(fraction: f64, exponent: i32) -> f64 {
    let power = |e: i32| f64::from_bits(((e + 1023) as u64) << 52);
    if exponent > 1025 {
        f64::INFINITY.copysign(fraction)
    } else if exponent < -1076 {
        0.0_f64.copysign(fraction)
    } else if exponent > 1023 {
        (fraction * power(exponent - 1023)) * power(1023)
    } else if exponent < -1022 {
        (fraction * power(exponent + 1022)) * f64::MIN_POSITIVE
    } else {
        fraction * power(exponent)
    }
}

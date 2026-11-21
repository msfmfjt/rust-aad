//! Experimental f64 graph with lazy branches and reusable primal/adjoint storage.
//! Not yet a performance replacement for AADC/XAD.
#![forbid(unsafe_code)]
use std::{
    error::Error,
    fmt,
    sync::atomic::{AtomicU64, Ordering},
};

mod blocks;
use blocks::Block;

#[cfg(feature = "simd")]
mod batch;
#[cfg(feature = "simd")]
pub use batch::Batch2;

const MAX_NODES: usize = u16::MAX as usize + 1;
static NEXT_GRAPH_ID: AtomicU64 = AtomicU64::new(1);

/// Graph-local handle; cannot be constructed outside the library.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(usize, u64);

#[derive(Clone, Copy, Debug)]
enum Op {
    Input(usize),
    Constant(f64),
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Div(usize, usize),
    Neg(usize),
    Exp(usize),
    Ln(usize),
    Greater(usize, usize),
    Min(usize, usize),
    Max(usize, usize),
    Select {
        condition: usize,
        positive: usize,
        zero: usize,
    },
}

impl Op {
    #[inline]
    fn value(self, inputs: &[f64], p: &[f64]) -> f64 {
        match self {
            Op::Input(k) => inputs[k],
            Op::Constant(v) => v,
            Op::Neg(a) => -p[a],
            Op::Exp(a) => p[a].exp(),
            Op::Ln(a) => p[a].ln(),
            Op::Select {
                condition,
                positive,
                zero,
            } => p[if p[condition] != 0.0 { positive } else { zero }],
            Op::Add(a, b) => p[a] + p[b],
            Op::Sub(a, b) => p[a] - p[b],
            Op::Mul(a, b) => p[a] * p[b],
            Op::Div(a, b) => p[a] / p[b],
            Op::Greater(a, b) => {
                if p[a] > p[b] {
                    1.0
                } else {
                    0.0
                }
            }
            Op::Min(a, b) => p[a].min(p[b]),
            Op::Max(a, b) => p[a].max(p[b]),
        }
    }
}

// The left-first postorder is observable through reverse accumulation and
// the first nonfinite result. Only reachable nodes belong to the schedule.
fn straight_line_order(nodes: &[Op], outputs: &[NodeId]) -> Option<Box<[u16]>> {
    let mut visited = vec![false; nodes.len()];
    let mut order = Vec::new();
    let mut stack = Vec::new();
    for root in outputs {
        stack.push((root.0, false));
        while let Some((i, finish)) = stack.pop() {
            if visited[i] {
                continue;
            }
            if !finish {
                stack.push((i, true));
                match nodes[i] {
                    Op::Input(_) | Op::Constant(_) => {}
                    Op::Neg(a) | Op::Exp(a) | Op::Ln(a) => stack.push((a, false)),
                    Op::Add(a, b)
                    | Op::Sub(a, b)
                    | Op::Mul(a, b)
                    | Op::Div(a, b)
                    | Op::Min(a, b)
                    | Op::Max(a, b)
                    | Op::Greater(a, b) => {
                        stack.push((b, false));
                        stack.push((a, false));
                    }
                    Op::Select { .. } => return None,
                }
            } else {
                visited[i] = true;
                order.push(i as u16);
            }
        }
    }
    Some(order.into_boxed_slice())
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
    fn unary(&mut self, a: NodeId, f: impl FnOnce(usize) -> Op) -> Result<NodeId, AadError> {
        self.check(a)?;
        Ok(self.push(f(a.0)))
    }
    fn binary(
        &mut self,
        a: NodeId,
        b: NodeId,
        f: impl FnOnce(usize, usize) -> Op,
    ) -> Result<NodeId, AadError> {
        self.check(a)?;
        self.check(b)?;
        Ok(self.push(f(a.0, b.0)))
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
            condition: condition.0,
            positive: positive.0,
            zero: zero.0,
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
        let straight_line = straight_line_order(&self.nodes, outputs).map(|order| StaticSchedule {
            blocks: Block::compile(&self.nodes, outputs, &order),
            order,
        });
        Ok(CompiledGraph {
            straight_line,
            nodes: self.nodes.into_boxed_slice(),
            outputs: outputs.into(),
            input_count: self.input_count,
            graph_id: self.graph_id,
        })
    }
}

struct StaticSchedule {
    order: Box<[u16]>,
    blocks: Box<[Block]>,
}

/// Immutable graph, shareable across workers. Each worker needs its own Workspace.
pub struct CompiledGraph {
    straight_line: Option<StaticSchedule>,
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
        let dynamic_nodes = if self.straight_line.is_some() {
            0
        } else {
            self.nodes.len()
        };
        Workspace {
            primal: vec![0.0; self.nodes.len()],
            adjoint: vec![0.0; self.nodes.len()],
            visited: vec![false; dynamic_nodes],
            output: vec![0.0; self.outputs.len()],
            input_adjoints: vec![0.0; self.input_count],
            stack: Vec::with_capacity(dynamic_nodes),
            active: Vec::with_capacity(dynamic_nodes),
            static_executed: 0,
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
        // Workspace fields are private and can only be created by this graph.
        if w.graph_id != self.graph_id {
            return Err(AadError::InvalidWorkspace);
        }
        Ok(())
    }
    // Suspend only the parent while descending to its first missing dependency.
    // On return, the completed child is memoised, including across output roots.
    fn eval(&self, id: NodeId, inputs: &[f64], w: &mut Workspace) -> Result<f64, AadError> {
        let mut i = id.0;
        if w.visited[i] {
            return Ok(w.primal[i]);
        }
        loop {
            let op = self.nodes[i];
            let missing = |j: usize| (!w.visited[j]).then_some(j);
            let child = match op {
                Op::Input(_) | Op::Constant(_) => None,
                Op::Neg(a) | Op::Exp(a) | Op::Ln(a) => missing(a),
                Op::Add(a, b)
                | Op::Sub(a, b)
                | Op::Mul(a, b)
                | Op::Div(a, b)
                | Op::Greater(a, b)
                | Op::Min(a, b)
                | Op::Max(a, b) => missing(a).or_else(|| missing(b)),
                Op::Select {
                    condition,
                    positive,
                    zero,
                } => missing(condition).or_else(|| {
                    missing(if w.primal[condition] != 0.0 {
                        positive
                    } else {
                        zero
                    })
                }),
            };
            if let Some(child) = child {
                w.stack.push(i as u16);
                i = child;
                continue;
            }
            let value = op.value(inputs, &w.primal);
            if !value.is_finite() {
                w.stack.clear();
                return Err(AadError::NonFiniteResult { node: i });
            }
            w.primal[i] = value;
            w.visited[i] = true;
            w.active.push(i as u16);
            match w.stack.pop() {
                Some(parent) => i = usize::from(parent),
                None => return Ok(value),
            }
        }
    }
    fn evaluate_slots(
        &self,
        order: &[u16],
        inputs: &[f64],
        w: &mut Workspace,
    ) -> Result<(), AadError> {
        for (completed, &slot) in order.iter().enumerate() {
            let i = usize::from(slot);
            let value = self.nodes[i].value(inputs, &w.primal);
            if !value.is_finite() {
                w.static_executed = completed;
                return Err(AadError::NonFiniteResult { node: i });
            }
            w.primal[i] = value;
        }
        Ok(())
    }
    /// Evaluate all outputs. The workspace owns returned values; each call overwrites them.
    pub fn evaluate<'a>(
        &self,
        inputs: &[f64],
        w: &'a mut Workspace,
    ) -> Result<&'a [f64], AadError> {
        self.preflight(inputs, w)?;
        if let Some(schedule) = &self.straight_line {
            if schedule.blocks.is_empty() || !self.evaluate_blocks(schedule, inputs, w) {
                self.evaluate_slots(&schedule.order, inputs, w)?;
            }
            w.static_executed = schedule.order.len();
            for (value, id) in w.output.iter_mut().zip(&self.outputs) {
                *value = w.primal[id.0];
            }
        } else {
            w.visited.fill(false);
            w.stack.clear();
            w.active.clear();
            for (k, &id) in self.outputs.iter().enumerate() {
                w.output[k] = self.eval(id, inputs, w)?;
            }
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
        match &self.straight_line {
            Some(schedule) if !schedule.blocks.is_empty() => self.reverse_blocks(schedule, w)?,
            other => {
                let order = other.as_ref().map_or(w.active.as_slice(), |s| &s.order);
                self.reverse_slots(order, &w.primal, &mut w.adjoint, &mut w.input_adjoints)?;
            }
        }
        Ok(&w.input_adjoints)
    }

    fn reverse_slots(
        &self,
        order: &[u16],
        p: &[f64],
        adj: &mut [f64],
        gradients: &mut [f64],
    ) -> Result<(), AadError> {
        for &slot in order.iter().rev() {
            let i = usize::from(slot);
            let seed = adj[i];
            if seed == 0.0 {
                continue;
            }
            // Every contribution targets an earlier executed node. Its final
            // sum is available here, including seeds from repeated outputs.
            if !seed.is_finite() {
                return Err(AadError::NonFiniteResult { node: i });
            }
            match self.nodes[i] {
                Op::Input(k) => gradients[k] += seed,
                Op::Constant(_) | Op::Greater(..) => {}
                Op::Add(a, b) => {
                    adj[a] += seed;
                    adj[b] += seed;
                }
                Op::Sub(a, b) => {
                    adj[a] += seed;
                    adj[b] -= seed;
                }
                Op::Mul(a, b) => {
                    adj[a] += seed * p[b];
                    adj[b] += seed * p[a];
                }
                Op::Div(a, b) => {
                    adj[a] += seed / p[b];
                    adj[b] += division_denominator_adjoint(seed, p[a], p[b]);
                }
                Op::Neg(a) => adj[a] -= seed,
                Op::Exp(a) => adj[a] += seed * p[i],
                Op::Ln(a) => adj[a] += seed / p[a],
                Op::Min(a, b) => {
                    if p[a] <= p[b] {
                        adj[a] += seed
                    } else {
                        adj[b] += seed
                    }
                }
                Op::Max(a, b) => {
                    if p[a] >= p[b] {
                        adj[a] += seed
                    } else {
                        adj[b] += seed
                    }
                }
                Op::Select {
                    condition,
                    positive,
                    zero,
                } => {
                    let selected = if p[condition] != 0.0 { positive } else { zero };
                    adj[selected] += seed;
                }
            }
        }
        Ok(())
    }
}
/// Mutable per-worker cache. Do not share between concurrent evaluations.
pub struct Workspace {
    primal: Vec<f64>,
    adjoint: Vec<f64>,
    visited: Vec<bool>,
    output: Vec<f64>,
    input_adjoints: Vec<f64>,
    stack: Vec<u16>,
    active: Vec<u16>,
    static_executed: usize,
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

    /// Number of forward nodes completed by the most recent evaluation.
    #[must_use]
    pub fn executed_nodes(&self) -> usize {
        self.static_executed + self.active.len()
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

//! Experimental f64 graph with lazy branches and reusable primal/adjoint storage.
//! Not yet a performance replacement for AADC/XAD.
#![forbid(unsafe_code)]
use std::{error::Error, fmt};

const MAX_NODES: usize = 65_536;

/// Graph-local handle; cannot be constructed outside the library.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(usize);

#[derive(Clone, Copy, Debug)]
enum Op {
    Input(usize), Constant(f64), Add(NodeId, NodeId), Sub(NodeId, NodeId),
    Mul(NodeId, NodeId), Div(NodeId, NodeId), Neg(NodeId), Exp(NodeId),
    Ln(NodeId), Greater(NodeId, NodeId), Min(NodeId, NodeId), Max(NodeId, NodeId),
    Select { condition: NodeId, positive: NodeId, zero: NodeId },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AadError {
    InvalidNode, EmptyOutputs, GraphTooLarge, InputShape, SeedShape,
    InvalidConstant, NonFiniteInput, NonFiniteResult { node: usize },
    InvalidWorkspace,
}
impl fmt::Display for AadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "AAD error: {self:?}") }
}
impl Error for AadError {}

/// Construct an immutable graph. Nodes are topologically ordered by construction.
#[derive(Default)]
pub struct GraphBuilder { nodes: Vec<Op>, input_count: usize }
impl GraphBuilder {
    pub fn new() -> Self { Self::default() }
    fn push(&mut self, op: Op) -> NodeId { let id = NodeId(self.nodes.len()); self.nodes.push(op); id }
    pub fn input(&mut self) -> NodeId { let i = self.input_count; self.input_count += 1; self.push(Op::Input(i)) }
    pub fn constant(&mut self, v: f64) -> Result<NodeId, AadError> {
        if !v.is_finite() { return Err(AadError::InvalidConstant); }
        Ok(self.push(Op::Constant(v)))
    }
    fn unary(&mut self, a: NodeId, f: impl FnOnce(NodeId) -> Op) -> Result<NodeId, AadError> {
        self.check(a)?; Ok(self.push(f(a)))
    }
    fn binary(&mut self, a: NodeId, b: NodeId, f: impl FnOnce(NodeId, NodeId) -> Op) -> Result<NodeId, AadError> {
        self.check(a)?; self.check(b)?; Ok(self.push(f(a,b)))
    }
    fn check(&self, node: NodeId) -> Result<(), AadError> {
        if node.0 >= self.nodes.len() { Err(AadError::InvalidNode) } else { Ok(()) }
    }
    pub fn add(&mut self, a: NodeId,b: NodeId)->Result<NodeId,AadError>{self.binary(a,b,Op::Add)}
    pub fn sub(&mut self, a: NodeId,b: NodeId)->Result<NodeId,AadError>{self.binary(a,b,Op::Sub)}
    pub fn mul(&mut self, a: NodeId,b: NodeId)->Result<NodeId,AadError>{self.binary(a,b,Op::Mul)}
    pub fn div(&mut self, a: NodeId,b: NodeId)->Result<NodeId,AadError>{self.binary(a,b,Op::Div)}
    pub fn min(&mut self, a: NodeId,b: NodeId)->Result<NodeId,AadError>{self.binary(a,b,Op::Min)}
    pub fn max(&mut self, a: NodeId,b: NodeId)->Result<NodeId,AadError>{self.binary(a,b,Op::Max)}
    pub fn greater(&mut self,a:NodeId,b:NodeId)->Result<NodeId,AadError>{self.binary(a,b,Op::Greater)}
    pub fn neg(&mut self,a:NodeId)->Result<NodeId,AadError>{self.unary(a,Op::Neg)}
    pub fn exp(&mut self,a:NodeId)->Result<NodeId,AadError>{self.unary(a,Op::Exp)}
    pub fn ln(&mut self,a:NodeId)->Result<NodeId,AadError>{self.unary(a,Op::Ln)}
    /// Lazy select; does not evaluate the inactive branch. A nonzero condition selects `positive`.
    pub fn select(&mut self, condition:NodeId,positive:NodeId,zero:NodeId)->Result<NodeId,AadError>{
        self.check(condition)?; self.check(positive)?; self.check(zero)?;
        Ok(self.push(Op::Select{condition,positive,zero}))
    }
    pub fn compile(self, outputs: &[NodeId]) -> Result<CompiledGraph,AadError> {
        if outputs.is_empty() { return Err(AadError::EmptyOutputs); }
        if self.nodes.len() > MAX_NODES { return Err(AadError::GraphTooLarge); }
        for &o in outputs { self.check(o)?; }
        Ok(CompiledGraph { nodes: self.nodes.into_boxed_slice(), outputs: outputs.into(), input_count:self.input_count })
    }
}

/// Immutable graph, shareable across workers. Each worker needs its own Workspace.
pub struct CompiledGraph { nodes: Box<[Op]>, outputs: Box<[NodeId]>, input_count: usize }
impl CompiledGraph {
    pub fn input_count(&self)->usize{self.input_count}
    pub fn output_count(&self)->usize{self.outputs.len()}
    pub fn node_count(&self)->usize{self.nodes.len()}
    pub fn workspace(&self)->Workspace{Workspace {
        primal:vec![0.0;self.nodes.len()], adjoint:vec![0.0;self.nodes.len()],
        visited:vec![false;self.nodes.len()], output:vec![0.0;self.outputs.len()],
        input_adjoints:vec![0.0;self.input_count],
    }}
    fn preflight(&self,inputs:&[f64],w:&Workspace)->Result<(),AadError>{
        if inputs.len()!=self.input_count{return Err(AadError::InputShape)}
        if inputs.iter().any(|v|!v.is_finite()){return Err(AadError::NonFiniteInput)}
        if w.primal.len()!=self.nodes.len()||w.adjoint.len()!=self.nodes.len()||w.visited.len()!=self.nodes.len()||w.output.len()!=self.outputs.len()||w.input_adjoints.len()!=self.input_count {
            return Err(AadError::InvalidWorkspace)
        }
        Ok(())
    }
    fn eval(&self, id:NodeId,inputs:&[f64],w:&mut Workspace)->Result<f64,AadError>{
        if w.visited[id.0]{return Ok(w.primal[id.0])}
        let value=match self.nodes[id.0]{
            Op::Input(i)=>inputs[i], Op::Constant(v)=>v,
            Op::Neg(a)=>-self.eval(a,inputs,w)?,
            Op::Exp(a)=>self.eval(a,inputs,w)?.exp(),
            Op::Ln(a)=>self.eval(a,inputs,w)?.ln(),
            Op::Select{condition,positive,zero}=>{
                let c=self.eval(condition,inputs,w)?;
                self.eval(if c!=0.0{positive}else{zero},inputs,w)?
            }
            other=>{
                let (a,b)=match other{
                    Op::Add(a,b)|Op::Sub(a,b)|Op::Mul(a,b)|Op::Div(a,b)|Op::Greater(a,b)|Op::Min(a,b)|Op::Max(a,b)=>(a,b),
                    _=>unreachable!(),
                };
                let x=self.eval(a,inputs,w)?;let y=self.eval(b,inputs,w)?;
                match other{
                    Op::Add(..)=>x+y,Op::Sub(..)=>x-y,Op::Mul(..)=>x*y,Op::Div(..)=>x/y,
                    Op::Greater(..)=>if x>y{1.0}else{0.0},
                    Op::Min(..)=>x.min(y),Op::Max(..)=>x.max(y),_=>unreachable!(),
                }
            }
        };
        if !value.is_finite(){return Err(AadError::NonFiniteResult{node:id.0})}
        w.primal[id.0]=value;w.visited[id.0]=true;Ok(value)
    }
    /// Evaluate all outputs. The workspace owns returned values; each call overwrites them.
    pub fn evaluate<'a>(&self,inputs:&[f64],w:&'a mut Workspace)->Result<&'a [f64],AadError>{
        self.preflight(inputs,w)?;w.visited.fill(false);
        for (k,&id) in self.outputs.iter().enumerate(){w.output[k]=self.eval(id,inputs,w)?;}
        Ok(&w.output)
    }
    /// Vector-Jacobian product: J(inputs)^T * output_seeds. Multiple outputs supported.
    pub fn vjp<'a>(&self,inputs:&[f64],output_seeds:&[f64],w:&'a mut Workspace)->Result<&'a [f64],AadError>{
        if output_seeds.len()!=self.outputs.len() || output_seeds.iter().any(|v|!v.is_finite()){
            return Err(AadError::SeedShape);
        }
        self.evaluate(inputs,w)?;
        w.adjoint.fill(0.0);w.input_adjoints.fill(0.0);
        for (&o,&seed) in self.outputs.iter().zip(output_seeds){w.adjoint[o.0]+=seed;}
        for i in (0..self.nodes.len()).rev(){
            if !w.visited[i]{continue}
            let seed=w.adjoint[i];if seed==0.0{continue}
            let p=&w.primal;
            match self.nodes[i]{
                Op::Input(k)=>w.input_adjoints[k]+=seed,Op::Constant(_)|Op::Greater(..)=>{},
                Op::Add(a,b)=>{w.adjoint[a.0]+=seed;w.adjoint[b.0]+=seed;},
                Op::Sub(a,b)=>{w.adjoint[a.0]+=seed;w.adjoint[b.0]-=seed;},
                Op::Mul(a,b)=>{w.adjoint[a.0]+=seed*p[b.0];w.adjoint[b.0]+=seed*p[a.0];},
                Op::Div(a,b)=>{w.adjoint[a.0]+=seed/p[b.0];w.adjoint[b.0]-=seed*p[a.0]/(p[b.0]*p[b.0]);},
                Op::Neg(a)=>w.adjoint[a.0]-=seed,
                Op::Exp(a)=>w.adjoint[a.0]+=seed*p[i],
                Op::Ln(a)=>w.adjoint[a.0]+=seed/p[a.0],
                Op::Min(a,b)=>{if p[a.0]<=p[b.0]{w.adjoint[a.0]+=seed}else{w.adjoint[b.0]+=seed}},
                Op::Max(a,b)=>{if p[a.0]>=p[b.0]{w.adjoint[a.0]+=seed}else{w.adjoint[b.0]+=seed}},
                Op::Select{condition,positive,zero}=>{
                    let selected=if p[condition.0]!=0.0{positive}else{zero};w.adjoint[selected.0]+=seed;
                }
            }
            if !w.adjoint[i].is_finite(){return Err(AadError::NonFiniteResult{node:i})}
        }
        if w.input_adjoints.iter().any(|v|!v.is_finite()){return Err(AadError::NonFiniteResult{node:0})}
        Ok(&w.input_adjoints)
    }
}
/// Mutable per-worker cache. Do not share between concurrent evaluations.
pub struct Workspace { primal:Vec<f64>,adjoint:Vec<f64>,visited:Vec<bool>,output:Vec<f64>,input_adjoints:Vec<f64> }

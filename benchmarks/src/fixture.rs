//! Portable numeric graphs. Node operands are zero-based indices into the program.
use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Op {
    Input(usize),
    Constant(f64),
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Max(usize, usize),
    Greater(usize, usize),
    Neg(usize),
    Log(usize),
    Select(usize, usize, usize),
}

#[derive(Default, Serialize)]
pub struct Program {
    pub nodes: Vec<Op>,
    pub outputs: Vec<usize>,
    pub seeds: Vec<f64>,
}
impl Program {
    fn push(&mut self, op: Op) -> usize {
        let i = self.nodes.len();
        self.nodes.push(op);
        i
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Case {
    Affine,
    Average(usize),
    Branch(usize),
    Width(usize),
}

impl Case {
    pub fn parse(id: &str) -> Option<Self> {
        if id == "affine32-v1" {
            return Some(Self::Affine);
        }
        let (name, n) = id.split_once('-')?;
        let n = n.strip_suffix("-v1")?.parse().ok()?;
        match (name, n) {
            ("average", 1 | 12 | 120) => Some(Self::Average(n)),
            ("branch", 1 | 50 | 99) => Some(Self::Branch(n)),
            ("width", 1 | 10 | 100 | 1000) => Some(Self::Width(n)),
            _ => None,
        }
    }
    pub fn inputs(self) -> usize {
        match self {
            Self::Affine => 2,
            Self::Average(n) | Self::Width(n) => n,
            Self::Branch(_) => 12,
        }
    }
    pub fn input(self, path: usize, j: usize) -> f64 {
        // All integer products fit in u32 for the registered maximum of 1M paths.
        match self {
            Self::Affine if j == 0 => ((path * 47 % 1021) as f64 - 510.) / 256.,
            Self::Affine => 1. + ((path * 31 % 127) as f64 - 63.) / 4096.,
            Self::Average(_) => {
                100. + ((path * 47 % 1021) as f64 - 510.) / 32.
                    + (((path % 197 + j * 17) % 197) as f64 - 98.) / 128.
            }
            Self::Branch(percent) => {
                let magnitude = 1. + ((path % 37 + j * 7) % 37) as f64 / 32.;
                if path % 100 < percent && j == (path / 100 + path % 100) % 12 {
                    magnitude
                } else {
                    -magnitude
                }
            }
            Self::Width(_) => (((path % 509 + j * 19) % 509) as f64 - 254.) / 256.,
        }
    }
    pub fn program(self) -> Program {
        use Op::*;
        let mut p = Program::default();
        for i in 0..self.inputs() {
            p.push(Input(i));
        }
        match self {
            Self::Affine => {
                let bias = p.push(Constant(1. / 64.));
                let mut s = 0;
                for _ in 0..32 {
                    let z = p.push(Mul(s, 1));
                    s = p.push(Add(z, bias));
                }
                let z = p.push(Mul(s, 0));
                p.outputs = vec![s, z];
                p.seeds = vec![1., -0.5];
            }
            Self::Average(n) => {
                let zero = p.push(Constant(0.));
                let mut sum = zero;
                for i in 0..n {
                    sum = p.push(Add(sum, i));
                }
                let weight = p.push(Constant(1. / n as f64));
                let mean = p.push(Mul(sum, weight));
                let strike = p.push(Constant(100.));
                let intrinsic = p.push(Sub(mean, strike));
                let out = p.push(Max(intrinsic, zero));
                p.outputs = vec![out];
                p.seeds = vec![1.];
            }
            Self::Branch(_) => {
                let zero = p.push(Constant(0.));
                let fraction = p.push(Constant(1. / 16.));
                let mut tail = zero;
                for i in (0..12).rev() {
                    let gate = p.push(Greater(i, zero));
                    let base = p.push(Constant(1. + i as f64 / 16.));
                    let log = p.push(Log(i));
                    let hit = p.push(Add(base, log));
                    let minus = p.push(Neg(i));
                    let log = p.push(Log(minus));
                    let coupon = p.push(Mul(fraction, log));
                    let miss = p.push(Add(coupon, tail));
                    tail = p.push(Select(gate, hit, miss));
                }
                p.outputs = vec![tail];
                p.seeds = vec![1.];
            }
            Self::Width(n) => {
                let zero = p.push(Constant(0.));
                let (mut sum, mut norm) = (zero, zero);
                for i in 0..n {
                    sum = p.push(Add(sum, i));
                    let square = p.push(Mul(i, i));
                    norm = p.push(Add(norm, square));
                }
                p.outputs = vec![sum, norm];
                p.seeds = vec![0.5, -0.25];
            }
        }
        p
    }
    /// Independent closed-form or forward-recurrence references for every coordinate.
    pub fn oracle(self, x: &[f64], y: &mut [f64], grad: &mut [f64]) {
        match self {
            Self::Affine => {
                let (x, q) = (x[0], x[1]);
                let (mut s, mut dx, mut dq) = (x, 1., 0.);
                for _ in 0..32 {
                    dq = dq * q + s;
                    dx *= q;
                    s = s * q + 1. / 64.;
                }
                y.copy_from_slice(&[s, s * x]);
                grad.copy_from_slice(&[dx * (1. - 0.5 * x) - 0.5 * s, dq * (1. - 0.5 * x)]);
            }
            Self::Average(n) => {
                let weight = 1. / n as f64;
                let intrinsic = x.iter().sum::<f64>() * weight - 100.;
                y[0] = intrinsic.max(0.);
                grad.fill(if intrinsic >= 0. { weight } else { 0. });
            }
            Self::Branch(_) => {
                grad.fill(0.);
                let mut sum = 0.;
                for (i, &v) in x.iter().enumerate() {
                    if v > 0. {
                        sum += 1. + i as f64 / 16. + v.ln();
                        grad[i] = 1. / v;
                        break;
                    }
                    sum += (-v).ln() / 16.;
                    grad[i] = 1. / (16. * v);
                }
                y[0] = sum;
            }
            Self::Width(_) => {
                y[0] = x.iter().sum();
                y[1] = x.iter().map(|v| v * v).sum();
                for (g, v) in grad.iter_mut().zip(x) {
                    *g = 0.5 - 0.5 * v;
                }
            }
        }
    }
}

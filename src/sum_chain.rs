//! A scalar expression of one ordered input sum and constant operands.
use crate::{Block, NodeId, Op};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Operator {
    Add,
    Sub,
    Mul,
    Min,
    Max,
}

#[derive(Clone)]
struct Step {
    op: Operator,
    constant: f64,
}

#[derive(Clone, Copy)]
struct AffineClamp {
    scale: f64,
    subtract: f64,
    bound: f64,
    floor: bool,
}
impl AffineClamp {
    fn compile(mut steps: &[Step]) -> Option<Self> {
        let mut take = |op| match steps.split_first() {
            Some((step, rest)) if step.op == op => {
                steps = rest;
                Some(step.constant)
            }
            _ => None,
        };
        let scale = take(Operator::Mul).unwrap_or(1.0);
        let subtract = take(Operator::Sub)
            .or_else(|| take(Operator::Add).map(|value| -value))
            .unwrap_or(0.0);
        let (bound, floor) = match steps {
            [step] if step.op == Operator::Min => (step.constant, false),
            [step] if step.op == Operator::Max => (step.constant, true),
            [] => (f64::NEG_INFINITY, true),
            _ => return None,
        };
        Some(Self {
            scale,
            subtract,
            bound,
            floor,
        })
    }

    #[inline]
    fn apply(self, sum: f64, limit: f64) -> Option<(f64, bool)> {
        let scaled = sum * self.scale;
        let shifted = scaled - self.subtract;
        if sum.abs() <= limit && scaled.abs() <= limit && shifted.abs() <= limit {
            let (value, active) = if self.floor {
                (shifted.max(self.bound), shifted >= self.bound)
            } else {
                (shifted.min(self.bound), shifted <= self.bound)
            };
            if value.abs() <= limit {
                return Some((value, active));
            }
        }
        None
    }
}
#[derive(Clone)]
pub(super) struct SumChain {
    affine: Option<AffineClamp>,
    initial: f64,
    limit: f64,
    scales: Box<[f64]>,
    steps: Box<[Step]>,
}

impl SumChain {
    pub(super) fn compile(
        nodes: &[Op],
        outputs: &[NodeId],
        blocks: &[Block],
        n: usize,
    ) -> Option<Self> {
        let [out] = outputs else { return None };
        let (mut slot, mut steps) = (out.0, Vec::new());
        let (initial_slot, sum) = blocks.iter().find_map(|block| match block {
            Block::InputSum {
                initial,
                output,
                inputs,
                shared: false,
                ..
            } if inputs.len() == n => Some((*initial, *output)),
            _ => None,
        })?;
        let Op::Constant(initial) = nodes[initial_slot] else {
            return None;
        };
        while slot != sum {
            let (op, left, right) = match nodes[slot] {
                Op::Add(a, b) => (Operator::Add, a, b),
                Op::Sub(a, b) => (Operator::Sub, a, b),
                Op::Mul(a, b) => (Operator::Mul, a, b),
                Op::Min(a, b) => (Operator::Min, a, b),
                Op::Max(a, b) => (Operator::Max, a, b),
                _ => return None,
            };
            let Op::Constant(constant) = nodes[right] else {
                return None;
            };
            steps.push(Step { op, constant });
            slot = left;
        }
        steps.reverse();
        let scales: Box<[_]> = steps
            .iter()
            .rev()
            .filter_map(|step| match step.op {
                Operator::Mul => Some(step.constant),
                _ => None,
            })
            .collect();
        // Reverse seeds grow only at multiplies. Bound each factor by a power
        // of two, then reserve bits for adding every contribution to one constant.
        let growth: u32 = scales
            .iter()
            .filter(|s| s.abs() > 1.0)
            .map(|s| ((s.abs().to_bits() >> 52) as u32 - 1023) + 1)
            .sum();
        let fan_in = (steps.len() + 1).next_power_of_two().ilog2();
        let exponent = 1023u32.checked_sub(growth + fan_in)? / 2;
        let limit = f64::from_bits(u64::from(1023 + exponent) << 52);
        Some(Self {
            affine: AffineClamp::compile(&steps),
            initial,
            limit,
            scales,
            steps: steps.into_boxed_slice(),
        })
    }

    #[cfg(feature = "simd")]
    pub(super) fn initial(&self) -> f64 {
        self.initial
    }

    fn within_bound(&self, value: f64) -> bool {
        value.abs() <= self.limit
    }

    // A winning constant cuts the only path from the output to the sum.
    // A tie keeps the left operand, exactly as in primitive reverse execution.
    pub(super) fn evaluate(&self, inputs: &[f64]) -> Option<(f64, bool)> {
        self.apply_sum(crate::ordered_sum::ordered_sum(self.initial, inputs))
    }

    pub(super) fn accepts_integer_sum(&self) -> bool {
        self.initial == 0.0
    }

    pub(super) fn apply_sum(&self, mut value: f64) -> Option<(f64, bool)> {
        if let Some(a) = self.affine {
            return a.apply(value, self.limit);
        }
        let mut active = true;
        if !self.within_bound(value) {
            return None;
        }
        for step in &self.steps {
            value = match step.op {
                Operator::Add => value + step.constant,
                Operator::Sub => value - step.constant,
                Operator::Mul => value * step.constant,
                Operator::Min => {
                    active &= value <= step.constant;
                    value.min(step.constant)
                }
                Operator::Max => {
                    active &= value >= step.constant;
                    value.max(step.constant)
                }
            };
            if !self.within_bound(value) {
                return None;
            }
        }
        Some((value, active))
    }

    pub(super) fn vjp(&self, inputs: &[f64], seed: f64) -> Option<(f64, f64)> {
        self.vjp_from_sum(crate::ordered_sum::ordered_sum(self.initial, inputs), seed)
    }

    pub(super) fn vjp_from_sum(&self, sum: f64, seed: f64) -> Option<(f64, f64)> {
        if !self.within_bound(seed) {
            return None;
        }
        if let Some(a) = self.affine {
            let (value, active) = a.apply(sum, self.limit)?;
            let gradient = if active { seed * a.scale } else { 0.0 };
            return Some((value, 0.0 + gradient));
        }
        let (value, active) = self.apply_sum(sum)?;
        // With bounded primals and seed, even all constant adjoints combined
        // remain finite. Out-of-range calls use the original checked schedule.
        let grad = if active {
            self.scales.iter().fold(seed, |g, scale| g * scale)
        } else {
            0.0
        };
        Some((value, 0.0 + grad))
    }
}

//! A scalar expression of one ordered input sum and constant operands.
use crate::{Block, NodeId, Op};

#[derive(Clone, Copy)]
enum Operator {
    Add,
    Sub,
    Mul,
    Min,
    Max,
}

struct Step {
    op: Operator,
    constant: f64,
}

pub(super) struct SumChain {
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
            initial,
            limit,
            scales,
            steps: steps.into_boxed_slice(),
        })
    }

    fn within_bound(&self, value: f64) -> bool {
        value.abs() <= self.limit
    }

    // A winning constant cuts the only path from the output to the sum.
    // A tie keeps the left operand, exactly as in primitive reverse execution.
    pub(super) fn evaluate(&self, inputs: &[f64]) -> Option<(f64, bool)> {
        let mut value = inputs.iter().fold(self.initial, |a, b| a + b);
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
        if !self.within_bound(seed) {
            return None;
        }
        let (value, active) = self.evaluate(inputs)?;
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

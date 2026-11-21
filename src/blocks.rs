//! Order-preserving loops for regular sections of a static graph.
use super::{AadError, CompiledGraph, NodeId, Op, StaticSchedule, Workspace};
use crate::moments::Moments;
use crate::recurrence::Recurrence;
use std::ops::{Add, Mul, Range};

pub(crate) trait BlockValue:
    Copy + Add<Output = Self> + Mul<Output = Self> + From<Self::Stored> + Into<Self::Stored>
{
    type Stored: Copy;
    const ZERO: Self;
    fn finite(self) -> bool;
}
impl BlockValue for f64 {
    type Stored = f64;
    const ZERO: Self = 0.0;
    fn finite(self) -> bool {
        self.is_finite()
    }
}
#[cfg(feature = "simd")]
impl BlockValue for wide::f64x2 {
    type Stored = [f64; 2];
    const ZERO: Self = Self::ZERO;
    fn finite(self) -> bool {
        self.is_finite().all()
    }
}

pub(super) enum Block {
    Scalar(Range<usize>),
    Moments(Moments),
    Recurrence(Recurrence),
    Dot {
        initial: usize,
        output: usize,
        left: Range<usize>,
        right: Range<usize>,
    },
    InputSum {
        initial: usize,
        output: usize,
        inputs: Range<usize>,
        slots: Range<usize>,
        shared: bool,
    },
}

impl Block {
    pub(super) fn compile(nodes: &[Op], outputs: &[NodeId], order: &[u16]) -> Box<[Self]> {
        if order.len() < 8 {
            return Box::default();
        }
        // Count output handles as uses, including multiple handles to one slot.
        let mut uses = vec![0usize; nodes.len()];
        for &slot in order {
            for child in nodes[usize::from(slot)].operands() {
                uses[child] += 1;
            }
        }
        for out in outputs {
            uses[out.0] += 1;
        }
        let mut blocks = Vec::new();
        let (mut scalar_start, mut i) = (0, 0);
        while i < order.len() {
            let remaining = &order[i..];
            let region = match nodes[usize::from(remaining[0])] {
                Op::Input(_) => Self::input_sum(nodes, &uses, remaining),
                Op::Mul(..) => Self::dot(nodes, &uses, remaining).or_else(|| {
                    Recurrence::compile(nodes, &uses, remaining)
                        .map(|(run, n)| (Self::Recurrence(run), n))
                }),
                _ => None,
            };
            if let Some((block, consumed)) = region {
                if scalar_start < i {
                    blocks.push(Self::Scalar(scalar_start..i));
                }
                if let Some(moments) = blocks
                    .last()
                    .and_then(|last| Self::moments(last, &block, &uses))
                {
                    blocks.pop();
                    blocks.push(Self::Moments(moments));
                } else {
                    blocks.push(block);
                }
                i += consumed;
                scalar_start = i;
            } else {
                i += 1;
            }
        }
        if blocks.is_empty() {
            return Box::default();
        }
        if scalar_start < order.len() {
            blocks.push(Self::Scalar(scalar_start..order.len()));
        }
        blocks.into_boxed_slice()
    }

    fn moments(sum: &Self, dot: &Self, uses: &[usize]) -> Option<Moments> {
        let Self::InputSum {
            initial: sum_initial,
            output: sum_output,
            inputs,
            slots,
            ..
        } = sum
        else {
            return None;
        };
        let Self::Dot {
            initial: square_initial,
            output: square_output,
            left,
            right,
        } = dot
        else {
            return None;
        };
        // The leaves have no other readers or output seeds. The second initializer
        // must already be available when the two passes begin.
        if slots != left
            || slots != right
            || square_initial == sum_output
            || !uses[slots.clone()].iter().all(|&n| n == 3)
        {
            return None;
        }
        Some(Moments {
            inputs: inputs.clone(),
            first_node: slots.start,
            sum_initial: *sum_initial,
            square_initial: *square_initial,
            sum_output: *sum_output,
            square_output: *square_output,
        })
    }

    // Every reachable adjoint has one defining step. Fused intermediates
    // have no outside consumers, and reverse never reads their scratch slots.
    pub(super) fn reset_ranges(
        blocks: &[Block],
        order: &[u16],
        slots: usize,
        map: impl Fn(usize) -> usize,
    ) -> Box<[Range<usize>]> {
        if blocks.is_empty() || slots < 32 {
            return Box::default();
        }
        let mut live = vec![false; slots];
        for block in blocks {
            match block {
                Self::Scalar(range) => {
                    for &id in &order[range.clone()] {
                        live[map(usize::from(id))] = true;
                    }
                }
                Self::InputSum {
                    output,
                    slots: leaves,
                    shared,
                    ..
                } => {
                    live[map(*output)] = true;
                    if *shared {
                        for leaf in leaves.clone() {
                            live[map(leaf)] = true;
                        }
                    }
                }
                Self::Recurrence(run) => live[map(run.output())] = true,
                Self::Dot { output, .. } => live[map(*output)] = true,
                Self::Moments(m) => {
                    live[map(m.sum_output)] = true;
                    live[map(m.square_output)] = true;
                }
            }
        }
        let mut ranges: Vec<Range<usize>> = Vec::new();
        for (slot, used) in live.into_iter().enumerate() {
            if used {
                match ranges.last_mut() {
                    Some(r) if slot < r.end + 8 => r.end = slot + 1,
                    _ => ranges.push(slot..slot + 1),
                }
            }
        }
        if ranges.iter().map(Range::len).sum::<usize>() > slots / 2 {
            Box::default()
        } else {
            ranges.into()
        }
    }

    // Fold products of two contiguous primal slices, including overlapping
    // slices. Only the last accumulator may have other consumers.
    fn dot(nodes: &[Op], uses: &[usize], order: &[u16]) -> Option<(Self, usize)> {
        let pair = |s: &[u16]| {
            let product = usize::from(*s.first()?);
            let dest = usize::from(*s.get(1)?);
            let (Op::Mul(a, b), Op::Add(prev, term)) = (nodes[product], nodes[dest]) else {
                return None;
            };
            (term == product && uses[product] == 1).then_some((a, b, prev, dest))
        };
        let (a, b, initial, mut output) = pair(order)?;
        let mut len = 1;
        for step in order[2..].chunks_exact(2) {
            let Some((left, right, prev, dest)) = pair(step) else {
                break;
            };
            if left != a + len || right != b + len || prev != output || uses[prev] != 1 {
                break;
            }
            output = dest;
            len += 1;
        }
        (len >= 4).then_some((
            Self::Dot {
                initial,
                output,
                left: a..a + len,
                right: b..b + len,
            },
            2 * len,
        ))
    }

    // Match a left fold whose next operand is a fresh input at each step.
    // An intermediate observed elsewhere terminates the run at that output.
    fn input_sum(nodes: &[Op], uses: &[usize], order: &[u16]) -> Option<(Self, usize)> {
        let pair = |s: &[u16]| {
            let leaf = usize::from(*s.first()?);
            let dest = usize::from(*s.get(1)?);
            let Op::Input(input) = nodes[leaf] else {
                return None;
            };
            let Op::Add(a, b) = nodes[dest] else {
                return None;
            };
            if b != leaf || a == leaf {
                return None;
            }
            Some((leaf, input, a, dest))
        };
        let (first_slot, first_input, initial, mut output) = pair(order)?;
        let mut len = 1;
        let mut shared = uses[first_slot] != 1;
        for s in order[2..].chunks_exact(2) {
            let Some((leaf, input, previous, dest)) = pair(s) else {
                break;
            };
            if previous != output
                || uses[previous] != 1
                || leaf != first_slot + len
                || input != first_input + len
            {
                break;
            }
            output = dest;
            len += 1;
            shared |= uses[leaf] != 1;
        }
        (len >= 4).then_some((
            Self::InputSum {
                initial,
                output,
                inputs: first_input..first_input + len,
                slots: first_slot..first_slot + len,
                shared,
            },
            len * 2,
        ))
    }
}

impl CompiledGraph {
    pub(super) fn evaluate_blocks(
        &self,
        schedule: &StaticSchedule,
        inputs: &[f64],
        w: &mut Workspace,
    ) -> bool {
        for block in &schedule.blocks {
            match block {
                Block::Recurrence(run) => {
                    if !run.forward(&mut w.primal) {
                        return false;
                    }
                }
                Block::Moments(m) => {
                    if !m.forward(inputs, &mut w.primal) {
                        return false;
                    }
                }
                Block::Scalar(range) => {
                    for &slot in &schedule.order[range.clone()] {
                        let i = usize::from(slot);
                        let value = self.nodes[i].value(inputs, &w.primal);
                        if !value.is_finite() {
                            return false;
                        }
                        w.primal[i] = value;
                    }
                }
                Block::Dot {
                    initial,
                    output,
                    left,
                    right,
                } => {
                    let sum = w.primal[left.clone()]
                        .iter()
                        .zip(&w.primal[right.clone()])
                        .fold(w.primal[*initial], |sum, (a, b)| sum + a * b);
                    if !sum.is_finite() {
                        return false;
                    }
                    w.primal[*output] = sum;
                }
                Block::InputSum {
                    initial,
                    output,
                    inputs: range,
                    slots,
                    shared,
                } => {
                    let values = &inputs[range.clone()];
                    if *shared {
                        w.primal[slots.clone()].copy_from_slice(values);
                    }
                    // Finite addends cannot turn an infinite sum finite. On
                    // overflow the caller replays scalar steps for the precise error.
                    let sum = crate::ordered_sum::ordered_sum(w.primal[*initial], values);
                    if !sum.is_finite() {
                        return false;
                    }
                    w.primal[*output] = sum;
                }
            }
        }
        true
    }

    pub(super) fn reverse_blocks(
        &self,
        schedule: &StaticSchedule,
        inputs: &[f64],
        w: &mut Workspace,
    ) -> Result<(), AadError> {
        for block in schedule.blocks.iter().rev() {
            match block {
                Block::Recurrence(run) => run
                    .reverse(&w.primal, &mut w.adjoint)
                    .map_err(|node| AadError::NonFiniteResult { node })?,
                Block::Moments(m) => m.reverse(
                    inputs,
                    &mut w.adjoint,
                    &mut w.input_adjoints,
                    m.sum_output,
                    m.square_output,
                )?,
                Block::Scalar(range) => self.reverse_slots(
                    &schedule.order[range.clone()],
                    &w.primal,
                    &mut w.adjoint,
                    &mut w.input_adjoints,
                )?,
                Block::Dot {
                    initial,
                    output,
                    left,
                    right,
                } => {
                    let seed = w.adjoint[*output];
                    if !seed.is_finite() {
                        return Err(AadError::NonFiniteResult { node: *output });
                    }
                    if seed == 0.0 {
                        continue;
                    }
                    for (a, b) in left.clone().zip(right.clone()).skip(1).rev() {
                        w.adjoint[a] += seed * w.primal[b];
                        w.adjoint[b] += seed * w.primal[a];
                    }
                    // The first Add is reversed before the first Mul; initial
                    // may alias one of its operands.
                    w.adjoint[*initial] += seed;
                    w.adjoint[left.start] += seed * w.primal[right.start];
                    w.adjoint[right.start] += seed * w.primal[left.start];
                }
                Block::InputSum {
                    initial,
                    output,
                    inputs,
                    slots,
                    shared,
                } => {
                    let seed = w.adjoint[*output];
                    if !seed.is_finite() {
                        return Err(AadError::NonFiniteResult { node: *output });
                    }
                    if *shared {
                        let gradients = &mut w.input_adjoints[inputs.clone()];
                        for (i, (g, previous)) in gradients
                            .iter_mut()
                            .zip(&w.adjoint[slots.clone()])
                            .enumerate()
                            .rev()
                        {
                            let value = previous + seed;
                            if !value.is_finite() {
                                return Err(AadError::NonFiniteResult {
                                    node: slots.start + i,
                                });
                            }
                            *g = 0.0 + value;
                        }
                    } else {
                        w.input_adjoints[inputs.clone()].fill(0.0 + seed);
                    }
                    if seed != 0.0 {
                        w.adjoint[*initial] += seed;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;

//! Translate scalar reduction blocks to the compact two-lane tape at preparation.
use super::{Batch2, f64x2};
use crate::moments::Moments;
use crate::recurrence::Recurrence;
use crate::{AadError, Block};
use std::ops::Range;

pub(super) enum Region {
    Steps(Range<usize>),
    Moments(Moments),
    Recurrence(Recurrence),
    Sum {
        initial: usize,
        output: usize,
        first_leaf: usize,
        inputs: Range<usize>,
        shared: bool,
    },
    Dot {
        initial: usize,
        output: usize,
        operands: Box<[[u16; 2]]>,
    },
}

impl Region {
    pub(super) fn prepare(block: &Block, slots: &[usize]) -> Self {
        match block {
            Block::Recurrence(run) => Self::Recurrence(run.compact(slots)),
            Block::Moments(m) => Self::Moments(m.compact(slots)),
            Block::Scalar(range) => Self::Steps(range.clone()),
            Block::InputSum {
                initial,
                output,
                inputs,
                slots: leaves,
                shared,
            } => Self::Sum {
                initial: slots[*initial],
                output: slots[*output],
                first_leaf: slots[leaves.start],
                inputs: inputs.clone(),
                shared: *shared,
            },
            Block::Dot {
                initial,
                output,
                left,
                right,
            } => Self::Dot {
                initial: slots[*initial],
                output: slots[*output],
                operands: left
                    .clone()
                    .zip(right.clone())
                    .map(|(a, b)| [slots[a] as u16, slots[b] as u16])
                    .collect(),
            },
        }
    }
}

impl Batch2 {
    pub(super) fn forward_regions(&mut self, inputs: &[[f64; 2]]) -> Result<(), AadError> {
        for region in &self.regions {
            let p = &mut self.primal;
            let (output, value) = match region {
                Region::Recurrence(run) => {
                    if !run.forward(p) {
                        return Err(AadError::NonFiniteResult {
                            node: self.steps[run.output()].source_node,
                        });
                    }
                    continue;
                }
                Region::Moments(m) => {
                    if !m.forward(inputs, p) {
                        return Err(AadError::NonFiniteResult {
                            node: self.steps[m.square_output].source_node,
                        });
                    }
                    continue;
                }
                Region::Steps(range) => {
                    Self::forward_steps(&self.steps[range.clone()], range.start, inputs, p)?;
                    continue;
                }
                Region::Sum {
                    initial,
                    output,
                    first_leaf,
                    inputs: indices,
                    shared,
                } => {
                    let values = &inputs[indices.clone()];
                    if *shared {
                        // The compact tape alternates an input and its consuming add.
                        for (slot, &pair) in p[*first_leaf..].iter_mut().step_by(2).zip(values) {
                            *slot = f64x2::new(pair);
                        }
                    }
                    (
                        *output,
                        values
                            .iter()
                            .fold(p[*initial], |s, &pair| s + f64x2::new(pair)),
                    )
                }
                Region::Dot {
                    initial,
                    output,
                    operands,
                } => (
                    *output,
                    operands.iter().fold(p[*initial], |s, &[a, b]| {
                        s + p[usize::from(a)] * p[usize::from(b)]
                    }),
                ),
            };
            if !value.is_finite().all() {
                return Err(AadError::NonFiniteResult {
                    node: self.steps[output].source_node,
                });
            }
            p[output] = value;
        }
        Ok(())
    }

    pub(super) fn reverse_regions(&mut self, inputs: &[[f64; 2]]) -> Result<(), AadError> {
        let adj = &mut self.adjoint;
        for region in self.regions.iter().rev() {
            match region {
                Region::Recurrence(run) => {
                    run.reverse(&self.primal, adj)
                        .map_err(|slot| AadError::NonFiniteResult {
                            node: self.steps[slot].source_node,
                        })?
                }
                Region::Moments(m) => m.reverse(
                    inputs,
                    adj,
                    &mut self.gradients,
                    self.steps[m.sum_output].source_node,
                    self.steps[m.square_output].source_node,
                )?,
                Region::Steps(range) => Self::reverse_steps(
                    &self.steps[range.clone()],
                    range.start,
                    &self.primal,
                    adj,
                    &mut self.gradients,
                )?,
                Region::Sum {
                    initial,
                    output,
                    first_leaf,
                    inputs,
                    shared,
                } => {
                    let seed = adj[*output];
                    if !seed.is_finite().all() {
                        return Err(AadError::NonFiniteResult {
                            node: self.steps[*output].source_node,
                        });
                    }
                    if *shared {
                        for (i, g) in self.gradients[inputs.clone()].iter_mut().enumerate().rev() {
                            let leaf = first_leaf + 2 * i;
                            let grad = adj[leaf] + seed;
                            if !grad.is_finite().all() {
                                return Err(AadError::NonFiniteResult {
                                    node: self.steps[leaf].source_node,
                                });
                            }
                            *g = grad.to_array();
                        }
                    } else {
                        self.gradients[inputs.clone()].fill(seed.to_array());
                    }
                    adj[*initial] += seed;
                }
                Region::Dot {
                    initial,
                    output,
                    operands,
                } => {
                    let seed = adj[*output];
                    if !seed.is_finite().all() {
                        return Err(AadError::NonFiniteResult {
                            node: self.steps[*output].source_node,
                        });
                    }
                    for (i, &[a, b]) in operands.iter().enumerate().rev() {
                        if i == 0 {
                            adj[*initial] += seed;
                        }
                        let (a, b) = (usize::from(a), usize::from(b));
                        adj[a] += seed * self.primal[b];
                        adj[b] += seed * self.primal[a];
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::GraphBuilder;

    #[test]
    fn compact_dot_handles_overlapping_operands_and_intermediate_outputs() {
        let mut b = GraphBuilder::new();
        let x: Vec<_> = (0..16).map(|_| b.input()).collect();
        let mut outputs = x.clone();
        let mut total = x[0];
        for (i, pair) in x.windows(2).enumerate() {
            let product = b.mul(pair[0], pair[1]).unwrap();
            total = b.add(total, product).unwrap();
            if i == 6 {
                outputs.push(product);
            }
        }
        outputs.extend([total, total]);
        let g = b.compile(&outputs).unwrap();
        let mut optimized = g.batch2().unwrap();
        let mut primitive = g.batch2().unwrap();
        assert!(
            optimized
                .regions
                .iter()
                .any(|r| matches!(r, super::Region::Dot { .. }))
        );
        primitive.regions = Box::default();
        let bits = |values: &[[f64; 2]]| {
            values
                .iter()
                .map(|v| v.map(f64::to_bits))
                .collect::<Vec<_>>()
        };
        for i in 0..100 {
            let x: [[f64; 2]; 16] = std::array::from_fn(|k| {
                [
                    ((k + i) as f64 * 0.125).sin(),
                    ((2 * k + i) as f64 * 0.25).cos(),
                ]
            });
            let seeds: [[f64; 2]; 19] = std::array::from_fn(|k| [((i + k) as f64).cos(), -0.0]);
            assert_eq!(
                optimized.evaluate(&x).map(bits),
                primitive.evaluate(&x).map(bits)
            );
            assert_eq!(
                optimized.vjp(&x, &seeds).map(bits),
                primitive.vjp(&x, &seeds).map(bits)
            );
        }
    }
}

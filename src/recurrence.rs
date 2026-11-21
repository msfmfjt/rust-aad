//! Repeated x <- x * scale + bias with a distinct, private intermediate per step.
use crate::{Op, blocks::BlockValue};

pub(super) struct Recurrence {
    initial: usize,
    scale: usize,
    bias: usize,
    outputs: Box<[u16]>,
}

impl Recurrence {
    pub(super) fn compile(nodes: &[Op], uses: &[usize], order: &[u16]) -> Option<(Self, usize)> {
        let pair = |s: &[u16]| {
            let product = usize::from(*s.first()?);
            let dest = *s.get(1)?;
            let (Op::Mul(prev, scale), Op::Add(term, bias)) =
                (nodes[product], nodes[usize::from(dest)])
            else {
                return None;
            };
            (term == product && uses[product] == 1 && scale != bias)
                .then_some((prev, scale, bias, dest))
        };
        let (initial, scale, bias, first) = pair(order)?;
        if initial == scale || initial == bias {
            return None;
        }
        let mut tail = first;
        let mut count = 1;
        for s in order[2..].chunks_exact(2) {
            let Some((prev, factor, offset, next)) = pair(s) else {
                break;
            };
            if prev != usize::from(tail) || uses[prev] != 1 || factor != scale || offset != bias {
                break;
            }
            tail = next;
            count += 1;
        }
        if count < 4 {
            return None;
        }
        let outputs = order[1..2 * count].iter().step_by(2).copied().collect();
        Some((
            Self {
                initial,
                scale,
                bias,
                outputs,
            },
            2 * count,
        ))
    }

    pub(super) fn output(&self) -> usize {
        usize::from(*self.outputs.last().unwrap())
    }

    #[cfg(feature = "simd")]
    pub(super) fn compact(&self, slots: &[usize]) -> Self {
        Self {
            initial: slots[self.initial],
            scale: slots[self.scale],
            bias: slots[self.bias],
            outputs: self
                .outputs
                .iter()
                .map(|&i| slots[usize::from(i)] as u16)
                .collect(),
        }
    }
    pub(super) fn forward<T: BlockValue>(&self, p: &mut [T]) -> bool {
        let (mut x, scale, bias) = (p[self.initial], p[self.scale], p[self.bias]);
        for &out in &self.outputs {
            x = x * scale + bias;
            p[usize::from(out)] = x;
        }
        x.finite()
    }
    pub(super) fn reverse<T: BlockValue>(&self, p: &[T], adj: &mut [T]) -> Result<(), usize> {
        let scale = p[self.scale];
        let mut seed = adj[self.output()];
        let (mut scale_bar, mut bias_bar) = (adj[self.scale], adj[self.bias]);
        for (i, &out) in self.outputs.iter().enumerate().rev() {
            if !seed.finite() {
                return Err(usize::from(out));
            }
            let previous = if i == 0 {
                self.initial
            } else {
                usize::from(self.outputs[i - 1])
            };
            bias_bar = bias_bar + seed;
            scale_bar = scale_bar + seed * p[previous];
            seed = T::ZERO + seed * scale;
        }
        adj[self.initial] = adj[self.initial] + seed;
        adj[self.scale] = scale_bar;
        adj[self.bias] = bias_bar;
        Ok(())
    }
}
